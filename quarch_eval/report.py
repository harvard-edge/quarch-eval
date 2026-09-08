import sys
from pathlib import Path
from typing import Any
import json

import fire
import pandas as pd
from loguru import logger

from . import datatypes as dt
from . import majority_utils
from . import hf_load


def load_frq_question_data(frq_dir: Path) -> dict:
    qid2data = {}
    frq_questions = frq_dir / "questions"
    for exam_dir in frq_questions.iterdir():
        # questions only:
        exam_file = exam_dir / f"{exam_dir.name}.json"
        exam_data = json.loads(exam_file.read_text())
        for question_data in exam_data:
            qid = question_data["question_id"]
            qid2data[qid] = question_data
    logger.info(f"Total frq qids: {len(qid2data)}")
    return qid2data


def load_skill_mapping(env_file: str | None, hf_dataset_override_yml: Path | None) -> dict[str, str]:
    envfile_path = Path(env_file) if env_file else None

    frq_dir = hf_load.get_quarchds_frq_dir(
        envfile_path=envfile_path,
        quarch_hf_config_override_yml=hf_dataset_override_yml,
    )
    frq_qid2data = load_frq_question_data(frq_dir)
    mcqs = hf_load.get_quarchds_mcqs(
        as_hf_ds=False,
        envfile_path=envfile_path,
        quarch_hf_config_override_yml=hf_dataset_override_yml,
    )

    skill_map: dict[str, str] = {}
    for qid, question_data in frq_qid2data.items():
        primary_skill = question_data.get('human_labeled_skills', {}).get('primary_skill')
        if primary_skill:
            skill_map[str(qid)] = primary_skill
    for mcq in mcqs:
        primary_skill = mcq.get('human_labeled_skills', {}).get('primary_skill')
        if primary_skill:
            skill_map[str(mcq['question_id'])] = primary_skill
    return skill_map


def load_topic_mapping(topics_file: str | None) -> dict[str, str]:
    if not topics_file:
        return {}
    path = Path(topics_file)
    if not path.exists():
        logger.error(f"Error: topics file not found at {path}")
        return {}

    try:
        df = pd.read_json(path, lines=True)
        if 'question_id' not in df.columns or 'topic' not in df.columns:
            logger.error(f"Error reading topics file {path}: missing question_id or topic field")
            return {}

        df = df[['question_id', 'topic']].dropna()
        return {
            str(qid).strip(): str(topic).strip()
            for qid, topic in zip(df['question_id'], df['topic'])
            if str(qid).strip() and str(topic).strip()
        }
    except Exception as e:
        logger.error(f"Error reading topics file {path}: {e}")
        return {}


def load_qid_subset(qid_subset: str | None) -> set[str] | None:
    if not qid_subset:
        return None
    path = Path(qid_subset)
    if not path.exists():
        logger.error(f"Error: QID subset file not found at {path}")
        sys.exit(1)
    return {line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def discover_models(output_path: Path, llms: list[str] | str | None) -> list[str]:
    if isinstance(llms, list):
        return [str(llm) for llm in llms]

    if llms is None or isinstance(llms, str) and llms.lower() == "all":
        models: set[str] = set()
        mcq_dir = output_path / 'mcq-evals'
        if mcq_dir.exists():
            models.update(p.stem for p in mcq_dir.glob('*.json'))

        frq_dir = output_path / 'llm-judge-responses'
        if frq_dir.exists():
            for p in frq_dir.glob('*_llm_judge_responses.json'):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        models.add(json.load(f)['student_llm_name'])
                except Exception as e:
                    logger.error(f"Error reading student_llm_name from {p}: {e}")
        return sorted(models)

    return [llms]


def calculate_mcq_question_accuracies(
    mcq_file: Path,
    selected_qids: set[str] | None,
    k: int | None,
    skill_map: dict[str, str],
    topic_map: dict[str, str],
) -> pd.DataFrame:
    if not mcq_file.exists():
        return pd.DataFrame(columns=['qid', 'accuracy', 'skill', 'topic'])

    try:
        with open(mcq_file, 'r', encoding='utf-8') as f:
            data = dt.MCQEvaluation.model_validate_json(f.read())

        records = []
        for group in data.results:
            qid = str(group.question_id)
            if selected_qids and qid not in selected_qids:
                continue

            attempts = group.attempts[:k] if k is not None else group.attempts
            if not attempts:
                logger.info(f"No attempts for MCQ question {qid}.")
                continue

            correct = sum(1 for a in attempts if a.is_correct)
            records.append({
                'qid': qid,
                'accuracy': correct / len(attempts),
                'skill': skill_map.get(qid),
                'topic': topic_map.get(qid),
            })
        return pd.DataFrame.from_records(records)
    except Exception as e:
        logger.error(f"Error processing MCQ results from {mcq_file}: {e}")
        return pd.DataFrame(columns=['qid', 'accuracy', 'skill', 'topic'])


def calculate_frq_question_accuracies(
    frq_dir: Path,
    model_name: str,
    selected_qids: set[str] | None,
    k: int | None,
    j: int | None,
    judge_llm_name: str | None,
    skill_map: dict[str, str],
    topic_map: dict[str, str],
) -> pd.DataFrame:
    if not frq_dir.exists():
        return pd.DataFrame(columns=['qid', 'accuracy', 'skill', 'topic'])

    pattern = f"_{model_name}_llm_judge_responses.json"
    records = []
    for file in frq_dir.glob(f"*{pattern}"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = dt.ExamResponses.model_validate_json(f.read())
            for qr in data.question_responses:
                qid = str(qr.question_id)
                if selected_qids and qid not in selected_qids:
                    continue

                responses = qr.responses[:k] if k is not None else qr.responses
                evaluated = 0
                correct = 0
                for i, resp in enumerate(responses):
                    if not resp.judgments:
                        logger.info(f"No judgments for response {i} in question {qid}.")
                        continue

                    is_correct = majority_utils.majority_vote_correct(resp, j, judge_llm_name)
                    if is_correct is None:
                        logger.info(f"No majority judgment for response {i} in question {qid}.")
                        continue

                    evaluated += 1
                    if is_correct:
                        correct += 1

                if evaluated:
                    records.append({
                        'qid': qid,
                        'accuracy': correct / evaluated,
                        'skill': skill_map.get(qid),
                        'topic': topic_map.get(qid),
                    })
                else:
                    logger.info(f"No evaluated responses for FRQ question {qid} in {file.name}.")
        except Exception as e:
            logger.error(f"Error processing FRQ file {file}: {e}")
    return pd.DataFrame.from_records(records, columns=['qid', 'accuracy', 'skill', 'topic'])


def format_percent(value: float | None) -> str:
    return f"{value * 100:6.2f}%" if value is not None else "N/A"


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    cell_lines: list[list[list[str]]] = [
        [str(cell).split('\n') for cell in row]
        for row in rows
    ]

    widths = [len(header) for header in headers]
    for row in cell_lines:
        for index, lines in enumerate(row):
            widths[index] = max(widths[index], max((len(line) for line in lines), default=0))

    lines: list[str] = []
    lines.append(' | '.join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    lines.append('-+-'.join('-' * widths[i] for i in range(len(headers))))
    for row in cell_lines:
        height = max(len(lines) for lines in row)
        for line_idx in range(height):
            row_cells = [
                (row[col][line_idx] if line_idx < len(row[col]) else '').ljust(widths[col])
                for col in range(len(headers))
            ]
            lines.append(' | '.join(row_cells))
    return '\n'.join(lines)

def run(
    output_dir: str = 'output',
    qid_subset: str | None = None,
    judge_llm_name: str | None = None,
    k: int | None = None,
    j: int | None = None,
    llms: list[str] | str | None = None,
    env_file: str | None = None,
    topics_file: str | None = None,
    hf_dataset_config_override: str | None = None,
):
    """
    Generates an accuracy report for one or more student LLM models.

    Options:
    - qid_subset: Path to a file containing newline-delimited question IDs to filter results.
    - judge_llm_name: Filter FRQ judgments to only consider those from this specific judge LLM. If unset, all judge LLMs are aggregated
    - k: Calculate accuracy based on the first 'k' student attempts/responses.
    - j: For LLM-as-a-judge questions, base correctness on the first 'j' judge calls.
    - llms: Optional explicit list of LLMs to report on, or "--llms=all" will return all models found in the output directory
            to encode a list via bash: --llms='["gpt-oss-120b", "claude-sonnet-4"]'
    - env_file: Optional path to a .env file (for HF_TOKEN) used to load groundtruth questions from HuggingFace,
      from which skill labels are derived.
    - topics_file: Optional JSONL file with question topics.
    - hf_dataset_config_override: YAML path to override the default HuggingFace location (ignore, only for internal testing)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.error(f"Error: output directory not found at {output_path}")
        return

    if hf_dataset_config_override is not None:
        hf_dataset_config_override: Path = Path(hf_dataset_config_override)
        logger.info(f"Using HuggingFace dataset config override from {hf_dataset_config_override}")
    selected_qids = load_qid_subset(qid_subset)
    skill_map = load_skill_mapping(env_file, hf_dataset_override_yml=hf_dataset_config_override)
    topic_map = load_topic_mapping(topics_file)
    models = discover_models(output_path, llms)
    if not models:
        logger.error('No models found to report on.')
        return

    overall_rows = []
    skill_rows: dict[str, dict[str, float]] = {}
    topic_rows: dict[str, dict[str, float]] = {}
    skill_count_rows: dict[str, dict[str, int]] = {}
    topic_count_rows: dict[str, dict[str, int]] = {}

    all_skills: set[str] = set()
    all_topics: set[str] = set()
    
    for model in models:
        mcq_df = calculate_mcq_question_accuracies(
            output_path / 'mcq-evals' / f'{model}.json',
            selected_qids,
            k,
            skill_map,
            topic_map,
        )
        frq_df = calculate_frq_question_accuracies(
            output_path / 'llm-judge-responses',
            model,
            selected_qids,
            k,
            j,
            judge_llm_name,
            skill_map,
            topic_map,
        )

        mcq_overall = mcq_df['accuracy'].mean() if not mcq_df.empty else None
        frq_overall = frq_df['accuracy'].mean() if not frq_df.empty else None
        mcq_count = len(mcq_df)
        frq_count = len(frq_df)

        overall_rows.append([
            model,
            f"{mcq_count}\n{format_percent(mcq_overall)}",
            f"{frq_count}\n{format_percent(frq_overall)}",
        ])

        merged_df = pd.concat([mcq_df, frq_df], ignore_index=True)
        if not merged_df.empty:
            skill_df = merged_df[merged_df['skill'].notna()]
            topic_df = merged_df[merged_df['topic'].notna()]

            if not skill_df.empty:
                skill_rows[model] = skill_df.groupby('skill')['accuracy'].mean().to_dict()
                skill_count_rows[model] = skill_df.groupby('skill')['qid'].nunique().to_dict()
                all_skills.update(skill_rows[model].keys())
                all_skills.update(skill_count_rows[model].keys())
            else:
                skill_count_rows[model] = {}

            if not topic_df.empty:
                topic_rows[model] = topic_df.groupby('topic')['accuracy'].mean().to_dict()
                topic_count_rows[model] = topic_df.groupby('topic')['qid'].nunique().to_dict()
                all_topics.update(topic_rows[model].keys())
                all_topics.update(topic_count_rows[model].keys())
            else:
                topic_count_rows[model] = {}

    logger.info(
        '\n' + '=' * 80
        + '\nAccuracy Report\n' + '=' * 80
        + '\n' + format_table(['model', 'mcq', 'frq'], overall_rows)
        + '\n' + '=' * 80
    )

    if skill_map and skill_rows:
        # Preferred skill ordering, then any remaining skills alphabetically
        preferred_order = ["recall", "analyze", "design", "implement"]
        remaining = sorted(s for s in all_skills if s not in preferred_order)
        sorted_skills = [s for s in preferred_order if s in all_skills] + remaining
        headers = ['model'] + sorted_skills + ['Reasoning Overall']
        rows = []
        for model in models:
            row = [model]
            # show counts and accuracies per skill
            for skill in sorted_skills:
                count = skill_count_rows.get(model, {}).get(skill, 0)
                acc = skill_rows.get(model, {}).get(skill)
                row.append(f"{count}\n{format_percent(acc)}")
            # Compute weighted Reasoning Overall over Analyze, Design, Implement
            reasoning_skills = ["analyze", "design", "implement"]
            num = 0.0
            denom = 0
            for s in reasoning_skills:
                acc = skill_rows.get(model, {}).get(s)
                cnt = skill_count_rows.get(model, {}).get(s, 0)
                if acc is not None and cnt > 0:
                    num += acc * cnt
                    denom += cnt
            reasoning_score = (num / denom) if denom > 0 else None
            row.append(f"{denom}\n{format_percent(reasoning_score)}")
            rows.append(row)
        logger.info(
            '\nAccuracy by skill:\n' + format_table(headers, rows)
            + '\n' + '=' * 80
        )

    if topic_map and topic_rows:
        sorted_topics = sorted(all_topics)
        headers = ['model'] + sorted_topics
        rows = []
        for model in models:
            row = [model]
            for topic in sorted_topics:
                count = topic_count_rows.get(model, {}).get(topic, 0)
                acc = topic_rows.get(model, {}).get(topic)
                row.append(f"{count}\n{format_percent(acc)}")
            rows.append(row)
        logger.info('\nAccuracy by topic:')
        logger.info('\n' + format_table(headers, rows))
        logger.info('=' * 80)

if __name__ == '__main__':
    fire.Fire(run)
