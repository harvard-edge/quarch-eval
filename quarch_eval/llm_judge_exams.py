import argparse
import json
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ml_collections.config_dict import ConfigDict

from dotenv import dotenv_values
from loguru import logger
import fire

from . import datatypes as dt
from . import llmconfig
from . import llm_infra
from . import majority_utils

# threading config
MAX_PARALLEL_PASS_WORKERS = 8
MAX_PARALLEL_QUESTION_WORKERS = 8

# --- Prompts ---
LLM_QUESTION_PROMPT = '''You are an expert computer architect taking an exam. You will be provided with a question and its context. Your task is to provide a clear, accurate, and well-reasoned answer to the question.

Please provide your answer in a structured format that clearly addresses the question. If the question involves calculations, show your work step-by-step. If it involves diagrams or tables, describe them clearly.

Remember to:
1. Read the question carefully and understand what is being asked
2. Use the provided context to inform your answer
3. Show your reasoning and work where appropriate
4. Be precise and accurate in your response
5. If you're unsure about something, acknowledge the uncertainty

Question Context:
{context}
{context_images_placeholder}

Question:
{question}

Please provide your answer:'''

LLM_JUDGE_PROMPT = '''You are an expert computer architect acting as an exam grader to evaluate the quality of an answer to a computer architecture question. You will be provided with:

1. The original question and context
2. The correct solution
3. A student's answer to the question

Your task is to carefully evaluate whether the student's answer is correct, partially correct, or incorrect by comparing it to the provided solution.

Evaluation criteria:
- CORRECT: The answer is accurate, complete, and demonstrates proper understanding
- PARTIALLY-CORRECT: The answer shows some understanding but has significant errors or is incomplete
- INCORRECT: The answer is fundamentally wrong or shows major misunderstandings

Consider:
- Mathematical accuracy
- Conceptual understanding
- Completeness of the response solely in relation to the question being asked
- Logical reasoning
- Whether the answer addresses what was actually asked

Be fair but rigorous in your evaluation. If you are unsure, err on the side of being more critical.

Question Context:
{context}
{context_images_placeholder}

Question:
{question}

Correct Solution:
{solution}
{solution_images_placeholder}

Student's Answer:
{student_answer}

Please evaluate the student's answer and provide your reasoning. At the end of your response, write exactly one of the following words in all caps on a new line: CORRECT, PARTIALLY-CORRECT, or INCORRECT. If you do not end your response with a new line with exactly one of these options, you will not be paid for your work.'''

# --- Utility Functions ---

def extract_judgment(text: str) -> str:
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line in ["CORRECT", "PARTIALLY-CORRECT", "INCORRECT"]:
            return line
    return "UNSURE"

def maybe_extract_reasoining_trace(text: str | None) -> str | None:
    """
    tested with gpt-oss-120b
    """
    if text is None:
        return None
    j = json.loads(text)
    try:
        choices = j.get("choices", [])
        if not choices:
            return None
        if len(choices) > 1:
            logger.warning("multiple choices found in reasoning trace extraction")
        message = choices[0].get("message", {})
        reasoning = message.get("reasoning", None)
        if reasoning is not None:
            return reasoning
        # it might also be under reasoning_content
        reasoning_content = message.get("reasoning_content", None)
        return reasoning_content
    except Exception:
        return None

def process_question(
    question_data: dict,
    quarch_ds_dir: Path,
    student_llm_name: str,
    judge_llm_name: str,
    exam_name: str,
    existing_question_responses: dt.QuestionResponse,
    n_retries: int,
    k_student: int,
    k_judge: int,
    llm_configs: ConfigDict
) -> dt.QuestionResponse | None:
    question_id = question_data["question_id"]
    assert question_id == existing_question_responses.question_id, f"mismatch {question_id=} {existing_question_responses.question_id=}" 

    student_llm_config = llm_configs[student_llm_name]
    judge_llm_config = llm_configs[judge_llm_name]

    # --- Get Student Answer ---
    logger.info(f"Processing {question_id=} with LLM: {student_llm_name=}")
    question_prompt = LLM_QUESTION_PROMPT.format(
        context=question_data.get("context", ""),
        question=question_data.get("question", ""),
        context_images_placeholder="\n".join(f"<CONTEXT_IMAGE_{{i+1}}>" for i in range(len(question_data.get("context_figures", [])))))

    # for readability, image paths in "context_figures" and "solution_figures" are formatted as 
    # [{exam_name}/{image_name}.png, ...] in each .json
    # however, depending on whether the caller of create_converse_messages() is a student or judge LLM, 
    # the parent image_dir is: quarch_ds_dir / {EITHER questions OR answers} / {exam_name} 
    # (judges get both context and solution images, students only get context images)

    student_image_paths = []
    for img_path_str in question_data.get("context_figures", []):
        img_path = Path(img_path_str)
        full_img_path = quarch_ds_dir / "questions" / exam_name / img_path.name
        student_image_paths.append(full_img_path)

    question_messages = llm_infra.create_converse_messages(
        text=question_prompt,
        image_paths=student_image_paths,
        llm_config=student_llm_config
    )

    def _student_closure(idx: int):
        """thread execution"""
        api_response = llm_infra.invoke_model(
            llm_name=student_llm_name,
            messages=question_messages,
            llm_config=student_llm_config,
            n_retries=n_retries,
        )
        return api_response

    if not question_messages:
        logger.error(f"Failed to construct prompt for question {question_id}")
        return None

    if len(existing_question_responses.responses) >= k_student:
        logger.info(f"All {k_student=} responses for question {question_id} already exist")
    else:
        # we might have more than k_student already, so guard with max
        student_requests_pending = max(0, k_student - len(existing_question_responses.responses))
        n_workers = min(student_requests_pending, MAX_PARALLEL_PASS_WORKERS)
        logger.info(f"dispatching {student_requests_pending} student requests for question {question_id}, {n_workers=} ")
        new_student_answers = []
        new_api_responses_rawstr = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_student_closure, i): i for i in range(student_requests_pending)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    api_response = fut.result()
                    student_answer_str = student_llm_config["response_parser_fn"](
                        api_response
                    )
                    new_student_answers.append(student_answer_str)
                    try:
                        new_api_responses_rawstr.append(api_response.model_dump_json())
                    except Exception:
                        logger.opt(exception=True).error(f"Failed to serialize raw response for student answer {i=} for {question_id=}")
                        new_api_responses_rawstr.append(None)
                    logger.info(f"Got additional student answer {i+1}/{student_requests_pending} (total {k_student=}) for {question_id=}")
                except Exception:
                    logger.opt(exception=True).error(f"Student answer {i=} failed for {question_id=}")

        for ans, resp_rawstr in zip(new_student_answers, new_api_responses_rawstr):
            if ans:
                qr = dt.Response(
                    student_response=ans,
                    student_apiresponse_raw=resp_rawstr,
                    student_reasoning_trace=maybe_extract_reasoining_trace(resp_rawstr),
                    judgments=None,
                    metrics=None
                )
                existing_question_responses.responses.append(qr)

    # --- Get Judge Verdicts ---
    def _judge_closure(response: dt.Response, response_idx: int) -> list[dt.Judgment]:
        """thread execution"""
        if response.judgments is None:
            existing_judgments_subset = []
        else:
            existing_judgments_subset = [j for j in response.judgments if j.judge_llm_name == judge_llm_name]
        if len(existing_judgments_subset) >= k_judge:
            logger.info(f"All {k_judge=} judgments for response {response_idx} already exist")
            return []
        elif majority_utils.majority_reached_for_planned_k(
            judgments=existing_judgments_subset,
            k_planned=k_judge,
            judge_llm_name=judge_llm_name,
            rebucket_partial_to_incorrect=False,
        ).reached:
            logger.info(f"Majority already reached for {response_idx} with planned {k_judge=} at {len(existing_judgments_subset)=}")
            return []

        judge_prompt = LLM_JUDGE_PROMPT.format(
            context=question_data.get("context", ""),
            question=question_data.get("question", ""),
            solution=question_data.get("solution", ""),
            student_answer=response.student_response,
            context_images_placeholder="\n".join(f"<CONTEXT_IMAGE_{{i+1}}>" for i in range(len(question_data.get("context_figures", [])))),
            solution_images_placeholder="\n".join(f"<SOLUTION_IMAGE_{{i+1}}>" for i in range(len(question_data.get("solution_figures", [])))))

        solution_image_paths = []
        for img_path_str in question_data.get("solution_figures", []):
            img_path = Path(img_path_str)
            # encrypted images are appended with .enc
            full_img_path = quarch_ds_dir / "answers" / exam_name / (img_path.name + ".enc")
            solution_image_paths.append(full_img_path)
        all_judge_images = student_image_paths + solution_image_paths

        judge_messages = llm_infra.create_converse_messages(
            text=judge_prompt,
            image_paths=all_judge_images,
            llm_config=judge_llm_config,
        )

        # TODO(mmaz) this sequentially invokes which makes k_judge>1 slow, parallelizing needed
        this_response_judge_requests_pending = max(0, k_judge - len(existing_judgments_subset))
        logger.info(f"dispatching {this_response_judge_requests_pending} judge requests for response {response_idx} of question {question_id=}")
        new_judgments_for_response = []
        for judge_ix in range(this_response_judge_requests_pending):
            judge_response = llm_infra.invoke_model(
                llm_name=judge_llm_name,
                messages=judge_messages,
                llm_config=judge_llm_config,
                n_retries=n_retries,
            )
            judge_text = judge_llm_config["response_parser_fn"](judge_response)
            if not judge_text:
                logger.error(f"Failed to get a valid judge response for question {question_id}")
                continue

            judgment = dt.Judgment(
                judge_llm_name=judge_llm_name,
                judge_instruction_template=LLM_JUDGE_PROMPT,
                judge_response=judge_text,
                judge_reasoning_trace=None,
                judge_generation_config=judge_llm_config.get("generation_config"),
                judgment=extract_judgment(judge_text),
            )
            new_judgments_for_response.append(judgment)
            logger.info(f"Got {judge_ix=}/{k_judge=} verdict for response {response_idx} of question {question_id=}: {judgment.judgment}")
            if majority_utils.majority_reached_for_planned_k(
                judgments=existing_judgments_subset + new_judgments_for_response,
                k_planned=k_judge,
                judge_llm_name=judge_llm_name,
                rebucket_partial_to_incorrect=False,
            ).reached:
                logger.info(f"Majority reached for response {response_idx} with planned {k_judge=} at {len(existing_judgments_subset) + len(new_judgments_for_response)=}")
                break  # stop early if majority reached
        return new_judgments_for_response

    # process judgments for each response in parallel
    judge_requests_for_all_responses_pending = 0
    for r in existing_question_responses.responses:
        if r.judgments is None:
            r.judgments = []
        existing_judgments_subset = [j for j in r.judgments if j.judge_llm_name == judge_llm_name]
        # we might have more than k_judge already, so guard with max
        judge_requests_for_all_responses_pending += max(0, k_judge - len(existing_judgments_subset))

    if judge_requests_for_all_responses_pending == 0:
        logger.info(
            f"All {k_judge=} judgments for all responses for question {question_id} already exist"
        )
    else:
        n_workers = min(judge_requests_for_all_responses_pending, MAX_PARALLEL_PASS_WORKERS)
        logger.info(
            f"dispatching {judge_requests_for_all_responses_pending} judge requests for question {question_id}, {n_workers=} "
        )
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_judge_closure, r, idx): (r, idx)
                for idx, r in enumerate(existing_question_responses.responses)
            }
            for fut in as_completed(futures):
                r, idx = futures[fut]
                try:
                    new_judgments = fut.result()
                    if new_judgments:
                        if r.judgments is None:
                            r.judgments = []
                        r.judgments.extend(new_judgments)
                    logger.info(
                        f"Got {len(new_judgments)} judge verdicts for response {idx} of question {question_id=}"
                    )
                except Exception:
                    logger.opt(exception=True).error(
                        f"Judge verdict failed for response {idx} of question {question_id=}"
                    )
    return existing_question_responses

# --- Main Logic ---
def main(
    exam_name: str,
    quarch_frqs_dir: str,
    output_dir: str = None,
    student_llm_name: str = "llama3-2-1b",
    judge_llm_name: str = "claude-3-7-thinking",
    k_student: int = 1,
    k_judge: int = 1,
    n_retries: int = 3,
    qid_subset: str | None = None,
    writelogs: bool = True,
    llm_config: str | None = None,
):
    """
    This module is invoked by run_all_judges.py in parallel for each exam and student LLM
    Args:
        exam_name: Name of the exam to process.
        quarch_frqs_dir: Path to QuArch Free Response QA dataset directory. 
        output_dir: Directory to save output files.
        student_llm_name: LLM model name to use for student answers.
        judge_llm_name: LLM model name to use for judging answers.
        k_student: Number of student responses to generate per question.
        k_judge: Number of judge evaluations to generate per student response.
        n_retries: Number of retries for LLM calls.
        qid_subset: Optional filename containing a subset of question IDs to process.
        writelogs: Whether to write logs to a file in output_dir/logs/ (default: enabled)
        llm_config: Optional path to YAML configuration file conforming to llmconfig.py
    """

    output_dir : Path = Path(output_dir) if output_dir is not None else Path("output")
    assert output_dir.is_dir(), f"Output directory not found: {output_dir}"

    responses_dir = output_dir / "llm-judge-responses"
    os.makedirs(responses_dir, exist_ok=True)

    if writelogs:
        logger.remove()
        logger.add(sys.stderr, level="INFO", colorize=True)
        logs_dir = output_dir / "logs"
        os.makedirs(logs_dir, exist_ok=True)
        logfile = logs_dir / f"llmjudge_{exam_name}_{student_llm_name}.log"
        logger.add(str(logfile), rotation="1 MB")

    quarch_frqs_dir : Path = Path(quarch_frqs_dir)
    questions_file = quarch_frqs_dir / "questions" / exam_name / f"{exam_name}.json"
    answers_file = quarch_frqs_dir / "answers" / exam_name / f"{exam_name}.json.enc"
    for f in [questions_file, answers_file]:
        if not f.is_file():
            logger.error(f"Required file not found: {f}")
            sys.exit(1)

    questions_data = json.loads(questions_file.read_text(encoding='utf-8'))
    answers_data_enc = answers_file.read_bytes()
    fernet = llm_infra.load_fernet()
    answers_data_json = json.loads(fernet.decrypt(answers_data_enc).decode('utf-8'))
    assert len(questions_data) == len(answers_data_json), "Mismatch in number of questions and answers"
    assert set(q["question_id"] for q in questions_data) == set(
        a["question_id"] for a in answers_data_json
    ), "Mismatch in question IDs between questions and answers"
    # merge questions and answers on shared key "question_id"
    answers_by_id = {a["question_id"]: a for a in answers_data_json}
    exam_data = [
        (q | answers_by_id[q["question_id"]])
        for q in questions_data
    ]

    llm_configs = llmconfig.load_llm_config(Path(llm_config) if llm_config else None)
    student_llm_config = llm_configs[student_llm_name]
    judge_llm_config = llm_configs[judge_llm_name]
    max_images = student_llm_config.get('max_input_images', 0)
    assert max_images >= 0, f"invalid config {max_images=}"

    if max_images == 0:
        questions_to_process = [q for q in exam_data if not q.get("context_figures")]
    elif max_images < float('inf'):
        questions_to_process = [q for q in exam_data if len(q.get("context_figures", [])) <= max_images]
    else: # max_images == float('inf'), no filtering needed
        questions_to_process = exam_data

    logger.info(f"Found {len(questions_to_process)} questions that passed model constraints for LLM '{student_llm_name}'.")

    output_file = responses_dir / f"{exam_name}_{student_llm_name}_llm_judge_responses.json"

    if os.path.exists(output_file):
        logger.info(f"Output file already exists: {output_file}")
        try:
            exam_responses = dt.ExamResponses.model_validate_json(Path(output_file).read_text(encoding='utf-8'))
            assert exam_responses.student_llm_name == student_llm_name, "LLM name mismatch in existing results"
            existing_results = exam_responses.question_responses
        except Exception:
            logger.opt(exception=True).error(f"Error reading existing results from {output_file}")
            sys.exit(1)
    else:
        existing_results = []

    # this is a simple sanity check against duplicate question IDs in existing results;
    # the worker thread is responsible for satisfying k_student and k_judge requests per question_id or early exiting
    existing_qids = [r.question_id for r in existing_results]
    if len(existing_qids) != len(set(existing_qids)):
        logger.error(f"Duplicate question IDs found in existing results in {output_file}")
        sys.exit(1)

    if qid_subset:
        qid_subset_f = Path(qid_subset)
        assert qid_subset_f.exists(), f"Question ID subset file not found: {qid_subset_f}"
        selected_qids = {line.strip() for line in qid_subset_f.read_text(encoding='utf-8').splitlines() if line.strip()}
        logger.info(f"Filtering to {len(selected_qids)} question IDs from {qid_subset_f}")
        questions_to_process = [q for q in questions_to_process if q["question_id"] in selected_qids]

    if not questions_to_process:
        logger.info("All questions have already been processed.")
        return

    logger.info(f"{len(questions_to_process)} questions to process for {exam_name=}.")

    n_workers = min(len(questions_to_process), MAX_PARALLEL_QUESTION_WORKERS)
    logger.info(f"Processing questions with {n_workers} parallel workers.")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for question in questions_to_process:
            existing_qr : dt.QuestionResponse | None = None
            for er in existing_results:
                if er.question_id == question["question_id"]:
                    existing_qr = er
                    break
            if existing_qr is None:
                existing_qr = dt.QuestionResponse(question_id=question["question_id"], responses=[])
            futures[pool.submit(
                process_question,
                question,
                quarch_frqs_dir,
                student_llm_name,
                judge_llm_name,
                exam_name,
                existing_qr,
                n_retries,
                k_student,
                k_judge,
                llm_configs
            )] = question["question_id"]

        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                res = fut.result()
                if res is None:
                    logger.error(f"Processing returned None for question {qid}")
                    continue
                # update or append the new QuestionResponse to existing_results
                for eix, er in enumerate(existing_results):
                    if er.question_id == res.question_id:
                        existing_results[eix] = res
                        break
                else:
                    existing_results.append(res)
                existing_results.sort(key=lambda x: x.question_id)
                logger.info(f"Completed processing for question {qid}")
                # persist intermediate results
                Path(output_file).write_text(
                    dt.ExamResponses(
                        exam_id=exam_name,
                        student_llm_name=student_llm_name,
                        student_instruction_template=LLM_QUESTION_PROMPT,
                        question_responses=existing_results,
                    ).model_dump_json(indent=2, exclude_none=True),
                    encoding="utf-8",
                )
                logger.info(f"\nIntermediate results saved to: {output_file}")
            except Exception:
                logger.opt(exception=True).error(f"Processing failed for question {qid}")

    logger.info(f"Finished {exam_name}, {len(existing_results)=} questions saved to: {output_file}")


if __name__ == '__main__':
    fire.Fire(main)
