#!/usr/bin/env python3
"""
Run llm-judge-exams.py across all [FINAL]*.json exams and LLM types, in parallel.

Examples:
  python run_llm_judge.py --skip_existing true
  python run_llm_judge.py --final_dir output/final --responses_dir output/llm-judge-responses
"""

from pathlib import Path
import subprocess
import collections
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import fire
from tqdm import tqdm
from dotenv import load_dotenv, dotenv_values
import huggingface_hub
from . import llmconfig
from . import hf_load


def dispatch(cmd: list[str]) -> tuple[int, str, str]:
    try:
        # check=False to avoid raising exceptions on non-zero exit codes
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", f"Exception running {cmd}: {e}"

Job = collections.namedtuple("Command", ["cmd", "exam_name", "llm"])

def run(
    skip_existing: bool = False,
    quarchds_dir: str | None = None,
    output_dir: str = "output",
    judge_module: str = "quarch_eval.llm_judge_exams",
    judge_llm_name: str = "qwen3-vl-235b-a22b",
    k_student: int = 1,
    k_judge: int = 1,
    qid_subset: str | None = None,
    llm_subset: str | None = None,
    writelogs: bool = True,
    python_executable: str = sys.executable,
    joblimit: int | None = None,
    smoothing: float = 0.8,
    llm_config: str | None = None,
    env_file: str | None = None,
):
    """Run LLM-as-judge evaluations on free-form questions.
    
    Args:
        skip_existing: If True, don't re-run when the output file already exists.
        quarchds_dir: Path to QuArch dataset free response (frqs) subdirectory (e.g., quarchds/frqs),
          containing questions/ and answers/ subdirs. If None, defaults to downloading from HuggingFace
        output_dir: Path to output directory where results are written. Defaults to "output"
        judge_module: Optional module path to llm_judge_exams.py 
        python_executable: Optional, Python interpreter to use for subprocesses.
        qid_subset: Optional path to a text file with one question ID per line
        llm_subset: Optional comma-separated string of LLM model names to run.
        writelogs: Whether to write log files to output/logs/ (--writelogs False to disable)
        judge_llm_name: LLM name to use for the judge LLM.
        k_student: Number of student LLM responses to generate per question.
        k_judge: Number of judge LLM evaluations to generate per student response.
        joblimit: Maximum number of parallel subprocesses to run. If None, defaults to number of jobs.
        smoothing: Smoothing factor for tqdm progress bar.
        llm_config: Path to a custom LLM configuration YAML file.
        env_file: Path to a .env file to load environment variables from.

    Returns:
        Process exit code (0 on success, non-zero if any subprocess failed or setup error).
    """
    if env_file:
        load_dotenv(env_file)

    # first assert the default is False for provider_is_safe_for_llm_as_a_judge, to 
    # prevent accidental use of unsafe APIs for judging
    assert not llmconfig.get_llm_config().provider_is_safe_for_llm_as_a_judge
    
    llm_configs = llmconfig.load_llm_config(Path(llm_config) if llm_config else None)

    assert judge_llm_name in llm_configs, f"{judge_llm_name=} missing"
    if not llm_configs[judge_llm_name].provider_is_safe_for_llm_as_a_judge:
        error_msg = (
            f"Judge LLM {judge_llm_name} is not marked as approved for LLM-as-a-Judge. "
            "If the API you intend to use guarantees it does not train on prompts "
            "(e.g., Amazon AWS Bedrock) or if you are self-hosting a judge LLM, "
            "set provider_is_safe_for_llm_as_a_judge: true in llmconfig.yaml for only that LLM"
        )

        logger.error(error_msg)
        sys.exit(1)
    if quarchds_dir is not None:
        quarchds_dir: Path = Path(quarchds_dir)
    else:
        quarchds_dir = hf_load.get_quarchds_frq_dir(envfile_path=Path(env_file) if env_file else None)
    output_dir: Path = Path(output_dir)
    if not quarchds_dir.is_dir():
        logger.error("QuArch free-response question (frqs) dataset directory '{}' not found.", quarchds_dir)
        sys.exit(1)
    questions_dir = quarchds_dir / "questions"
    answers_dir = quarchds_dir / "answers"
    if not questions_dir.is_dir() or not answers_dir.is_dir():
        logger.error("questions/ and answers/ subdirectories not found in '{}'.", quarchds_dir)
        sys.exit(1)
    if not output_dir.is_dir():
        logger.error("Output directory '{}' not found.", output_dir)
        sys.exit(1)
    responses_path = output_dir / "llm-judge-responses"

    extra_args = []
    qid_subset_path = None
    if qid_subset is not None:
        qid_subset_path = Path(qid_subset)
        if not qid_subset_path.is_file():
            logger.error("Question ID subset file '{}' not found.", qid_subset_path)
            sys.exit(1)
        extra_args.extend(["--qid_subset", str(qid_subset_path)])
    if writelogs:
        extra_args.extend(["--writelogs", "True"])
        logname = "run_all_judges.log"
        logger.add(logname, rotation="5 MB")
        logger.info(f"::::: New run_all_judges run: {judge_llm_name=}, {k_student=}, {k_judge=}, {qid_subset=}, {llm_subset=}, {skip_existing=} {joblimit=}")
    else:
        extra_args.extend(["--writelogs", "False"])

    extra_args.extend(["--judge_llm_name", judge_llm_name])
    extra_args.extend(["--k_student", str(k_student)])
    extra_args.extend(["--k_judge", str(k_judge)])
    extra_args.extend(["--llm_config", str(llm_config)])

    # Ensure output directory exists
    responses_path.mkdir(parents=True, exist_ok=True)

    # If qid_subset is passed, we only want to run on the exams that are specified.
    allowed_exam_names = None
    if qid_subset_path:
        qid_data = Path(qid_subset_path).read_text()
        qids = [line.strip() for line in qid_data.splitlines() if line.strip()]
        # A qid is like `exam_name/problem/part`.
        # The part before the first slash is the exam name.
        allowed_exam_names = {qid.split("/")[0] for qid in qids}
        logger.info("Running only for exams present in {}: {}", qid_subset, allowed_exam_names)

    # grab all {exam_name}.json files in each subdir of quarchds_dir
    exam_files = list(sorted(questions_dir.glob("**/*.json")))
    if not exam_files:
        logger.warning("No exam files found in '{}'.", quarchds_dir)
        sys.exit(1)
    # sanity check that each question json has a corresponding encrypted answers file (ending in json.enc)
    for exam_path in exam_files:
        answer_path = answers_dir / exam_path.relative_to(questions_dir)
        answer_path_enc = answer_path.with_suffix(answer_path.suffix + ".enc")
        if not answer_path_enc.is_file():
            logger.error("Encrypted answer file '{}' not found for exam '{}'.", answer_path_enc, exam_path)
            sys.exit(1)

    jobs: list[Job] = []

    all_llms = llm_configs.keys()
    if llm_subset:
        subset_list = [llm.strip() for llm in llm_subset.split(',')]
        llms_to_run = []
        for llm in subset_list:
            if llm in all_llms:
                llms_to_run.append(llm)
            else:
                logger.warning("LLM '{}' from --llm_subset not found in llmconfig.py, skipping.", llm)
    else:
        llms_to_run = all_llms

    for exam_path in exam_files:
        exam_name = exam_path.stem

        if allowed_exam_names and exam_name not in allowed_exam_names:
            logger.debug("Skipping exam {} as it is not in the allowed_exam_names list.", exam_name)
            continue

        for student_llm_name in llms_to_run:
            out_file = responses_path / f"{exam_name}_{student_llm_name}_llm_judge_responses.json"

            if skip_existing and out_file.exists():
                logger.info("[SKIP] {} already exists.", out_file)
                continue

            logger.info("[RUN] {} with LLM: {}", exam_name, student_llm_name)
            cmd = [
                python_executable,
                "-m",
                judge_module,
                "--exam_name",
                exam_name,
                "--quarch_frqs_dir",
                str(quarchds_dir),
                "--output_dir",
                str(output_dir),
                "--student_llm_name",
                student_llm_name,
            ] + extra_args
            jobs.append(Job(cmd=cmd, exam_name=exam_name, llm=student_llm_name))

    if len(jobs) == 0:
        logger.info("No exams to process. Exiting.")
        return
    failed = 0
    if joblimit is None:
        joblimit = len(jobs)
    logger.info("run_all_judges: launching {} across {} processes", len(jobs), joblimit)

    with ThreadPoolExecutor(max_workers=joblimit) as ex:
        # futures = {ex.submit(p.communicate): p for p in procs}
        futures = {ex.submit(dispatch, job.cmd): job for job in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Running jobs", unit="job", smoothing=smoothing):
            job = futures[fut]
            rc, stdout, stderr = fut.result()
            if stdout:
                # print raw stdout
                print(stdout)
            if stderr:
                # logger.error("[stderr] {}", stderr.strip())
                print(stderr)
            if rc != 0:
                failed += 1
                logger.error("Subprocess {} exited with code {}", job, rc)

    if failed:
        logger.error("Completed with {} failed subprocess(es).", failed)
        sys.exit(1)

    logger.success("All exams have been processed for all LLMs.")

if __name__ == "__main__":
    fire.Fire(run)
