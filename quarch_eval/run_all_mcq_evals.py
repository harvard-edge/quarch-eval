#!/usr/bin/env python3
"""
Run llm_mcq_eval.py across all or a subset of LLMs, in parallel.

Examples:
  python run_all_mcq_evals.py --llm_subset gpt-4o,claude-sonnet-4
  python run_all_mcq_evals.py --k 3
"""

from pathlib import Path
import subprocess
import collections
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import fire
from tqdm import tqdm
from . import llmconfig
from dotenv import load_dotenv

def dispatch(cmd: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", f"Exception running {cmd}: {e}"

Job = collections.namedtuple("Command", ["cmd", "llm"])

def run(
    llm_subset: str | None = None,
    output_dir: str = "output/mcq-evals",
    eval_module: str = "quarch_eval.llm_mcq_eval",
    k: int = 1,
    num_workers: int = 4,
    joblimit: int | None = None,
    python_executable: str = sys.executable,
    llm_config: str | None = None,
    env_file: str | None = None,
):
    """Run evaluations on multiple-choice questions.
    
    Args:
        llm_subset: Optional comma-separated string of LLM model names to run.
        output_dir: Where to write the evaluation results.
        eval_module: The Python module to invoke for each LLM.
        k: The target number of attempts for pass@k evaluation.
        num_workers: The number of parallel threads for the inner script (question processing).
        joblimit: The number of parallel processes for the driver script (LLM processing).
        python_executable: Python interpreter to use for subprocesses.
        llm_config: Path to a custom LLM configuration YAML file.
        env_file: Path to a .env file to load environment variables from.
    """
    if env_file:
        load_dotenv(env_file)
        
    llm_configs = llmconfig.load_llm_config(Path(llm_config) if llm_config else None)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_llms = llm_configs.keys()
    if llm_subset:
        subset_list = [llm.strip() for llm in llm_subset.split(',')]
        llms_to_run = [llm for llm in subset_list if llm in all_llms]
        skipped = set(subset_list) - set(llms_to_run)
        if skipped:
            logger.warning(f"LLMs from --llm_subset not found in llmconfig, skipping: {sorted(skipped)}")
    else:
        llms_to_run = list(all_llms)

    jobs: list[Job] = []
    for llm in llms_to_run:
        logger.info(f"[QUEUE] Evaluation for LLM: {llm} with target k={k}")
        cmd = [
            python_executable, "-m", eval_module, 
            "--llm_name", llm, 
            "--output_dir", str(output_path), 
            "-k", str(k),
            "--num_workers", str(num_workers),
        ]
        if env_file:
            cmd += ["--env_file", env_file]
        if llm_config:
            cmd += ["--llm_config", llm_config]
        jobs.append(Job(cmd=cmd, llm=llm))

    if not jobs:
        logger.info("No evaluations to run. Exiting.")
        return

    if joblimit is None:
        joblimit = os.cpu_count() or 1
    
    logger.info(f"Launching {len(jobs)} evaluation(s) across {min(joblimit, len(jobs))} parallel processes.")

    failed_jobs = []
    with ThreadPoolExecutor(max_workers=joblimit) as executor:
        futures = {executor.submit(dispatch, job.cmd): job for job in jobs}
        
        for future in tqdm(as_completed(futures), total=len(jobs), desc="Running Evaluations"):
            job = futures[future]
            try:
                rc, stdout, stderr = future.result()
                if rc != 0:
                    failed_jobs.append(job)
                    logger.error(f"Job for LLM '{job.llm}' failed with exit code {rc}.")
                    logger.error(f"  STDOUT:\n{stdout}")
                    logger.error(f"  STDERR:\n{stderr}")
            except Exception as e:
                failed_jobs.append(job)
                logger.error(f"Job for LLM '{job.llm}' threw an exception: {e}")

    if failed_jobs:
        logger.error(f"Completed with {len(failed_jobs)} failed evaluation(s): {[job.llm for job in failed_jobs]}")
        sys.exit(1)

    logger.success("All evaluations completed successfully.")

if __name__ == "__main__":
    fire.Fire(run)
