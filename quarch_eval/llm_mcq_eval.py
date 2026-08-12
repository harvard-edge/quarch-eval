import argparse
import json
import os
import re
import sys
from pathlib import Path
from dotenv import dotenv_values
from typing import Literal, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import huggingface_hub
import datasets
import jsonlines
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from .datatypes import MCQResult, MCQResultGroup, MCQEvaluation
from . import llm_infra
from . import llmconfig
from . import hf_load

# import code

MCQ_PROMPT = '''You are an expert computer architect solving multiple choice questions. Please read the following question and select the best answer from the choices provided.

Question: {question}

Choices:
{choices}

Please conclude your response with a JSON object containing your final answer. The JSON object must match this schema: {{"final_answer": "<A, B, C, or D>"}}. 
'''

class MCQChoice(BaseModel):
    final_answer: Literal['A', 'B', 'C', 'D'] = Field(description="The final selected answer for the multiple-choice question.")

def parse_structured_answer(response_text: str) -> str:
    """Finds and parses a JSON object from the LLM's response to extract the final answer."""

    last_brace_index = response_text.rfind('{')
    if last_brace_index == -1:
        return ""
        
    json_text_to_try = response_text[last_brace_index:]
    
    match = re.search(r"(\{.*?\})", json_text_to_try, re.DOTALL)
    if not match:
        return ""
        
    json_text = match.group(1).strip()
    
    try:
        data = json.loads(json_text)
        choice = MCQChoice.model_validate(data)
        return choice.final_answer
    except (json.JSONDecodeError, ValidationError):
        return ""


def fix_boxed_answer(response_text: str) -> str:
    """Looks for any letters that are boxed using the '\boxed{}' latex syntax."""
    
    match = re.search(r"\\boxed{([A-D])}", response_text)
    if match:
        return match.group(1)
    return ""

def fix_match_option_text(response_text: str, options: List[str]) -> str:
    """
    Checks if the json entry "final_answer" entry matches any of the 4 options of the question.
    """

    last_brace_index = response_text.rfind('{')
    if last_brace_index == -1:
        return ""
        
    json_text_to_try = response_text[last_brace_index:]
    
    match = re.search(r"(\{.*?\})", json_text_to_try, re.DOTALL)
    if not match:
        return ""
        
    json_text = match.group(1).strip()

    try:
        data = json.loads(json_text)
        if "final_answer" not in data:
            return ""
        
        llm_answer_text = data["final_answer"]
        
        matching_options = []
        for i, option_text in enumerate(options):
            if str(llm_answer_text).strip() == option_text.strip():
                matching_options.append(chr(ord('A') + i))
        
        if len(matching_options) == 1:
            return matching_options[0]
            
    except (json.JSONDecodeError, ValidationError, TypeError):
        return ""
        
    return ""

def parse_and_clean_answer(response_text: str, options: List[str]) -> str:
    """
    Tries to parse the structured answer, and if it fails, applies a series of cleaning heuristics.
    """
    if (response_text == ""):
        logger.warning(f"No response text provided")
        return ""

    answer = parse_structured_answer(response_text)
    if answer:
        return answer

    answer = fix_boxed_answer(response_text)
    if answer:
        return answer

    answer = fix_match_option_text(response_text, options)
    if answer:
        return answer
        
    logger.warning(f"Could not determine answer from response: {response_text}")
    return ""

def process_item(item: Dict[str, Any], llm_name: str, llm_config: Dict[str, Any], num_attempts: int) -> MCQResultGroup:
    """Processes a single dataset item, running the LLM `num_attempts` times."""
    question_id = item['question_id']
    question = item['question']
    options = item['options']
    correct_answer_letter = item['answer']

    formatted_choices = "\n".join([f"{chr(ord('A') + j)}) {choice}" for j, choice in enumerate(options)])
    prompt = MCQ_PROMPT.format(question=question, choices=formatted_choices)

    messages = llm_infra.create_converse_messages(
        text=prompt, image_paths=[], llm_config=llm_config
    )
    if not messages:
        logger.error(f"Failed to create messages for question {question_id}.")
        return MCQResultGroup(question_id=question_id, attempts=[])

    attempts = []
    for _ in range(num_attempts):
        response = llm_infra.invoke_model(llm_name, messages, llm_config)
        response_text = llm_config["response_parser_fn"](response)
        if (response_text == ""):
            continue
        llm_answer_letter = parse_and_clean_answer(response_text, options)

        is_correct = llm_answer_letter == correct_answer_letter if llm_answer_letter else False

        attempts.append(MCQResult(
            is_correct=is_correct,
            llm_answer=llm_answer_letter,
            llm_response_text=response_text
        ))
    
    return MCQResultGroup(
        question_id=question_id,
        attempts=attempts
    )

def save_results(intermediate_results: Dict[str, MCQResultGroup], output_filename: Path, llm_name: str) -> MCQEvaluation:
    result_groups = list(intermediate_results.values())
    total_correct_questions = sum(1 for group in result_groups if any(attempt.is_correct for attempt in group.attempts))

    evaluation_result = MCQEvaluation(
        llm_name=llm_name,
        correct_count=total_correct_questions,
        results=result_groups
    )

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(evaluation_result.model_dump_json(indent=2))
    logger.info(f"Results saved to {output_filename}")
    return evaluation_result

def report_accuracy(evaluation: MCQEvaluation, k: int, num_mcqs: int) -> None:
    total_correct_questions = sum(1 for group in evaluation.results if any(attempt.is_correct for attempt in group.attempts))
    accuracy = (total_correct_questions / num_mcqs) * 100 if num_mcqs > 0 else 0
    logger.info(f"pass@{k} Accuracy: {accuracy:.2f}% ({total_correct_questions}/{num_mcqs})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple-choice questions from the Harvard-Edge/QuArch_v1_unique_ids dataset.")
    parser.add_argument("--llm_name", type=str, required=True, help="Which LLM to use for evaluation.")
    parser.add_argument("--output_dir", type=str, default="output/mcq-evals", help="Directory to save evaluation results.")
    parser.add_argument("--num_questions", type=int, default=None, help="Number of questions to evaluate (optional, for testing).")
    parser.add_argument("-k", type=int, default=1, help="Target number of attempts for pass@k evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker threads.")
    parser.add_argument("--llm_config", type=str, default="", help="Path to llm config.")
    parser.add_argument("--env_file", type=str, default=None, help="Path to a .env file to load environment variables from.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_filename = f"eval_{args.llm_name}.log"
    logger.add(output_dir / log_filename, rotation="1 MB")

    # Check for existing results to implement incremental evaluation
    output_filename = output_dir / f"{args.llm_name}.json"
    existing_results = {}
    if output_filename.exists():
        try:
            existing_data = MCQEvaluation.model_validate_json(output_filename.read_text(encoding='utf-8'))
            for group in existing_data.results:
                existing_results[group.question_id] = group
            logger.info(f"Loaded {len(existing_results)} existing results from {output_filename}")
        except Exception as e:
            logger.error(f"Could not parse existing results file {output_filename}, starting fresh. Error: {e}")

    dataset = hf_load.get_quarchds_mcqs(as_hf_ds=True, envfile_path=args.env_file)

    if args.num_questions:
        dataset = dataset.select(range(args.num_questions))

    logger.info(f"Starting pass@{args.k} evaluation for LLM: {args.llm_name} on  with {args.num_workers} workers")
    llm_name = args.llm_name
    llm_configs = llmconfig.load_llm_config(Path(args.llm_config) if args.llm_config else None)
    llm_config = llm_configs[llm_name]

    # Prepare jobs for parallel execution
    jobs_to_run = []
    for item in dataset:
        qid = item['question_id']
        num_existing_attempts = len(existing_results.get(qid, MCQResultGroup(question_id=qid, attempts=[])).attempts)
        new_attempts_needed = args.k - num_existing_attempts
        if new_attempts_needed > 0:
            jobs_to_run.append((item, new_attempts_needed))

    if not jobs_to_run:
        logger.info("No new attempts needed for any question. Evaluation is up to date.")
        report_accuracy(
            evaluation=MCQEvaluation(
                llm_name=llm_name,
                correct_count=sum(1 for group in existing_results.values() if any(attempt.is_correct for attempt in group.attempts)),
                results=list(existing_results.values())
            ),
            k=args.k,
            num_mcqs=len(dataset)
        )
        return
    else:
        logger.info(f"Found {len(jobs_to_run)} questions that need additional attempts.")
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_item, item, llm_name, llm_config, attempts_needed) for item, attempts_needed in jobs_to_run]

            for future in tqdm(as_completed(futures), total=len(jobs_to_run), desc="Processing questions"):
                try:
                    new_result_group = future.result()
                except Exception as e:
                    logger.exception(f"An MCQ worker failed: {e}")
                    continue
                qid = new_result_group.question_id
                logger.info(f"Processed question_id {qid}, obtained {len(new_result_group.attempts)} new attempts.")
                if qid in existing_results:
                    existing_results[qid].attempts.extend(new_result_group.attempts)
                else:
                    existing_results[qid] = new_result_group

                _ = save_results(
                    intermediate_results=existing_results,
                    output_filename=output_filename,
                    llm_name=llm_name,
                )
    logger.info("All questions processed. Finalizing results.")
    final_eval = save_results(
        intermediate_results=existing_results,
        output_filename=output_filename,
        llm_name=llm_name,
    )
    report_accuracy(final_eval, args.k, len(dataset))



if __name__ == "__main__":
    main()
