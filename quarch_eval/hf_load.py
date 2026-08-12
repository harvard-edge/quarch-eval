import sys
import os
from pathlib import Path
from typing import ClassVar

import yaml
from dotenv import load_dotenv
import huggingface_hub
import datasets
import jsonlines
from loguru import logger
from ml_collections.config_dict import ConfigDict

from quarch_eval import llm_infra


def get_quarch_hf_config() -> ConfigDict:
    config = ConfigDict()
    # fmt: off
    config.dataset_name = "Harvard-Edge/quarchds"
    config.frq_dir = "quarchds/frqs"
    config.mcq_questions = "quarchds/mcqs/test_questions.jsonl"
    config.mcq_answers = "quarchds/mcqs/test_answers.jsonl.enc"
    # fmt: on
    return config


def get_quarch_hf_cfg_with_override(override_yml: Path | None = None) -> ConfigDict:
    cfg = get_quarch_hf_config()
    if override_yml is not None:
        override_cfg = yaml.safe_load(override_yml.read_text())
        cfg.update(override_cfg)
        logger.info("Using overridden quarch_hf_config: {}", cfg)
    return cfg


def get_hf_token(envfile_path: Path | None = None) -> str:
    if envfile_path is not None:
        load_dotenv(envfile_path)
    else:
        load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not found in .env file. Aborting.")
        raise ValueError("HF_TOKEN not found in .env file.")
    return hf_token


def get_quarchds_frq_dir(
    quarch_hf_config_override_yml: Path | None = None, envfile_path: Path | None = None
) -> Path:
    hf_token = get_hf_token(envfile_path)
    cfg = get_quarch_hf_cfg_with_override(quarch_hf_config_override_yml)

    logger.info("Downloading QuArch frqs from HuggingFace")
    snapshot_path = huggingface_hub.snapshot_download(
        repo_id=cfg.dataset_name,
        repo_type="dataset",
        allow_patterns=[cfg.frq_dir + "/**"],
        token=hf_token,
    )
    quarchds_dir = Path(snapshot_path) / cfg.frq_dir
    logger.info("QuArch frqs downloaded to {}", quarchds_dir)
    return quarchds_dir


def get_quarchds_mcqs(
    as_hf_ds: bool,
    quarch_hf_config_override_yml: Path | None = None,
    envfile_path: Path | None = None,
) -> datasets.Dataset | list[dict]:
    cfg = get_quarch_hf_cfg_with_override(quarch_hf_config_override_yml)
    try:
        hf_token = get_hf_token(envfile_path)

        questions_f = huggingface_hub.hf_hub_download(
            repo_id=cfg.dataset_name,
            repo_type="dataset",
            filename=cfg.mcq_questions,
            token=hf_token,
        )
        with jsonlines.open(questions_f) as reader:
            questions = list(reader)
        answers_f = huggingface_hub.hf_hub_download(
            repo_id=cfg.dataset_name,
            repo_type="dataset",
            filename=cfg.mcq_answers,
            token=hf_token,
        )
        answers_bytes = llm_infra.decrypt_file(llm_infra.load_fernet(), Path(answers_f))
        answers = llm_infra.decode_jsonl_from_bytes(answers_bytes)
        qid2answer = {ans["question_id"]: dict(answer=ans["answer"]) for ans in answers}
        missing_qids = {q["question_id"] for q in questions} - qid2answer.keys()
        if len(missing_qids) > 0:
            logger.error(f"No answer found for question_ids {missing_qids=}")
            sys.exit(1)
        qas = [q | qid2answer[q["question_id"]] for q in questions]
        if not as_hf_ds:
            return qas
        dataset = datasets.Dataset.from_list(qas)
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset '{cfg.dataset_name}': {e}")
        raise
