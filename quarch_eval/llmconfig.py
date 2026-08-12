from loguru import logger
import yaml
from pathlib import Path
import importlib.util
import sys
from typing import Optional

from ml_collections.config_dict import ConfigDict

def get_llm_config() -> ConfigDict:
    config = ConfigDict()

    config.litellm_name = ""
    config.api_base = ""
    config.api_key_env_var = "" # Name of API key env variable name if not default

    config.max_input_images = float("inf")
    config.image_max_dim = None

    config.response_parser = "default"
    config.response_parser_fn = None

    config.use_responses_api = False

    # NOTE: this flag must be False by default! override it only for approved APIs in llmconfig.yaml
    # Only trusted model APIs which guarantee no training on prompts (such as AWS Bedrock) can be used 
    # for LLM-as-a-Judge, as judge models are provided groundtruth QuArch answers in their prompts.
    config.provider_is_safe_for_llm_as_a_judge = False # DO NOT EDIT

    config.generation_kwargs = dict()

    return config


def _parse_litellm_response(response) -> str:
    """Helper to parse response from litellm."""
    if not response:
        return ""
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError):
        logger.opt(exception=True).error("Error parsing litellm response")
        return ""

def _parse_litellm_responses_api_response(response) -> str:
    if not response:
        return ""
    try:
        pieces = []
        output = response.output
        for item in output:
            content = item.content
            for c in content:
                if c.type == "output_text":
                    pieces.append(c.text)
        return "".join(pieces)
    except Exception:
        logger.opt(exception=True).error("Error parsing OpenAI response")
        return ""


def _parse_qwen3thinking_response(response) -> str:
    """drops everything before and including </think>"""
    if not response:
        return ""
    try:
        content = response.choices[0].message.content
        assert "</think>" in content, "Expected </think> tag in Qwen3-Thinking response"
        return content.split("</think>")[-1]
    except (AttributeError, IndexError, KeyError, AssertionError):
        logger.opt(exception=True).error("Error parsing qwen3 thinking response")
        return ""


# Built-in response parsers
RESPONSE_PARSERS = {
    "default": _parse_litellm_response,
    "litellm_responses_api": _parse_litellm_responses_api_response,
    "qwen3thinking": _parse_qwen3thinking_response,
}


def load_llm_config(llm_config_path: Optional[Path] = None) -> ConfigDict:
    config_path = llm_config_path
    if not config_path:
        config_path = Path(__file__).parent / "llmconfig.yaml"
    elif not config_path.exists():
        logger.warning(
            f"Custom LLM config file not found at {config_path}, falling back to default."
        )
        config_path = Path(__file__).parent / "llmconfig.yaml"

    with open(config_path, "r") as f:
        llm_configs_from_yaml = yaml.safe_load(f) or {}

    llm_configs = ConfigDict()

    for llm_name, yaml_config in llm_configs_from_yaml.items():
        config = get_llm_config()
        config.update(yaml_config)
        if not config.litellm_name:
            logger.warning(f"LLM config for {llm_name} missing litellm_name field, skipping.")
            continue
        llm_configs[llm_name] = config

    _process_parsers(llm_configs)

    return llm_configs


def _process_parsers(config: ConfigDict):
    """Processes the response_parser_fn field for each LLM configuration."""
    for llm_name, llm_config in config.items():
        parser_name = llm_config.get("response_parser")
        if not parser_name:
            llm_config.response_parser_fn = RESPONSE_PARSERS["default"]
            continue

        if parser_name in RESPONSE_PARSERS:
            llm_config.response_parser_fn = RESPONSE_PARSERS[parser_name]
        else:
            logger.warning(
                f"Response parser '{parser_name}' not found for LLM '{llm_name}'. Using default parser."
            )
            llm_config.response_parser_fn = RESPONSE_PARSERS["default"]
