from pathlib import Path
import tempfile

import fire
import requests
from dotenv import load_dotenv
from loguru import logger
from ml_collections.config_dict import ConfigDict

from . import llmconfig
from . import llm_infra

def invoke_multimodal(llm_name: str, config: ConfigDict) -> None:
    url = "https://apod.nasa.gov/apod/image/2309/TheLargeMagellanicCloud1024.jpg"

    with tempfile.TemporaryDirectory() as tmpdir:
        imgpth = Path(tmpdir) / "test.jpg"

        response = requests.get(url)
        response.raise_for_status()

        imgpth.write_bytes(response.content)
        logger.info(f"Downloaded {url} to {imgpth}")

        messages = llm_infra.create_converse_messages(
            text="What is in this image?",
            image_paths=[imgpth],
            llm_config=config,
        )

        response = llm_infra.invoke_model(llm_name, messages, config)
        
        if response:
            logger.success(f"LLM response:\n{response}")
            logger.info(f"Parsed response:\n{config.response_parser_fn(response)}")
        else:
            logger.error("LLM invocation failed.")

def run(
    llm_name: str,
    prompt: str = "Tell me a computer architecture joke.",
    llm_config: str | None = None,
    env_file: str | None = None,
    test_multimodal: bool = False,
):
    """
    Run a smoke test for a specified LLM.

    Args:
        llm_name: The name of the LLM to test.
        prompt: The prompt to send to the LLM.
        llm_config: Path to a custom LLM configuration YAML file.
        env_file: Path to a .env file to load environment variables from.
        test_multimodal: submit an image to the test llm
    """
    if env_file:
        load_dotenv(env_file)

    llm_configs = llmconfig.load_llm_config(Path(llm_config) if llm_config else None)

    if llm_name not in llm_configs:
        logger.error(f"LLM '{llm_name}' not found in configuration.")
        return

    config = llm_configs[llm_name]
    
    messages = [{"role": "user", "content": prompt}]
    
    logger.info(f"Testing LLM '{llm_name}' with prompt: '{prompt}'")
    
    response = llm_infra.invoke_model(llm_name, messages, config)
    
    if response:
        logger.success(f"LLM response:\n{response}")
        logger.info(f"Parsed response:\n{config.response_parser_fn(response)}")
    else:
        logger.error("LLM invocation failed.")

    if test_multimodal:
        logger.info("multimodal test:")
        invoke_multimodal(llm_name, config)

if __name__ == '__main__':
    fire.Fire(run)
