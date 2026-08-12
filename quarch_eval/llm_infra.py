import time
import base64
import os
import io
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm
from cryptography.fernet import Fernet
import jsonlines
from loguru import logger
from PIL import Image
from ml_collections.config_dict import ConfigDict

from . import llmconfig
from . import harmony_adapter


@lru_cache(maxsize=1)
def load_fernet() -> Fernet:
    secret = os.getenv("QUARCH_FERNET_SECRET")
    if not secret:
        logger.error("QUARCH_FERNET_SECRET not found in environment")
        raise ValueError("QUARCH_FERNET_SECRET not found in environment")
    return Fernet(secret.encode("ascii"))

def decrypt_file(fernet: Fernet, infile: Path) -> bytes:
    assert infile.is_file(), f"{infile} does not exist"
    encrypted: bytes = infile.read_bytes()
    decrypted: bytes = fernet.decrypt(encrypted)
    return decrypted

def decode_jsonl_from_bytes(data: bytes, encoding="utf-8") -> list[dict]:
    text_stream = io.StringIO(data.decode(encoding))
    with jsonlines.Reader(text_stream) as reader:
        return list(reader)

def _open_image_detached_from_source(img: Image.Image) -> Image.Image:
    """
    Image.open is lazy, this returns an image in memory without depending on an underlying file object or buffer
    """
    img.load()          # force decode
    copy = img.copy()   # detach from file/buffer
    img.close()
    return copy

def _open_image_maybe_encrypted(image_path: Path) -> Image.Image:
    """
    Returns a fully-loaded PIL Image, regardless of whether the source is
    an encrypted file (…*.enc) or a regular image file. Never writes to disk.
    """
    if image_path.suffix == ".enc":
        encrypted: bytes = image_path.read_bytes()
        decrypted: bytes = load_fernet().decrypt(encrypted)
        bio = BytesIO(decrypted)
        try:
            img = Image.open(bio)
            return _open_image_detached_from_source(img)
        finally:
            bio.close()
    else:
        with Image.open(image_path) as img:
            return _open_image_detached_from_source(img)

def sleep_logic(attempt: int, sleep_interval_sec: int = 3):
    logger.warning(f"Sleeping {sleep_interval_sec} sec before retrying...")
    time.sleep(sleep_interval_sec)

# --- Image Utilities ---

def get_image_bytes(image_path: Path, max_dim: Optional[int] = None) -> Optional[bytes]:
    try:
        if not image_path.is_file():
            logger.warning(f"Image file not found {image_path}")
            return None
        img = _open_image_maybe_encrypted(image_path)

        if max_dim and (img.width > max_dim or img.height > max_dim):
            scale = min(max_dim / img.width, max_dim / img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Could not load image {image_path}: {e}")
        return None

# --- Message Creation ---

def create_converse_messages(text: str, image_paths: list[Path], llm_config: dict) -> Optional[list[dict]]:
    if llm_config.generation_kwargs.get("use_harmony_adapter", False):
        assert len(image_paths) == 0, "images not supported with harmony adapter"
        return [dict(role="user", content=harmony_adapter.gpt_oss_high_reasoning(text))]
    content = [{"type": "text", "text": text}]
    max_dim = llm_config.get("image_max_dim")
    
    for img_path in image_paths:
        img_bytes = get_image_bytes(img_path, max_dim=max_dim)
        if img_bytes:
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                }
            )
        else:
            logger.warning(f"Could not process image, skipping: {img_path}")

    return [{"role": "user", "content": content}]

# --- Model Invocation ---

def invoke_model(
    llm_name: str, messages: list[dict], llm_config: ConfigDict, n_retries: int = 3
) -> Any:
    model_name_in_litellm = llm_config.litellm_name
    
    if llm_config.use_responses_api:
        kwargs = {
            "model": model_name_in_litellm,
            "input": messages,
        }
    else:
        kwargs = {
            "model": model_name_in_litellm,
            "messages": messages,
        }

    if llm_config.api_base:
        kwargs['api_base'] = llm_config.api_base

    if llm_config.api_key_env_var:
        kwargs['api_key'] = os.getenv(llm_config.api_key_env_var)

    # Add any additional generation kwargs, converting ConfigDict to dict
    generation_kwargs = llm_config.generation_kwargs.to_dict()
    
    # Exclude internal-only keys
    internal_keys = ['use_harmony_adapter']
    for key, value in generation_kwargs.items():
        if key not in internal_keys:
            kwargs[key] = value

    attempts = max(1, n_retries)
    for attempt in range(1, attempts + 1):
        try:

            if llm_config.use_responses_api:
                response = litellm.responses(**kwargs)
            else:
                response = litellm.completion(**kwargs)

            return response
        except Exception as e:
            logger.warning(f"Attempt {attempt} ERROR: Can't invoke model '{model_name_in_litellm}' with litellm: {e}")
        if attempt >= attempts:
            logger.error(f"Failed to invoke model '{model_name_in_litellm}' after {attempts} attempts")
            return None
        sleep_logic(attempt)
        logger.warning(f"Attempt {attempt} failed, retrying model {model_name_in_litellm}")
