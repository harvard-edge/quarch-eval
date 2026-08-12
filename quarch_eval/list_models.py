from pathlib import Path
from . import llmconfig
import fire

def run(llm_config: str | None = None):
    """
    Lists the available LLM models from the config file.
    """
    try:
        llm_configs = llmconfig.load_llm_config(Path(llm_config) if llm_config else None)
        print("\nAvailable LLM Models:")
        print("=" * 45)
        for model in sorted(llm_configs.keys()):
            print(f"- {model}")
        print("=" * 45 + "\n")
    except Exception as e:
        print(f"Error loading LLM configs: {e}")

if __name__ == '__main__':
    fire.Fire(run)
