# QuArch: A Benchmark for Evaluating LLM Reasoning in Computer Architecture (ICML 2026)

* Website: https://quarch.ai
* ICML 2026 talk: https://icml.cc/virtual/2026/poster/60614 (OpenReview: https://openreview.net/forum?id=yU6X1XZl8t)
* ArXiv: https://arxiv.org/abs/2510.22087

To cite our work:

```
@inproceedings{prakash2026quarch,
  title={QuArch: A Benchmark for Evaluating LLM Reasoning in Computer Architecture},
  author={Prakash, Shvetank and Cheng, Andrew and Tschand, Arya and Mazumder, Mark and Gohil, Varun and Ma, Jeffrey and Yik, Jason and Wan, Zishen and Quaye, Jessica and Alvanaki, Elisavet Lydia and Kumar, Avinash and Mazumdar, Chandrashis and Khare, Tuhin and Ingare, Alexander and Uchendu, Ikechukwu and Ghosal, Radhika and Tyagi, Abhishek and Wang, Chenyu and Garavagno, Andrea Mattia and Gu, Sarah and Guo, Alice and Hur, Grace and Carloni, Luca and Krishna, Tushar and Nayak, Ankita and Yazdanbakhsh, Amir and Reddi, Vijay Janapa},
  booktitle={Forty-third International Conference on Machine Learning (ICML)},
  year={2026},
  url={https://openreview.net/forum?id=yU6X1XZl8t}
}
```

This repository contains the **evaluation harness** for QuArch. In order to access the benchmark questions and answers, please follow the instructions on https://quarch.ai

## Installation

### Prerequisites

-   Python 3.8 or higher

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/harvard-edge/quarch-eval.git
    cd quarch-eval
    ```
2.  **Install the package in editable mode:**
    ```bash
    pip install -e .
    ```
    Using editable mode (`-e`) is recommended because it allows you to easily add custom response parsers by directly modifying the package files.

## Configuration

`quarch-eval` uses YAML files for LLM configuration.

### Base Configuration

The package includes a default [`llmconfig.yaml`](quarch_eval/llmconfig.yaml) file that contains a standard set of tested LLM configurations. You can use this as a template for your own configurations.

### Custom Configuration

To use your own set of LLMs or override the defaults, create a custom YAML file (e.g., `my_llms.yaml`). You can then pass this file to the CLI using the `--llm_config` flag.

When you provide a custom configuration file, it is used **instead of** the default one. Example configuration LLMs will not be available.

For more information, see [Custom LLM Configuration](#custom-llm-configuration).

### `quarch-eval` subcommands

The `quarch-eval` package provides a command-line interface with five main subcommands.

* `judges` — run LLM-as-a-judge evaluation on free-response questions (FRQs) and save to disk
* `mcq` — run evaluation on multiple-choice questions (MCQs)
* `report` — generate an accuracy report from existing on-disk evaluation results
* `test-llm` — smoke-test a single LLM configuration
* `list-models` — list the LLM models available in the current config

### Environment Variables and API Keys

`quarch-eval judges`, `quarch-eval mcq`, `quarch-eval test-llm`, and `quarch-eval report` CLI commands accept a `--env_file` argument, which allows you to load environment variables from a `.env` file. This is the recommended way to manage secrets like API keys.

By default, `quarch-eval` relies on `litellm`'s standard environment variable detection for API keys. For instance, the OpenAI API key will be automatically sourced from the `OPENAI_API_KEY` environment variable.

If a specific model requires a non-standard API key variable, you can specify it in your custom `llmconfig.yaml` using the `api_key_env_var` field.

**Example `.env` file:**
```
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-..."
MY_CUSTOM_API_KEY="some-secret-key"
```

You would then run a command like this:
```bash
quarch-eval judges --env_file /some/location/.env --llm_subset "gpt-oss-120b" ...
```

## Usage


### 1. Free-Response Question (FRQ) Answering Evaluation (`judges`)

This command runs the LLM-as-a-judge evaluation process.

```bash
quarch-eval judges --help
```

**Example Usage:**

To run the judge evaluation for all free response questions using a custom config:

```bash
quarch-eval judges \
  --llm_config my_llms.yaml \
  --llm_subset "my-custom-model,gpt-oss-120b" \
  --k_student 3 \
  --k_judge 3
```

See `--help` for all options available

### 2. Multiple-Choice Question (MCQ) Evaluation (`mcq`)

This command runs the multiple-choice question evaluation.

```bash
quarch-eval mcq --help
```

**Example Usage:**

To run MCQ evaluation for a specific LLM with pass@k=3 using a custom config:

```bash
quarch-eval mcq \
  --llm_config my_llms.yaml \
  --llm_subset "my-custom-model" \
  -k 3 \
  --num_workers 4
```

### 3. Reporting evaluation results (`report`)

This command aggregates on-disk `judges`/`mcq` output into an accuracy report.

**Example Usage:**

To report accuracy for all discovered LLMs across three student attempts (`k=3`) and consensus across 3 judges for FRQs (`j=3`):

```bash
quarch-eval report --llms all --k 3 --j 3
```

### 4. Testing an LLM (`test-llm`)

You can quickly test an LLM configuration using the `test_llm.py` script.

**Example Usage:**

To run a smoke test for an LLM named "gpt-oss-120b" with a custom configuration file:

```bash
quarch-eval test-llm --env_file .env --llm_config my_llms.yaml --llm_name gpt-oss-120b
```

### 5. Listing available models (`list-models`)

This command prints the LLM names available in the current config.

**Example Usage:**

```bash
quarch-eval list-models --llm_config my_llms.yaml
```

## Custom LLM Configuration

### Custom Configuration File

As mentioned in the `Configuration` section, you can provide a path to a custom YAML file for LLM configurations using the `--llm_config` option.

The format of this file is a dictionary of LLM configurations. You can refer to `quarch_eval/llmconfig.yaml` for examples.

**Full YAML Configuration Fields:**

Each LLM entry in your YAML file should be a dictionary with the following potential fields:

-   `litellm_name` (string, **required**): The name of the model as expected by `litellm` (e.g., `"openai/gpt-4o"`, `"bedrock/us.meta.llama3-2-1b-instruct-v1:0"`).
-   `api_base` (string, optional): The base URL for the LLM API endpoint (e.g., `"http://localhost:1234/v1"`).
-   `api_key_env_var` (string, optional): The name of the environment variable that holds the API key for this model if it's not the default one expected by `litellm`.
-   `max_input_images` (integer or float, optional): The maximum number of input images the model can handle. Use `0` for text-only models, `float("inf")` for models with unlimited image input (default).
-   `image_max_dim` (integer, optional): The maximum dimension (width or height) for input images, if applicable.
-   `response_parser` (string, optional, default: `"default"`): The name of a registered response parser function in `quarch_eval/llmconfig.py`.
-   `use_responses_api` (boolean, optional, default: `false`): Set to `true` if the model uses the `litellm responses` API structure.
-   `generation_kwargs` (dictionary, optional): A dictionary of additional keyword arguments to pass directly to the `litellm` generation call for this model (e.g., `max_tokens`, `temperature`).
-   `provider_is_safe_for_llm_as_a_judge` (boolean, optional, default: `false`): Only trusted model APIs which guarantee no training on prompts (such as AWS Bedrock) can be used for LLM-as-a-Judge, as judge models are provided groundtruth QuArch answers in their prompts.

**Example (`my_llms.yaml`):**
```yaml
my-custom-model:
  litellm_name: "openai/custom-llm-v1" # The model name litellm expects
  api_base: "http://localhost:1234/v1"
  response_parser: "my_custom_parser" # Name of a custom parser

another-model:
  litellm_name: "gemini/gemini-2.5-pro"
  generation_kwargs:
    max_tokens: 100000
```

### Custom Response Parsers

If your LLM's response needs special parsing, you can add a custom parser function. This requires modifying the package files, which is why `pip install -e .` is the recommended installation method.

To add a custom parser:

1.  **Open `quarch_eval/llmconfig.py`**.
2.  **Define your parser function**. It should take one argument (the response object from `litellm`) and return a string.

    ```python
    def _my_custom_parser(response) -> str:
        """
        A custom function to parse the response from 'my-custom-model'.
        """
        try:
            # Your custom parsing logic here
            return response['some']['nested']['field']
        except (TypeError, KeyError):
            return ""
    ```

3.  **Register your parser function** by adding it to the `RESPONSE_PARSERS` dictionary. Use a unique name for your parser.

    ```python
    RESPONSE_PARSERS = {
        "default": _parse_litellm_response,
        "litellm_responses_api": _parse_litellm_responses_api_response,
        "qwen3thinking": _parse_qwen3thinking_response,
        "my_custom_parser": _my_custom_parser, # Add your parser here
    }
    ```

4.  **Use your parser** in your custom LLM configuration YAML file by setting the `response_parser` field to the name you registered (e.g., `my_custom_parser`).

