*This activity has been created as part of the 42 curriculum by mshowaia.*

# Call Me Maybe

## Description

**Call Me Maybe** is an introduction to function calling in Large Language Models (LLMs). The goal is to make a small language model (Qwen/Qwen3-0.6B) translate natural language requests into structured function calls instead of answering the question directly.

For example, given the prompt `"What is the sum of 2 and 3?"`, the program does not answer `5`. It produces a structured function call:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

Small language models can be unreliable when asked to generate valid JSON using prompts alone. To address this, the project uses **constrained decoding**. During generation, the model's token logits are restricted according to the constraints, allowing only tokens that are valid for the current generation state.

The program reads function definitions and natural language prompts, then generates a JSON file containing one function call for each prompt.

## Instructions

### Requirements

- Python 3.10 or later.
- [uv](https://github.com/astral-sh/uv) for dependency management.
- The `llm_sdk` package provided with the project, located in `llm_sdk/`.

### Installation

```bash
make install
```

This installs the project dependencies defined in `pyproject.toml` and `uv.lock`.

### Execution

Run the program using:

```bash
make run
```

Or directly:

```bash
uv run python -m src
```

By default, the program reads:

- `data/input/functions_definition.json` — available function definitions.
- `data/input/function_calling_tests.json` — natural language prompts.

The generated results are written to:

- `data/output/function_calling_results.json`

Custom paths can be provided using command-line arguments:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Other Makefile Rules

| Rule | Description |
|---|---|
| `make install` | Install project dependencies. |
| `make run` | Run the program. |
| `make debug` | Run the program with the Python debugger (`pdb`). |
| `make clean` | Remove temporary files and caches. |
| `make lint` | Run `flake8` and `mypy` with the required flags. |
| `make lint-strict` | Run `flake8` and `mypy` in strict mode. |

### Project Structure

```text
.
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── .flake8
├── .gitignore
├── data/
│   ├── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
│   
├── llm_sdk/
└── src/
    ├── __init__.py
    ├── __main__.py
    ├── call_state.py
    ├── constraints.py
    ├── decoding.py
    ├── definitions.py
    ├── instructions.py
    ├── json_parser.py
    ├── model_access.py
    └── vocabulary.py
```

### Source Files

| File | Responsibility |
|---|---|
| `src/__main__.py` | Application entry point and command-line execution. |
| `src/definitions.py` | Defines and validates function definitions. |
| `src/instructions.py` | Builds the instructions and prompt given to the model. |
| `src/model_access.py` | Provides access to the language model through `llm_sdk`. |
| `src/vocabulary.py` | Handles the model vocabulary and token-to-text mapping. |
| `src/call_state.py` | Tracks the current state of function-call generation. |
| `src/constraints.py` | Defines the rules used to validate generated content and restrict tokens. |
| `src/decoding.py` | Implements the constrained decoding process. |
| `src/json_parser.py` | Handles and validates the generated function-call JSON. |

## Example Usage

Run the program with the default input files:

```bash
make run
```

Run it with explicit paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

An example entry in the generated `data/output/function_calling_results.json` file:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

## Algorithm Explanation

The constrained decoding process works token by token:

1. The function definitions and user prompt are provided to the model.
2. The model calculates the logits for the next token.
3. The current generation state is checked against the defined constraints.
4. Invalid token logits are masked so that only allowed tokens can be selected.
5. The token with the highest score among the allowed tokens is selected.
6. The selected token is appended to the generated sequence, and the generation state is updated.
7. The process continues until the function call is complete.

The constraints are used to control the generated function call, including its structure, function name, parameter names, and parameter types.

## Design Decisions

- **Separation of responsibilities:** The project is divided into small modules, each with a specific responsibility, such as function definitions, model access, vocabulary handling, call state, constraints, and decoding.
- **Constraints separated from decoding:** The rules that determine which tokens are allowed are kept separate from the generation process. This makes the constraints easier to understand and maintain.
- **Model access isolation:** All interaction with the language model is handled through `model_access.py`, which uses the provided `llm_sdk` package.
- **Data-driven function definitions:** Available functions are loaded from `functions_definition.json`, allowing the program to work with different function catalogs that follow the expected format.
- **Generation state management:** The `CallState` class tracks the progress of generation and stores the generated text and token IDs.

## Performance Analysis

The project was evaluated using the provided Moulinette test set.

- **Accuracy:** 100% on the tested public set.
- **Output format:** Generated results follow the required JSON structure.
- **Execution time:** The execution is designed to meet the project requirement of completing within five minutes.

## Challenges Faced

- **Understanding constrained decoding:** The model produces logits over its vocabulary rather than directly producing a function call. Understanding how to control token selection through logit masking was an important part of the project.
- **Tokenization:** Tokens do not necessarily correspond to individual characters. A token can represent multiple characters or parts of a word, which makes token-level validation more complex.
- **Maintaining generation state:** The generation process requires tracking what has already been generated to determine which tokens are valid next.
- **Function selection:** The model must select a function that exists in the provided function definitions.
- **Typed parameters:** Generated parameter values must respect the types specified in the function definitions.
- **Performance:** Checking possible tokens during generation can be computationally expensive, so handling the vocabulary efficiently is important.

## Testing Strategy

The project is tested using the provided function definitions and test prompts.

The testing process includes:

- Running the program on the provided prompts.
- Checking whether the correct function is selected.
- Checking whether the generated parameters match the expected values.
- Verifying that the output is valid JSON and contains the required fields: `prompt`, `name`, and `parameters`.
- Running the Moulinette evaluation.
- Checking code quality using `make lint` and `make lint-strict`.

## Resources

- Subject of the project: *Call Me Maybe — Introduction to Function Calling in LLMs* (42).
- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B).
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers).
- [Pydantic documentation](https://docs.pydantic.dev/).
- [uv documentation](https://docs.astral.sh/uv/).

### Use of AI

AI was used as a learning and development assistant throughout the project:

- To understand concepts related to function calling, tokenization, logits, and constrained decoding.
- To clarify Python concepts and help troubleshoot implementation issues.
- To assist with reviewing and improving the README documentation.
