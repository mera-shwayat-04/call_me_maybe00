"""Command-line entry point for constrained function calling."""

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from src.constraints import FunctionCallConstraint
from src.decoding import DEFAULT_MAX_TOKENS, decode_call
from src.definitions import FunctionCatalog
from src.json_parser import parse_json
from src.model_access import ModelClient
from src.vocabulary import build_char_index, load_token_mapping


def parse_arguments() -> argparse.Namespace:
    """Parse the supported command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate function calls with constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output results JSON file.",
    )
    return parser.parse_args()


def read_json(path: str) -> object:
    """Read one strict JSON document with clear error messages."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Could not read {path}: {error}") from error

    try:
        return parse_json(text)
    except ValueError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def load_catalog(path: str) -> FunctionCatalog:
    """Load and validate the function catalog."""
    data = read_json(path)
    try:
        return FunctionCatalog.model_validate(data)
    except ValidationError as error:
        raise ValueError(
            f"Invalid function definitions in {path}: {error}"
        ) from error


def load_prompts(path: str) -> list[str]:
    """Load the prompt array while preserving every supplied string."""
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Input file {path} must contain a JSON array")

    prompts: list[str] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Input item {index} must be a JSON object"
            )
        if "prompt" not in item:
            raise ValueError(f"Input item {index} has no prompt")
        prompt = item["prompt"]
        if not isinstance(prompt, str):
            raise ValueError(
                f"Input item {index} prompt must be a string"
            )
        prompts.append(prompt)
    return prompts


def validate_result(
    result: object,
    request: str,
    catalog: FunctionCatalog,
) -> dict[str, object]:
    """Validate a generated call against the live catalog."""
    constraint = FunctionCallConstraint(catalog=catalog)
    return constraint.validate_call(result, request)


def write_results(
    path: str,
    results: list[dict[str, object]],
) -> None:
    """Atomically write the result array, preserving existing output."""
    output_path = Path(path)
    if output_path.is_dir():
        raise ValueError(f"Output path is a directory: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                results,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _same_path(first: str, second: str) -> bool:
    """Return whether two paths resolve to the same location."""
    return Path(first).resolve() == Path(second).resolve()


def main() -> int:
    """Run every prompt, validate the results, and save one JSON array."""
    args = parse_arguments()
    functions_path = str(args.functions_definition)
    input_path = str(args.input)
    output_path = str(args.output)

    try:
        if _same_path(output_path, functions_path):
            raise ValueError(
                "Output path must differ from functions definition"
            )
        if _same_path(output_path, input_path):
            raise ValueError("Output path must differ from input path")

        catalog = load_catalog(functions_path)
        prompts = load_prompts(input_path)

        if not prompts:
            write_results(output_path, [])
            return 0

        model = ModelClient()
        token_mapping = load_token_mapping(model)
        char_index = build_char_index(token_mapping)

        results: list[dict[str, object]] = []
        total = len(prompts)
        failed = 0
        for index, prompt in enumerate(prompts, start=1):
            print(
                f"Processing prompt {index}/{total}...",
                file=sys.stderr,
            )
            try:
                raw = decode_call(
                    prompt,
                    catalog.root,
                    model,
                    token_mapping,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    char_index=char_index,
                )
                results.append(validate_result(raw, prompt, catalog))
            except Exception as error:
                failed += 1
                print(
                    f"Warning: request {index}/{total} failed for "
                    f"{prompt!r}: {error}",
                    file=sys.stderr,
                )
                results.append(
                    {"prompt": prompt, "name": None, "parameters": {}}
                )

        if failed:
            print(
                f"Warning: {failed}/{total} request(s) fell back to null.",
                file=sys.stderr,
            )
        write_results(output_path, results)
        return 0
    except KeyboardInterrupt:
        print("Error: interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
