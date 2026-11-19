import json

from src.definitions import FunctionSpec


def _describe(function: FunctionSpec) -> str:
    """Format one function compactly for the model."""
    parameters = ",".join(
        f"{name}:{specification.type}"
        for name, specification in function.parameters.items()
    )
    return (
        f"{function.name}({parameters}): {function.description}"
    )


def compose_instructions(
    request: str,
    definitions: list[FunctionSpec],
    allow_null: bool = False,
) -> str:
    """Build instructions for selecting a function from the catalog."""
    lines = [
        "Pick the one function that matches the request.",
        "Reply with the function name exactly as listed, and nothing"
        " else.",
    ]
    if allow_null:
        lines.append(
            "If none of the functions can serve the request, reply null."
        )
    lines.append("")
    lines.extend(f"- {_describe(function)}" for function in definitions)
    lines.extend(
        [
            "",
            f"Request: {json.dumps(request, ensure_ascii=False)}",
            "",
        ]
    )
    return "\n".join(lines)


def compose_relevance_instructions(
    request: str,
    definitions: list[FunctionSpec],
    function_name: str,
) -> str:
    """Build the prompt that verifies the selected function."""
    return (
        compose_instructions(request, definitions, allow_null=True)
        + f"\nFunction: {function_name}\nRelevant? "
    )


def compose_value_instructions(
    request: str,
    function: FunctionSpec,
    parameter: str,
    value_type: str,
) -> str:
    """Build a focused prompt for extracting one argument value."""
    return "\n".join(
        [
            "Fill one argument for the function from the request.",
            "Use the declared argument type.",
            "Write a regex pattern without / delimiters, flags, anchors,"
            " or word boundaries.",
            f"- {_describe(function)}",
            "",
            f"Request: {json.dumps(request, ensure_ascii=False)}",
            "",
            f"Argument {parameter} ({value_type}):",
        ]
    )
