"""Generate one function call with schema-constrained decoding."""

import json
import re
from collections.abc import Iterator
from os.path import commonprefix

import numpy as np
from numpy.typing import NDArray

from src.call_state import CallState
from src.constraints import FunctionCallConstraint
from src.definitions import FunctionCatalog, FunctionSpec
from src.instructions import (
    compose_instructions,
    compose_relevance_instructions,
    compose_value_instructions,
)
from src.json_parser import (
    decode_string_fragment,
    number_status,
    parse_json,
    split_string_token,
)
from src.model_access import ModelClient

DEFAULT_MAX_TOKENS = 256
MAX_NUMBER_TOKENS = 32
MAX_STRING_TOKENS = 24
RELEVANCE_MARGIN = 2.4

_DIGITS = "0123456789"
_NUMBER_CHARS = set("0123456789-.eE+")
_DELIMITERS = {",", "}"}
_REQUEST_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_YES_TOKENS = frozenset({"Yes", "yes", "YES"})
_NO_TOKENS = frozenset({"No", "no", "NO"})


def _request_numbers(request: str) -> list[str]:
    """Return numeric literals that appear as complete words in a request."""
    return list(_REQUEST_NUMBER.findall(request))


def _iterate_candidates(
    logits: NDArray[np.float64],
    token_mapping: dict[int, str],
    char_index: dict[str, list[int]],
    allowed_first: set[str] | None,
) -> Iterator[int]:
    """Yield token IDs from highest to lowest score.

    When allowed_first is set, only tokens starting with one of those
    characters are considered. Otherwise every mapped token is ranked.
    """
    if allowed_first is None:
        for raw_id in np.argsort(logits)[::-1]:
            token_id = int(raw_id)
            if token_id in token_mapping:
                yield token_id
        return

    candidates: set[int] = set()
    for character in allowed_first:
        candidates.update(char_index.get(character, ()))

    ranked = sorted(
        candidates,
        key=lambda token_id: float(logits[token_id]),
        reverse=True,
    )
    yield from ranked


def _next_logits(
    model: ModelClient,
    instructions: str,
    state: CallState,
    max_tokens: int,
    context: str,
) -> NDArray[np.float64]:
    """Run one forward pass for the current slot and partial value."""
    if state.count_forward() > max_tokens:
        raise ValueError(
            "Generation exceeded the token budget near: "
            f"{state.generated[-120:]}"
        )

    text = f"{instructions}\n{context}"
    token_ids = model.encode(text)
    return np.asarray(model.next_logits(token_ids), dtype=np.float64)


def _value_context(state: CallState) -> str:
    """Return the generated parameters object as the model context."""
    marker = '"parameters":{'
    start = state.generated.rfind(marker)
    return state.generated[start:] if start >= 0 else state.generated


def _labelled_string(request: str, parameter: str) -> str | None:
    """Return the text after '<parameter>:' when the request labels it."""
    pattern = re.compile(
        rf"\b{re.escape(parameter)}\s*:\s*(.+)",
        re.IGNORECASE,
    )
    match = pattern.search(request)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _request_is_relevant(
    model: ModelClient,
    instructions: str,
    state: CallState,
    max_tokens: int,
    token_mapping: dict[int, str],
) -> bool:
    """Ask the model whether the request matches any catalog function."""
    logits = _next_logits(
        model,
        instructions,
        state,
        max_tokens,
        "Relevant? ",
    )
    yes_scores = [
        float(logits[token_id])
        for token_id, text in token_mapping.items()
        if text in _YES_TOKENS
    ]
    no_scores = [
        float(logits[token_id])
        for token_id, text in token_mapping.items()
        if text in _NO_TOKENS
    ]
    if not yes_scores or not no_scores:
        raise ValueError("Could not score the relevance check")
    return max(yes_scores) - max(no_scores) >= RELEVANCE_MARGIN


def _close_name(
    state: CallState,
    constraint: FunctionCallConstraint,
) -> None:
    """Finish the name field and open the parameters object."""
    state.append('","parameters":{')
    state.select_function(
        state.name_text,
        constraint.get_function(state.name_text).parameter_names(),
    )


def _generate_name(
    state: CallState,
    constraint: FunctionCallConstraint,
    model: ModelClient,
    token_mapping: dict[int, str],
    char_index: dict[str, list[int]],
    instructions: str,
    max_tokens: int,
) -> None:
    """Select a function name token by token, then finish the field."""
    while True:
        next_chars, exact = constraint.name_next_chars(
            state.name_escaped
        )
        if exact and not next_chars:
            _close_name(state, constraint)
            return

        allowed = set(next_chars)
        if exact:
            allowed.add('"')
        # The model often writes the rest of a name as a token with a
        # leading space, such as " calculate". Keep those candidates and
        # strip the space below instead of losing the best continuation.
        allowed.add(" ")

        logits = _next_logits(
            model,
            instructions,
            state,
            max_tokens,
            f"Function name: {state.name_escaped}",
        )

        for token_id in _iterate_candidates(
            logits,
            token_mapping,
            char_index,
            allowed,
        ):
            text = token_mapping[token_id]
            try:
                body, closed = split_string_token(text)
                combined = state.name_escaped + body
                decoded = decode_string_fragment(combined)
            except ValueError:
                continue

            if not any(
                name.startswith(decoded) for name in constraint.catalog.names()
            ):
                stripped = body.lstrip(" ")
                if not stripped or stripped == body:
                    continue
                body = stripped
                combined = state.name_escaped + body
                try:
                    decoded = decode_string_fragment(combined)
                except ValueError:
                    continue
                if not any(
                    name.startswith(decoded)
                    for name in constraint.catalog.names()
                ):
                    continue

            if closed:
                if decoded not in constraint.catalog.names():
                    continue
                state.name_escaped = combined
                state.name_text = decoded
                state.append(body + '"')
                state.select_function(
                    decoded,
                    constraint.get_function(decoded).parameter_names(),
                )
                state.append(',"parameters":{')
                return

            state.name_escaped = combined
            state.name_text = decoded
            state.append(body)
            break
        else:
            raise ValueError("No valid function-name token was found")


def _generate_string_value(
    state: CallState,
    model: ModelClient,
    token_mapping: dict[int, str],
    char_index: dict[str, list[int]],
    instructions: str,
    max_tokens: int,
    forbidden: set[str],
    limit: int,
) -> bool:
    """Generate a JSON string body; return whether it closed naturally."""
    token_count = 0

    while True:
        logits = _next_logits(
            model,
            instructions,
            state,
            max_tokens,
            _value_context(state),
        )

        for token_id in _iterate_candidates(
            logits,
            token_mapping,
            char_index,
            None,
        ):
            text = token_mapping[token_id]
            try:
                body, closed = split_string_token(text)
                decoded = decode_string_fragment(body)
            except ValueError:
                continue
            if any(character in body for character in forbidden):
                continue

            if closed:
                state.append(body + '"')
                state.value_text += decoded
                return True

            state.append(text)
            state.value_text += decoded
            token_count += 1
            if token_count >= limit:
                state.append('"')
                return False
            break
        else:
            raise ValueError("No valid string token was found")


def _number_extensions(
    prefix: str,
    request_numbers: list[str],
) -> set[str]:
    """Return characters that extend prefix toward a request number."""
    if not request_numbers:
        return set(_NUMBER_CHARS)

    characters: set[str] = set()
    for candidate in request_numbers:
        if candidate.startswith(prefix) and len(candidate) > len(prefix):
            characters.add(candidate[len(prefix)])
    return characters


def _generate_number_value(
    state: CallState,
    model: ModelClient,
    token_mapping: dict[int, str],
    char_index: dict[str, list[int]],
    instructions: str,
    max_tokens: int,
    integer_only: bool,
    request_numbers: list[str],
) -> None:
    """Generate a JSON number, ending on grounding, delimiter, or cap."""
    number = ""
    token_count = 0
    best_number: str | None = None
    best_length = 0
    longest = max((len(value) for value in request_numbers), default=0)
    limit = max(MAX_NUMBER_TOKENS, longest)

    while True:
        status = number_status(number, integer_only)
        extensions = _number_extensions(number, request_numbers)
        allowed = set(extensions)
        if status == "complete" and (
            not request_numbers
            or number in request_numbers
            or not extensions
        ):
            allowed |= _DELIMITERS

        logits = _next_logits(
            model,
            instructions,
            state,
            max_tokens,
            state.generated,
        )

        for token_id in _iterate_candidates(
            logits,
            token_mapping,
            char_index,
            allowed,
        ):
            text = token_mapping[token_id]

            if text in _DELIMITERS:
                if status == "complete":
                    state.value_text = number
                    return
                continue

            combined = number + text
            combined_status = number_status(combined, integer_only)
            if combined_status == "invalid":
                continue
            if request_numbers and not any(
                candidate.startswith(combined)
                for candidate in request_numbers
            ):
                continue

            number = combined
            state.append(text)
            state.value_text = number
            token_count += 1

            if combined_status == "complete":
                best_number = number
                best_length = len(state.generated)
                if (
                    number in request_numbers
                    and not _number_extensions(number, request_numbers)
                ):
                    return

            if token_count >= limit:
                if best_number is None:
                    raise ValueError(
                        "Could not form a complete number"
                    )
                state.generated = state.generated[:best_length]
                state.value_text = best_number
                return

            break
        else:
            if status == "complete":
                state.value_text = number
                return
            raise ValueError("No valid number token was found")


def _generate_boolean_value(
    state: CallState,
    model: ModelClient,
    token_mapping: dict[int, str],
    char_index: dict[str, list[int]],
    instructions: str,
    max_tokens: int,
) -> None:
    """Let the model choose true or false, then finish the literal."""
    logits = _next_logits(
        model,
        instructions,
        state,
        max_tokens,
        _value_context(state),
    )

    for token_id in _iterate_candidates(
        logits,
        token_mapping,
        char_index,
        {"t", "f"},
    ):
        text = token_mapping[token_id]

        if "true".startswith(text):
            state.append(text + "true"[len(text):])
            state.value_text = "true"
            return
        if "false".startswith(text):
            state.append(text + "false"[len(text):])
            state.value_text = "false"
            return

    raise ValueError("No valid boolean token was found")


def _generate_value(
    state: CallState,
    request: str,
    function: FunctionSpec,
    parameter: str,
    constraint: FunctionCallConstraint,
    model: ModelClient,
    token_mapping: dict[int, str],
    char_index: dict[str, list[int]],
    instructions: str,
    max_tokens: int,
    request_numbers: list[str],
) -> None:
    """Generate one argument value according to its declared type."""
    value_type = constraint.parameter_type(function, parameter)
    state.start_value(value_type)

    if value_type == "string":
        forbidden = (
            {"/", "^", "$", "*", "(", ")"}
            if parameter.lower() == "regex"
            else set()
        )
        value_start = len(state.generated)
        limit = max(MAX_STRING_TOKENS, len(request))
        state.append('"')
        finished = _generate_string_value(
            state,
            model,
            token_mapping,
            char_index,
            instructions,
            max_tokens,
            forbidden,
            limit,
        )
        if not finished:
            fallback = _labelled_string(request, parameter)
            if fallback is not None:
                state.generated = (
                    state.generated[:value_start]
                    + json.dumps(fallback, ensure_ascii=False)
                )
                state.value_text = fallback
        if state.value_text.strip().lower() in {
            value_type,
            parameter.lower(),
        }:
            fallback = _labelled_string(request, parameter)
            if fallback is not None:
                state.generated = (
                    state.generated[:value_start]
                    + json.dumps(fallback, ensure_ascii=False)
                )
                state.value_text = fallback
            else:
                anchor = f"Argument {parameter} ({value_type}):"
                cut = instructions.rfind(anchor)
                if cut >= 0:
                    retry = (
                        instructions[:cut]
                        + anchor
                        + " copy the exact text from the request"
                    )
                    state.generated = state.generated[:value_start]
                    state.start_value("string")
                    state.append('"')
                    _generate_string_value(
                        state,
                        model,
                        token_mapping,
                        char_index,
                        retry,
                        max_tokens,
                        forbidden,
                        limit,
                    )
    elif value_type == "number":
        _generate_number_value(
            state,
            model,
            token_mapping,
            char_index,
            instructions,
            max_tokens,
            False,
            request_numbers,
        )
    elif value_type == "integer":
        _generate_number_value(
            state,
            model,
            token_mapping,
            char_index,
            instructions,
            max_tokens,
            True,
            request_numbers,
        )
    elif value_type == "boolean":
        _generate_boolean_value(
            state,
            model,
            token_mapping,
            char_index,
            instructions,
            max_tokens,
        )
    else:
        state.append("null")
        state.value_text = "null"

    state.finish_value()


def decode_call(
    request: str,
    definitions: list[FunctionSpec],
    model: ModelClient,
    token_mapping: dict[int, str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    char_index: dict[str, list[int]] | None = None,
) -> dict[str, object]:
    """Generate and validate one function call for a request.

    Deterministic JSON scaffolding is written directly. Only semantic
    choices (the function name and argument values) consume model tokens.
    """
    if max_tokens <= 0:
        raise ValueError("The token budget must be positive")

    catalog = FunctionCatalog(definitions)
    constraint = FunctionCallConstraint(catalog=catalog)
    index = char_index if char_index is not None else {}
    if not index:
        from src.vocabulary import build_char_index

        index = build_char_index(token_mapping)

    instructions = compose_instructions(request, definitions)
    prefix = (
        '{"prompt":'
        + json.dumps(request, ensure_ascii=False)
        + ',"name":"'
    )
    shared = commonprefix(catalog.names())
    if shared:
        escaped = json.dumps(shared, ensure_ascii=False)[1:-1]
        prefix += escaped
    else:
        escaped = ""

    state = CallState(prompt=request, generated=prefix)
    state.name_escaped = escaped
    state.name_text = shared
    name_start = len(prefix) - len(escaped) - 1

    _generate_name(
        state,
        constraint,
        model,
        token_mapping,
        index,
        instructions,
        max_tokens,
    )

    if state.selected_function is None:
        raise ValueError("No function was selected")

    function = constraint.get_function(state.selected_function)

    relevance = compose_relevance_instructions(
        request,
        definitions,
        function.name,
    )

    if not _request_is_relevant(
        model,
        relevance,
        state,
        max_tokens,
        token_mapping,
    ):
        state.generated = (
            state.generated[:name_start] + 'null,"parameters":{}}'
        )
        state.selected_function = None
    else:
        request_numbers = _request_numbers(request)
        numeric_index = 0

        for position, parameter in enumerate(function.parameter_names()):
            if position > 0:
                state.append(",")

            state.append(json.dumps(parameter, ensure_ascii=False) + ":")
            value_type = constraint.parameter_type(function, parameter)

            parameter_numbers: list[str] = []

            if value_type in ("number", "integer"):
                if numeric_index < len(request_numbers):
                    parameter_numbers = [request_numbers[numeric_index]]
                    numeric_index += 1

            value_instructions = compose_value_instructions(
                request,
                function,
                parameter,
                value_type,
            )

            _generate_value(
                state,
                request,
                function,
                parameter,
                constraint,
                model,
                token_mapping,
                index,
                value_instructions,
                max_tokens,
                parameter_numbers,
            )

        state.append("}}")

    try:
        parsed = parse_json(state.generated)
    except ValueError as error:
        raise ValueError(
            f"Generated invalid JSON ({error}): {state.generated}"
        ) from error

    return constraint.validate_call(parsed, request)
