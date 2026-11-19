import json
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

NumberStatus = Literal["invalid", "incomplete", "complete"]

_DIGITS = "0123456789"
_HEX = "0123456789abcdefABCDEF"
_SIMPLE_ESCAPES = '"\\/bfnrt'


def _reject_constant(value: str) -> None:
    """Reject NaN and Infinity, which Python accepts but JSON does not."""
    raise ValueError(f"Invalid JSON constant: {value}")


def parse_json(text: str) -> object:
    """Parse one strict JSON document.

    Raises:
        ValueError: If the document is malformed or contains a non-standard
            constant such as NaN or Infinity.
    """
    parsed: Any = json.loads(text, parse_constant=_reject_constant)
    return cast(object, parsed)


def decode_string_fragment(body: str) -> str:
    """Decode a JSON string body that has no surrounding quotes.

    The complete body must end on an escape boundary; partial escapes are
    rejected so a token can never be committed in an ambiguous state.

    Raises:
        ValueError: If the body is not a valid JSON string body.
    """
    try:
        decoded: Any = json.loads(f'"{body}"')
    except json.JSONDecodeError as error:
        raise ValueError("Invalid JSON string fragment") from error
    if not isinstance(decoded, str):
        raise ValueError("Fragment did not decode to a string")
    return decoded


def split_string_token(text: str) -> tuple[str, bool]:
    """Split a token into a string body and whether it closes the string.

    A token may contain ordinary characters, complete escape sequences, and
    at most one unescaped closing quote. Any characters after that quote
    belong to the JSON structure that the decoder writes itself, so they are
    discarded.

    Raises:
        ValueError: If a control character, malformed escape sequence, or a
            second unescaped quote is found before the closing quote.
    """
    if not text:
        raise ValueError("Empty token")

    index = 0
    quote_index: int | None = None
    while index < len(text):
        character = text[index]
        if character == "\\":
            if index + 1 >= len(text):
                raise ValueError("Incomplete escape sequence")
            escape = text[index + 1]
            if escape == "u":
                end = index + 6
                if end > len(text):
                    raise ValueError("Incomplete Unicode escape")
                if any(char not in _HEX for char in text[index + 2:end]):
                    raise ValueError("Invalid Unicode escape")
                index = end
                continue
            if escape not in _SIMPLE_ESCAPES:
                raise ValueError("Invalid escape sequence")
            index += 2
            continue
        if character == '"':
            quote_index = index
            break
        if character < " ":
            raise ValueError("Unescaped control character")
        index += 1

    if quote_index is None:
        body = text
        closed = False
    else:
        body = text[:quote_index]
        closed = True

    decode_string_fragment(body)
    return body, closed


def number_status(text: str, integer_only: bool) -> NumberStatus:
    """Classify a JSON number prefix as invalid, incomplete, or complete."""
    if not text:
        return "incomplete"

    index = 0
    length = len(text)
    if text[0] == "-":
        index = 1
        if index == length:
            return "incomplete"

    if index >= length:
        return "invalid"

    if text[index] == "0":
        index += 1
    elif text[index] in "123456789":
        index += 1
        while index < length and text[index] in _DIGITS:
            index += 1
    else:
        return "invalid"

    if index < length and text[index] == ".":
        if integer_only:
            return "invalid"
        index += 1
        if index == length:
            return "incomplete"
        if text[index] not in _DIGITS:
            return "invalid"
        while index < length and text[index] in _DIGITS:
            index += 1

    if index < length and text[index] in "eE":
        if integer_only:
            return "invalid"
        index += 1
        if index == length:
            return "incomplete"
        if text[index] in "+-":
            index += 1
            if index == length:
                return "incomplete"
        if index >= length or text[index] not in _DIGITS:
            return "invalid"
        while index < length and text[index] in _DIGITS:
            index += 1

    if index != length:
        return "invalid"
    return "complete"


class JSONParser(BaseModel):
    """Validate strict JSON documents.

    Prefix-level constraints live in the generation pipeline; this model is
    the final independent parser used before a result is accepted.
    """

    model_config = ConfigDict(extra="forbid")

    def accepts(self, text: str) -> bool:
        """Return whether text is one complete, strict JSON document."""
        try:
            parse_json(text)
        except ValueError:
            return False
        return True

    def is_complete(self, text: str) -> bool:
        """Return whether text is a complete JSON document."""
        return self.accepts(text)
