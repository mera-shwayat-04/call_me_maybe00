"""Load the model vocabulary and convert pieces into generated text.

Qwen uses byte-level byte-pair encoding: a vocabulary piece is a sequence of
Unicode characters that represent bytes, not literal output. This module
reverses that mapping and decodes the resulting bytes as UTF-8.
"""

import json
from pathlib import Path

from src.model_access import ModelClient


def _bytes_to_unicode() -> dict[int, str]:
    """Return the standard byte-to-Unicode table used by byte-level BPE."""
    values = (
        list(range(33, 127))
        + list(range(161, 173))
        + list(range(174, 256))
    )
    characters = values[:]
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters)))


_BYTE_TO_CHAR = _bytes_to_unicode()
_CHAR_TO_BYTE = {
    character: byte for byte, character in _BYTE_TO_CHAR.items()
}


def _piece_to_bytes(piece: str) -> bytes | None:
    """Convert one vocabulary piece to bytes, or None if unmappable."""
    data = bytearray()
    for character in piece:
        byte = _CHAR_TO_BYTE.get(character)
        if byte is None:
            return None
        data.append(byte)
    return bytes(data)


def _read_vocabulary(path: str) -> dict[str, int]:
    """Read and validate the vocabulary JSON object."""
    vocabulary_path = Path(path)
    if not vocabulary_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")

    try:
        with vocabulary_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read vocabulary: {error}") from error

    if not isinstance(data, dict) or not data:
        raise ValueError("Vocabulary must be a non-empty JSON object")

    for token, token_id in data.items():
        if not isinstance(token, str):
            raise ValueError("Vocabulary contains a non-string token")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("Vocabulary contains a non-integer ID")
        if token_id < 0:
            raise ValueError("Vocabulary contains a negative ID")

    if len(set(data.values())) != len(data):
        raise ValueError("Vocabulary contains duplicate token IDs")

    return data


def load_token_mapping(model: ModelClient) -> dict[int, str]:
    """Map token IDs to the text they contribute to the output.

    Tokens whose bytes are not a complete UTF-8 sequence are not usable as
    standalone JSON contributions and are omitted. They are byte fragments of
    characters that also exist as complete tokens.
    """
    vocabulary = _read_vocabulary(model.vocabulary_path())
    mapping: dict[int, str] = {}

    for piece, token_id in vocabulary.items():
        raw = _piece_to_bytes(piece)
        if raw is None:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text:
            mapping[token_id] = text

    if not mapping:
        raise ValueError("Vocabulary produced no usable tokens")
    return mapping


def build_char_index(
    token_mapping: dict[int, str],
) -> dict[str, list[int]]:
    """Index token IDs by the first character of their contribution."""
    index: dict[str, list[int]] = {}
    for token_id, text in token_mapping.items():
        index.setdefault(text[0], []).append(token_id)
    return index
