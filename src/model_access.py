from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model


class ModelClient(BaseModel):
    """Wrap the SDK so the rest of the program never touches it directly."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Qwen/Qwen3-0.6B"

    _model: Small_LLM_Model = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        """Load the model once when the client is constructed."""
        try:
            self._model = Small_LLM_Model(model_name=self.model_name)
        except Exception as error:
            raise RuntimeError(
                f"Could not load model {self.model_name}: {error}"
            ) from error

    def encode(self, text: str) -> list[int]:
        """Tokenize text into a flat list of token IDs."""
        if not text:
            raise ValueError("Cannot encode empty text")

        encoded = self._model.encode(text)
        batches: list[list[int]] = encoded.tolist()
        if not batches or not batches[0]:
            raise RuntimeError("The model returned no input tokens")
        return [int(token_id) for token_id in batches[0]]

    def next_logits(self, token_ids: list[int]) -> list[float]:
        """Return one score per vocabulary entry for the next token."""
        if not token_ids:
            raise ValueError("Cannot request logits without input tokens")

        logits = self._model.get_logits_from_input_ids(token_ids)
        if not logits:
            raise RuntimeError("The model returned no logits")
        return logits

    def vocabulary_path(self) -> str:
        """Return the public path to the model vocabulary file."""
        return self._model.get_path_to_vocab_file()

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs through the public SDK method."""
        return self._model.decode(token_ids)
