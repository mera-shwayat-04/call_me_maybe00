from typing import Literal

from pydantic import BaseModel, Field

Stage = Literal["name", "value", "done"]
ValueKind = Literal["string", "number", "integer", "boolean", "null"]


class CallState(BaseModel):
    """Track the deterministic template and the slot being generated."""

    prompt: str
    generated: str
    stage: Stage = "name"
    name_escaped: str = ""
    name_text: str = ""
    selected_function: str | None = None
    parameter_names: list[str] = Field(default_factory=list)
    parameter_index: int = 0
    value_kind: ValueKind | None = None
    value_text: str = ""
    forward_count: int = 0

    def append(self, text: str) -> None:
        """Append generated text to the JSON document under construction."""
        self.generated += text

    def count_forward(self) -> int:
        """Record one model forward pass and return the new count."""
        self.forward_count += 1
        return self.forward_count

    def select_function(
        self,
        name: str,
        parameter_names: list[str],
    ) -> None:
        """Record the selected function and prepare its first parameter."""
        self.selected_function = name
        self.parameter_names = parameter_names
        self.parameter_index = 0
        if parameter_names:
            self.stage = "value"
        else:
            self.stage = "done"

    def current_parameter(self) -> str | None:
        """Return the parameter currently being generated, if any."""
        if self.parameter_index < len(self.parameter_names):
            return self.parameter_names[self.parameter_index]
        return None

    def is_last_parameter(self) -> bool:
        """Return whether the current parameter is the final one."""
        return self.parameter_index == len(self.parameter_names) - 1

    def start_value(self, kind: ValueKind) -> None:
        """Start generating the value of the current parameter."""
        self.value_kind = kind
        self.value_text = ""

    def finish_value(self) -> None:
        """Finish the current value and move to the next parameter."""
        self.value_kind = None
        self.value_text = ""
        self.parameter_index += 1
        if self.parameter_index >= len(self.parameter_names):
            self.stage = "done"

    def finish(self) -> None:
        """Mark the whole call as complete."""
        self.stage = "done"
