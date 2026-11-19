import json
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    RootModel,
    field_validator,
    model_validator,
)

ArgumentType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
    "null",
]


class ArgumentSpec(BaseModel):
    """Describe one argument accepted by a function."""

    model_config = ConfigDict(extra="ignore")

    type: ArgumentType


class ReturnSpec(BaseModel):
    """Describe the value returned by a function."""

    model_config = ConfigDict(extra="ignore")

    type: ArgumentType


class FunctionSpec(BaseModel):
    """Describe one callable function from the supplied catalog."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    parameters: dict[str, ArgumentSpec]
    returns: ReturnSpec

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject names that are empty or only whitespace."""
        if not value.strip():
            raise ValueError("Function name cannot be empty")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(
        cls,
        value: dict[str, ArgumentSpec],
    ) -> dict[str, ArgumentSpec]:
        """Reject empty parameter names."""
        for name in value:
            if not name.strip():
                raise ValueError("Parameter name cannot be empty")
        return value

    def parameter_names(self) -> list[str]:
        """Return parameter names in declaration order."""
        return list(self.parameters)

    def escaped_name(self) -> str:
        """Return the JSON-escaped contents of the function name."""
        return json.dumps(self.name, ensure_ascii=False)[1:-1]


class FunctionCatalog(RootModel[list[FunctionSpec]]):
    """Hold a validated, non-empty set of unique functions."""

    @model_validator(mode="after")
    def validate_catalog(self) -> "FunctionCatalog":
        """Reject empty catalogs and duplicate function names."""
        if not self.root:
            raise ValueError("Function catalog cannot be empty")

        seen: set[str] = set()
        for function in self.root:
            if function.name in seen:
                raise ValueError(
                    f"Duplicate function name: {function.name}"
                )
            seen.add(function.name)
        return self

    def names(self) -> list[str]:
        """Return all function names in catalog order."""
        return [function.name for function in self.root]

    def get(self, name: str) -> FunctionSpec:
        """Return a function by exact name or raise a clear error."""
        for function in self.root:
            if function.name == name:
                return function
        raise ValueError(f"Unknown function name: {name}")
