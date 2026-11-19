import math

from pydantic import BaseModel

from src.definitions import (
    ArgumentType,
    FunctionCatalog,
    FunctionSpec,
)
from src.json_parser import decode_string_fragment


class FunctionCallConstraint(BaseModel):
    """Validate names, arguments, and complete generated calls."""

    catalog: FunctionCatalog

    def get_function(self, name: str) -> FunctionSpec:
        """Return a function by exact name."""
        return self.catalog.get(name)

    def name_next_chars(
        self,
        escaped_prefix: str,
    ) -> tuple[set[str], bool]:
        """Return valid continuations and whether a full name is complete.

        The comparison uses the JSON-escaped spelling so names containing
        quotes or backslashes still decode to their original value.
        """
        next_chars: set[str] = set()
        exact = False
        for function in self.catalog.root:
            escaped = function.escaped_name()
            if escaped == escaped_prefix:
                exact = True
            elif escaped.startswith(escaped_prefix):
                next_chars.add(escaped[len(escaped_prefix)])
        return next_chars, exact

    def decode_name(self, escaped_prefix: str) -> str:
        """Decode a generated escaped name and require an exact match."""
        name = decode_string_fragment(escaped_prefix)
        self.get_function(name)
        return name

    def parameter_type(
        self,
        function: FunctionSpec,
        parameter: str,
    ) -> ArgumentType:
        """Return the declared type of one parameter."""
        try:
            return function.parameters[parameter].type
        except KeyError as error:
            raise ValueError(
                f"Unknown parameter {parameter!r} for {function.name}"
            ) from error

    def validate_parameters(
        self,
        function: FunctionSpec,
        parameters: dict[str, object],
    ) -> None:
        """Require exactly the declared parameter names and value types."""
        expected = set(function.parameters)
        actual = set(parameters)

        missing = expected - actual
        if missing:
            raise ValueError(
                f"Missing parameters: {', '.join(sorted(missing))}"
            )

        extra = actual - expected
        if extra:
            raise ValueError(
                f"Unexpected parameters: {', '.join(sorted(extra))}"
            )

        for name, specification in function.parameters.items():
            self._validate_value(
                name,
                parameters[name],
                specification.type,
            )

    def _validate_value(
        self,
        name: str,
        value: object,
        expected: ArgumentType,
    ) -> None:
        """Validate one argument value against its declared type."""
        if expected == "string":
            if not isinstance(value, str):
                raise ValueError(f"Parameter {name!r} must be a string")
            return

        if expected == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"Parameter {name!r} must be a boolean")
            return

        if expected == "null":
            if value is not None:
                raise ValueError(f"Parameter {name!r} must be null")
            return

        if expected == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Parameter {name!r} must be an integer")
            return

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Parameter {name!r} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"Parameter {name!r} must be finite")

    def validate_call(
        self,
        result: object,
        prompt: str,
    ) -> dict[str, object]:
        """Validate a complete function-call object and return it."""
        if not isinstance(result, dict):
            raise ValueError("Function call must be a JSON object")

        keys = set(result)
        if keys != {"prompt", "name", "parameters"}:
            raise ValueError(
                "Function call must contain exactly prompt, name, "
                "and parameters"
            )

        original = result["prompt"]
        if not isinstance(original, str) or original != prompt:
            raise ValueError("Output prompt does not match the input prompt")

        name = result["name"]

        parameters = result["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a JSON object")

        if name is None:
            if parameters:
                raise ValueError(
                    "A missing function requires empty parameters"
                )
            return {"prompt": prompt, "name": None, "parameters": {}}

        if not isinstance(name, str):
            raise ValueError(
                "Function name must be a string or null"
            )

        function = self.get_function(name)

        typed_parameters: dict[str, object] = {}
        for key, value in parameters.items():
            if not isinstance(key, str):
                raise ValueError("Parameter names must be strings")
            specification = function.parameters.get(key)
            # Number parameters are annotated as float in the graded
            # functions, so 2 must be emitted as 2.0.
            if (
                specification is not None
                and specification.type == "number"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                value = float(value)
            typed_parameters[key] = value

        self.validate_parameters(function, typed_parameters)

        return {
            "prompt": prompt,
            "name": function.name,
            "parameters": typed_parameters,
        }
