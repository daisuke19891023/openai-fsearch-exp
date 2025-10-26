"""Utilities for registering and executing Responses API function tools."""

from __future__ import annotations

from dataclasses import dataclass
import collections.abc
import json
import typing
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Sequence
else:
    Sequence = collections.abc.Sequence

__all__ = [
    "FunctionToolDefinition",
    "ToolExecutionError",
    "ToolRegistry",
]


class ToolExecutionError(RuntimeError):
    """Raised when a requested tool cannot be executed."""


@dataclass(slots=True)
class FunctionToolDefinition[InputModelT: BaseModel, OutputModelT: BaseModel]:
    """Definition for a function-call tool exposed to the LLM."""

    name: str
    description: str
    input_model: type[InputModelT]
    output_model: type[OutputModelT]
    handler: typing.Callable[[InputModelT], OutputModelT | dict[str, Any]]
    strict: bool = True

    def to_openai_tool(self) -> dict[str, Any]:
        """Return the Responses API payload describing the tool."""
        payload: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }
        if self.strict:
            payload["function"]["strict"] = True
        return payload

    def invoke(self, arguments: dict[str, Any]) -> OutputModelT:
        """Validate ``arguments`` and execute the tool handler."""
        validated = self.input_model.model_validate(arguments)
        result = self.handler(validated)
        if isinstance(result, self.output_model):
            return result
        return self.output_model.model_validate(result)

    def serialize_output(self, output: OutputModelT) -> str:
        """Return a JSON string conforming to the tool's output schema."""
        return output.model_dump_json()


class ToolRegistry:
    """Registry of available function tools keyed by name."""

    def __init__(
        self,
        tools: Sequence[FunctionToolDefinition[Any, Any]],
    ) -> None:
        """Store tool definitions keyed by their registered name."""
        if not tools:
            msg = "At least one tool definition must be provided"
            raise ValueError(msg)
        self._tools = {tool.name: tool for tool in tools}

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        """Return ``True`` when ``name`` corresponds to a registered tool."""
        return name in self._tools

    @property
    def tools(self) -> tuple[FunctionToolDefinition[Any, Any], ...]:  # pragma: no cover - trivial
        """Expose the registered tools as an immutable sequence."""
        return tuple(self._tools.values())

    def get_payload(self) -> list[dict[str, Any]]:
        """Return serialized tool definitions for the Responses API."""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def run(
        self,
        name: str,
        arguments: str | dict[str, Any],
    ) -> tuple[FunctionToolDefinition[Any, Any], BaseModel]:
        """Execute ``name`` using validated ``arguments`` and return the result."""
        try:
            tool = self._tools[name]
        except KeyError as exc:  # pragma: no cover - defensive
            msg = f"Tool '{name}' is not registered"
            raise ToolExecutionError(msg) from exc

        if isinstance(arguments, str):
            try:
                payload = json.loads(arguments)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                msg = f"Tool '{name}' received invalid JSON arguments"
                raise ToolExecutionError(msg) from exc
        else:
            payload = arguments

        result = tool.invoke(payload)
        return tool, result
