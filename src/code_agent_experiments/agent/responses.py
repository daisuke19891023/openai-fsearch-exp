"""Integration helpers for the OpenAI Responses API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from typing import Any, Protocol, TYPE_CHECKING, TypeGuard, cast

from code_agent_experiments.domain.models import ToolCall, ToolLimits, ToolName

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from openai.types.responses import Response
    from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
    from openai.types.responses.response_output_message import ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText
    from code_agent_experiments.agent.tooling import ToolRegistry


class ResponsesAPIProtocol(Protocol):
    """Typed subset of the OpenAI Responses API client."""

    def create(self, **kwargs: Any) -> Response:  # pragma: no cover - protocol method
        """Create a new Responses API request."""
        ...

    def submit_tool_outputs(
        self,
        *,
        response_id: str,
        tool_outputs: list[dict[str, str]],
    ) -> Response:  # pragma: no cover - protocol method
        """Submit tool outputs back to the Responses API."""
        ...


class OpenAIClientProtocol(Protocol):
    """Minimal client protocol exposing the Responses API surface."""

    @property
    def responses(self) -> Any:  # pragma: no cover - protocol method
        """Return the Responses API client."""
        ...


def _is_function_tool_call(item: Any) -> TypeGuard[ResponseFunctionToolCall]:
    """Return ``True`` when ``item`` is a function tool call payload."""
    return getattr(item, "type", None) == "function_call"


def _is_output_message(item: Any) -> TypeGuard[ResponseOutputMessage]:
    """Return ``True`` when ``item`` represents an assistant message."""
    return getattr(item, "type", None) == "message"


def _is_output_text(content: Any) -> TypeGuard[ResponseOutputText]:
    """Return ``True`` when ``content`` is an output text payload."""
    return getattr(content, "type", None) == "output_text"

__all__ = ["AgentRunResult", "ResponsesAgent"]


@dataclass(slots=True)
class AgentRunResult:
    """Final result produced by the Responses agent runner."""

    response_id: str
    output_text: str
    tool_calls: list[ToolCall]
    limit_reached: bool


class ResponsesAgent:
    """Drive the OpenAI Responses API with locally registered tools."""

    def __init__(
        self,
        client: OpenAIClientProtocol,
        *,
        model: str,
        tool_registry: ToolRegistry,
        tool_limits: ToolLimits,
        system_prompt: str | None = None,
    ) -> None:
        """Initialise a Responses agent runner with the provided configuration."""
        self._client = client
        self._model = model
        self._tool_registry = tool_registry
        self._tool_limits = tool_limits
        self._system_prompt = system_prompt
        self._previous_response_id: str | None = None

    def run(self, query: str) -> AgentRunResult:
        """Execute ``query`` through the Responses API and return the result."""
        input_items = self._build_inputs(query)
        responses_api = cast("ResponsesAPIProtocol", self._client.responses)
        response = responses_api.create(
            model=self._model,
            input=input_items,
            tools=self._tool_registry.get_payload(),
            parallel_tool_calls=False,
            max_tool_calls=self._tool_limits.max_calls,
            previous_response_id=self._previous_response_id,
        )

        tool_logs: list[ToolCall] = []
        limit_reached = False
        tool_calls_executed = 0

        while True:
            tool_calls = [
                item
                for item in response.output or []
                if _is_function_tool_call(item)
            ]
            if not tool_calls:
                break

            if tool_calls_executed >= self._tool_limits.max_calls:
                limit_reached = True
                break

            tool_outputs: list[dict[str, str]] = []
            for tool_call in tool_calls:
                if tool_calls_executed >= self._tool_limits.max_calls:
                    limit_reached = True
                    break

                log_entry, output_payload = self._execute_tool(tool_call)
                tool_logs.append(log_entry)
                tool_outputs.append(
                    {
                        "tool_call_id": tool_call.id or tool_call.call_id,
                        "output": output_payload,
                    },
                )
                tool_calls_executed += 1

            if limit_reached or not tool_outputs:
                break

            response = responses_api.submit_tool_outputs(
                response_id=response.id,
                tool_outputs=tool_outputs,
            )

        output_text = self._collect_output_text(response)
        if limit_reached and not output_text:
            output_text = (
                "Tool execution limit reached before the model completed its response."
            )

        self._previous_response_id = response.id

        return AgentRunResult(
            response_id=response.id,
            output_text=output_text,
            tool_calls=tool_logs,
            limit_reached=limit_reached,
        )

    def _build_inputs(self, query: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if self._system_prompt:
            items.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._system_prompt}],
                },
            )
        items.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": query}],
            },
        )
        return items

    def _execute_tool(self, tool_call: ResponseFunctionToolCall) -> tuple[ToolCall, str]:
        started_at = datetime.now(UTC)
        arguments = json.loads(tool_call.arguments or "{}")
        tool, result = self._tool_registry.run(tool_call.name, arguments)
        finished_at = datetime.now(UTC)
        output_text = tool.serialize_output(result)

        log_entry = ToolCall(
            name=cast("ToolName", tool.name),
            args=arguments,
            started_at=started_at,
            ended_at=finished_at,
            success=True,
            stdout_preview=output_text[:8000],
            stderr_preview=None,
            token_in=0,
            token_out=0,
            latency_ms=int((finished_at - started_at).total_seconds() * 1000),
        )
        return log_entry, output_text

    def _collect_output_text(self, response: Response) -> str:
        text_parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            if not _is_output_message(item):
                continue
            for content in getattr(item, "content", []) or []:
                if not _is_output_text(content):
                    continue
                text = getattr(content, "text", "")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return "\n".join(text_parts)

