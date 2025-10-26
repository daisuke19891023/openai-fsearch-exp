"""Agents SDK integration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING, Any, cast

from agents.agent import Agent, AgentBase
from agents.items import ItemHelpers
from agents.model_settings import ModelSettings
from agents.run import DEFAULT_AGENT_RUNNER, RunConfig
from agents.tool import FunctionTool
from agents.tracing import set_tracing_export_api_key
from pydantic import BaseModel

from code_agent_experiments.domain.models import ToolCall, ToolLimits, ToolName

from .responses import AgentRunResult
from .tooling import FunctionToolDefinition, ToolExecutionError, ToolRegistry

__all__ = ["AgentsSDKAgent"]


if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from agents.run_context import RunContextWrapper
    from agents.tool import Tool
    from agents.tool_context import ToolContext


def _tool_call_list_factory() -> list[ToolCall]:
    """Return an empty list with the correct ``ToolCall`` type annotation."""
    return []


@dataclass(slots=True)
class _AgentsRunState:
    """Mutable run state shared across Agents SDK tool invocations."""

    tool_limits: ToolLimits
    tool_calls: list[ToolCall] = field(default_factory=_tool_call_list_factory)
    tool_calls_executed: int = 0
    limit_reached: bool = False

    def record(
        self,
        *,
        name: ToolName,
        args: dict[str, Any],
        started_at: datetime,
        ended_at: datetime | None,
        success: bool,
        stdout: str | None,
        stderr: str | None,
    ) -> None:
        """Append a ``ToolCall`` telemetry entry and update counters."""
        latency_ms: int | None = None
        if ended_at is not None:
            latency_ms = int((ended_at - started_at).total_seconds() * 1000)

        self.tool_calls.append(
            ToolCall(
                name=name,
                args=args,
                started_at=started_at,
                ended_at=ended_at,
                success=success,
                stdout_preview=stdout[:8000] if stdout else None,
                stderr_preview=stderr[:8000] if stderr else None,
                token_in=0,
                token_out=0,
                latency_ms=latency_ms,
            ),
        )
        self.tool_calls_executed += 1
        if self.tool_calls_executed >= self.tool_limits.max_calls:
            self.limit_reached = True


class AgentsSDKAgent:
    """Execute a query using the OpenAI Agents SDK while recording telemetry."""

    def __init__(
        self,
        *,
        model: str,
        tool_registry: ToolRegistry,
        tool_limits: ToolLimits,
        system_prompt: str | None = None,
        workflow_name: str = "code-agent-experiments.agents-sdk",
        trace_export_api_key: str | None = None,
        tracing_enabled: bool = True,
    ) -> None:
        """Store configuration shared across runs."""
        self._model = model
        self._tool_registry = tool_registry
        self._tool_limits = tool_limits
        self._system_prompt = system_prompt
        self._workflow_name = workflow_name
        self._trace_export_api_key = trace_export_api_key
        self._tracing_enabled = tracing_enabled
        self._trace_configured = False
        self._previous_response_id: str | None = None
        self._conversation_id: str | None = None

    def run(
        self,
        query: str,
        *,
        workflow_name: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        """Execute ``query`` and return the aggregated result."""
        if self._trace_export_api_key and not self._trace_configured:
            set_tracing_export_api_key(self._trace_export_api_key)
            self._trace_configured = True

        state = _AgentsRunState(tool_limits=self._tool_limits)
        tools: list[Tool] = [
            cast("Tool", self._build_function_tool(tool_definition, state))
            for tool_definition in self._tool_registry.tools
        ]
        agent = Agent(
            name="code-search-agent",
            instructions=self._system_prompt,
            tools=tools,
            model=self._model,
            model_settings=ModelSettings(parallel_tool_calls=False),
            reset_tool_choice=False,
        )

        run_config = RunConfig(
            workflow_name=workflow_name or self._workflow_name,
            trace_metadata=trace_metadata or {},
            tracing_disabled=not self._tracing_enabled,
            model=self._model,
            model_settings=ModelSettings(parallel_tool_calls=False),
        )

        result = DEFAULT_AGENT_RUNNER.run_sync(
            agent,
            query,
            context=state,
            max_turns=max(self._tool_limits.max_calls + 1, 1),
            run_config=run_config,
            previous_response_id=self._previous_response_id,
            conversation_id=self._conversation_id,
        )

        self._previous_response_id = result.last_response_id
        output_text = self._collect_output_text(result)

        return AgentRunResult(
            response_id=result.last_response_id or "",
            output_text=output_text,
            tool_calls=state.tool_calls,
            limit_reached=state.limit_reached,
        )

    def _build_function_tool(
        self,
        tool_definition: FunctionToolDefinition[Any, Any],
        state: _AgentsRunState,
    ) -> FunctionTool:
        """Convert a ``FunctionToolDefinition`` into an Agents SDK tool."""

        async def _on_invoke(
            _ctx: ToolContext[Any],
            raw_arguments: str,
        ) -> Any:
            started_at = datetime.now(UTC)
            try:
                parsed_arguments: Any = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                finished_at = datetime.now(UTC)
                state.record(
                    name=cast("ToolName", tool_definition.name),
                    args={},
                    started_at=started_at,
                    ended_at=finished_at,
                    success=False,
                    stdout=None,
                    stderr=str(exc),
                )
                message = "Invalid JSON arguments"
                raise ToolExecutionError(message) from exc

            if not isinstance(parsed_arguments, dict):
                finished_at = datetime.now(UTC)
                message = "Tool arguments must be provided as a JSON object"
                state.record(
                    name=cast("ToolName", tool_definition.name),
                    args={},
                    started_at=started_at,
                    ended_at=finished_at,
                    success=False,
                    stdout=None,
                    stderr=message,
                )
                raise ToolExecutionError(message)

            args_payload = cast("dict[str, Any]", parsed_arguments)

            if state.tool_calls_executed >= state.tool_limits.max_calls:
                finished_at = datetime.now(UTC)
                message = (
                    "Tool execution limit reached before the model completed its response."
                )
                state.record(
                    name=cast("ToolName", tool_definition.name),
                    args=args_payload,
                    started_at=started_at,
                    ended_at=finished_at,
                    success=False,
                    stdout=None,
                    stderr=message,
                )
                return {"error": message}

            try:
                _, result = self._tool_registry.run(tool_definition.name, args_payload)
            except Exception as exc:  # pragma: no cover - defensive
                finished_at = datetime.now(UTC)
                state.record(
                    name=cast("ToolName", tool_definition.name),
                    args=args_payload,
                    started_at=started_at,
                    ended_at=finished_at,
                    success=False,
                    stdout=None,
                    stderr=str(exc),
                )
                raise

            finished_at = datetime.now(UTC)
            payload = result.model_dump()
            stdout = result.model_dump_json()
            state.record(
                name=cast("ToolName", tool_definition.name),
                args=args_payload,
                started_at=started_at,
                ended_at=finished_at,
                success=True,
                stdout=stdout,
                stderr=None,
            )
            return payload

        def _is_enabled(
            run_context: RunContextWrapper[Any],
            _agent: AgentBase,
        ) -> bool:
            context_state = cast("_AgentsRunState", run_context.context)
            return context_state.tool_calls_executed < context_state.tool_limits.max_calls

        return FunctionTool(
            name=tool_definition.name,
            description=tool_definition.description,
            params_json_schema=tool_definition.input_model.model_json_schema(),
            on_invoke_tool=_on_invoke,
            strict_json_schema=tool_definition.strict,
            is_enabled=_is_enabled,
        )

    @staticmethod
    def _collect_output_text(result: Any) -> str:
        """Extract assistant text from an Agents SDK run result."""
        text = ItemHelpers.text_message_outputs(result.new_items)
        if text.strip():
            return text.strip()

        final_output = result.final_output
        if isinstance(final_output, str):
            return final_output
        if isinstance(final_output, BaseModel):
            return final_output.model_dump_json()
        if isinstance(final_output, (dict, list)):
            return json.dumps(final_output, ensure_ascii=False)
        return str(final_output)
