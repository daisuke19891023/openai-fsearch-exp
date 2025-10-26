from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from agents.items import MessageOutputItem, ModelResponse
from agents.run_context import RunContextWrapper
from agents.tool_context import ToolContext
from agents.usage import Usage
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

from code_agent_experiments.agent import AgentsSDKAgent
from code_agent_experiments.agent.builtin_tools import create_ripgrep_tool
from code_agent_experiments.agent.tooling import ToolRegistry
from code_agent_experiments.domain.models import ToolLimits


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a small repository with multiple TODO markers."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "alpha.py").write_text("""# TODO: investigate alpha branch\n""", encoding="utf-8")
    (repo / "beta.py").write_text("""# TODO: review beta output\n""", encoding="utf-8")
    (repo / "gamma.py").write_text("""# TODO: clean up gamma\n""", encoding="utf-8")
    return repo


class _StubRunResult:
    """Provide the minimal ``RunResult`` interface needed by the agent wrapper."""

    def __init__(self, agent: Any, state: Any, input_text: str, response: ModelResponse) -> None:
        self.input = input_text
        message = response.output[0]
        assert isinstance(message, ResponseOutputMessage)
        self.new_items = [MessageOutputItem(agent=agent, raw_item=message)]
        self.raw_responses = [response]
        self.final_output = "All done."
        self._last_agent = agent
        self.input_guardrail_results: list[Any] = []
        self.output_guardrail_results: list[Any] = []
        self.tool_input_guardrail_results: list[Any] = []
        self.tool_output_guardrail_results: list[Any] = []
        self.context_wrapper = RunContextWrapper(context=state)
        self._response = response

    @property
    def last_agent(self) -> Any:  # pragma: no cover - accessor for completeness
        return self._last_agent

    @property
    def last_response_id(self) -> str | None:
        return self._response.response_id


class _StubRunner:
    """Mimic the Agents SDK runner by invoking registered tools."""

    def __init__(self, repository: Path) -> None:
        self._repository = repository
        self.calls = 0

    def run_sync(
        self,
        starting_agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> _StubRunResult:
        state = kwargs["context"]
        tool = starting_agent.tools[0]

        async def _invoke(pattern: str, call_id: str) -> None:
            args = {"pattern": pattern, "root": str(self._repository), "max_count": 5}
            ctx = ToolContext(
                context=state,
                tool_name=tool.name,
                tool_call_id=call_id,
                tool_arguments=json.dumps(args),
            )
            await tool.on_invoke_tool(ctx, json.dumps(args))

        for idx, pattern in enumerate(["TODO", "TODO", "TODO"]):
            asyncio.run(_invoke(pattern, f"tc_{idx}"))

        message = ResponseOutputMessage(
            id="msg_1",
            role="assistant",
            status="completed",
            type="message",
            content=[ResponseOutputText(type="output_text", text="All done.", annotations=[])],
        )
        response = ModelResponse(
            output=[message],
            usage=Usage(),
            response_id=f"resp_{self.calls}",
        )
        self.calls += 1
        return _StubRunResult(starting_agent, state, input_text, response)


def test_agents_sdk_agent_executes_tools(monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
    """Running via the Agents SDK records structured tool telemetry."""
    runner = _StubRunner(sample_repo)
    monkeypatch.setattr(
        "code_agent_experiments.agent.agents_sdk.DEFAULT_AGENT_RUNNER",
        runner,
    )
    registry = ToolRegistry([create_ripgrep_tool()])
    agent = AgentsSDKAgent(
        model="gpt-test",
        tool_registry=registry,
        tool_limits=ToolLimits(max_calls=3),
        workflow_name="code-agent-experiments.tests",
    )

    result = agent.run(
        "Find TODO markers",
        trace_metadata={"repository": str(sample_repo)},
    )

    assert result.output_text == "All done."
    assert len(result.tool_calls) == 3
    first = result.tool_calls[0]
    assert first.name == "ripgrep"
    assert first.args["pattern"] == "TODO"
    assert Path(first.args["root"]) == sample_repo
    assert first.stdout_preview is not None
    assert result.response_id == "resp_0"
    assert result.limit_reached


def test_trace_export_configured_once(monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
    """Trace export API keys are configured exactly once across runs."""
    runner = _StubRunner(sample_repo)
    monkeypatch.setattr(
        "code_agent_experiments.agent.agents_sdk.DEFAULT_AGENT_RUNNER",
        runner,
    )
    calls: list[str] = []

    def _record_api_key(value: str) -> None:
        calls.append(value)

    monkeypatch.setattr(
        "code_agent_experiments.agent.agents_sdk.set_tracing_export_api_key",
        _record_api_key,
    )
    registry = ToolRegistry([create_ripgrep_tool()])
    agent = AgentsSDKAgent(
        model="gpt-test",
        tool_registry=registry,
        tool_limits=ToolLimits(max_calls=3),
        trace_export_api_key="test-key",
    )

    agent.run("First run", trace_metadata={})
    agent.run("Second run", trace_metadata={})

    assert calls == ["test-key"]
