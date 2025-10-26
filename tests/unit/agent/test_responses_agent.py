from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast

import pytest
from openai.types.responses import Response
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

from code_agent_experiments.agent import ResponsesAgent
from code_agent_experiments.agent.builtin_tools import create_ripgrep_tool
from code_agent_experiments.agent.tooling import ToolRegistry
from code_agent_experiments.domain.models import ToolLimits


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a small repository containing a TODO marker."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("""# TODO: investigate this branch\n""", encoding="utf-8")
    return repo


@dataclass
class _StubResponsesAPI:
    create_responses: list[Response]
    submit_responses: list[Response]

    def __post_init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []
        self.create_kwargs: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Response:
        self.create_kwargs.append(kwargs)
        return self.create_responses.pop(0)

    def submit_tool_outputs(self, *, response_id: str, tool_outputs: list[dict[str, str]]) -> Response:
        self.submitted.append({"response_id": response_id, "tool_outputs": tool_outputs})
        return self.submit_responses.pop(0)


@dataclass
class _StubOpenAIClient:
    responses: _StubResponsesAPI


def _make_tool_call_response(repo: Path) -> Response:
    return Response(
        id="resp_1",
        model="gpt-test",
        created_at=int(datetime.now(tz=UTC).timestamp()),
        object="response",
        status="in_progress",
        output=[
            ResponseFunctionToolCall(
                id="tc_1",
                call_id="tc_1",
                name="ripgrep",
                arguments=json.dumps(
                    {
                        "pattern": "TODO",
                        "root": str(repo),
                        "max_count": 5,
                    },
                ),
                type="function_call",
            ),
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _make_final_message_response(text: str) -> Response:
    message = ResponseOutputMessage(
        id="msg_1",
        role="assistant",
        status="completed",
        type="message",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )
    return Response(
        id="resp_1",
        model="gpt-test",
        created_at=int(datetime.now(tz=UTC).timestamp()),
        object="response",
        status="completed",
        output=[message],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def test_ripgrep_tool_schema_contains_expected_fields() -> None:
    """Ripgrep tool definition exposes expected JSON schema information."""
    tool = create_ripgrep_tool()
    payload = tool.to_openai_tool()

    assert payload["function"]["name"] == "ripgrep"
    parameters = payload["function"]["parameters"]
    assert "pattern" in parameters["properties"]
    assert parameters["properties"]["root"]["type"] == "string"
    assert parameters["required"] == ["pattern", "root"]


def test_agent_executes_tool_and_returns_message(sample_repo: Path) -> None:
    """The agent runs a tool call and returns the final assistant message."""
    tool_registry = ToolRegistry([create_ripgrep_tool()])
    initial = _make_tool_call_response(sample_repo)
    final = _make_final_message_response("Search complete.")
    stub_api = _StubResponsesAPI(create_responses=[initial], submit_responses=[final])
    client = cast("Any", _StubOpenAIClient(responses=stub_api))
    agent = ResponsesAgent(
        client,
        model="gpt-test",
        tool_registry=tool_registry,
        tool_limits=ToolLimits(max_calls=3),
        system_prompt="Use ripgrep to look for TODO markers.",
    )

    result = agent.run("Find TODO markers")

    assert result.output_text == "Search complete."
    assert not result.limit_reached
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "ripgrep"
    assert call.args["pattern"] == "TODO"
    assert Path(call.args["root"]) == sample_repo.resolve()
    assert call.stdout_preview is not None
    assert "target.py" in call.stdout_preview

    assert stub_api.submitted, "expected tool outputs to be submitted"
    submission = stub_api.submitted[0]
    assert submission["response_id"] == "resp_1"
    payload = submission["tool_outputs"][0]
    output = json.loads(payload["output"])
    assert output["matches"], "tool output should include matches"


def test_agent_respects_tool_call_limit(sample_repo: Path) -> None:
    """Tool execution halts when the configured limit is exceeded."""
    tool_registry = ToolRegistry([create_ripgrep_tool()])
    first = _make_tool_call_response(sample_repo)
    second = _make_tool_call_response(sample_repo)
    stub_api = _StubResponsesAPI(create_responses=[first], submit_responses=[second])
    client = cast("Any", _StubOpenAIClient(responses=stub_api))
    agent = ResponsesAgent(
        client,
        model="gpt-test",
        tool_registry=tool_registry,
        tool_limits=ToolLimits(max_calls=1),
    )

    result = agent.run("Investigate")

    assert result.limit_reached
    assert len(result.tool_calls) == 1
    assert "limit" in result.output_text.lower()
    # Only a single submission should occur even though the follow-up requested more calls.
    assert len(stub_api.submitted) == 1
