"""Builtin tool definitions that wrap local Python functionality."""

from __future__ import annotations

from pathlib import Path
from code_agent_experiments.agent.tooling import FunctionToolDefinition
from code_agent_experiments.domain.tool_schemas import RipgrepToolInput, RipgrepToolOutput
from code_agent_experiments.tools.shell import run_ripgrep

__all__ = ["create_ripgrep_tool"]


def create_ripgrep_tool() -> FunctionToolDefinition[RipgrepToolInput, RipgrepToolOutput]:
    """Return the ripgrep tool wired to the local Python fallback implementation."""

    def _handler(payload: RipgrepToolInput) -> RipgrepToolOutput:
        root = Path(payload.root).expanduser().resolve()
        matches = run_ripgrep(
            payload.pattern,
            root,
            globs=payload.globs or None,
            ignore_case=payload.ignore_case,
            max_count=payload.max_count,
        )
        return RipgrepToolOutput.from_runtime(matches, limit=payload.max_count)

    return FunctionToolDefinition(
        name="ripgrep",
        description=(
            "Search repository files using ripgrep-compatible regular expressions."
        ),
        input_model=RipgrepToolInput,
        output_model=RipgrepToolOutput,
        handler=_handler,
    )

