"""Tests for configuration loading utilities."""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import pytest

from code_agent_experiments.config import (
    LoadError,
    load_environment,
    load_run_configs,
    load_scenarios,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_load_run_configs_with_env_substitution(tmp_path: Path) -> None:
    """It loads run configs while applying environment substitutions."""
    env_file = tmp_path / "runs.env"
    env_file.write_text("""MODEL_NAME=gpt-4.1-mini\nREPO_ROOT=/repos/demo\n""", encoding="utf-8")

    run_yaml = tmp_path / "runs.yaml"
    run_yaml.write_text(
        """
        runs:
          - id: run-${RUN_ID:-001}
            model:
              name: ${MODEL_NAME}
              temperature: 0.2
            tools_enabled:
              - ripgrep
              - fd
            chunking:
              type: fixed
              size_tokens: 256
              overlap_tokens: 32
              languages: [python]
            seed: 42
        """,
        encoding="utf-8",
    )

    result = load_run_configs(run_yaml, env_files=[env_file], overrides={"RUN_ID": "007"})
    assert result.source == run_yaml
    assert len(result.items) == 1
    run = result.items[0]
    assert run.id == "run-007"
    assert run.model.name == "gpt-4.1-mini"
    assert math.isclose(run.model.temperature, 0.2, rel_tol=1e-9)
    assert run.tools_enabled == ["ripgrep", "fd"]
    assert run.chunking.size_tokens == 256
    assert run.seed == 42


def test_load_scenarios_supports_plain_list(tmp_path: Path) -> None:
    """It accepts YAML files with top-level lists."""
    env_file = tmp_path / "scenarios.env"
    env_file.write_text("""REPO_ROOT=/workspace/project\n""", encoding="utf-8")

    scenarios_yaml = tmp_path / "scenarios.yaml"
    scenarios_yaml.write_text(
        """
        - id: ${SCENARIO_ID}
          repo_path: ${REPO_ROOT}
          query: ${QUERY:-Locate failure}
          language: python
          ground_truth_files:
            - src/app.py
            - tests/test_app.py
        """,
        encoding="utf-8",
    )

    result = load_scenarios(
        scenarios_yaml,
        env_files=[env_file],
        overrides={"SCENARIO_ID": "scenario-123"},
    )
    scenario = result.items[0]
    assert scenario.id == "scenario-123"
    assert scenario.repo_path == "/workspace/project"
    assert scenario.query == "Locate failure"
    assert scenario.ground_truth_files == ["src/app.py", "tests/test_app.py"]


def test_missing_environment_variable_raises(tmp_path: Path) -> None:
    """It raises an error when a placeholder lacks a corresponding value."""
    run_yaml = tmp_path / "invalid.yaml"
    run_yaml.write_text(
        """
        runs:
          - id: run-1
            model:
              name: ${MISSING}
            tools_enabled: [ripgrep]
        """,
        encoding="utf-8",
    )

    with pytest.raises(LoadError):
        load_run_configs(run_yaml)


def test_load_environment_merges_overrides(tmp_path: Path) -> None:
    """Environment files merge in order, allowing overrides."""
    env_a = tmp_path / "a.env"
    env_b = tmp_path / "b.env"
    env_a.write_text("""KEY_A=foo\nSHARED=first\n""", encoding="utf-8")
    env_b.write_text("""KEY_B=bar\nSHARED=second\n""", encoding="utf-8")

    values = load_environment([env_a, env_b], base={"BASE": "base"})
    assert values == {"BASE": "base", "KEY_A": "foo", "SHARED": "second", "KEY_B": "bar"}


def test_missing_env_file_raises(tmp_path: Path) -> None:
    """It raises when an expected env file does not exist."""
    missing_path = tmp_path / "missing.env"
    with pytest.raises(LoadError):
        load_environment([missing_path])


def test_placeholder_defaults_are_applied(tmp_path: Path) -> None:
    """Placeholders fall back to defaults when variables are absent."""
    run_yaml = tmp_path / "runs.yaml"
    run_yaml.write_text(
        """
        runs:
          - id: run-1
            model:
              name: ${MODEL_NAME:-gpt-4o-mini}
            tools_enabled: [ripgrep]
        """,
        encoding="utf-8",
    )

    result = load_run_configs(run_yaml)
    assert result.items[0].model.name == "gpt-4o-mini"
