"""Tests for the SWE-bench scenario importer."""

from __future__ import annotations

import csv
import json

import pytest
from typing import TYPE_CHECKING

from code_agent_experiments.setup.swe_bench import ingest_swe_bench_scenarios


if TYPE_CHECKING:
    from pathlib import Path


def test_ingest_swe_bench_scenarios_produces_ten_unique_cases(
    tmp_path: Path,
) -> None:
    """Ensure the importer yields 10 scenarios with unique ground truth files."""
    json_path = _write_instances(tmp_path)
    csv_path = _write_metadata(tmp_path)
    repo_root = tmp_path / "repos"

    scenarios = ingest_swe_bench_scenarios(
        json_path,
        csv_path=csv_path,
        repo_root=repo_root,
        limit=10,
    )

    assert len(scenarios) == 10
    for index, scenario in enumerate(scenarios):
        assert scenario.ground_truth_files, "ground_truth_files should not be empty"
        assert len(scenario.ground_truth_files) == len(
            set(scenario.ground_truth_files),
        ), "ground_truth_files should not contain duplicates"
        assert scenario.repo_path == str(repo_root / "org__repo")
        assert scenario.id == f"swe-{index}"

    first_metadata = scenarios[0].metadata
    assert first_metadata["base_commit"] == "commit-0"
    assert first_metadata["environment"] == "py38"
    assert first_metadata["split"] == "train"
    assert first_metadata["notes"] == "dataset-row-0"


def test_ingest_swe_bench_requires_enough_valid_instances(
    tmp_path: Path,
) -> None:
    """Verify an error is raised when not enough valid instances exist."""
    json_path = tmp_path / "instances.json"
    json_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "instance_id": "invalid",
                        "repo": "org__repo",
                        "patch": "",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Requested 1 scenarios"):
        ingest_swe_bench_scenarios(json_path, limit=1)


def _write_instances(tmp_path: Path) -> Path:
    entries: list[dict[str, object]] = []
    for index in range(12):
        patch_lines = [
            f"diff --git a/src/module_{index}.py b/src/module_{index}.py",
            f"--- a/src/module_{index}.py",
            f"+++ b/src/module_{index}.py",
            "@@ -1 +1 @@",
            "-old",
            "+new",
        ]
        if index % 2 == 0:
            patch_lines.extend(
                [
                    f"diff --git a/tests/test_module_{index}.py b/tests/test_module_{index}.py",
                    f"--- a/tests/test_module_{index}.py",
                    f"+++ b/tests/test_module_{index}.py",
                ],
            )
        patch = "\n".join(patch_lines)
        entries.append(
            {
                "instance_id": f"swe-{index}",
                "repo": "org__repo",
                "patch": patch,
                "problem_statement": f"Fix bug {index}",
                "base_commit": f"commit-{index}",
                "fail_tests": [f"tests/test_module_{index}.py::test_case"],
                "hints": ["consider edge cases"],
                "split": "train",
                "difficulty": "medium",
                "issue_url": f"https://example.com/{index}",
                "language": "python",
            },
        )

    # Include an invalid entry that should be ignored because the patch is empty.
    entries.append(
        {
            "instance_id": "invalid-entry",
            "repo": "org__repo",
            "patch": "",
        },
    )

    path = tmp_path / "instances.json"
    path.write_text(json.dumps({"instances": entries}), encoding="utf-8")
    return path


def _write_metadata(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance_id", "environment", "notes"])
        writer.writeheader()
        for index in range(12):
            writer.writerow(
                {
                    "instance_id": f"swe-{index}",
                    "environment": "py38",
                    "notes": f"dataset-row-{index}",
                },
            )
    return path

