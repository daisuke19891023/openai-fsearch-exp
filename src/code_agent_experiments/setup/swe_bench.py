"""Utilities to ingest SWE-bench scenarios into the internal format."""

from __future__ import annotations

from collections import OrderedDict
import csv
import json
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from code_agent_experiments.domain.models import Scenario

__all__ = ["ingest_swe_bench_scenarios"]


_DIFF_MIN_PARTS = 4


class SWEbenchInstance(BaseModel):
    """Subset of the SWE-bench instance schema used for ingestion."""

    instance_id: str
    repo: str
    patch: str
    problem_statement: str | None = None
    base_commit: str | None = None
    fail_tests: list[str] | None = None
    hints: list[str] | None = None
    environment: str | None = None
    split: str | None = None
    difficulty: str | None = None
    title: str | None = None
    issue_url: str | None = None
    language: str | None = None

    def to_scenario(
        self,
        *,
        repo_root: str | Path | None,
        extra_metadata: dict[str, str] | None,
    ) -> Scenario:
        """Convert the instance into a :class:`Scenario`."""
        ground_truth_files = _extract_ground_truth_files(self.patch)
        if not ground_truth_files:
            message = f"Instance {self.instance_id} has no ground truth files"
            raise ValueError(message)

        metadata: dict[str, Any] = {
            "repo": self.repo,
            "base_commit": self.base_commit,
            "fail_tests": self.fail_tests,
            "hints": self.hints,
            "environment": self.environment,
            "split": self.split,
            "difficulty": self.difficulty,
            "issue_url": self.issue_url,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        metadata = {key: value for key, value in metadata.items() if value}

        query = (
            self.problem_statement
            or self.title
            or f"SWE-bench instance {self.instance_id}"
        )

        language = self.language or "python"
        repo_path = _resolve_repo_path(self.repo, repo_root)

        return Scenario(
            id=self.instance_id,
            repo_path=repo_path,
            query=query,
            language=language,
            ground_truth_files=ground_truth_files,
            metadata=metadata,
        )


def ingest_swe_bench_scenarios(
    json_path: str | Path,
    *,
    csv_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    limit: int = 10,
) -> list[Scenario]:
    """Ingest SWE-bench instances and convert them into :class:`Scenario` objects."""
    instances = _load_instances(json_path)
    metadata_by_id = _load_metadata(csv_path)

    scenarios: list[Scenario] = []
    for instance in instances:
        metadata = metadata_by_id.get(instance.instance_id)
        try:
            scenario = instance.to_scenario(
                repo_root=repo_root,
                extra_metadata=metadata,
            )
        except ValueError:
            continue
        scenarios.append(scenario)
        if limit and len(scenarios) >= limit:
            break

    if limit and len(scenarios) < limit:
        message = (
            f"Requested {limit} scenarios but only {len(scenarios)} valid instances were found"
        )
        raise ValueError(message)

    return scenarios


def _load_instances(json_path: str | Path) -> list[SWEbenchInstance]:
    path = Path(json_path)
    if not path.exists():
        message = f"SWE-bench JSON file not found: {path}"
        raise FileNotFoundError(message)

    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    candidates: list[dict[str, Any]]
    if isinstance(payload, list):
        candidates = cast("list[dict[str, Any]]", payload)
    elif isinstance(payload, dict):
        mapping = cast("dict[str, Any]", payload)
        maybe_items = mapping.get("instances") or mapping.get("items")
        if isinstance(maybe_items, list):
            candidates = cast("list[dict[str, Any]]", maybe_items)
        else:
            message = "SWE-bench JSON must contain a list of instances"
            raise TypeError(message)
    else:
        message = "Unsupported SWE-bench JSON structure"
        raise TypeError(message)

    instances: list[SWEbenchInstance] = []
    for index, item in enumerate(candidates):
        try:
            instance = SWEbenchInstance.model_validate(item)
        except ValidationError as exc:  # pragma: no cover - validation errors are rare
            message = f"Invalid SWE-bench entry at index {index}: {exc}"
            raise ValueError(message) from exc
        instances.append(instance)
    return instances


def _load_metadata(csv_path: str | Path | None) -> dict[str, dict[str, str]]:
    if csv_path is None:
        return {}
    path = Path(csv_path)
    if not path.exists():
        message = f"SWE-bench CSV file not found: {path}"
        raise FileNotFoundError(message)

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        metadata: dict[str, dict[str, str]] = {}
        for row in reader:
            identifier = _extract_identifier(row)
            if not identifier:
                continue
            cleaned = {key: value for key, value in row.items() if value}
            if cleaned:
                metadata[identifier] = cleaned
    return metadata


def _extract_identifier(row: dict[str, str | None]) -> str | None:
    for key in ("instance_id", "id", "problem_id"):
        value = row.get(key)
        if value:
            return value
    return None


def _extract_ground_truth_files(patch: str) -> list[str]:
    files: OrderedDict[str, None] = OrderedDict()
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= _DIFF_MIN_PARTS:
                for candidate in parts[2:4]:
                    path = _normalise_diff_path(candidate)
                    if path:
                        files.setdefault(path, None)
        elif line.startswith(("+++ ", "--- ")):
            path = _normalise_diff_path(line[4:])
            if path:
                files.setdefault(path, None)
    return list(files.keys())


def _normalise_diff_path(value: str) -> str | None:
    value = value.strip()
    if value in {"/dev/null", "a/null", "b/null"}:
        return None
    if value.startswith(("a/", "b/")):
        return value[2:]
    return value or None


def _resolve_repo_path(repo: str, repo_root: str | Path | None) -> str:
    if repo_root is None:
        return repo
    return str(Path(repo_root) / repo)

