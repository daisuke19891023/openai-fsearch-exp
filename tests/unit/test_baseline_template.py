from pathlib import Path
from typing import Any, cast

import yaml


def test_baseline_template_includes_expected_profiles() -> None:
    """Baseline template exposes the three comparison profiles."""
    template_path = Path("docs/templates/baseline_runs.yaml")
    assert template_path.exists()
    raw_payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert isinstance(raw_payload, dict)
    payload = cast(dict[str, Any], raw_payload)
    runs = payload.get("runs")
    assert isinstance(runs, list)
    runs_list = cast(list[Any], runs)
    run_ids: set[str] = set()
    for entry_obj in runs_list:
        if not isinstance(entry_obj, dict):
            continue
        entry = cast(dict[str, Any], entry_obj)
        identifier = entry.get("id")
        if isinstance(identifier, str):
            run_ids.add(identifier)
    assert {"rg-only", "faiss-fixed", "hybrid"} <= run_ids
