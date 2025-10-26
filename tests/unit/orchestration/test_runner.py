from pathlib import Path

from code_agent_experiments.domain.models import ModelConfig, RunConfig, Scenario
from code_agent_experiments.orchestration import ScenarioRunner


def _create_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    target = repo / "src" / "app.py"
    target.write_text("""def add(a, b):\n    return a + b\n""", encoding="utf-8")
    return repo


def test_runner_executes_replicates_and_persists(tmp_path: Path) -> None:
    """Run replicated scenarios and ensure artefacts exist."""
    repo_path = _create_repo(tmp_path)
    run = RunConfig(
        id="rg-baseline",
        model=ModelConfig(name="offline"),
        tools_enabled=["ripgrep"],
        replicate=2,
    )
    scenario = Scenario(
        id="scenario-1",
        repo_path=str(repo_path),
        query="add function",
        ground_truth_files=["src/app.py"],
    )

    output_dir = tmp_path / "artifacts"
    runner = ScenarioRunner([run], [scenario], output_dir)
    summaries = runner.execute()

    assert len(summaries) == 2
    run_ids = {summary.run_id for summary in summaries}
    assert run_ids == {"rg-baseline-rep01", "rg-baseline-rep02"}

    for summary in summaries:
        assert summary.metrics
        metrics_path = summary.metrics_path
        assert metrics_path.exists()
        record_dir = metrics_path.parent / "records"
        record_files = list(record_dir.glob("*.json"))
        assert record_files
        for record_file in record_files:
            contents = record_file.read_text(encoding="utf-8")
            assert "scenario-1#r" in contents



def test_runner_handles_missing_repository(tmp_path: Path) -> None:
    """Gracefully handle scenarios that reference missing repositories."""
    run = RunConfig(
        id="rg-baseline",
        model=ModelConfig(name="offline"),
        tools_enabled=["ripgrep"],
    )
    scenario = Scenario(
        id="missing-repo",
        repo_path=str(tmp_path / "does-not-exist"),
        query="anything",
        ground_truth_files=["src/app.py"],
    )

    runner = ScenarioRunner([run], [scenario], tmp_path / "artifacts")
    summaries = runner.execute()
    assert len(summaries) == 1
    metrics = summaries[0].metrics[0]
    assert metrics.recall_at_k == {5: 0.0, 10: 0.0, 20: 0.0}
    assert metrics.mrr == 0.0
