"""Persistence layer for experiment artefacts backed by SQLite."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import sqlite3
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    from pathlib import Path
    from code_agent_experiments.domain.models import Metrics, RetrievalRecord, RunConfig, Scenario


@dataclass(frozen=True, slots=True)
class FailurePayload:
    """Structured representation of a failure event."""

    run_id: str
    scenario_id: str
    replicate_index: int
    error: str
    timestamp: datetime

    def to_row(self) -> tuple[str, str, int, str, str, str]:
        """Return a SQLite row payload for this failure."""
        now = datetime.now(tz=UTC).isoformat()
        return (
            self.run_id,
            self.scenario_id,
            self.replicate_index,
            self.error,
            self.timestamp.isoformat(),
            now,
        )


class SQLiteExperimentStorage:
    """SQLite-backed persistence helper for experiment runs."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path) -> None:
        """Configure storage against ``db_path``."""
        self._db_path = db_path

    def initialize(self) -> None:
        """Initialise the database file and apply migrations if required."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            self._ensure_schema(connection)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._db_path)
        try:
            connection.row_factory = sqlite3.Row
            yield connection
        finally:
            connection.close()

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        """Ensure that the SQLite schema is materialised and up to date."""
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY CHECK(version > 0)
                )
                """,
            )
            row = connection.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1",
            ).fetchone()
            if row is None:
                self._apply_initial_schema(connection)
                connection.execute(
                    "INSERT INTO schema_version(version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )
            elif row["version"] != self.SCHEMA_VERSION:
                message = (
                    "Unsupported experiment storage schema version: "
                    f"{row['version']}"
                )
                raise RuntimeError(message)

    def _apply_initial_schema(self, connection: sqlite3.Connection) -> None:
        """Create the initial schema for experiment persistence."""
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_configs (
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                written_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scenarios (
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                written_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS retrieval_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                payload TEXT NOT NULL,
                written_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_records_run ON retrieval_records(run_id);
            CREATE INDEX IF NOT EXISTS idx_retrieval_records_scenario ON retrieval_records(scenario_id);
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                payload TEXT NOT NULL,
                written_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_scenario ON metrics(scenario_id);
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                replicate_index INTEGER NOT NULL,
                error TEXT NOT NULL,
                failure_at TEXT NOT NULL,
                written_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_failures_run ON failures(run_id);
            CREATE INDEX IF NOT EXISTS idx_failures_scenario ON failures(scenario_id);
            """,
        )

    def upsert_run_configs(self, configs: Iterable[RunConfig]) -> None:
        """Persist ``configs`` ensuring duplicates are updated."""
        now = datetime.now(tz=UTC).isoformat()
        rows = [
            (config.id, config.model_dump_json(), now)
            for config in configs
        ]
        if not rows:
            return
        with self._connect() as connection, connection:
            connection.executemany(
                """
                INSERT INTO run_configs(id, payload, written_at)
                VALUES(?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    payload = excluded.payload,
                    written_at = excluded.written_at
                """,
                rows,
            )

    def upsert_scenarios(self, scenarios: Iterable[Scenario]) -> None:
        """Persist ``scenarios`` ensuring duplicates are updated."""
        now = datetime.now(tz=UTC).isoformat()
        rows = [
            (scenario.id, scenario.model_dump_json(), now)
            for scenario in scenarios
        ]
        if not rows:
            return
        with self._connect() as connection, connection:
            connection.executemany(
                """
                INSERT INTO scenarios(id, payload, written_at)
                VALUES(?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    payload = excluded.payload,
                    written_at = excluded.written_at
                """,
                rows,
            )

    def insert_retrieval_record(self, record: RetrievalRecord) -> None:
        """Insert a retrieval ``record`` entry."""
        now = datetime.now(tz=UTC).isoformat()
        with self._connect() as connection, connection:
            connection.execute(
                """
                INSERT INTO retrieval_records(
                    run_id,
                    scenario_id,
                    strategy,
                    elapsed_ms,
                    payload,
                    written_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.scenario_id,
                    record.strategy,
                    record.elapsed_ms,
                    record.model_dump_json(),
                    now,
                ),
            )

    def insert_metrics(self, metrics: Metrics) -> None:
        """Insert a ``metrics`` entry."""
        now = datetime.now(tz=UTC).isoformat()
        with self._connect() as connection, connection:
            connection.execute(
                """
                INSERT INTO metrics(
                    run_id,
                    scenario_id,
                    strategy,
                    payload,
                    written_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    metrics.run_id,
                    metrics.scenario_id,
                    metrics.strategy,
                    metrics.model_dump_json(),
                    now,
                ),
            )

    def insert_failure(self, payload: FailurePayload) -> None:
        """Insert a failure telemetry ``payload`` entry."""
        with self._connect() as connection, connection:
            connection.execute(
                """
                INSERT INTO failures(
                    run_id,
                    scenario_id,
                    replicate_index,
                    error,
                    failure_at,
                    written_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                payload.to_row(),
            )

    @property
    def db_path(self) -> Path:
        """Return the configured database path."""
        return self._db_path
