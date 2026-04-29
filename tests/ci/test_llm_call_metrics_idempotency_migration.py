# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression checks for the OMN-10332 idempotency migration."""

import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARD = (
    REPO_ROOT
    / "docker/migrations/forward/071_add_llm_call_metrics_idempotency_unique.sql"
)
ROLLBACK = (
    REPO_ROOT
    / "docker/migrations/rollback/rollback_071_add_llm_call_metrics_idempotency_unique.sql"
)


def _normalized_sql(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split()).lower()


def test_forward_migration_enforces_session_event_idempotency() -> None:
    sql = _normalized_sql(FORWARD)

    assert "create unique index if not exists" in sql
    assert "idx_llm_call_metrics_idempotency_unique" in sql
    assert "on llm_call_metrics" in sql
    assert "model_id" in sql
    assert "session_id" in sql
    assert "coalesce(run_id, '')" in sql
    assert "input_hash" in sql
    assert "where input_hash is not null" in sql


def test_rollback_migration_removes_idempotency_index() -> None:
    sql = _normalized_sql(ROLLBACK)

    assert "drop index if exists idx_llm_call_metrics_idempotency_unique" in sql


def test_idempotency_key_prevents_duplicate_durable_rows() -> None:
    """Duplicate logical LLM metric writes collapse to one durable row."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE llm_call_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            session_id TEXT,
            run_id TEXT,
            input_hash TEXT
        );

        CREATE UNIQUE INDEX idx_llm_call_metrics_idempotency_unique
            ON llm_call_metrics (
                model_id,
                session_id,
                COALESCE(run_id, ''),
                input_hash
            )
            WHERE input_hash IS NOT NULL;
        """
    )

    params = ("gpt-4o", "session-123", None, "sha256-same-input")
    insert_sql = """
        INSERT INTO llm_call_metrics (
            model_id, session_id, run_id, input_hash
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT DO NOTHING
    """

    conn.execute(insert_sql, params)
    conn.execute(insert_sql, params)

    row_count = conn.execute("SELECT COUNT(*) FROM llm_call_metrics").fetchone()[0]
    assert row_count == 1
