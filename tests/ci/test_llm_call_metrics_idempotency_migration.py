# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression checks for the OMN-10332 idempotency migration."""

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
