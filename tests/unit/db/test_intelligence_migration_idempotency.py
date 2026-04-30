# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static checks for idempotent omniintelligence migrations."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INTELLIGENCE_MIGRATIONS = REPO_ROOT / "docker" / "migrations" / "intelligence"
INTELLIGENCE_MIGRATION_RUNNER = REPO_ROOT / "scripts" / "run-intelligence-migrations.sh"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("migration_name", "table_name", "constraint_name"),
    [
        (
            "009_add_signature_hash.sql",
            "learned_patterns",
            "unique_signature_hash_domain_version",
        ),
        (
            "018_migrate_routing_feedback_scores_outcome_raw.sql",
            "routing_feedback_scores",
            "uq_routing_feedback_scores_session",
        ),
    ],
)
def test_unique_constraint_migrations_guard_partial_replay(
    migration_name: str,
    table_name: str,
    constraint_name: str,
) -> None:
    migration_sql = (INTELLIGENCE_MIGRATIONS / migration_name).read_text(
        encoding="utf-8",
    )

    assert "pg_constraint" in migration_sql
    assert f"conname = '{constraint_name}'" in migration_sql
    assert f"conrelid = '{table_name}'::regclass" in migration_sql
    assert f"ADD CONSTRAINT {constraint_name}" in migration_sql


@pytest.mark.unit
def test_intelligence_migration_runner_provisions_cross_repo_idempotency_table() -> (
    None
):
    runner = INTELLIGENCE_MIGRATION_RUNNER.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS idempotency_records" in runner
    assert "idx_idempotency_records_processed_at" in runner
    assert "idx_idempotency_records_domain" in runner
    assert "idx_idempotency_records_correlation_id" in runner
    assert "Cross-repo idempotency table ready" in runner
