# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for migration 066: add quality_score column to routing_outcomes (OMN-8666).

Validates:
1. Migration SQL file exists.
2. Rollback SQL file exists.
3. Migration is idempotent (ADD COLUMN IF NOT EXISTS).
4. quality_score column is DOUBLE PRECISION.
5. Migration updates the sentinel with schema_version = '066'.
6. Rollback drops the column.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent

MIGRATION_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "066_add_quality_score_to_routing_outcomes.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_066_add_quality_score_to_routing_outcomes.sql"
)


@pytest.mark.unit
class TestMigration066Files:
    """Validate that required migration files exist."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_FILE.exists(), f"Migration file not found: {MIGRATION_FILE}"

    def test_rollback_file_exists(self) -> None:
        assert ROLLBACK_FILE.exists(), f"Rollback file not found: {ROLLBACK_FILE}"


@pytest.mark.unit
class TestMigration066Schema:
    """Validate the schema changes declared in migration 066."""

    def test_migration_targets_routing_outcomes(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "routing_outcomes" in sql, (
            "Migration must target the routing_outcomes table"
        )

    def test_migration_adds_quality_score_column(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "quality_score" in sql, "Migration must add the quality_score column"

    def test_quality_score_is_double_precision(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "DOUBLE PRECISION" in sql, (
            "quality_score column must be declared as DOUBLE PRECISION"
        )

    def test_migration_is_idempotent(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "ADD COLUMN IF NOT EXISTS" in sql, (
            "Migration must use ADD COLUMN IF NOT EXISTS for idempotency"
        )


@pytest.mark.unit
class TestMigration066Sentinel:
    """Validate the migration sentinel update."""

    def test_migration_updates_sentinel(self) -> None:
        sql = MIGRATION_FILE.read_text().upper()
        assert "MIGRATIONS_COMPLETE" in sql, (
            "Migration must update db_metadata.migrations_complete"
        )
        assert "TRUE" in sql

    def test_migration_sets_schema_version(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "'066'" in sql, "Migration must set schema_version = '066'"


@pytest.mark.unit
class TestMigration066Rollback:
    """Validate the rollback file."""

    def test_rollback_drops_quality_score_column(self) -> None:
        sql = ROLLBACK_FILE.read_text().upper()
        assert "DROP COLUMN" in sql, "Rollback must drop the quality_score column"
        assert "QUALITY_SCORE" in sql

    def test_rollback_reverts_schema_version(self) -> None:
        sql = ROLLBACK_FILE.read_text()
        assert "'065'" in sql, "Rollback must revert schema_version to '065'"
