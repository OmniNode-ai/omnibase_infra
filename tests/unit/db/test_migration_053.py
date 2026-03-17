# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for migration 053: create context_audit_events table (OMN-5239).

Validates:
1. Migration SQL file exists.
2. Rollback SQL file exists.
3. Migration is idempotent (CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS).
4. Table schema contains all required columns.
5. All 4 required indexes are declared.
6. Migration updates the sentinel with schema_version = '053'.
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
    / "053_create_context_audit_events.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_053_create_context_audit_events.sql"
)

REQUIRED_COLUMNS = [
    "id",
    "task_id",
    "parent_task_id",
    "correlation_id",
    "contract_id",
    "event_type",
    "enforcement_level",
    "enforcement_action",
    "violation_details",
    "context_tokens_used",
    "context_budget_tokens",
    "return_tokens",
    "return_max_tokens",
    "created_at",
]

REQUIRED_INDEXES = [
    "idx_context_audit_correlation",
    "idx_context_audit_task",
    "idx_context_audit_type",
    "idx_context_audit_created",
]


@pytest.mark.unit
class TestMigration053Files:
    """Validate that required migration files exist."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_FILE.exists(), f"Migration file not found: {MIGRATION_FILE}"

    def test_rollback_file_exists(self) -> None:
        assert ROLLBACK_FILE.exists(), f"Rollback file not found: {ROLLBACK_FILE}"


@pytest.mark.unit
class TestMigration053Schema:
    """Validate the schema declared in migration 053."""

    def test_migration_creates_context_audit_events_table(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "context_audit_events" in sql, (
            "Migration must create the context_audit_events table"
        )

    def test_migration_is_idempotent_table(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "CREATE TABLE IF NOT EXISTS" in sql, (
            "Migration must use CREATE TABLE IF NOT EXISTS for idempotency"
        )

    def test_migration_is_idempotent_indexes(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "CREATE INDEX IF NOT EXISTS" in sql, (
            "Migration must use CREATE INDEX IF NOT EXISTS for idempotency"
        )

    def test_required_columns_present(self) -> None:
        sql = MIGRATION_FILE.read_text()
        for column in REQUIRED_COLUMNS:
            assert column in sql, (
                f"Migration must declare column '{column}' in context_audit_events"
            )

    def test_created_at_has_default(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "created_at" in sql
        assert "DEFAULT NOW()" in sql, "created_at must have DEFAULT NOW()"

    def test_violation_details_is_jsonb(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "violation_details" in sql
        assert "JSONB" in sql, "violation_details must be of type JSONB"

    def test_primary_key_is_bigserial(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "BIGSERIAL" in sql, (
            "id column must use BIGSERIAL for auto-incrementing primary key"
        )


@pytest.mark.unit
class TestMigration053Indexes:
    """Validate all 4 required indexes are declared."""

    def test_all_required_indexes_present(self) -> None:
        sql = MIGRATION_FILE.read_text()
        for index in REQUIRED_INDEXES:
            assert index in sql, f"Migration must declare index '{index}'"

    def test_index_count(self) -> None:
        # Count only non-comment lines that create indexes
        lines = MIGRATION_FILE.read_text().splitlines()
        index_count = sum(
            1
            for line in lines
            if "CREATE INDEX IF NOT EXISTS" in line
            and not line.lstrip().startswith("--")
        )
        assert index_count == len(REQUIRED_INDEXES), (
            f"Migration must declare exactly {len(REQUIRED_INDEXES)} indexes, "
            f"found {index_count}"
        )


@pytest.mark.unit
class TestMigration053Sentinel:
    """Validate the migration sentinel update."""

    def test_migration_updates_sentinel(self) -> None:
        sql = MIGRATION_FILE.read_text().upper()
        assert "MIGRATIONS_COMPLETE" in sql, (
            "Migration must update db_metadata.migrations_complete"
        )
        assert "TRUE" in sql

    def test_migration_sets_schema_version(self) -> None:
        sql = MIGRATION_FILE.read_text()
        assert "'053'" in sql, "Migration must set schema_version = '053'"

    def test_rollback_drops_table(self) -> None:
        sql = ROLLBACK_FILE.read_text().upper()
        assert "DROP TABLE" in sql, "Rollback must drop the context_audit_events table"
        assert "CONTEXT_AUDIT_EVENTS" in sql
