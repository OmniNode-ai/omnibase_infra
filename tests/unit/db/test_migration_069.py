# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for migration 069: create remote_task_state table (OMN-9631)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent

MIGRATION_FILE = (
    REPO_ROOT / "docker" / "migrations" / "forward" / "069_create_remote_task_state.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_069_drop_remote_task_state.sql"
)

REQUIRED_COLUMNS = [
    "task_id",
    "invocation_kind",
    "protocol",
    "target_ref",
    "remote_task_handle",
    "correlation_id",
    "status",
    "last_remote_status",
    "last_emitted_event_type",
    "submitted_at",
    "updated_at",
    "completed_at",
    "error",
]

LIFECYCLE_VALUES = [
    "SUBMITTED",
    "ACCEPTED",
    "PROGRESS",
    "ARTIFACT",
    "COMPLETED",
    "FAILED",
    "TIMED_OUT",
    "CANCELED",
]


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


@pytest.mark.unit
class TestMigration069Files:
    """Validate that required migration files exist."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_FILE.exists(), f"Migration file not found: {MIGRATION_FILE}"

    def test_rollback_file_exists(self) -> None:
        assert ROLLBACK_FILE.exists(), f"Rollback file not found: {ROLLBACK_FILE}"


@pytest.mark.unit
class TestMigration069Schema:
    """Validate the remote_task_state schema declared by migration 069."""

    def test_creates_remote_task_state_table(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bCREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+public\.remote_task_state\b",
            sql,
            re.IGNORECASE,
        ), "Migration must CREATE TABLE IF NOT EXISTS public.remote_task_state"

    def test_required_columns_present(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        for column in REQUIRED_COLUMNS:
            assert re.search(rf"\b{column}\b", sql), (
                f"Migration must declare column {column}"
            )

    def test_task_id_is_uuid_primary_key(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\btask_id\s+UUID\s+PRIMARY\s+KEY\b",
            sql,
            re.IGNORECASE,
        ), "task_id must be UUID PRIMARY KEY"

    def test_invocation_kind_constraint_matches_core_values(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert "'agent'" in sql
        assert "'model'" in sql
        assert "remote_task_state_invocation_kind_check" in sql

    def test_protocol_constraint_matches_current_core_values(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert "remote_task_state_protocol_check" in sql
        assert "'A2A'" in sql
        assert "'MCP'" not in sql
        assert "'HTTP'" not in sql

    def test_status_constraint_contains_lifecycle_values(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert "remote_task_state_status_check" in sql
        for value in LIFECYCLE_VALUES:
            assert f"'{value}'" in sql

    def test_last_emitted_event_type_constraint_contains_lifecycle_values(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert "remote_task_state_last_emitted_event_type_check" in sql
        for value in LIFECYCLE_VALUES:
            assert f"'{value}'" in sql

    def test_timestamps_have_no_default_now(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        for column in ("submitted_at", "updated_at", "completed_at"):
            declaration = re.search(
                rf"\b{column}\b[^,\n]*",
                sql,
                re.IGNORECASE,
            )
            assert declaration, f"Missing timestamp column {column}"
            assert "DEFAULT NOW()" not in declaration.group(0).upper()


@pytest.mark.unit
class TestMigration069Indexes:
    """Validate required indexes for query paths."""

    def test_status_index_present(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bCREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+idx_remote_task_state_status\b",
            sql,
            re.IGNORECASE,
        ), "Migration must declare idx_remote_task_state_status"

    def test_correlation_id_index_present(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bCREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+idx_remote_task_state_correlation_id\b",
            sql,
            re.IGNORECASE,
        ), "Migration must declare idx_remote_task_state_correlation_id"


@pytest.mark.unit
class TestMigration069Sentinel:
    """Validate migration sentinel updates."""

    def test_migration_updates_sentinel(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bUPDATE\b[^;]*\bdb_metadata\b",
            sql,
            re.IGNORECASE,
        ), "Migration must UPDATE db_metadata"
        assert re.search(
            r"\bmigrations_complete\s*=\s*TRUE\b",
            sql,
            re.IGNORECASE,
        ), "Migration must set migrations_complete = TRUE"

    def test_migration_sets_schema_version(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bschema_version\s*=\s*'069'",
            sql,
            re.IGNORECASE,
        ), "Migration must set schema_version = '069'"


@pytest.mark.unit
class TestMigration069Rollback:
    """Validate rollback SQL."""

    def test_rollback_drops_table(self) -> None:
        sql = _strip_comments(ROLLBACK_FILE.read_text())
        assert re.search(
            r"\bDROP\s+TABLE\s+IF\s+EXISTS\s+public\.remote_task_state\b",
            sql,
            re.IGNORECASE,
        ), "Rollback must DROP TABLE IF EXISTS public.remote_task_state"

    def test_rollback_reverts_schema_version(self) -> None:
        sql = _strip_comments(ROLLBACK_FILE.read_text())
        assert re.search(
            r"\bschema_version\s*=\s*'068'",
            sql,
            re.IGNORECASE,
        ), "Rollback must revert schema_version to '068'"
