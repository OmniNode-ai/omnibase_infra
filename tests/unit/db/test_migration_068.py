# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for migration 068: create landing_content_projection table (OMN-9661).

Validates:
1. Migration SQL file exists.
2. Rollback SQL file exists.
3. CREATE TABLE uses IF NOT EXISTS (idempotency).
4. Composite PK (content_kind, schema_version_major) declared.
5. Required columns present with correct types.
6. last_applied_event_id is NOT NULL.
7. last_applied_offset is NOT NULL DEFAULT 0.
8. CHECK constraint for non-negative offset.
9. CHECK constraint for non-negative sequence.
10. Two indexes declared.
11. Sentinel updated to schema_version = '068'.
12. Rollback drops the table.
13. Rollback reverts sentinel to '067'.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent

MIGRATION_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "068_create_landing_content_projection.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_068_create_landing_content_projection.sql"
)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


@pytest.mark.unit
class TestMigration068Files:
    def test_migration_file_exists(self) -> None:
        assert MIGRATION_FILE.exists(), f"Migration file not found: {MIGRATION_FILE}"

    def test_rollback_file_exists(self) -> None:
        assert ROLLBACK_FILE.exists(), f"Rollback file not found: {ROLLBACK_FILE}"


@pytest.mark.unit
class TestMigration068Schema:
    def test_creates_table_if_not_exists(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bCREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\b",
            sql,
            re.IGNORECASE,
        ), "Migration must use CREATE TABLE IF NOT EXISTS"

    def test_table_name(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\blanding_content_projection\b",
            sql,
            re.IGNORECASE,
        ), "Migration must reference landing_content_projection table"

    def test_composite_primary_key(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bPRIMARY\s+KEY\s*\([^)]*content_kind[^)]*schema_version_major[^)]*\)",
            sql,
            re.IGNORECASE,
        ), "Migration must declare composite PK (content_kind, schema_version_major)"

    def test_data_column_is_jsonb(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bdata\s+JSONB\b",
            sql,
            re.IGNORECASE,
        ), "data column must be JSONB"

    def test_last_applied_event_id_not_null(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\blast_applied_event_id\s+UUID\s+NOT\s+NULL\b",
            sql,
            re.IGNORECASE,
        ), "last_applied_event_id must be UUID NOT NULL"

    def test_last_applied_offset_not_null_default_zero(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\blast_applied_offset\s+BIGINT\s+NOT\s+NULL\s+DEFAULT\s+0\b",
            sql,
            re.IGNORECASE,
        ), "last_applied_offset must be BIGINT NOT NULL DEFAULT 0"

    def test_check_constraint_valid_offset(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bvalid_offset\b",
            sql,
            re.IGNORECASE,
        ), "Migration must declare valid_offset CHECK constraint"

    def test_check_constraint_valid_sequence(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bvalid_sequence\b",
            sql,
            re.IGNORECASE,
        ), "Migration must declare valid_sequence CHECK constraint"

    def test_two_indexes_declared(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        matches = re.findall(
            r"\bCREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\b",
            sql,
            re.IGNORECASE,
        )
        assert len(matches) >= 2, "Migration must declare at least 2 indexes"


@pytest.mark.unit
class TestMigration068Sentinel:
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
            r"\bschema_version\s*=\s*'068'",
            sql,
            re.IGNORECASE,
        ), "Migration must set schema_version = '068'"


@pytest.mark.unit
class TestMigration068Rollback:
    def test_rollback_drops_table(self) -> None:
        sql = _strip_comments(ROLLBACK_FILE.read_text())
        assert re.search(
            r"\bDROP\s+TABLE\b[^;]*\blanding_content_projection\b",
            sql,
            re.IGNORECASE,
        ), "Rollback must DROP TABLE landing_content_projection"

    def test_rollback_reverts_schema_version(self) -> None:
        sql = _strip_comments(ROLLBACK_FILE.read_text())
        assert re.search(
            r"\bschema_version\s*=\s*'067'",
            sql,
            re.IGNORECASE,
        ), "Rollback must revert schema_version to '067'"
