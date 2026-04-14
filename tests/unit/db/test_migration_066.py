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

import re
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


def _strip_comments(sql: str) -> str:
    """Remove SQL line comments (--) and block comments (/* */) before asserting."""
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


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
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bALTER\s+TABLE\b[^;]*\brouting_outcomes\b",
            sql,
            re.IGNORECASE,
        ), "Migration must ALTER TABLE routing_outcomes"

    def test_migration_adds_quality_score_column(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bADD\s+COLUMN\b[^;]*\bquality_score\b",
            sql,
            re.IGNORECASE,
        ), "Migration must ADD COLUMN quality_score"

    def test_quality_score_is_double_precision(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bquality_score\s+DOUBLE\s+PRECISION\b",
            sql,
            re.IGNORECASE,
        ), "quality_score column must be declared as DOUBLE PRECISION"

    def test_migration_is_idempotent(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\b",
            sql,
            re.IGNORECASE,
        ), "Migration must use ADD COLUMN IF NOT EXISTS for idempotency"


@pytest.mark.unit
class TestMigration066Sentinel:
    """Validate the migration sentinel update."""

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
            r"\bschema_version\s*=\s*'066'",
            sql,
            re.IGNORECASE,
        ), "Migration must set schema_version = '066'"


@pytest.mark.unit
class TestMigration066Rollback:
    """Validate the rollback file."""

    def test_rollback_drops_quality_score_column(self) -> None:
        sql = _strip_comments(ROLLBACK_FILE.read_text())
        assert re.search(
            r"\bDROP\s+COLUMN\b[^;]*\bquality_score\b",
            sql,
            re.IGNORECASE,
        ), "Rollback must DROP COLUMN quality_score"

    def test_rollback_reverts_schema_version(self) -> None:
        sql = _strip_comments(ROLLBACK_FILE.read_text())
        assert re.search(
            r"\bschema_version\s*=\s*'065'",
            sql,
            re.IGNORECASE,
        ), "Rollback must revert schema_version to '065'"
