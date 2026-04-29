# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for migration 072: add LLM call metric attribution (OMN-10333)."""

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
    / "072_add_llm_call_metrics_attribution.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_072_add_llm_call_metrics_attribution.sql"
)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


@pytest.mark.unit
class TestMigration072Files:
    """Validate required migration files exist."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_FILE.exists(), f"Migration file not found: {MIGRATION_FILE}"

    def test_rollback_file_exists(self) -> None:
        assert ROLLBACK_FILE.exists(), f"Rollback file not found: {ROLLBACK_FILE}"


@pytest.mark.unit
class TestMigration072Schema:
    """Validate LLM call metric attribution schema changes."""

    def test_adds_nullable_attribution_columns(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        for column in ("repo_name", "machine_id"):
            assert re.search(
                rf"\bADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+{column}\s+VARCHAR\(255\)",
                sql,
                re.IGNORECASE,
            ), f"Migration must add nullable VARCHAR(255) column {column}"

    def test_partial_indexes_present(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        expected = {
            "idx_llm_call_metrics_repo_name": "repo_name",
            "idx_llm_call_metrics_machine_id": "machine_id",
        }
        for index_name, column in expected.items():
            assert re.search(
                rf"\bCREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+{index_name}\b"
                rf".*?\bON\s+llm_call_metrics\s*\(\s*{column}\s*\)"
                rf".*?\bWHERE\s+{column}\s+IS\s+NOT\s+NULL\b",
                sql,
                re.IGNORECASE | re.DOTALL,
            ), f"Migration must create partial index {index_name}"


@pytest.mark.unit
class TestMigration072Rollback:
    """Validate rollback SQL."""

    def test_rollback_drops_indexes_and_columns(self) -> None:
        sql = _strip_comments(ROLLBACK_FILE.read_text())
        for index_name in (
            "idx_llm_call_metrics_machine_id",
            "idx_llm_call_metrics_repo_name",
        ):
            assert re.search(
                rf"\bDROP\s+INDEX\s+IF\s+EXISTS\s+{index_name}\b",
                sql,
                re.IGNORECASE,
            ), f"Rollback must drop index {index_name}"

        for column in ("machine_id", "repo_name"):
            assert re.search(
                rf"\bDROP\s+COLUMN\s+IF\s+EXISTS\s+{column}\b",
                sql,
                re.IGNORECASE,
            ), f"Rollback must drop column {column}"
