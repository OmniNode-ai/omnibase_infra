# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for migration 074: create savings_estimates table (OMN-10340)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent
MIGRATION_FILE = (
    REPO_ROOT / "docker" / "migrations" / "forward" / "074_create_savings_estimates.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_074_create_savings_estimates.sql"
)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


@pytest.mark.unit
class TestMigration074Files:
    def test_migration_file_exists(self) -> None:
        assert MIGRATION_FILE.exists()

    def test_rollback_file_exists(self) -> None:
        assert ROLLBACK_FILE.exists()


@pytest.mark.unit
class TestMigration074Schema:
    def test_creates_savings_estimates_table(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bCREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+savings_estimates\b",
            sql,
            re.IGNORECASE,
        )

    def test_required_columns_present(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        for column in (
            "id",
            "event_timestamp",
            "session_id",
            "model_local",
            "model_cloud_baseline",
            "local_cost_usd",
            "cloud_cost_usd",
            "savings_usd",
            "repo_name",
            "machine_id",
            "created_at",
        ):
            assert re.search(rf"\b{column}\b", sql), f"missing {column}"

    def test_numeric_money_columns_have_six_decimal_places(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        for column in ("local_cost_usd", "cloud_cost_usd", "savings_usd"):
            assert re.search(
                rf"\b{column}\s+NUMERIC\(14,\s*6\)\s+NOT\s+NULL\b",
                sql,
                re.IGNORECASE,
            )

    def test_constraints_match_projection_idempotency(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert "savings_consistency" in sql
        assert "savings_usd = cloud_cost_usd - local_cost_usd" in sql
        assert "unique_savings_estimate_event" in sql
        for column in (
            "session_id",
            "event_timestamp",
            "model_local",
            "model_cloud_baseline",
        ):
            assert column in sql

    def test_indexes_present(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        for index in (
            "idx_savings_estimates_event_ts",
            "idx_savings_estimates_session",
            "idx_savings_estimates_model_local",
        ):
            assert index in sql


@pytest.mark.unit
class TestMigration074Rollback:
    def test_rollback_drops_table_and_indexes(self) -> None:
        sql = _strip_comments(ROLLBACK_FILE.read_text())
        assert "DROP TABLE IF EXISTS savings_estimates" in sql
        for index in (
            "idx_savings_estimates_event_ts",
            "idx_savings_estimates_session",
            "idx_savings_estimates_model_local",
        ):
            assert index in sql
