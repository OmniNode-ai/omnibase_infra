# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for migration 073: add GPU time fields (OMN-10338)."""

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
    / "073_add_llm_call_metrics_gpu_fields.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_073_add_llm_call_metrics_gpu_fields.sql"
)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


@pytest.mark.unit
class TestMigration073Files:
    def test_migration_file_exists(self) -> None:
        assert MIGRATION_FILE.exists()

    def test_rollback_file_exists(self) -> None:
        assert ROLLBACK_FILE.exists()


@pytest.mark.unit
class TestMigration073Schema:
    def test_adds_gpu_metric_columns(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        expected_patterns = {
            "gpu_seconds": r"gpu_seconds\s+NUMERIC\(10,\s*3\)",
            "gpu_type": r"gpu_type\s+VARCHAR\(64\)",
            "gpu_count": r"gpu_count\s+SMALLINT",
            "compute_usage_source": r"compute_usage_source\s+usage_source_type",
        }
        for column, pattern in expected_patterns.items():
            assert re.search(
                rf"\bADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+{pattern}",
                sql,
                re.IGNORECASE,
            ), f"Migration must add {column}"

    def test_adds_compute_cost_to_aggregates(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bALTER\s+TABLE\s+llm_cost_aggregates\b.*"
            r"\bADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+compute_cost_usd\s+"
            r"NUMERIC\(14,\s*6\)\s+NOT\s+NULL\s+DEFAULT\s+0",
            sql,
            re.IGNORECASE | re.DOTALL,
        )

    def test_gpu_type_partial_index_present(self) -> None:
        sql = _strip_comments(MIGRATION_FILE.read_text())
        assert re.search(
            r"\bCREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+"
            r"idx_llm_call_metrics_gpu_type\b.*"
            r"\bON\s+llm_call_metrics\s*\(\s*gpu_type\s*\).*"
            r"\bWHERE\s+gpu_type\s+IS\s+NOT\s+NULL\b",
            sql,
            re.IGNORECASE | re.DOTALL,
        )


@pytest.mark.unit
class TestMigration073Rollback:
    def test_rollback_drops_index_and_columns(self) -> None:
        sql = _strip_comments(ROLLBACK_FILE.read_text())
        assert re.search(
            r"\bDROP\s+INDEX\s+IF\s+EXISTS\s+idx_llm_call_metrics_gpu_type\b",
            sql,
            re.IGNORECASE,
        )
        for column in (
            "compute_usage_source",
            "gpu_count",
            "gpu_type",
            "gpu_seconds",
        ):
            assert re.search(
                rf"\bDROP\s+COLUMN\s+IF\s+EXISTS\s+{column}\b",
                sql,
                re.IGNORECASE,
            ), f"Rollback must drop {column}"
        assert re.search(
            r"\bALTER\s+TABLE\s+llm_cost_aggregates\b.*"
            r"\bDROP\s+COLUMN\s+IF\s+EXISTS\s+compute_cost_usd\b",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
