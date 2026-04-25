# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for migration 069: create remote_task_state."""

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
    / "rollback_069_create_remote_task_state.sql"
)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


@pytest.mark.unit
def test_migration_069_files_exist() -> None:
    assert MIGRATION_FILE.exists()
    assert ROLLBACK_FILE.exists()


@pytest.mark.unit
def test_migration_069_creates_remote_task_state_table() -> None:
    sql = _strip_comments(MIGRATION_FILE.read_text())
    assert "CREATE TABLE IF NOT EXISTS remote_task_state" in sql
    assert "target_ref" in sql
    assert "remote_task_handle" in sql
    assert "idx_remote_task_state_status" in sql
    assert "idx_remote_task_state_correlation_id" in sql
    assert "schema_version = '069'" in sql


@pytest.mark.unit
def test_migration_069_declares_lifecycle_check_constraints() -> None:
    sql = _strip_comments(MIGRATION_FILE.read_text())
    assert "chk_remote_task_state_status" in sql
    assert "chk_remote_task_state_last_emitted" in sql
    assert "TIMED_OUT" in sql
    assert "CANCELED" in sql


@pytest.mark.unit
def test_migration_069_rollback_reverts_schema_version() -> None:
    sql = _strip_comments(ROLLBACK_FILE.read_text())
    assert "DROP TABLE IF EXISTS remote_task_state" in sql
    assert "schema_version = '068'" in sql
