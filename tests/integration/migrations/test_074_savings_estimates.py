# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration-facing migration contract checks for savings_estimates."""

from __future__ import annotations

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


@pytest.mark.integration
def test_074_savings_estimates_forward_declares_projection_contract() -> None:
    sql = MIGRATION_FILE.read_text()

    assert "CREATE TABLE IF NOT EXISTS savings_estimates" in sql
    assert "NUMERIC(14, 6)" in sql
    assert "CONSTRAINT savings_consistency" in sql
    assert "CONSTRAINT unique_savings_estimate_event" in sql
    assert "idx_savings_estimates_event_ts" in sql
    assert "idx_savings_estimates_session" in sql
    assert "idx_savings_estimates_model_local" in sql


@pytest.mark.integration
def test_074_savings_estimates_rollback_removes_projection_table() -> None:
    sql = ROLLBACK_FILE.read_text()

    assert "DROP TABLE IF EXISTS savings_estimates" in sql
