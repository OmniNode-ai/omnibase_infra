# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit checks for migration 076 savings_estimates provenance columns."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "076_add_savings_estimate_provenance.sql"
)
ROLLBACK = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_076_add_savings_estimate_provenance.sql"
)


def test_076_forward_adds_provenance_columns_and_constraints() -> None:
    sql = FORWARD.read_text(encoding="utf-8").lower()

    assert "alter table savings_estimates" in sql
    assert (
        "add column if not exists usage_source text not null default 'unknown'" in sql
    )
    assert "add column if not exists estimation_method text" in sql
    assert "add column if not exists source_payload_hash text" in sql
    assert "chk_savings_estimates_usage_source" in sql
    assert "measured" in sql
    assert "estimated" in sql
    assert "unknown" in sql


def test_076_rollback_removes_provenance_columns_and_constraints() -> None:
    sql = ROLLBACK.read_text(encoding="utf-8").lower()

    assert "drop constraint if exists chk_savings_estimates_usage_source" in sql
    assert "drop column if exists usage_source" in sql
    assert "drop column if exists estimation_method" in sql
    assert "drop column if exists source_payload_hash" in sql
