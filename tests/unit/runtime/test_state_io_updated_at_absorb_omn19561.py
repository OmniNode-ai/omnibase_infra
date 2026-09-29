# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Duplicate absorbs preserve the delegation completion-bound clock (OMN-19561)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FORWARD_DIR = Path(__file__).resolve().parents[3] / "docker" / "migrations" / "forward"
MIGRATION = (
    FORWARD_DIR / "109_delegation_workflow_state_updated_at_only_on_transition.sql"
)


@pytest.mark.unit
def test_migration_preserves_updated_at_when_transition_fields_are_unchanged() -> None:
    sql = re.sub(r"--[^\n]*", "", MIGRATION.read_text(encoding="utf-8"))
    sql = " ".join(sql.upper().split())
    assert (
        "CREATE OR REPLACE FUNCTION PUBLIC.REFRESH_DELEGATION_WORKFLOW_STATE_UPDATED_AT()"
        in sql
    )
    for column in ("STATE", "PAYLOAD", "TENANT_ID"):
        assert f"NEW.{column} IS NOT DISTINCT FROM OLD.{column}" in sql
    assert "THEN NEW.UPDATED_AT = OLD.UPDATED_AT;" in sql
    assert "ELSE NEW.UPDATED_AT = NOW();" in sql
