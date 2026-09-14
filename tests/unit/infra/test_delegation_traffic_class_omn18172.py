# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18172 typed delegation traffic-class persistence pins."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = (
    REPO_ROOT
    / "docker/migrations/forward/106_add_delegation_workflow_state_traffic_class.sql"
)


@pytest.mark.unit
def test_traffic_class_is_stored_from_authoritative_provenance() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "ALTER TABLE delegation_workflow_state" in sql
    assert "traffic_class TEXT" in sql
    assert "GENERATED ALWAYS AS" in sql
    assert "payload #>> '{request,provenance,traffic_class}'" in sql
    assert ") STORED" in sql
    assert "COALESCE(" in sql and "'unclassified'" in sql


@pytest.mark.unit
def test_traffic_class_rejects_values_outside_the_canonical_enum() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "CHECK (traffic_class IN ('unclassified', 'organic', 'synthetic'))" in sql


@pytest.mark.unit
def test_migration_never_materializes_prompt_text() -> None:
    """The reader gets a classifier column, never a second copy of the prompt."""
    executable = "\n".join(
        line
        for line in MIGRATION.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )

    assert "prompt" not in executable.lower()
