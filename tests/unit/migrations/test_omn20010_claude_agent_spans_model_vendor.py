# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20010 vendor identity for claude_hook_events migration 0003.

The migration adds the nullable model, description and workflow_phase columns
to omninode_internal.claude_agent_spans. The bytes are vendored from omnimarket
node_projection_claude_hook_events, and the table-level grants in 0001 already
cover the new columns.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_claude_hook_events"
_FILE = "0003_add_span_model_and_description.sql"
_SHA256 = "1f1065f78ae810ca414f05e853edac5d8cb2612e2a31c30bb600e4732d74e1f8"
_SQL = _FORWARD / "nodes" / _NODE / _FILE


def test_vendor_bytes_are_pinned() -> None:
    assert hashlib.sha256(_SQL.read_bytes()).hexdigest() == _SHA256


def test_manifest_row_binds_the_bytes() -> None:
    artifact = f"nodes/{_NODE}/{_FILE}"
    rows = {
        row.split("\t", 1)[0]: row.split("\t")
        for row in (_FORWARD / "_ledger" / "application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if row.strip()
    }
    assert rows[artifact] == [
        artifact,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "omninode_internal",
        f"node:{_NODE}:{_FILE}",
        _SHA256,
    ]


def test_the_migration_is_declared_expand_only() -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_FILE}"] == "expand-only"


def test_the_migration_adds_exactly_the_three_nullable_span_columns() -> None:
    statements = "\n".join(
        line
        for line in _SQL.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )
    for column in ("model", "description", "workflow_phase"):
        assert f"ADD COLUMN IF NOT EXISTS {column} TEXT;" in statements
    assert "DROP" not in statements
    assert "DO $$" not in statements
