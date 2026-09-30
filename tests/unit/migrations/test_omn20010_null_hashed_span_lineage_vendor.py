# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20010 vendor identity for claude_hook_events migration 0004.

The migration nulls sha256-valued model, description and workflow_phase on
omninode_internal.claude_agent_spans, left there while the emit drainer ran an
omnimarket whose capture redaction hashed the lineage agent fields. The bytes
are vendored from omnimarket node_projection_claude_hook_events.
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
_FILE = "0004_null_hashed_span_lineage.sql"
_SHA256 = "4122394679bffd61da47600caf9698b6502efdd4d32a7befa9f9165f56fb7247"
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


def test_the_migration_is_declared_forward_only() -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_FILE}"] == "forward-only"


def test_the_migration_is_static_sql_that_only_nulls_hashed_span_columns() -> None:
    statements = "\n".join(
        line
        for line in _SQL.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )
    for column in ("model", "description", "workflow_phase"):
        assert f"SET {column} = NULL" in statements
        assert f"WHERE {column} LIKE 'sha256:%'" in statements
    assert statements.count("UPDATE omninode_internal.claude_agent_spans") == 3
    assert "DO $$" not in statements
    assert "EXECUTE" not in statements
    assert "DELETE" not in statements
    assert "DROP" not in statements
