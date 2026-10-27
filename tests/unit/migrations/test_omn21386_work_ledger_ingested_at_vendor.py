# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-21386: pin the work-ledger ingestion-time vendor bytes and declarations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_work_ledger"
_VENDOR = _FORWARD / "nodes" / _NODE
_SEQUENCE = "0003_work_ledger_seq.sql"
_INGESTED_AT = "0006_work_ledger_ingested_at.sql"
# sha256 of the omnimarket#4408 source bytes at 858a0c67dba2197df290508a839c9021ee4a7e75.
_SHA256 = "d38170eb852f86533de9fcbcad0131d3370c6a80f0999a5ef281cd274a6eab49"
_TABLE = "omninode_internal.work_ledger_rows"


def _statements(filename: str) -> list[str]:
    sql = re.sub(r"--[^\n]*", "", (_VENDOR / filename).read_text(encoding="utf-8"))
    return [" ".join(s.split()) for s in sql.split(";") if s.strip()]


def test_vendor_bytes_and_manifest_binding_are_exact() -> None:
    path = _VENDOR / _INGESTED_AT
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _SHA256
    artifact = f"nodes/{_NODE}/{_INGESTED_AT}"
    rows = [
        row.split("\t")
        for row in (_FORWARD / "_ledger" / "application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if row.split("\t", 1)[0] == artifact
    ]
    assert rows == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{_INGESTED_AT}",
            _SHA256,
        ]
    ]


def test_migration_class_is_forward_only() -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_INGESTED_AT}"] == "forward-only"


def test_nullable_column_is_added_before_its_insert_default() -> None:
    column, default = _statements(_INGESTED_AT)
    assert column == (
        f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ"
    )
    assert default == (
        f"ALTER TABLE {_TABLE} ALTER COLUMN ingested_at SET DEFAULT clock_timestamp()"
    )
    # Existing rows keep NULL: no NOT NULL, no default on the ADD, no backfill.
    assert not re.search(r"\b(NOT NULL|DEFAULT|UPDATE|DELETE)\b", column)
    assert not re.search(r"\b(UPDATE|DELETE|DROP)\b", default)


def test_it_applies_after_the_sequence_migration() -> None:
    order = sorted(path.name for path in _VENDOR.glob("*.sql"))
    assert order.index(_SEQUENCE) < order.index(_INGESTED_AT)
