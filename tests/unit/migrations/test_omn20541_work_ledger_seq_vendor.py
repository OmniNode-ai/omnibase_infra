# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20541: pin the work-ledger sequence vendor bytes and prerequisites."""

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
_CREATE = "0000_create_work_ledger.sql"
_SEQUENCE = "0003_work_ledger_seq.sql"
# sha256 of the omnimarket source bytes vendored for OMN-20541.
_SHA256 = "29360c6c1478484f486a8d6cae61ac253e1a5e09d4b7438074b760df627fe443"
_TABLE = "omninode_internal.work_ledger_rows"


def _statements(filename: str) -> list[str]:
    sql = re.sub(r"--[^\n]*", "", (_VENDOR / filename).read_text(encoding="utf-8"))
    return [" ".join(s.split()) for s in sql.split(";") if s.strip()]


def test_vendor_bytes_and_manifest_binding_are_exact() -> None:
    path = _VENDOR / _SEQUENCE
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _SHA256
    artifact = f"nodes/{_NODE}/{_SEQUENCE}"
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
            f"node:{_NODE}:{_SEQUENCE}",
            _SHA256,
        ]
    ]


def test_migration_class_is_expand_only() -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_SEQUENCE}"] == "expand-only"


def test_only_nullable_column_and_non_unique_partial_index_are_added() -> None:
    column, index = _statements(_SEQUENCE)
    assert column == f"ALTER TABLE {_TABLE} ADD COLUMN IF NOT EXISTS ledger_seq BIGINT"
    assert index == (
        f"CREATE INDEX IF NOT EXISTS idx_work_ledger_rows_seq ON {_TABLE} "
        "(ledger_id, ledger_seq) WHERE ledger_seq IS NOT NULL"
    )
    assert (
        "NOT NULL" not in column
    )  # The index predicate does not constrain the column.
    assert not re.search(r"\b(DEFAULT|UNIQUE|UPDATE|DELETE)\b", f"{column} {index}")


def test_only_the_table_and_ledger_id_from_0000_are_needed() -> None:
    order = sorted(path.name for path in _VENDOR.glob("*.sql"))
    assert order.index(_CREATE) < order.index(_SEQUENCE)
    create = " ".join(_statements(_CREATE))
    assert re.search(
        rf"CREATE TABLE IF NOT EXISTS {re.escape(_TABLE)} \([^;]*?\bledger_id TEXT\b",
        create,
    )
    assert set(
        re.findall(r"(?:ALTER TABLE|ON) (\w+\.\w+)", " ".join(_statements(_SEQUENCE)))
    ) == {_TABLE}
