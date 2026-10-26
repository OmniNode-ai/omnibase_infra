# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the interim vendored compaction current-context migrations.

omnimarket#4430 (OMN-21090, TA.13 stage a) declares compaction_current_context
and its cursor sequence in omnimarket's ownership manifest and carries no SQL.
Until the omnimarket writer node carries the two node-owned migrations, they
are held here and the digests pin the reviewed bytes: any edit to a vendored
file fails the pin, and the ledger checksum is checked against the file's live
hash, not only the constant. These checks also cover the ledger's shape,
migration classes, the absence of a FORCE fence, and the interim grant
declaration that the topology derives the omninode_runtime grant from.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_LEDGER = _FORWARD / "_ledger/application-migrations.tsv"
_LEDGER_COLUMNS = 6
_VENDORED = [
    (
        "node_projection_compaction_current_context/0000_create_compaction_current_context.sql",
        "6ab5da2ae5b0eda9730c43d509e225c48a9f1b0ceccb2491273aa30d3d7e5d21",
        "forward-only",
    ),
    (
        "node_projection_compaction_current_context/0001_grant_omninode_runtime_compaction_current_context.sql",
        "8b6d3da9c9c6e786a3c91219ce195036b1fc70bc45e1cf6b7053c9fd563360b5",
        "expand-only",
    ),
]


@pytest.mark.parametrize(("path", "sha256", "klass"), _VENDORED)
def test_vendored_bytes_match_the_reviewed_pin(
    path: str, sha256: str, klass: str
) -> None:
    assert (
        hashlib.sha256((_FORWARD / "nodes" / path).read_bytes()).hexdigest() == sha256
    )


@pytest.mark.parametrize(("path", "sha256", "klass"), _VENDORED)
def test_the_vendor_has_exactly_one_checksum_bound_ledger_row(
    path: str, sha256: str, klass: str
) -> None:
    node, name = path.split("/")
    artifact = f"nodes/{path}"
    rows = [line.split("\t") for line in _LEDGER.read_text().splitlines()]
    assert [row for row in rows if row[0] == artifact] == [
        [
            artifact,
            f"node:{node}",
            f"node:{node}",
            "omninode_internal",
            f"node:{node}:{name}",
            sha256,
        ]
    ]


def test_the_ledger_is_well_formed_tsv() -> None:
    text = _LEDGER.read_text()
    assert text.endswith("\n")
    lines = text.splitlines()
    assert all(line.strip() for line in lines)
    rows = [line.split("\t") for line in lines]
    assert all(len(row) == _LEDGER_COLUMNS for row in rows)
    assert all(field and field == field.strip() for row in rows for field in row)
    artifacts = [row[0] for row in rows]
    assert len(artifacts) == len(set(artifacts))


@pytest.mark.parametrize(("path", "sha256", "klass"), _VENDORED)
def test_the_ledger_checksum_is_the_live_file_digest(
    path: str, sha256: str, klass: str
) -> None:
    artifact = f"nodes/{path}"
    rows = [line.split("\t") for line in _LEDGER.read_text().splitlines()]
    (row,) = [row for row in rows if row[0] == artifact]
    assert row[5] == hashlib.sha256((_FORWARD / artifact).read_bytes()).hexdigest()


@pytest.mark.parametrize(("path", "sha256", "klass"), _VENDORED)
def test_the_vendor_is_declared_with_its_class(
    path: str, sha256: str, klass: str
) -> None:
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{path}"] == klass


def test_neither_current_context_migration_is_fenced() -> None:
    fenced = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    ids = {entry["id"] for entry in fenced["fenced_node_migrations"]}
    for path, _, _ in _VENDORED:
        node, name = path.split("/")
        assert f"node:{node}:{name}" not in ids


def test_the_interim_grant_declaration_names_the_vendored_create() -> None:
    matches = [
        declaration
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name == "compaction_current_context"
    ]
    assert len(matches) == 1
    table = matches[0].table
    assert table.database_ref == "application"
    assert table.schema == "omninode_internal"
    assert table.access == "read_write"
    assert table.role == "compaction_current_context"
    assert table.migration == (
        "docker/migrations/forward/nodes/node_projection_compaction_current_context/"
        "0000_create_compaction_current_context.sql"
    )
