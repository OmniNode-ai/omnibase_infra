# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the interim vendored compaction receipts migrations.

omnimarket#4347 (OMN-21336 stage a) declares compaction_receipts and its cursor
sequence in omnimarket's ownership manifest, but it dropped the two SQL files
before merging, so no omnimarket tree carries them yet. Until the omnimarket
writer node does, these files are held here and the digests pin the reviewed
bytes: any edit to a vendored file fails the pin, and the ledger checksum is
checked against the file's live hash, not only the constant. These checks also
cover the ledger's shape, migration classes, the absence of a FORCE fence, and
the interim grant declaration that the topology derives the omninode_runtime
grant from.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    VENDORED_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_LEDGER = _FORWARD / "_ledger/application-migrations.tsv"
_LEDGER_COLUMNS = 6
_VENDORED = [
    (
        "node_projection_compaction_receipts/0000_create_compaction_receipts.sql",
        "0e43ebcf212a747931682d85460f43d15a43b97e9885fb01cd1520b9631c1030",
        "forward-only",
    ),
    (
        "node_projection_compaction_receipts/0001_grant_omninode_runtime_compaction_receipts.sql",
        "89402ff7fe70ba9a7ea5804dcdadfed090fc29c311c124bef598f41b24efd95f",
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


def test_neither_receipts_migration_is_fenced() -> None:
    fenced = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    ids = {entry["id"] for entry in fenced["fenced_node_migrations"]}
    for path, _, _ in _VENDORED:
        node, name = path.split("/")
        assert f"node:{node}:{name}" not in ids


def test_the_interim_grant_declaration_names_the_vendored_create() -> None:
    matches = [
        declaration
        for declaration in VENDORED_TABLE_DECLARATIONS
        if declaration.table.name == "compaction_receipts"
    ]
    assert len(matches) == 1
    table = matches[0].table
    assert table.database_ref == "application"
    assert table.schema == "omninode_internal"
    assert table.access == "read_write"
    assert table.role == "compaction_receipts"
    assert table.migration == (
        "docker/migrations/forward/nodes/node_projection_compaction_receipts/"
        "0000_create_compaction_receipts.sql"
    )
