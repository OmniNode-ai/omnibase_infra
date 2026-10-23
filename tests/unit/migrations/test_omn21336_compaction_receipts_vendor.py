# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the compaction receipts migrations to the measured omnimarket bytes.

The digests were measured from omnimarket#4347 (OMN-21336 stage a), the change
that declares compaction_receipts and its cursor sequence in omnimarket's
ownership manifest. Source and vendor were compared byte for byte before
pinning. These checks cover deployment identity, ledger checksums, migration
classes, the absence of a FORCE fence, and the interim grant declaration that
the topology derives the omninode_runtime grant from.
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
def test_vendored_bytes_match_the_pinned_omnimarket_source(
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
    rows = [
        line.split("\t")
        for line in (_FORWARD / "_ledger/application-migrations.tsv")
        .read_text()
        .splitlines()
        if line.strip()
    ]
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
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
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
