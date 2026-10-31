# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the vendored receipt-gate pin friction migrations (OMN-20134).

omnimarket's node_projection_receipt_gate_pin_friction_rows (stacked on
omnimarket#4633) owns the relation and declares it in its own contract
(``db_io.db_tables``). The omnimarket node-migration vendor-parity gate wants this
repository to hold the two SQL files first, so they are held here and the digests
pin the reviewed bytes: any edit to a vendored file fails the pin, and the ledger
checksum is checked against the file's live hash, not only the constant. These
checks also cover the ledger's shape, migration classes, the absence of a FORCE
fence, and the vendored ``db_io.yaml`` declaration that the topology derives the
omninode_runtime grant from. No principal or privilege is listed here or in the
derivation module: the grant is derived from the declared relation and access.
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
        "node_projection_receipt_gate_pin_friction_rows/0000_create_receipt_gate_pin_friction.sql",
        "67abe6f42bc4f238f10de2bcd2c3236ca2300b08beaf79de0b2c86c2151f5678",
        "forward-only",
    ),
    (
        "node_projection_receipt_gate_pin_friction_rows/0001_grant_omninode_runtime_receipt_gate_pin_friction.sql",
        "1a6fe6a9a3b2619f469f092d1be49b46f5faaa853973147a8d7467bd6b3b19b9",
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


def test_neither_pin_friction_migration_is_fenced() -> None:
    fenced = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    ids = {entry["id"] for entry in fenced["fenced_node_migrations"]}
    for path, _, _ in _VENDORED:
        node, name = path.split("/")
        assert f"node:{node}:{name}" not in ids


def test_the_vendored_declaration_names_the_vendored_create() -> None:
    matches = [
        declaration
        for declaration in VENDORED_TABLE_DECLARATIONS
        if declaration.table.name == "receipt_gate_pin_friction"
    ]
    assert len(matches) == 1
    assert matches[0].node == "vendored:node_projection_receipt_gate_pin_friction_rows"
    table = matches[0].table
    assert table.database_ref == "application"
    assert table.schema == "omninode_internal"
    assert table.access == "read_write"
    assert table.role == "receipt_gate_pin_friction"
    assert table.migration == (
        "docker/migrations/forward/nodes/node_projection_receipt_gate_pin_friction_rows/"
        "0000_create_receipt_gate_pin_friction.sql"
    )


def test_the_grant_migration_gives_the_runtime_exactly_what_read_write_derives() -> (
    None
):
    sql = (
        _FORWARD
        / "nodes"
        / "node_projection_receipt_gate_pin_friction_rows"
        / "0001_grant_omninode_runtime_receipt_gate_pin_friction.sql"
    ).read_text()
    assert "GRANT SELECT, INSERT, UPDATE" in sql
    assert "ON omninode_internal.receipt_gate_pin_friction" in sql
    assert "TO omninode_runtime" in sql
    assert "DELETE" not in sql.replace("DELETE is refused", "")
