# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deployment identity of the delegation_capability_cells node migrations.

The digests were measured from the omnimarket source the vendor was copied from,
omnimarket commit 8f91988e9 (omnimarket#4380). Source and vendor were compared
byte for byte before pinning. Failure modes: a vendored file drifts from its
source, loses its checksum-bound ledger row, is declared with a class other than
the one the class checker reads from its statements, the FORCE step reaches a
lane unfenced, or the interim grant declaration is missing.
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
_NODE = "node_projection_delegation_capability_map"

_VENDORED = [
    (
        "0000_create_delegation_capability_cells.sql",
        "cd7e33e472e746638ebf10cfebb60e415041456eaaecba9f5d0f5f27fbb12a9b",
        "forward-only",
    ),
    (
        "0001_grant_tenant_projection_writer_delegation_capability_cells.sql",
        "f59915357fc05262968a709719b532ad9d8a664b52f3ea7f28469eda39c43500",
        "expand-only",
    ),
    (
        "0002_force_rls_delegation_capability_cells.sql",
        "f3472c62a468de9892e8b6908aa39bf4edfbd61ac7b898cc0687d1e7b6863d55",
        "forward-only",
    ),
]


def _ledger_rows() -> list[list[str]]:
    return [
        line.split("\t")
        for line in (_FORWARD / "_ledger/application-migrations.tsv")
        .read_text()
        .splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize(("name", "sha256", "klass"), _VENDORED)
def test_vendored_bytes_match_the_pinned_omnimarket_source(
    name: str, sha256: str, klass: str
) -> None:
    artifact = _FORWARD / "nodes" / _NODE / name
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == sha256


@pytest.mark.parametrize(("name", "sha256", "klass"), _VENDORED)
def test_the_vendor_has_exactly_one_checksum_bound_ledger_row(
    name: str, sha256: str, klass: str
) -> None:
    artifact = f"nodes/{_NODE}/{name}"
    live = hashlib.sha256((_FORWARD / artifact).read_bytes()).hexdigest()
    assert [row for row in _ledger_rows() if row[0] == artifact] == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "tenant",
            f"node:{_NODE}:{name}",
            live,
        ]
    ]


@pytest.mark.parametrize(("name", "sha256", "klass"), _VENDORED)
def test_the_vendor_is_declared_with_its_class(
    name: str, sha256: str, klass: str
) -> None:
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{_NODE}/{name}"] == klass


def test_only_the_force_step_is_fenced() -> None:
    fenced = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    ids = {entry["id"] for entry in fenced["fenced_node_migrations"]}
    assert f"node:{_NODE}:0002_force_rls_delegation_capability_cells.sql" in ids
    assert f"node:{_NODE}:0000_create_delegation_capability_cells.sql" not in ids
    assert (
        f"node:{_NODE}:0001_grant_tenant_projection_writer_delegation_capability_cells.sql"
        not in ids
    )


def test_the_interim_declaration_grants_read_write_under_its_role() -> None:
    (declaration,) = [
        d
        for d in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if d.table.name == "delegation_capability_cells"
    ]
    assert declaration.node == "legacy_migration:delegation_capability_cells"
    assert (declaration.table.access, declaration.table.role) == (
        "read_write",
        "delegation_capability_cells",
    )
    assert declaration.table.migration == (
        f"docker/migrations/forward/nodes/{_NODE}/"
        "0000_create_delegation_capability_cells.sql"
    )
