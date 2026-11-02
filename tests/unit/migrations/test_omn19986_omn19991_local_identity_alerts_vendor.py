# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deployment identity of the local_identity and local_alerts node migrations.

Infra half only, not page acceptance. The digests were measured from the
omnimarket sources the vendor was copied from: local_alerts from omnimarket#4803
(head ceef59db4), local_identity from the OMN-19986 omnimarket branch that adds
the same files. Source and vendor were compared byte for byte before pinning.
Failure modes: a vendored file drifts from its source, loses its checksum-bound
ledger row, is declared with a class other than the one the class checker reads
from its statements, the FORCE step reaches a lane unfenced, or the interim grant
declaration is missing.
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
        "node_projection_local_identity",
        "0000_create_local_identity.sql",
        "2d95a5f1ded67bc3a9472afac8563906542c1579b203ecff1c46014599daa77a",
        "forward-only",
    ),
    (
        "node_projection_local_identity",
        "0001_grant_tenant_projection_writer_local_identity.sql",
        "b9fe1b1798368cd68f3ac7c9af4ef63f9a0a7018eafda508a87e145c4e47b4c8",
        "expand-only",
    ),
    (
        "node_projection_local_identity",
        "0002_force_rls_local_identity.sql",
        "61f8fefd270edcd98f502d9d56fce2867fae82518dff076555e6b3dec0248126",
        "forward-only",
    ),
    (
        "node_projection_local_alerts",
        "0000_create_local_alerts.sql",
        "6e2756c85562968895be097dad7fd7f98eada2d26158642ef848e97528b96a6f",
        "forward-only",
    ),
    (
        "node_projection_local_alerts",
        "0001_grant_tenant_projection_writer_local_alerts.sql",
        "4dec2b43dbda1b0e6099ccb64b3c5df8b7f3c161172581871af6f5dd76920315",
        "expand-only",
    ),
    (
        "node_projection_local_alerts",
        "0002_force_rls_local_alerts.sql",
        "e6416102cef1ba4cc5baf032d9e763fc453f4481bb6e0729793ce401b5e922b5",
        "forward-only",
    ),
]

_DIGESTS = [(node, name, sha256) for node, name, sha256, _ in _VENDORED]
_CLASSES = [(node, name, klass) for node, name, _, klass in _VENDORED]

_TABLES = [
    ("local_identity", "node_projection_local_identity"),
    ("local_alerts", "node_projection_local_alerts"),
]


@pytest.mark.parametrize(("node", "name", "sha256"), _DIGESTS)
def test_vendored_bytes_match_the_pinned_omnimarket_source(
    node: str, name: str, sha256: str
) -> None:
    artifact = _FORWARD / "nodes" / node / name
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == sha256


@pytest.mark.parametrize(("node", "name", "sha256"), _DIGESTS)
def test_the_vendor_has_exactly_one_checksum_bound_ledger_row(
    node: str, name: str, sha256: str
) -> None:
    artifact = f"nodes/{node}/{name}"
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
            "tenant",
            f"node:{node}:{name}",
            sha256,
        ]
    ]


@pytest.mark.parametrize(("node", "name", "klass"), _CLASSES)
def test_the_vendor_is_declared_with_its_class(
    node: str, name: str, klass: str
) -> None:
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{node}/{name}"] == klass


@pytest.mark.parametrize(("table", "node"), _TABLES)
def test_only_the_force_step_is_fenced(table: str, node: str) -> None:
    fenced = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    ids = {entry["id"] for entry in fenced["fenced_node_migrations"]}
    assert f"node:{node}:0002_force_rls_{table}.sql" in ids
    assert f"node:{node}:0000_create_{table}.sql" not in ids
    assert f"node:{node}:0001_grant_tenant_projection_writer_{table}.sql" not in ids


@pytest.mark.parametrize(("table", "node"), _TABLES)
def test_the_interim_grant_declaration_names_the_vendored_create(
    table: str, node: str
) -> None:
    matches = [
        entry
        for entry in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if entry.table.name == table
    ]
    assert len(matches) == 1
    declared = matches[0].table
    assert declared.migration == (
        f"docker/migrations/forward/nodes/{node}/0000_create_{table}.sql"
    )
    assert (declared.access, declared.role, declared.schema) == (
        "read_write",
        table,
        "public",
    )
