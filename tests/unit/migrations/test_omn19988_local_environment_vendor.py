# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deployment identity of the local_environment node migrations, not page acceptance.

The digests were measured from the omnimarket source the vendor was copied from,
omnimarket commit 956949072 (omnimarket#4044). Source and vendor were compared
byte for byte before pinning. Failure modes: a vendored file drifts from its
source, loses its checksum-bound ledger row, is declared with a class other than
the one the class checker reads from its statements, or the FORCE step reaches a
lane unfenced.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_NODE = "node_projection_local_environment"

_VENDORED = [
    (
        "0000_create_local_environment.sql",
        "449833bf5b6727d95471b99dbeafe0c8288f886023c21697903da1b9908dd821",
        "forward-only",
    ),
    (
        "0001_grant_tenant_projection_writer_local_environment.sql",
        "62e0024122b691e064d5e08b0d3eb6880544a5ca09c2a71b2f3cee03b0024150",
        "expand-only",
    ),
    (
        "0002_force_rls_local_environment.sql",
        "0a429ce454c5042c1e5c7d77dcb835fdb17872ff68e7d4ce73e37102719e5d16",
        "forward-only",
    ),
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
            f"node:{_NODE}",
            f"node:{_NODE}",
            "tenant",
            f"node:{_NODE}:{name}",
            sha256,
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
    assert f"node:{_NODE}:0002_force_rls_local_environment.sql" in ids
    assert f"node:{_NODE}:0000_create_local_environment.sql" not in ids
    assert (
        f"node:{_NODE}:0001_grant_tenant_projection_writer_local_environment.sql"
        not in ids
    )
