# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin delegation capability-map migrations from omnimarket#4380, 677d2941ff7e.

The reviewed source digests pin the vendored bytes. The create migration also
restores its four CHECK constraints on a drifted table, so it differs from the
source by those guarded blocks. The ledger must bind each
artifact exactly once and match its live digest, and the shipped instances
must carry the tenant_projection_writer grant the pinned contract derives.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_NODE = "node_projection_delegation_capability_map"
_VENDOR = _FORWARD / "nodes" / _NODE
_LEDGER = _FORWARD / "_ledger/application-migrations.tsv"
_CREATE = "0000_create_delegation_capability_cells.sql"
_GRANT = "0001_grant_tenant_projection_writer_delegation_capability_cells.sql"
_FORCE_RLS = "0002_force_rls_delegation_capability_cells.sql"
_TABLE = "delegation_capability_cells"
# Every shipped topology instance, read from the package rather than named here.
_PROFILES = sorted(
    path.stem
    for path in (_ROOT / "src/omnibase_infra/topology/instances").glob("*.yaml")
)
_VENDORED = [
    (
        _CREATE,
        "04ab2f892baf98b55c34570ffc1bb1849f7a7bca496bfd8808a832ba72a7cc5d",
        "forward-only",
    ),
    (
        _GRANT,
        "f59915357fc05262968a709719b532ad9d8a664b52f3ea7f28469eda39c43500",
        "expand-only",
    ),
    (
        _FORCE_RLS,
        "f3472c62a468de9892e8b6908aa39bf4edfbd61ac7b898cc0687d1e7b6863d55",
        "forward-only",
    ),
]


@pytest.mark.parametrize(("filename", "sha256", "klass"), _VENDORED)
def test_vendored_bytes_match_the_reviewed_pin(
    filename: str, sha256: str, klass: str
) -> None:
    assert hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest() == sha256


def test_the_ledger_is_well_formed_tsv() -> None:
    text = _LEDGER.read_text(encoding="utf-8")
    assert text.endswith("\n")
    lines = text.splitlines()
    assert all(line.strip() for line in lines)
    rows = [line.split("\t") for line in lines]
    assert all(len(row) == 6 for row in rows)
    assert all(field and field == field.strip() for row in rows for field in row)
    artifacts = [row[0] for row in rows]
    assert len(artifacts) == len(set(artifacts))


@pytest.mark.parametrize(("filename", "sha256", "klass"), _VENDORED)
def test_the_vendor_has_exactly_one_checksum_bound_ledger_row(
    filename: str, sha256: str, klass: str
) -> None:
    artifact = f"nodes/{_NODE}/{filename}"
    rows = [line.split("\t") for line in _LEDGER.read_text().splitlines()]
    assert [row for row in rows if row[0] == artifact] == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "tenant",
            f"node:{_NODE}:{filename}",
            sha256,
        ]
    ]


@pytest.mark.parametrize(("filename", "sha256", "klass"), _VENDORED)
def test_the_ledger_checksum_is_the_live_file_digest(
    filename: str, sha256: str, klass: str
) -> None:
    artifact = f"nodes/{_NODE}/{filename}"
    rows = [line.split("\t") for line in _LEDGER.read_text().splitlines()]
    (row,) = [row for row in rows if row[0] == artifact]
    assert row[5] == hashlib.sha256((_FORWARD / artifact).read_bytes()).hexdigest()


@pytest.mark.parametrize(("filename", "sha256", "klass"), _VENDORED)
def test_the_vendor_is_declared_with_its_class(
    filename: str, sha256: str, klass: str
) -> None:
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{_NODE}/{filename}"] == klass


def test_only_the_force_rls_step_is_fenced() -> None:
    manifest = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    entries = [
        entry
        for entry in manifest["fenced_node_migrations"]
        if entry["id"].startswith(f"node:{_NODE}:")
    ]
    assert entries == [{"id": f"node:{_NODE}:{_FORCE_RLS}", "ticket": "OMN-20169"}]


@pytest.mark.parametrize("profile", _PROFILES)
def test_the_shipped_instance_carries_the_contract_derived_table_grant(
    profile: str,
) -> None:
    # The supplemental bridge was retired by the pin advance to 3aa4e8b26af6;
    # the pinned omnimarket contract derives this grant now, so the shipped
    # instance is what is left to pin.
    principal = "tenant_projection_writer"
    instance = yaml.safe_load(
        (
            _ROOT / "src/omnibase_infra/topology/instances" / f"{profile}.yaml"
        ).read_text()
    )
    shipped = [
        entry
        for entry in instance["databases"]["application"]["principals"][principal][
            "grants"
        ]
        if entry["object_type"] == "TABLE"
        and entry["schema"] == "public"
        and _TABLE in entry["objects"]
    ]
    assert len(shipped) == 1
    assert set(shipped[0]["privileges"]) == {"SELECT", "INSERT", "UPDATE"}
    assert shipped[0]["objects"] == sorted(shipped[0]["objects"])
