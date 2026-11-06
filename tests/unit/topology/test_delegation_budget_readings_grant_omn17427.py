# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The delegation budget readings table binds to the runtime role the way the lab runtime does (OMN-17427)."""

import hashlib
from pathlib import Path

import pytest
import yaml

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_NODE = "node_projection_delegation_budget_readings"
_TABLE = "delegation_budget_readings"
# The pinned omnimarket contract (node_projection_delegation_budget_readings)
# declares this relation; the supplemental bridge that used to stand in for it
# was retired by the pin advance to 3aa4e8b26af6.
_CREATE = f"docker/migrations/forward/nodes/{_NODE}/0000_create_{_TABLE}.sql"
_DECLARED_TABLE = ModelDbTableDeclaration(
    name=_TABLE,
    database_ref="application",
    schema="omninode_internal",
    migration=_CREATE,
    access="read_write",
    role=_TABLE,
)

# Shipped bytes of the vendored migrations; re-pin with the ledger row when the
# omnimarket source changes.
_EXPECTED = {
    "0000_create_delegation_budget_readings.sql": "835bc4e3b5e9f93c0ca0bea530e02cf616eb27c56b0397903827fc89fb83e595",
    "0001_grant_omninode_runtime_delegation_budget_readings.sql": "7dec2aba11f649d51da66d6968ba78d51f2a2b2ef47cd624048c2fe1c4ea4b90",
}


@pytest.mark.parametrize(
    "profile", tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))
)
def test_the_write_uses_only_the_runtime_table_grant(profile: str) -> None:
    topology = load_topology_profile(profile)
    target = _resolve_projection_database_target((_DECLARED_TABLE,), topology)
    assert target.database_refs == ("application",)
    assert target.physical_schemas == ("omninode_internal",)
    principal = topology.databases["application"].principals["omninode_runtime"]
    grants = [g for g in principal.grants if _TABLE in g.objects]
    assert len(grants) == 1
    assert grants[0].schema == "omninode_internal"
    assert {p.value for p in grants[0].privileges} == {"INSERT", "SELECT", "UPDATE"}
    for name, other in topology.databases["application"].principals.items():
        if name != "omninode_runtime":
            assert not any(_TABLE in g.objects for g in other.grants)


def test_the_declared_table_has_delivered_migrations() -> None:
    create = _ROOT / _DECLARED_TABLE.migration
    assert create.is_file()
    grant = create.parent / "0001_grant_omninode_runtime_delegation_budget_readings.sql"
    assert grant.is_file()
    # SQL privilege semantics are exercised by the live-apply migration proof,
    # not inferred from statement prefixes or a partial SQL parser here.


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("0000_create_delegation_budget_readings.sql", "forward-only"),
        ("0001_grant_omninode_runtime_delegation_budget_readings.sql", "expand-only"),
    ],
)
def test_the_vendor_ledger_binds_live_bytes_and_migration_class(
    name: str, kind: str
) -> None:
    relative = f"nodes/{_NODE}/{name}"
    artifact = _ROOT / "docker/migrations/forward" / relative
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert digest == _EXPECTED[name]
    ledger = _ROOT / "docker/migrations/forward/_ledger/application-migrations.tsv"
    rows = [row.split("\t") for row in ledger.read_text().splitlines()]
    assert [row for row in rows if row[0] == relative] == [
        [
            relative,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{name}",
            digest,
        ]
    ]
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/{relative}"] == kind
