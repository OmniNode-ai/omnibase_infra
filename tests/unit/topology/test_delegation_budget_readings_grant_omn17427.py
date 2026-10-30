# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The delegation budget readings table binds to the runtime role the way the lab runtime does (OMN-17427)."""

import hashlib
import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_NODE = "node_projection_delegation_budget_readings"
_TABLE = "delegation_budget_readings"

# Shipped bytes of the vendored migrations; re-pin with the ledger row when the
# omnimarket source changes.
_EXPECTED = {
    "0000_create_delegation_budget_readings.sql": "61aad2b0f460eebf9a7c26cb6e4794baf884abd49c9ad3ca03a29d3a621495f2",
    "0001_grant_omninode_runtime_delegation_budget_readings.sql": "7dec2aba11f649d51da66d6968ba78d51f2a2b2ef47cd624048c2fe1c4ea4b90",
}


@pytest.mark.parametrize(
    "profile", tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))
)
def test_the_write_uses_only_the_runtime_table_grant(profile: str) -> None:
    topology = load_topology_profile(profile)
    (declaration,) = (
        d for d in LEGACY_MIGRATION_TABLE_DECLARATIONS if d.table.name == _TABLE
    )
    target = _resolve_projection_database_target((declaration.table,), topology)
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


def test_the_interim_declaration_has_delivered_migrations() -> None:
    (declaration,) = (
        d for d in LEGACY_MIGRATION_TABLE_DECLARATIONS if d.table.name == _TABLE
    )
    assert declaration.table.access == "read_write"
    assert declaration.table.role == _TABLE
    create = _ROOT / declaration.table.migration
    assert create.is_file()
    grant = create.parent / "0001_grant_omninode_runtime_delegation_budget_readings.sql"
    # The complete executable SQL: comments stripped, then every statement that
    # can change privileges or objects, in any case, must be one of the three.
    executable = re.sub(r"--[^\n]*", "", grant.read_text())
    statements = [" ".join(s.split()) for s in executable.split(";") if s.strip()]
    mutating = [
        s
        for s in statements
        if re.match(r"(grant|revoke|alter|create|drop)\b", s, re.IGNORECASE)
    ]
    assert mutating == [
        "GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime",
        "GRANT SELECT, INSERT, UPDATE ON omninode_internal.delegation_budget_readings TO omninode_runtime",
        "GRANT USAGE ON SEQUENCE omninode_internal.delegation_budget_readings_projection_cursor_seq TO omninode_runtime",
    ]
    assert not re.search(r"default\s+privileges", executable, re.IGNORECASE)


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
