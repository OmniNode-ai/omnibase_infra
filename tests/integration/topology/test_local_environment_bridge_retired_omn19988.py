# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""local_environment's grant survives the retirement of its interim bridge (OMN-19988).

The omnimarket contract pin advanced past node_projection_local_environment
(OMN-17292), so the hand-written LEGACY_MIGRATION_TABLE_DECLARATIONS entry that
covered the infra-first window is gone. The grant must now come from the node
contract's own ``db_io.db_tables`` declaration, and the shipped topology
instances must still give tenant_projection_writer exactly INSERT, SELECT and
UPDATE on public.local_environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    derive_table_grants,
    load_contract_declarations,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELATION = "local_environment"
_SCHEMA = "public"
_DATABASE_REF = "application"
_PRINCIPAL = "tenant_projection_writer"
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})
_INSTANCES_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
# Every instance the generator renders and the deploy ships.
_SHIPPED_INSTANCES = tuple(sorted(path.stem for path in _INSTANCES_DIR.glob("*.yaml")))

# The db_io block omnimarket's node_projection_local_environment contract
# declares at the advanced pin.
_NODE_CONTRACT = {
    "name": "node_projection_local_environment",
    "db_io": {
        "db_tables": [
            {
                "name": _RELATION,
                "database_ref": _DATABASE_REF,
                "schema": _SCHEMA,
                "migration": "0000_create_local_environment.sql",
                "access": "read_write",
                "role": "local_environment",
            }
        ]
    },
}


def _instance_grants(profile: str) -> list[dict[str, Any]]:
    path = _INSTANCES_DIR / f"{profile}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    principal = document["databases"][_DATABASE_REF]["principals"][_PRINCIPAL]
    grants = principal["grants"]
    assert isinstance(grants, list), f"{profile}'s {_PRINCIPAL} has no grants list"
    return grants


def test_the_shipped_instances_are_found() -> None:
    assert len(_SHIPPED_INSTANCES) >= 3, (
        f"found {_SHIPPED_INSTANCES!r} under {_INSTANCES_DIR}; the grant "
        "assertions below would cover too few shipped instances."
    )


def test_the_interim_bridge_is_retired() -> None:
    carried = [
        declaration.node
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name == _RELATION
    ]
    assert carried == [], (
        f"{_RELATION} is still carried by {carried!r} in "
        "LEGACY_MIGRATION_TABLE_DECLARATIONS although the pinned omnimarket "
        "node contract declares it. Delete the interim bridge."
    )


@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
def test_the_node_contract_derives_the_writer_grant(
    profile: str, tmp_path: Path
) -> None:
    node_dir = tmp_path / "node_projection_local_environment"
    node_dir.mkdir()
    (node_dir / "contract.yaml").write_text(
        yaml.safe_dump(_NODE_CONTRACT), encoding="utf-8"
    )
    derived = derive_table_grants(
        load_topology_profile(profile),
        (*load_contract_declarations(tmp_path), *LEGACY_MIGRATION_TABLE_DECLARATIONS),
    )
    matching = [
        grant
        for grant in derived.grants.get(_PRINCIPAL, ())
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == _SCHEMA
        and _RELATION in grant.objects
    ]
    assert len(matching) == 1, (
        f"derivation for {profile} produced {len(matching)} {_PRINCIPAL} TABLE "
        f"grants for {_SCHEMA}.{_RELATION}, not one."
    )
    privileges = frozenset(privilege.value for privilege in matching[0].privileges)
    assert privileges == _REQUIRED_PRIVILEGES, (
        f"derivation for {profile} gives {_PRINCIPAL} {sorted(privileges)} on "
        f"{_SCHEMA}.{_RELATION}, expected {sorted(_REQUIRED_PRIVILEGES)}."
    )


@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
def test_the_shipped_instance_keeps_the_writer_grant(profile: str) -> None:
    matching = [
        grant
        for grant in _instance_grants(profile)
        if isinstance(grant, dict)
        and grant.get("object_type") == "TABLE"
        and grant.get("schema") == _SCHEMA
        and _RELATION in tuple(grant.get("objects") or ())
    ]
    assert len(matching) == 1, (
        f"{profile} ships {len(matching)} {_PRINCIPAL} TABLE grants for "
        f"{_SCHEMA}.{_RELATION}, not one. Regenerate the topology grants."
    )
    privileges = frozenset(matching[0].get("privileges") or ())
    assert privileges == _REQUIRED_PRIVILEGES, (
        f"{profile} ships {sorted(privileges)} for {_PRINCIPAL} on "
        f"{_SCHEMA}.{_RELATION}, expected {sorted(_REQUIRED_PRIVILEGES)}."
    )
