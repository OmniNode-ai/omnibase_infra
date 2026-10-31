# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The shipped activation declaration derives the grant runtime wiring needs."""

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP
from omnibase_infra.topology.table_grant_derivation import (
    VENDORED_TABLE_DECLARATIONS,
    WRITE_PRIVILEGES,
    derive_table_grants,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_TABLE = "compaction_activations"


@pytest.mark.parametrize(
    "profile", tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))
)
def test_shipped_activation_declaration_derives_the_runtime_binding(
    profile: str,
) -> None:
    declarations = tuple(
        d for d in VENDORED_TABLE_DECLARATIONS if d.table.name == _TABLE
    )
    assert len(declarations) == 1
    (declaration,) = declarations
    table = declaration.table
    assert table.database_ref == "application"
    assert table.schema == "omninode_internal"
    assert table.access == "read_write"
    assert table.role == _TABLE
    assert declaration.contract_path == Path(table.migration)
    assert (_ROOT / table.migration).is_file()

    topology = load_topology_profile(profile)
    derived = derive_table_grants(topology, declarations, database_ref="application")
    assert not derived.unmappable
    assert set(derived.grants) == {"omninode_runtime"}
    (grant,) = derived.grants["omninode_runtime"]
    assert grant.schema == table.schema
    assert set(grant.objects) == {_TABLE}
    assert set(grant.privileges) == WRITE_PRIVILEGES

    # Derivation and the checked-in artifact must agree before runtime wiring.
    shipped = topology.databases["application"].principals["omninode_runtime"]
    shipped_grants = tuple(g for g in shipped.grants if _TABLE in g.objects)
    assert len(shipped_grants) == 1
    assert shipped_grants[0].schema == grant.schema
    assert set(shipped_grants[0].privileges) == set(grant.privileges)
    _resolve_projection_database_target((table,), topology)

    # Removing the declaration must remove its derived privilege, even though
    # this topology's checked-in grants still contain the positive control.
    empty = derive_table_grants(topology, (), database_ref="application")
    assert not empty.unmappable
    assert not any(_TABLE in g.objects for gs in empty.grants.values() for g in gs)
