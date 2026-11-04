# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The shipped context_audit_events declaration derives the grant the node needs."""

from pathlib import Path

import pytest

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    WRITE_PRIVILEGES,
    derive_table_grants,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_TABLE = "context_audit_events"
_PRINCIPAL = "tenant_projection_writer"


@pytest.mark.parametrize(
    "profile", tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))
)
def test_shipped_declaration_derives_the_projection_writer_grant(
    profile: str,
) -> None:
    declarations = tuple(
        d for d in LEGACY_MIGRATION_TABLE_DECLARATIONS if d.table.name == _TABLE
    )
    assert len(declarations) == 1
    (declaration,) = declarations
    table = declaration.table
    assert table.database_ref == "application"
    assert table.schema == "public"
    assert table.access == "read_write"
    assert (_ROOT / table.migration).is_file()

    topology = load_topology_profile(profile)
    derived = derive_table_grants(topology, declarations, database_ref="application")
    assert not derived.unmappable
    assert _PRINCIPAL in derived.grants
    (grant,) = (g for g in derived.grants[_PRINCIPAL] if _TABLE in g.objects)
    assert set(grant.privileges) == WRITE_PRIVILEGES

    # The checked-in instance must agree with the derivation.
    shipped = topology.databases["application"].principals[_PRINCIPAL]
    shipped_grants = tuple(g for g in shipped.grants if _TABLE in g.objects)
    assert len(shipped_grants) == 1
    assert shipped_grants[0].schema == grant.schema
    assert set(shipped_grants[0].privileges) == set(grant.privileges)

    # Removing the declaration removes the derived privilege.
    empty = derive_table_grants(topology, (), database_ref="application")
    assert not any(_TABLE in g.objects for gs in empty.grants.values() for g in gs)
