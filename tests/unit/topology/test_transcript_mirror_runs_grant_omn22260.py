# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""transcript_mirror_runs is written by the reducer's principal and by no other (OMN-22260).

The reducer ``node_projection_transcript_mirror_run`` is the sole writer. It runs in the
platform runtime, whose application-database principal is ``omninode_runtime``. Every shipped
profile must grant that principal exactly SELECT, INSERT and UPDATE on the table, and no other
principal any privilege on it.
"""

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    ContractTableDeclaration,
)

pytestmark = pytest.mark.unit

_TABLE = "transcript_mirror_runs"
_WRITER = "omninode_runtime"
_PROFILES = tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))


def _declaration() -> ContractTableDeclaration:
    (declaration,) = (
        d for d in LEGACY_MIGRATION_TABLE_DECLARATIONS if d.table.name == _TABLE
    )
    return declaration


@pytest.mark.parametrize("profile", _PROFILES)
def test_only_the_runtime_principal_holds_any_grant_on_the_table(profile: str) -> None:
    topology = load_topology_profile(profile)
    target = _resolve_projection_database_target((_declaration().table,), topology)
    assert target.database_refs == ("application",)
    assert target.physical_schemas == ("omninode_internal",)
    principals = topology.databases["application"].principals
    holders = {
        name
        for name, principal in principals.items()
        if any(_TABLE in grant.objects for grant in principal.grants)
    }
    assert holders == {_WRITER}
    grants = [g for g in principals[_WRITER].grants if _TABLE in g.objects]
    assert len(grants) == 1
    assert grants[0].schema == "omninode_internal"
    assert {p.value for p in grants[0].privileges} == {"INSERT", "SELECT", "UPDATE"}


def test_the_interim_declaration_is_read_write_with_a_table_named_role() -> None:
    declaration = _declaration()
    assert declaration.table.access == "read_write"
    assert declaration.table.role == _TABLE
    assert declaration.table.schema == "omninode_internal"
    migration = declaration.contract_path
    assert migration.name == "0000_create_transcript_mirror_runs.sql"
    assert migration.parent.name == "node_projection_transcript_mirror_run"
