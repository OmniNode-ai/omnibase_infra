# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The infra routing decisions table resolves and is granted through every shipped topology profile (OMN-21935)."""

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP
from omnibase_infra.topology.table_grant_derivation import (
    STATE_IO_TABLE_DECLARATIONS,
    derive_table_grants,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_TABLE = "infra_routing_decisions"
_PRINCIPAL = "role_omnibase_infra"
_PROFILES = tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))


def _declaration():
    (declaration,) = (d for d in STATE_IO_TABLE_DECLARATIONS if d.table.name == _TABLE)
    return declaration


@pytest.mark.parametrize("profile", _PROFILES)
def test_declared_table_resolves_to_the_omnibase_infra_database(profile: str) -> None:
    declaration = _declaration()
    assert (_ROOT / declaration.table.migration).is_file()
    target = _resolve_projection_database_target(
        (declaration.table,), load_topology_profile(profile)
    )
    assert target.database_refs == ("omnibase_infra",)
    assert target.physical_schemas == ("public",)
    assert target.physical_database


@pytest.mark.parametrize("profile", _PROFILES)
def test_derivation_over_the_shipped_profile_grants_the_runtime_principal(
    profile: str,
) -> None:
    topology = load_topology_profile(profile)
    derived = derive_table_grants(
        topology, STATE_IO_TABLE_DECLARATIONS, database_ref="omnibase_infra"
    )
    granted = {
        principal
        for principal, grants in derived.grants.items()
        for grant in grants
        if _TABLE in grant.objects
    }
    assert granted == {_PRINCIPAL}
