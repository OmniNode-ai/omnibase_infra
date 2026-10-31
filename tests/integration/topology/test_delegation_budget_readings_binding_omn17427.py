# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The delegation budget readings table resolves through every shipped topology profile (OMN-17427)."""

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

_ROOT = Path(__file__).resolve().parents[3]
_TABLE = "delegation_budget_readings"


@pytest.mark.parametrize(
    "profile", tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))
)
def test_declared_table_resolves_to_application_database(profile: str) -> None:
    (declaration,) = (
        d for d in LEGACY_MIGRATION_TABLE_DECLARATIONS if d.table.name == _TABLE
    )
    assert (_ROOT / declaration.table.migration).is_file()
    target = _resolve_projection_database_target(
        (declaration.table,), load_topology_profile(profile)
    )
    assert target.database_refs == ("application",)
    assert target.physical_schemas == ("omninode_internal",)
    assert target.physical_database
