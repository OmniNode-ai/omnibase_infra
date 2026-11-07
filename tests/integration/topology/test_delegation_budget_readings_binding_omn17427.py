# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The delegation budget readings table resolves through every shipped topology profile (OMN-17427)."""

from pathlib import Path

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP

_ROOT = Path(__file__).resolve().parents[3]
_NODE = "node_projection_delegation_budget_readings"
_TABLE = "delegation_budget_readings"
# OMN-21970: the supplemental bridge is retired by the pin advance to 3aa4e8b26af6, whose
# node_projection_delegation_budget_readings/contract.yaml (db_io.db_tables, lines 60-65) declares
# database_ref application, schema omninode_internal, access read_write, role delegation_budget_readings.
# This repo may not import omnimarket (OMN-19998), so the values are restated here; the OMN-15361
# enforcement job derives every grant from that pinned contract and fails if they ever differ.
_DECLARED_TABLE = ModelDbTableDeclaration(
    name=_TABLE,
    database_ref="application",
    schema="omninode_internal",
    migration=f"docker/migrations/forward/nodes/{_NODE}/0000_create_{_TABLE}.sql",
    access="read_write",
    role=_TABLE,
)


@pytest.mark.parametrize(
    "profile", tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))
)
def test_declared_table_resolves_to_application_database(profile: str) -> None:
    assert (_ROOT / _DECLARED_TABLE.migration).is_file()
    target = _resolve_projection_database_target(
        (_DECLARED_TABLE,), load_topology_profile(profile)
    )
    assert target.database_refs == ("application",)
    assert target.physical_schemas == ("omninode_internal",)
    assert target.physical_database
