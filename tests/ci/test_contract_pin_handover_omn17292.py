# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contract pin handover retires bridges without losing runtime grants."""

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
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[2]
_PIN = "917f721fcdfb30339cfb4532257d0d2e7a4cb4d4"
_RETIRED = {"delegation_routing_feedback", "delegation_budget_applied_events"}


def test_contract_and_sibling_pins_name_the_handover_commit() -> None:
    contract_pin = yaml.safe_load(
        (_ROOT / ".github/omnimarket-contract-pin.yaml").read_text()
    )
    sibling_pins = yaml.safe_load((_ROOT / ".github/sibling-pins.yaml").read_text())
    assert contract_pin["omnimarket_contract_ref"] == _PIN
    assert sibling_pins["pins"]["omnimarket"] == _PIN


def test_contract_owned_relations_have_no_supplemental_bridge() -> None:
    supplemental = {item.table.name for item in LEGACY_MIGRATION_TABLE_DECLARATIONS}
    assert not (_RETIRED & supplemental)
    assert "demo_readiness_latest" in supplemental


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_retired_bridges_keep_their_runtime_write_grants(profile: str) -> None:
    topology = load_topology_profile(profile)
    for name in sorted(_RETIRED):
        declaration = ModelDbTableDeclaration(
            name=name,
            database_ref="application",
            schema="public",
            access="read_write",
            role=name,
            migration=f"{name}.sql",
        )
        _resolve_projection_database_target((declaration,), topology)
