# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests verifying that the registration orchestrator plugin wires topic_router correctly.

Two-level proof:
1. The contract.yaml published_events section contains the expected event-to-topic mappings.
2. The plugin module exposes _TOPIC_ROUTER built from the contract, and passes it to
   DispatchResultApplier via topic_router=_TOPIC_ROUTER.

OMN-4883 / OMN-4880
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.contract_topic_router import (
    build_topic_router_from_contract,
)

_CONTRACT_PATH = (
    Path(__file__).parents[4]
    / "src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml"
)


@pytest.mark.unit
def test_contract_published_events_produces_expected_router_entries() -> None:
    """The registration orchestrator contract produces a router with all declared events."""
    contract_data = yaml.safe_load(_CONTRACT_PATH.read_text())
    router = build_topic_router_from_contract(contract_data)
    # Must contain the core events declared in published_events
    assert "ModelNodeRegistrationAccepted" in router
    assert "ModelNodeBecameActive" in router
    assert "ModelNodeLivenessExpired" in router
    assert "ModelNodeRegistrationRejected" in router
    # Must not contain spurious entries (count matches contract)
    expected_count = len(
        [
            e
            for e in contract_data.get("published_events", [])
            if e.get("event_type") and e.get("topic")
        ]
    )
    assert len(router) == expected_count


@pytest.mark.unit
def test_contract_router_contains_registration_accepted_topic() -> None:
    """ModelNodeRegistrationAccepted must map to the correct declared topic."""
    contract_data = yaml.safe_load(_CONTRACT_PATH.read_text())
    router = build_topic_router_from_contract(contract_data)
    assert router.get("ModelNodeRegistrationAccepted") == (
        "onex.evt.platform.node-registration-accepted.v1"
    )


@pytest.mark.unit
def test_plugin_topic_router_constant_matches_contract_and_is_wired() -> None:
    """Two-part wiring proof.

    Part A: plugin._TOPIC_ROUTER matches build_topic_router_from_contract output.
    Part B: plugin source contains topic_router=_TOPIC_ROUTER at the DispatchResultApplier
            construction site.
    """
    import omnibase_infra.nodes.node_registration_orchestrator.plugin as plugin_module

    # Part A: module constant shape matches contract
    contract_data = yaml.safe_load(_CONTRACT_PATH.read_text())
    expected_router = build_topic_router_from_contract(contract_data)
    assert expected_router == plugin_module._TOPIC_ROUTER, (
        "plugin._TOPIC_ROUTER does not match what build_topic_router_from_contract produces. "
        "Re-check the module-level initialization."
    )

    # Part B: plugin source passes _TOPIC_ROUTER to DispatchResultApplier
    plugin_source = Path(plugin_module.__file__).read_text()
    assert "topic_router=_TOPIC_ROUTER" in plugin_source, (
        "plugin.py does not pass topic_router=_TOPIC_ROUTER to DispatchResultApplier. "
        "The router is built but not wired."
    )
