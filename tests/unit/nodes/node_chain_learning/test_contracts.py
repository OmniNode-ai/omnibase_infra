# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for chain learning contract YAML files.

Validates that each node's contract.yaml has required fields,
correct node_type, and well-formed handler routing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Base path for chain learning node contracts
_NODES_DIR = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra" / "nodes"

_CHAIN_NODES = [
    "node_chain_retrieval_effect",
    "node_chain_replay_compute",
    "node_chain_store_effect",
    "node_chain_verify_reducer",
    "node_chain_orchestrator",
]

_EXPECTED_NODE_TYPES = {
    "node_chain_retrieval_effect": "EFFECT_GENERIC",
    "node_chain_replay_compute": "COMPUTE_GENERIC",
    "node_chain_store_effect": "EFFECT_GENERIC",
    "node_chain_verify_reducer": "REDUCER_GENERIC",
    "node_chain_orchestrator": "ORCHESTRATOR_GENERIC",
}


def _load_contract(node_name: str) -> dict:
    contract_path = _NODES_DIR / node_name / "contract.yaml"
    assert contract_path.exists(), f"Missing contract: {contract_path}"
    with open(contract_path) as f:
        return yaml.safe_load(f)


@pytest.mark.unit
class TestChainLearningContracts:
    """Validate contract YAML structure for all chain learning nodes."""

    @pytest.mark.parametrize("node_name", _CHAIN_NODES)
    def test_contract_exists(self, node_name: str) -> None:
        contract_path = _NODES_DIR / node_name / "contract.yaml"
        assert contract_path.exists()

    @pytest.mark.parametrize("node_name", _CHAIN_NODES)
    def test_required_fields_present(self, node_name: str) -> None:
        contract = _load_contract(node_name)
        for field in (
            "name",
            "node_type",
            "contract_version",
            "node_version",
            "description",
        ):
            assert field in contract, f"{node_name}: missing '{field}'"

    @pytest.mark.parametrize("node_name", _CHAIN_NODES)
    def test_correct_node_type(self, node_name: str) -> None:
        contract = _load_contract(node_name)
        expected = _EXPECTED_NODE_TYPES[node_name]
        assert contract["node_type"] == expected, (
            f"{node_name}: expected {expected}, got {contract['node_type']}"
        )

    @pytest.mark.parametrize("node_name", _CHAIN_NODES)
    def test_handler_routing_or_state_machine_present(self, node_name: str) -> None:
        contract = _load_contract(node_name)
        has_routing = "handler_routing" in contract
        has_fsm = "state_machine" in contract
        assert has_routing or has_fsm, (
            f"{node_name}: missing both handler_routing and state_machine"
        )
        if has_routing:
            routing = contract["handler_routing"]
            assert "routing_strategy" in routing
            assert "handlers" in routing
            assert len(routing["handlers"]) > 0

    @pytest.mark.parametrize("node_name", _CHAIN_NODES)
    def test_contract_name_matches_directory(self, node_name: str) -> None:
        contract = _load_contract(node_name)
        assert contract["name"] == node_name

    @pytest.mark.parametrize("node_name", _CHAIN_NODES)
    def test_contract_version_structure(self, node_name: str) -> None:
        contract = _load_contract(node_name)
        version = contract["contract_version"]
        for part in ("major", "minor", "patch"):
            assert part in version, f"{node_name}: missing version.{part}"
            assert isinstance(version[part], int)


@pytest.mark.unit
class TestEffectNodeContracts:
    """Additional tests specific to EFFECT node contracts."""

    @pytest.mark.parametrize(
        "node_name",
        ["node_chain_retrieval_effect", "node_chain_store_effect"],
    )
    def test_event_bus_topics(self, node_name: str) -> None:
        contract = _load_contract(node_name)
        assert "event_bus" in contract, f"{node_name}: missing event_bus"
        bus = contract["event_bus"]
        assert "subscribe_topics" in bus
        assert "publish_topics" in bus
        for topic in bus["subscribe_topics"] + bus["publish_topics"]:
            assert topic.startswith("onex."), f"Bad topic prefix: {topic}"

    @pytest.mark.parametrize(
        "node_name",
        ["node_chain_retrieval_effect", "node_chain_store_effect"],
    )
    def test_error_handling_configured(self, node_name: str) -> None:
        contract = _load_contract(node_name)
        assert "error_handling" in contract, f"{node_name}: missing error_handling"
        eh = contract["error_handling"]
        assert "retry_policy" in eh
        assert "circuit_breaker" in eh
        assert eh["circuit_breaker"]["enabled"] is True


@pytest.mark.unit
class TestReducerContract:
    """Tests specific to the reducer contract FSM definition."""

    def test_state_machine_defined(self) -> None:
        contract = _load_contract("node_chain_verify_reducer")
        assert "state_machine" in contract

    def test_initial_state_is_pending(self) -> None:
        contract = _load_contract("node_chain_verify_reducer")
        assert contract["state_machine"]["initial_state"] == "pending"

    def test_terminal_states(self) -> None:
        contract = _load_contract("node_chain_verify_reducer")
        terminals = contract["state_machine"]["terminal_states"]
        assert set(terminals) == {"complete", "failed"}

    def test_all_transitions_reference_valid_states(self) -> None:
        contract = _load_contract("node_chain_verify_reducer")
        sm = contract["state_machine"]
        state_names = {s["state_name"] for s in sm["states"]}
        for t in sm["transitions"]:
            assert t["from_state"] in state_names, (
                f"Unknown from_state: {t['from_state']}"
            )
            assert t["to_state"] in state_names, f"Unknown to_state: {t['to_state']}"


@pytest.mark.unit
class TestOrchestratorContract:
    """Tests specific to the orchestrator contract."""

    def test_workflow_coordination_present(self) -> None:
        contract = _load_contract("node_chain_orchestrator")
        assert "workflow_coordination" in contract

    def test_execution_graph_nodes(self) -> None:
        contract = _load_contract("node_chain_orchestrator")
        graph = contract["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]
        node_ids = {n["node_id"] for n in graph["nodes"]}
        assert node_ids == {"retrieve", "replay", "verify", "store"}

    def test_subscribes_to_all_internal_topics(self) -> None:
        contract = _load_contract("node_chain_orchestrator")
        topics = contract["event_bus"]["subscribe_topics"]
        assert len(topics) >= 3  # command + internal event topics

    def test_routing_strategy_is_payload_type_match(self) -> None:
        contract = _load_contract("node_chain_orchestrator")
        assert contract["handler_routing"]["routing_strategy"] == "payload_type_match"
