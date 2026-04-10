# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for node_onboarding_orchestrator contract.yaml structure and node.py (OMN-8280)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_NODE_DIR = (
    Path(__file__).parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_onboarding_orchestrator"
)
_CONTRACT_FILE = _NODE_DIR / "contract.yaml"


@pytest.fixture
def contract_data() -> dict:
    """Load the node contract as a dict."""
    assert _CONTRACT_FILE.exists(), f"contract.yaml not found at {_CONTRACT_FILE}"
    with _CONTRACT_FILE.open() as f:
        return yaml.safe_load(f)


class TestNodeOnboardingOrchestratorContract:
    """Structural validation of contract.yaml and node.py for node_onboarding_orchestrator."""

    def test_contract_file_exists(self) -> None:
        """contract.yaml must exist in node directory."""
        assert _CONTRACT_FILE.exists(), f"contract.yaml not found at {_CONTRACT_FILE}"

    def test_required_fields_present(self, contract_data) -> None:
        """Required top-level contract fields must all be present."""
        required = {
            "name",
            "node_type",
            "contract_version",
            "node_version",
            "input_model",
            "output_model",
        }
        for field in required:
            assert field in contract_data, (
                f"Required field '{field}' missing from contract.yaml"
            )

    def test_node_type_is_orchestrator(self, contract_data) -> None:
        """node_type must be ORCHESTRATOR_GENERIC."""
        assert contract_data["node_type"] == "ORCHESTRATOR_GENERIC", (
            f"Expected node_type 'ORCHESTRATOR_GENERIC', got '{contract_data['node_type']}'"
        )

    def test_event_bus_topics_non_empty(self, contract_data) -> None:
        """event_bus must declare non-empty subscribe and publish topic lists."""
        event_bus = contract_data.get("event_bus", {})
        assert event_bus, "event_bus block missing from contract.yaml"
        subscribe = event_bus.get("subscribe_topics", [])
        publish = event_bus.get("publish_topics", [])
        assert len(subscribe) > 0, "event_bus.subscribe_topics must be non-empty"
        assert len(publish) > 0, "event_bus.publish_topics must be non-empty"

    def test_input_model_fields(self, contract_data) -> None:
        """input_model must declare name and module."""
        im = contract_data.get("input_model", {})
        assert "name" in im, "input_model.name missing"
        assert "module" in im, "input_model.module missing"
        assert im["name"] == "ModelOnboardingInput"

    def test_output_model_fields(self, contract_data) -> None:
        """output_model must declare name and module."""
        om = contract_data.get("output_model", {})
        assert "name" in om, "output_model.name missing"
        assert "module" in om, "output_model.module missing"
        assert om["name"] == "ModelOnboardingOutput"

    def test_node_py_importable(self) -> None:
        """NodeOnboardingOrchestrator must import cleanly from node.py."""
        from omnibase_infra.nodes.node_onboarding_orchestrator.node import (
            NodeOnboardingOrchestrator,
        )

        assert NodeOnboardingOrchestrator is not None

    def test_node_extends_node_orchestrator(self) -> None:
        """NodeOnboardingOrchestrator must subclass NodeOrchestrator."""
        from omnibase_core.nodes import NodeOrchestrator
        from omnibase_infra.nodes.node_onboarding_orchestrator.node import (
            NodeOnboardingOrchestrator,
        )

        assert issubclass(NodeOnboardingOrchestrator, NodeOrchestrator), (
            "NodeOnboardingOrchestrator must extend NodeOrchestrator"
        )
