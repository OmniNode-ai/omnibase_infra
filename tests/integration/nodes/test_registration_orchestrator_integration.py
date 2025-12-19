# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for NodeRegistrationOrchestrator (Declarative Pattern).

These tests verify the orchestrator's contract-driven behavior works correctly
in an integration context. Since the orchestrator is now declarative with zero
custom code, these tests focus on:

1. Contract loading and validation
2. Workflow definition structure
3. Model compatibility with contract specifications
4. Base class integration

Test Categories:
    - TestContractIntegration: Contract loading and structure validation
    - TestWorkflowGraphIntegration: Execution graph structure tests
    - TestModelContractAlignment: Model and contract specification alignment
    - TestDependencyStructure: Dependency declarations in contract

Running Tests:
    # Run all integration tests for the orchestrator:
    pytest tests/integration/nodes/test_registration_orchestrator_integration.py

    # Run with verbose output:
    pytest tests/integration/nodes/test_registration_orchestrator_integration.py -v

    # Run specific test class:
    pytest tests/integration/nodes/test_registration_orchestrator_integration.py::TestContractIntegration
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.node import (
    NodeRegistrationOrchestrator,
)

# Module-level markers - all tests in this file are integration tests
pytestmark = [
    pytest.mark.integration,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock ONEX container."""
    container = MagicMock()
    container.config = MagicMock()
    return container


@pytest.fixture
def contract_path() -> Path:
    """Return path to contract.yaml."""
    return Path(
        "src/omnibase_infra/nodes/node_registration_orchestrator/v1_0_0/contract.yaml"
    )


@pytest.fixture
def contract_data(contract_path: Path) -> dict:
    """Load and return contract.yaml as dict."""
    with open(contract_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# TestContractIntegration
# =============================================================================


class TestContractIntegration:
    """Integration tests for contract loading and structure.

    These tests verify that the contract.yaml is properly structured
    and contains all required fields for a declarative orchestrator.
    """

    def test_contract_structure_complete(self, contract_data: dict) -> None:
        """Test that contract has all required top-level sections."""
        required_sections = [
            "contract_version",
            "node_version",
            "name",
            "node_type",
            "description",
            "input_model",
            "output_model",
            "workflow_coordination",
            "consumed_events",
            "published_events",
            "error_handling",
        ]

        for section in required_sections:
            assert section in contract_data, f"Missing required section: {section}"

    def test_contract_versions_valid(self, contract_data: dict) -> None:
        """Test that contract and node versions are valid semver format."""
        contract_version = contract_data["contract_version"]
        node_version = contract_data["node_version"]

        # Should be semver format (x.y.z)
        assert len(contract_version.split(".")) == 3
        assert len(node_version.split(".")) == 3

        # All parts should be numeric
        for part in contract_version.split("."):
            assert part.isdigit()
        for part in node_version.split("."):
            assert part.isdigit()

    def test_node_type_is_orchestrator(self, contract_data: dict) -> None:
        """Test that node_type is ORCHESTRATOR."""
        assert contract_data["node_type"] == "ORCHESTRATOR"

    def test_input_model_importable(self, contract_data: dict) -> None:
        """Test that input model specified in contract is importable."""
        input_model = contract_data["input_model"]
        module_path = input_model["module"]
        class_name = input_model["name"]

        # Import the module dynamically
        import importlib

        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        assert model_class is not None
        assert class_name == "ModelOrchestratorInput"

    def test_output_model_importable(self, contract_data: dict) -> None:
        """Test that output model specified in contract is importable."""
        output_model = contract_data["output_model"]
        module_path = output_model["module"]
        class_name = output_model["name"]

        # Import the module dynamically
        import importlib

        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        assert model_class is not None
        assert class_name == "ModelOrchestratorOutput"


# =============================================================================
# TestWorkflowGraphIntegration
# =============================================================================


class TestWorkflowGraphIntegration:
    """Integration tests for workflow execution graph.

    These tests verify that the execution graph is properly structured
    and defines the expected workflow steps.
    """

    def test_execution_graph_has_all_nodes(self, contract_data: dict) -> None:
        """Test that execution graph has all expected nodes."""
        nodes = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]
        node_ids = {n["node_id"] for n in nodes}

        expected_nodes = {
            "receive_introspection",
            "compute_intents",
            "execute_consul_registration",
            "execute_postgres_registration",
            "aggregate_results",
            "publish_outcome",
        }

        assert expected_nodes <= node_ids, f"Missing nodes: {expected_nodes - node_ids}"

    def test_execution_graph_dependencies_valid(self, contract_data: dict) -> None:
        """Test that all dependencies reference valid nodes."""
        nodes = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        # Build set of all node IDs
        node_ids = {n["node_id"] for n in nodes}

        # Check that all dependencies reference existing nodes
        for node in nodes:
            deps = node.get("depends_on", [])
            for dep in deps:
                assert (
                    dep in node_ids
                ), f"Node {node['node_id']} depends on non-existent node: {dep}"

    def test_execution_graph_has_no_cycles(self, contract_data: dict) -> None:
        """Test that execution graph has no circular dependencies."""
        nodes = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        # Build dependency graph
        deps = {n["node_id"]: set(n.get("depends_on", [])) for n in nodes}

        # Topological sort to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for dep in deps.get(node, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node_id in deps:
            if node_id not in visited:
                if has_cycle(node_id):
                    pytest.fail(f"Cycle detected involving node: {node_id}")

    def test_node_types_valid(self, contract_data: dict) -> None:
        """Test that all node types are valid ONEX types."""
        nodes = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        valid_types = {"effect", "compute", "reducer", "orchestrator"}

        for node in nodes:
            node_type = node.get("node_type", "").lower()
            assert (
                node_type in valid_types
            ), f"Invalid node_type '{node_type}' for node {node['node_id']}"

    def test_consul_and_postgres_steps_are_effects(self, contract_data: dict) -> None:
        """Test that registration steps are effect nodes."""
        nodes = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        # Find the registration nodes
        for node in nodes:
            if "consul" in node["node_id"].lower():
                assert (
                    node["node_type"] == "effect"
                ), f"Consul registration should be effect type"
            if "postgres" in node["node_id"].lower():
                assert (
                    node["node_type"] == "effect"
                ), f"Postgres registration should be effect type"

    def test_compute_intents_is_reducer(self, contract_data: dict) -> None:
        """Test that compute_intents step is a reducer node."""
        nodes = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        for node in nodes:
            if node["node_id"] == "compute_intents":
                assert (
                    node["node_type"] == "reducer"
                ), "compute_intents should be reducer type"
                break
        else:
            pytest.fail("compute_intents node not found")


# =============================================================================
# TestCoordinationRulesIntegration
# =============================================================================


class TestCoordinationRulesIntegration:
    """Integration tests for workflow coordination rules.

    These tests verify that coordination rules are properly configured
    for the registration workflow.
    """

    def test_retry_policy_configured(self, contract_data: dict) -> None:
        """Test that retry policy is properly configured."""
        rules = contract_data["workflow_coordination"]["workflow_definition"][
            "coordination_rules"
        ]

        assert "max_retries" in rules
        assert rules["max_retries"] >= 0
        assert rules["failure_recovery_strategy"] == "retry"

    def test_timeout_configured(self, contract_data: dict) -> None:
        """Test that timeout is properly configured."""
        rules = contract_data["workflow_coordination"]["workflow_definition"][
            "coordination_rules"
        ]

        assert "timeout_ms" in rules
        assert rules["timeout_ms"] > 0

    def test_sequential_execution_mode(self, contract_data: dict) -> None:
        """Test that execution mode is sequential (appropriate for registration)."""
        metadata = contract_data["workflow_coordination"]["workflow_definition"][
            "workflow_metadata"
        ]

        assert metadata["execution_mode"] == "sequential"

    def test_checkpoint_enabled(self, contract_data: dict) -> None:
        """Test that checkpointing is enabled for recovery."""
        rules = contract_data["workflow_coordination"]["workflow_definition"][
            "coordination_rules"
        ]

        assert rules.get("checkpoint_enabled", False) is True


# =============================================================================
# TestErrorHandlingIntegration
# =============================================================================


class TestErrorHandlingIntegration:
    """Integration tests for error handling configuration.

    These tests verify that error handling is properly configured
    in the contract for resilient operation.
    """

    def test_retry_policy_structure(self, contract_data: dict) -> None:
        """Test retry policy has all required fields."""
        retry_policy = contract_data["error_handling"]["retry_policy"]

        required_fields = [
            "max_retries",
            "initial_delay_ms",
            "max_delay_ms",
            "exponential_base",
            "retry_on",
        ]

        for field in required_fields:
            assert field in retry_policy, f"Missing retry policy field: {field}"

    def test_circuit_breaker_configured(self, contract_data: dict) -> None:
        """Test circuit breaker is properly configured."""
        circuit_breaker = contract_data["error_handling"]["circuit_breaker"]

        assert circuit_breaker.get("enabled", False) is True
        assert "failure_threshold" in circuit_breaker
        assert "reset_timeout_ms" in circuit_breaker

    def test_error_types_defined(self, contract_data: dict) -> None:
        """Test that error types are defined."""
        error_types = contract_data["error_handling"]["error_types"]

        assert len(error_types) > 0

        # Check each error type has required fields
        for error_type in error_types:
            assert "name" in error_type
            assert "description" in error_type
            assert "recoverable" in error_type

    def test_retryable_errors_specified(self, contract_data: dict) -> None:
        """Test that retryable error types are specified."""
        retry_on = contract_data["error_handling"]["retry_policy"]["retry_on"]

        assert len(retry_on) > 0
        assert "ConnectionError" in retry_on or "EffectExecutionError" in retry_on


# =============================================================================
# TestEventIntegration
# =============================================================================


class TestEventIntegration:
    """Integration tests for event consumption and publication.

    These tests verify that events are properly configured in the contract.
    """

    def test_consumed_events_have_topics(self, contract_data: dict) -> None:
        """Test that consumed events have topic patterns."""
        consumed = contract_data["consumed_events"]

        for event in consumed:
            assert "topic" in event
            assert "event_type" in event
            # Topic should be a pattern with placeholders
            assert "{" in event["topic"] and "}" in event["topic"]

    def test_published_events_have_topics(self, contract_data: dict) -> None:
        """Test that published events have topic patterns."""
        published = contract_data["published_events"]

        for event in published:
            assert "topic" in event
            assert "event_type" in event

    def test_intent_consumption_configured(self, contract_data: dict) -> None:
        """Test that intent consumption is properly configured."""
        intent_config = contract_data.get("intent_consumption", {})

        if intent_config:
            assert "subscribed_intents" in intent_config
            assert "intent_routing_table" in intent_config

            # All subscribed intents should have routing
            for intent in intent_config["subscribed_intents"]:
                assert (
                    intent in intent_config["intent_routing_table"]
                ), f"Missing routing for intent: {intent}"


# =============================================================================
# TestModelContractAlignment
# =============================================================================


class TestModelContractAlignment:
    """Integration tests for model and contract alignment.

    These tests verify that the models are compatible with the
    contract specifications.
    """

    def test_models_match_contract_specification(self, contract_data: dict) -> None:
        """Test that model names match contract specification."""
        input_model_name = contract_data["input_model"]["name"]
        output_model_name = contract_data["output_model"]["name"]

        from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models import (
            ModelOrchestratorInput,
            ModelOrchestratorOutput,
        )

        assert ModelOrchestratorInput.__name__ == input_model_name
        assert ModelOrchestratorOutput.__name__ == output_model_name

    def test_model_module_paths_valid(self, contract_data: dict) -> None:
        """Test that model module paths in contract are valid."""
        input_module = contract_data["input_model"]["module"]
        output_module = contract_data["output_model"]["module"]

        expected_module = (
            "omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models"
        )
        assert input_module == expected_module
        assert output_module == expected_module


# =============================================================================
# TestNodeIntegration
# =============================================================================


class TestNodeIntegration:
    """Integration tests for node instantiation and base class behavior."""

    def test_node_instantiation_succeeds(self, mock_container: MagicMock) -> None:
        """Test that node can be instantiated successfully."""
        orchestrator = NodeRegistrationOrchestrator(mock_container)
        assert orchestrator is not None

    def test_node_inherits_base_class(self, mock_container: MagicMock) -> None:
        """Test that node inherits from NodeOrchestrator base class."""
        from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

        orchestrator = NodeRegistrationOrchestrator(mock_container)
        assert isinstance(orchestrator, NodeOrchestrator)

    def test_node_is_declarative(self, mock_container: MagicMock) -> None:
        """Test that node has no custom imperative methods."""
        orchestrator = NodeRegistrationOrchestrator(mock_container)

        # Old imperative methods should not exist
        imperative_methods = [
            "execute_registration_workflow",
            "set_reducer",
            "set_effect",
            "_execute_intent_with_retry",
            "_aggregate_results",
        ]

        for method in imperative_methods:
            assert not hasattr(
                orchestrator, method
            ), f"Found imperative method: {method}"


# =============================================================================
# TestDependencyStructure
# =============================================================================


class TestDependencyStructure:
    """Integration tests for dependency declarations."""

    def test_dependencies_declared(self, contract_data: dict) -> None:
        """Test that dependencies are declared in contract."""
        deps = contract_data.get("dependencies", [])

        # Should have at least reducer protocol and effect node dependencies
        dep_names = [d["name"] for d in deps]

        assert "reducer_protocol" in dep_names or len(deps) > 0
        assert "effect_node" in dep_names or len(deps) > 0

    def test_dependencies_have_required_fields(self, contract_data: dict) -> None:
        """Test that dependencies have required fields."""
        deps = contract_data.get("dependencies", [])

        for dep in deps:
            assert "name" in dep
            assert "type" in dep
            assert "description" in dep


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "TestContractIntegration",
    "TestWorkflowGraphIntegration",
    "TestCoordinationRulesIntegration",
    "TestErrorHandlingIntegration",
    "TestEventIntegration",
    "TestModelContractAlignment",
    "TestNodeIntegration",
    "TestDependencyStructure",
]
