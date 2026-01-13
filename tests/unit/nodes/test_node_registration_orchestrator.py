# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for NodeRegistrationOrchestrator (Declarative Pattern).

The orchestrator uses the declarative pattern where workflow behavior
is 100% driven by contract.yaml. These tests verify:
1. Correct inheritance from NodeOrchestrator
2. Contract can be loaded and validated
3. No custom workflow methods exist (declarative purity)
4. Models are exported for backward compatibility
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from omnibase_core.enums import EnumNodeKind

from omnibase_infra.nodes.node_registration_orchestrator.node import (
    NodeRegistrationOrchestrator,
)

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Test Fixtures
# =============================================================================
# Note: simple_mock_container, contract_path, and contract_data fixtures are
# provided by tests/unit/nodes/conftest.py - no local definition needed.


# =============================================================================
# TestDeclarativeOrchestratorPattern
# =============================================================================


class TestDeclarativeOrchestratorPattern:
    """Tests verifying the declarative pattern is correctly implemented."""

    def test_inherits_from_node_orchestrator(
        self, simple_mock_container: MagicMock
    ) -> None:
        """Test that orchestrator inherits from NodeOrchestrator."""
        from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

        orchestrator = NodeRegistrationOrchestrator(simple_mock_container)
        assert isinstance(orchestrator, NodeOrchestrator)

    def test_no_custom_workflow_methods(self, simple_mock_container: MagicMock) -> None:
        """Test that no custom workflow methods exist (declarative purity).

        The refactored orchestrator should not have the old imperative methods:
        - execute_registration_workflow
        - _execute_intent_with_retry
        - _aggregate_results
        - _create_error_output
        - set_reducer
        - set_effect

        These methods were removed in favor of contract-driven behavior.
        """
        orchestrator = NodeRegistrationOrchestrator(simple_mock_container)

        # These methods should NOT exist (were removed in refactor)
        assert not hasattr(orchestrator, "execute_registration_workflow")
        assert not hasattr(orchestrator, "_execute_intent_with_retry")
        assert not hasattr(orchestrator, "_aggregate_results")
        assert not hasattr(orchestrator, "_create_error_output")
        assert not hasattr(orchestrator, "set_reducer")
        assert not hasattr(orchestrator, "set_effect")

    def test_no_custom_config_attributes(
        self, simple_mock_container: MagicMock
    ) -> None:
        """Test that old custom config attributes were removed.

        The declarative pattern uses contract.yaml for configuration,
        not instance attributes for config, reducer, effect, etc.
        """
        orchestrator = NodeRegistrationOrchestrator(simple_mock_container)

        # Old imperative pattern attributes should not exist
        assert not hasattr(orchestrator, "_config")
        assert not hasattr(orchestrator, "_reducer")
        assert not hasattr(orchestrator, "_effect")

    def test_no_custom_state_properties(self, simple_mock_container: MagicMock) -> None:
        """Test that old state properties were removed.

        The declarative pattern handles state through the base class,
        not through custom properties like reducer_state.
        """
        orchestrator = NodeRegistrationOrchestrator(simple_mock_container)

        # Old state properties should not exist
        assert not hasattr(orchestrator, "reducer_state")

    def test_no_factory_method(self) -> None:
        """Test that the create() factory method was removed.

        The declarative pattern uses standard __init__ instantiation
        instead of async factory methods.
        """
        assert not hasattr(NodeRegistrationOrchestrator, "create")

    def test_only_minimal_methods_defined(
        self, simple_mock_container: MagicMock
    ) -> None:
        """Test that only minimal methods are defined in the class.

        The declarative pattern should have minimal code - just __init__
        that calls super().__init__. All other behavior comes from base class.

        OMN-1102: Removed timeout/heartbeat methods - now fully declarative.
        Handler routing is driven entirely by contract.yaml.
        """
        # Get methods defined directly on NodeRegistrationOrchestrator
        # (not inherited from base classes)
        own_methods = [
            name
            for name in dir(NodeRegistrationOrchestrator)
            if not name.startswith("_")
            and callable(getattr(NodeRegistrationOrchestrator, name, None))
            and name in NodeRegistrationOrchestrator.__dict__
        ]

        # Should have NO public methods defined directly on the class
        # (all behavior inherited from NodeOrchestrator and driven by contract.yaml)
        assert own_methods == [], (
            f"Declarative node should have no custom public methods. "
            f"Found: {own_methods}"
        )


# =============================================================================
# TestInheritedBehavior
# =============================================================================


class TestInheritedBehavior:
    """Tests for behavior inherited from NodeOrchestrator base class."""

    def test_has_container_reference(self, simple_mock_container: MagicMock) -> None:
        """Test that container is stored (from base class)."""
        orchestrator = NodeRegistrationOrchestrator(simple_mock_container)

        # Base class should store container reference
        # The exact attribute name depends on NodeOrchestrator implementation
        assert hasattr(orchestrator, "_container") or hasattr(orchestrator, "container")

    def test_is_node_instance(self, simple_mock_container: MagicMock) -> None:
        """Test that orchestrator is a valid ONEX node instance."""
        orchestrator = NodeRegistrationOrchestrator(simple_mock_container)

        # Should be a proper class instance
        assert orchestrator is not None
        assert isinstance(orchestrator, NodeRegistrationOrchestrator)


# =============================================================================
# TestContractValidation
# =============================================================================


class TestContractValidation:
    """Tests for contract.yaml validation."""

    def test_contract_yaml_exists(self, contract_path: Path) -> None:
        """Test that contract.yaml exists."""
        assert contract_path.exists(), f"Contract not found at {contract_path}"

    def test_contract_yaml_valid_yaml(self, contract_data: dict) -> None:
        """Test that contract.yaml is valid YAML with required fields."""
        assert contract_data is not None
        assert "name" in contract_data
        assert "node_type" in contract_data
        assert contract_data["node_type"] == EnumNodeKind.ORCHESTRATOR.name

    def test_contract_has_required_metadata(self, contract_data: dict) -> None:
        """Test that contract has required metadata fields."""
        assert "contract_version" in contract_data
        assert "node_version" in contract_data
        assert "description" in contract_data

    def test_contract_has_input_output_models(self, contract_data: dict) -> None:
        """Test that contract defines input and output models."""
        assert "input_model" in contract_data
        assert "output_model" in contract_data

        input_model = contract_data["input_model"]
        output_model = contract_data["output_model"]

        assert "name" in input_model
        assert "module" in input_model
        assert "name" in output_model
        assert "module" in output_model

    def test_contract_has_workflow_definition(self, contract_data: dict) -> None:
        """Test that contract has workflow_definition."""
        assert "workflow_coordination" in contract_data
        wf = contract_data["workflow_coordination"]
        assert "workflow_definition" in wf
        assert "execution_graph" in wf["workflow_definition"]
        assert "nodes" in wf["workflow_definition"]["execution_graph"]

    def test_workflow_has_required_nodes(self, contract_data: dict) -> None:
        """Test that workflow has all 8 required execution graph nodes.

        The registration orchestrator workflow requires these nodes in order:
        1. receive_introspection - Receive introspection or tick event
        2. read_projection - Read current registration state from projection (OMN-930)
        3. evaluate_timeout - Evaluate timeout using injected time (OMN-973)
        4. compute_intents - Compute registration intents via reducer
        5. execute_consul_registration - Execute Consul registration
        6. execute_postgres_registration - Execute PostgreSQL registration
        7. aggregate_results - Aggregate registration results
        8. publish_outcome - Publish registration outcome event

        This test ensures all 8 nodes are present with exact matching.
        """
        nodes = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]
        node_ids = {n["node_id"] for n in nodes}

        # All 8 required execution graph nodes per C1 requirements
        expected_nodes = {
            "receive_introspection",
            "read_projection",
            "evaluate_timeout",
            "compute_intents",
            "execute_consul_registration",
            "execute_postgres_registration",
            "aggregate_results",
            "publish_outcome",
        }

        # Strict equality check - must have exactly these nodes
        assert expected_nodes == node_ids, (
            f"Execution graph nodes mismatch.\n"
            f"Missing: {expected_nodes - node_ids}\n"
            f"Extra: {node_ids - expected_nodes}"
        )

    def test_workflow_has_metadata(self, contract_data: dict) -> None:
        """Test that workflow has required metadata."""
        wf_def = contract_data["workflow_coordination"]["workflow_definition"]
        assert "workflow_metadata" in wf_def

        metadata = wf_def["workflow_metadata"]
        assert "workflow_name" in metadata
        assert "workflow_version" in metadata
        assert "description" in metadata

    def test_workflow_has_coordination_rules(self, contract_data: dict) -> None:
        """Test that workflow has coordination rules.

        All coordination settings are consolidated in coordination_rules,
        including execution_mode which was previously in workflow_metadata.
        """
        wf_def = contract_data["workflow_coordination"]["workflow_definition"]
        assert "coordination_rules" in wf_def

        rules = wf_def["coordination_rules"]
        # Execution mode configuration
        assert "execution_mode" in rules
        assert "parallel_execution_allowed" in rules
        assert "max_parallel_branches" in rules
        # Failure handling
        assert "failure_recovery_strategy" in rules
        assert "max_retries" in rules
        assert "recovery_enabled" in rules
        # Timeout configuration
        assert "timeout_ms" in rules
        # Checkpoint and state persistence
        assert "checkpoint_enabled" in rules
        assert "checkpoint_interval_ms" in rules
        assert "state_persistence_enabled" in rules
        # Rollback configuration
        assert "rollback_enabled" in rules

    def test_contract_has_error_handling(self, contract_data: dict) -> None:
        """Test that contract defines error handling configuration."""
        assert "error_handling" in contract_data

        error_handling = contract_data["error_handling"]
        assert "retry_policy" in error_handling
        assert "circuit_breaker" in error_handling

    def test_contract_has_consumed_events(self, contract_data: dict) -> None:
        """Test that contract defines consumed events."""
        assert "consumed_events" in contract_data
        consumed = contract_data["consumed_events"]

        assert len(consumed) > 0
        # Should consume introspection events
        event_types = [e["event_type"] for e in consumed]
        assert "NodeIntrospectionEvent" in event_types

    def test_contract_has_published_events(self, contract_data: dict) -> None:
        """Test that contract defines published events."""
        assert "published_events" in contract_data
        published = contract_data["published_events"]

        assert len(published) > 0
        # Should publish registration result events
        event_types = [e["event_type"] for e in published]
        assert "NodeRegistrationResultEvent" in event_types


# =============================================================================
# TestModelsExport
# =============================================================================


class TestModelsExport:
    """Tests for backward-compatible model exports."""

    def test_models_still_exported(self) -> None:
        """Test that models are still exported for backward compatibility.

        Even though the orchestrator is now declarative, the models
        should still be importable for use by other components.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelIntentExecutionResult,
            ModelOrchestratorConfig,
            ModelOrchestratorInput,
            ModelOrchestratorOutput,
        )

        assert ModelOrchestratorConfig is not None
        assert ModelOrchestratorInput is not None
        assert ModelOrchestratorOutput is not None
        assert ModelIntentExecutionResult is not None

    def test_models_are_importable_from_package(self) -> None:
        """Test that models can be imported from the models package."""
        # This tests the __init__.py exports
        from omnibase_infra.nodes.node_registration_orchestrator import models

        assert hasattr(models, "ModelOrchestratorConfig")
        assert hasattr(models, "ModelOrchestratorInput")
        assert hasattr(models, "ModelOrchestratorOutput")
        assert hasattr(models, "ModelIntentExecutionResult")


# =============================================================================
# TestNodeInstantiation
# =============================================================================


class TestNodeInstantiation:
    """Tests for node instantiation."""

    def test_instantiation_with_container(
        self, simple_mock_container: MagicMock
    ) -> None:
        """Test that node can be instantiated with a container."""
        orchestrator = NodeRegistrationOrchestrator(simple_mock_container)
        assert orchestrator is not None

    def test_instantiation_requires_container(self) -> None:
        """Test that instantiation requires a container argument."""
        with pytest.raises(TypeError):
            NodeRegistrationOrchestrator()  # type: ignore[call-arg]

    def test_multiple_instances_independent(
        self, simple_mock_container: MagicMock
    ) -> None:
        """Test that multiple instances are independent."""
        orch1 = NodeRegistrationOrchestrator(simple_mock_container)
        orch2 = NodeRegistrationOrchestrator(simple_mock_container)

        assert orch1 is not orch2
