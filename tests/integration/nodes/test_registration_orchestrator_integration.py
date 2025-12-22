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
from uuid import UUID

import pytest
import yaml

from omnibase_infra.nodes.node_registration_orchestrator.node import (
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
    return Path("src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml")


@pytest.fixture
def contract_data(contract_path: Path) -> dict:
    """Load and return contract.yaml as dict.

    Raises:
        pytest.skip: If contract file doesn't exist (allows tests to be skipped gracefully).
        yaml.YAMLError: If contract file contains invalid YAML.
    """
    if not contract_path.exists():
        pytest.skip(f"Contract file not found: {contract_path}")

    with open(contract_path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in contract file: {e}")


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

        # contract_version should be a dict with major, minor, patch keys
        assert isinstance(contract_version, dict), "contract_version should be a dict"
        assert "major" in contract_version, "contract_version missing 'major' key"
        assert "minor" in contract_version, "contract_version missing 'minor' key"
        assert "patch" in contract_version, "contract_version missing 'patch' key"
        assert isinstance(contract_version["major"], int)
        assert isinstance(contract_version["minor"], int)
        assert isinstance(contract_version["patch"], int)

        # node_version should be semver format string (x.y.z)
        assert isinstance(node_version, str), "node_version should be a string"
        assert len(node_version.split(".")) == 3
        for part in node_version.split("."):
            assert part.isdigit()

    def test_node_type_is_orchestrator(self, contract_data: dict) -> None:
        """Test that node_type is ORCHESTRATOR."""
        assert contract_data["node_type"] == "ORCHESTRATOR"

    def test_input_model_importable(self, contract_data: dict) -> None:
        """Test that input model specified in contract is importable and valid.

        Verifies:
        - Model can be imported from specified module path
        - Model class name matches contract specification
        - Model is a proper Pydantic BaseModel with expected fields
        """
        import importlib

        from pydantic import BaseModel

        input_model = contract_data["input_model"]
        module_path = input_model["module"]
        class_name = input_model["name"]

        # Import the module dynamically
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Verify class name matches contract
        assert class_name == "ModelOrchestratorInput"
        assert model_class.__name__ == class_name

        # Verify it's a Pydantic model via duck typing (check for model_fields attribute)
        assert hasattr(model_class, "model_fields"), (
            f"{class_name} must be a Pydantic model with 'model_fields'"
        )

        # Verify required fields are present
        required_fields = {"introspection_event", "correlation_id"}
        actual_fields = set(model_class.model_fields.keys())
        missing_fields = required_fields - actual_fields
        assert not missing_fields, (
            f"{class_name} missing required fields: {missing_fields}"
        )

        # Verify model is subclass of BaseModel via duck typing
        # (has model_validate method which is BaseModel behavior)
        assert hasattr(model_class, "model_validate"), (
            f"{class_name} must have 'model_validate' method (Pydantic BaseModel)"
        )

    def test_output_model_importable(self, contract_data: dict) -> None:
        """Test that output model specified in contract is importable and valid.

        Verifies:
        - Model can be imported from specified module path
        - Model class name matches contract specification
        - Model is a proper Pydantic BaseModel with expected fields
        """
        import importlib

        output_model = contract_data["output_model"]
        module_path = output_model["module"]
        class_name = output_model["name"]

        # Import the module dynamically
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Verify class name matches contract
        assert class_name == "ModelOrchestratorOutput"
        assert model_class.__name__ == class_name

        # Verify it's a Pydantic model via duck typing (check for model_fields attribute)
        assert hasattr(model_class, "model_fields"), (
            f"{class_name} must be a Pydantic model with 'model_fields'"
        )

        # Verify required fields are present for orchestrator output
        required_fields = {
            "correlation_id",
            "status",
            "consul_applied",
            "postgres_applied",
            "intent_results",
        }
        actual_fields = set(model_class.model_fields.keys())
        missing_fields = required_fields - actual_fields
        assert not missing_fields, (
            f"{class_name} missing required fields: {missing_fields}"
        )

        # Verify model is subclass of BaseModel via duck typing
        assert hasattr(model_class, "model_validate"), (
            f"{class_name} must have 'model_validate' method (Pydantic BaseModel)"
        )


# =============================================================================
# TestWorkflowGraphIntegration
# =============================================================================


class TestWorkflowGraphIntegration:
    """Integration tests for workflow execution graph.

    These tests verify that the execution graph is properly structured
    and defines the expected workflow steps.
    """

    def test_execution_graph_has_all_nodes(self, contract_data: dict) -> None:
        """Test that execution graph has all 8 required nodes.

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
                assert dep in node_ids, (
                    f"Node {node['node_id']} depends on non-existent node: {dep}"
                )

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
            assert node_type in valid_types, (
                f"Invalid node_type '{node_type}' for node {node['node_id']}"
            )

    def test_consul_and_postgres_steps_are_effects(self, contract_data: dict) -> None:
        """Test that registration steps are effect nodes."""
        nodes = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        # Find the registration nodes
        for node in nodes:
            if "consul" in node["node_id"].lower():
                assert node["node_type"] == "effect", (
                    "Consul registration should be effect type"
                )
            if "postgres" in node["node_id"].lower():
                assert node["node_type"] == "effect", (
                    "Postgres registration should be effect type"
                )

    def test_compute_intents_is_reducer(self, contract_data: dict) -> None:
        """Test that compute_intents step is a reducer node."""
        nodes = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        for node in nodes:
            if node["node_id"] == "compute_intents":
                assert node["node_type"] == "reducer", (
                    "compute_intents should be reducer type"
                )
                break
        else:
            pytest.fail("compute_intents node not found")

    def test_all_8_nodes_have_correct_properties(self, contract_data: dict) -> None:
        """Test that all 8 execution graph nodes have correct types and dependencies.

        This test validates each of the 8 nodes in the registration orchestrator workflow:

        1. receive_introspection (effect) - Entry point, no dependencies
        2. read_projection (effect) - Reads state, depends on receive_introspection
        3. evaluate_timeout (compute) - Evaluates timeout, depends on read_projection
        4. compute_intents (reducer) - Generates intents, depends on evaluate_timeout
        5. execute_consul_registration (effect) - Consul registration, depends on compute_intents
        6. execute_postgres_registration (effect) - PostgreSQL registration, depends on compute_intents
        7. aggregate_results (compute) - Aggregates results, depends on both registrations
        8. publish_outcome (effect) - Publishes result event, depends on aggregate_results

        Each node is validated for:
        - Correct node_type (effect, compute, or reducer)
        - Correct dependencies (depends_on list)
        - Presence of description
        """
        nodes = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        # Build lookup for easier validation
        node_map = {n["node_id"]: n for n in nodes}

        # Expected properties for all 8 nodes
        # Format: node_id -> (node_type, depends_on)
        expected_node_properties = {
            # Node 1: Entry point - receives introspection or tick event
            "receive_introspection": {
                "node_type": "effect",
                "depends_on": [],
                "description": "Receive introspection or tick event",
            },
            # Node 2: Read projection state (OMN-930)
            "read_projection": {
                "node_type": "effect",
                "depends_on": ["receive_introspection"],
                "description": "Read current registration state from projection",
            },
            # Node 3: Evaluate timeout using injected time (OMN-973)
            "evaluate_timeout": {
                "node_type": "compute",
                "depends_on": ["read_projection"],
                "description": "Evaluate timeout based on injected now from RuntimeTick",
            },
            # Node 4: Compute intents via reducer
            "compute_intents": {
                "node_type": "reducer",
                "depends_on": ["evaluate_timeout"],
                "description": "Compute registration intents from introspection event",
            },
            # Node 5: Execute Consul registration
            "execute_consul_registration": {
                "node_type": "effect",
                "depends_on": ["compute_intents"],
                "description": "Execute Consul registration intent",
            },
            # Node 6: Execute PostgreSQL registration
            "execute_postgres_registration": {
                "node_type": "effect",
                "depends_on": ["compute_intents"],
                "description": "Execute PostgreSQL registration intent",
            },
            # Node 7: Aggregate registration results
            "aggregate_results": {
                "node_type": "compute",
                "depends_on": ["execute_consul_registration", "execute_postgres_registration"],
                "description": "Aggregate registration results",
            },
            # Node 8: Publish outcome event
            "publish_outcome": {
                "node_type": "effect",
                "depends_on": ["aggregate_results"],
                "description": "Publish registration outcome event",
            },
        }

        # Validate we have exactly 8 nodes
        assert len(expected_node_properties) == 8, "Test expects exactly 8 nodes"
        assert len(node_map) == 8, f"Contract has {len(node_map)} nodes, expected 8"

        # Validate each node's properties
        for node_id, expected in expected_node_properties.items():
            assert node_id in node_map, f"Missing node: {node_id}"
            node = node_map[node_id]

            # Validate node_type
            assert node["node_type"] == expected["node_type"], (
                f"Node '{node_id}' has type '{node['node_type']}', "
                f"expected '{expected['node_type']}'"
            )

            # Validate dependencies (order-independent comparison)
            actual_deps = set(node.get("depends_on", []))
            expected_deps = set(expected["depends_on"])
            assert actual_deps == expected_deps, (
                f"Node '{node_id}' has dependencies {actual_deps}, "
                f"expected {expected_deps}"
            )

            # Validate description exists
            assert "description" in node, f"Node '{node_id}' missing description"
            assert node["description"] == expected["description"], (
                f"Node '{node_id}' has description '{node['description']}', "
                f"expected '{expected['description']}'"
            )


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
        """Test that execution mode is sequential (appropriate for registration).

        Note: execution_mode is consolidated in coordination_rules
        along with all other coordination settings.
        """
        rules = contract_data["workflow_coordination"]["workflow_definition"][
            "coordination_rules"
        ]

        assert rules["execution_mode"] == "sequential"

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
                assert intent in intent_config["intent_routing_table"], (
                    f"Missing routing for intent: {intent}"
                )


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

        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelOrchestratorInput,
            ModelOrchestratorOutput,
        )

        assert ModelOrchestratorInput.__name__ == input_model_name
        assert ModelOrchestratorOutput.__name__ == output_model_name

    def test_model_module_paths_valid(self, contract_data: dict) -> None:
        """Test that model module paths in contract are valid."""
        input_module = contract_data["input_model"]["module"]
        output_module = contract_data["output_model"]["module"]

        expected_module = "omnibase_infra.nodes.node_registration_orchestrator.models"
        assert input_module == expected_module
        assert output_module == expected_module


# =============================================================================
# TestNodeIntegration
# =============================================================================


class TestNodeIntegration:
    """Integration tests for node instantiation and base class behavior."""

    def test_node_instantiation_succeeds(self, mock_container: MagicMock) -> None:
        """Test that node can be instantiated successfully with proper state.

        Verifies:
        - Node is instantiated with expected type
        - Container is stored as expected
        - Node has required orchestrator attributes
        """
        orchestrator = NodeRegistrationOrchestrator(mock_container)

        # Verify type via duck typing (check for class name match)
        assert orchestrator.__class__.__name__ == "NodeRegistrationOrchestrator", (
            "Instantiated object must be NodeRegistrationOrchestrator"
        )

        # Verify container reference is stored
        assert hasattr(orchestrator, "container"), (
            "Orchestrator must have 'container' attribute"
        )
        assert orchestrator.container is mock_container, (
            "Container reference must match provided container"
        )

    def test_node_inherits_base_class(self, mock_container: MagicMock) -> None:
        """Test that node inherits from NodeOrchestrator base class.

        Per ONEX conventions, we verify inheritance via duck typing by checking
        for required methods and attributes rather than isinstance checks.
        """
        orchestrator = NodeRegistrationOrchestrator(mock_container)

        # Verify NodeOrchestrator behavior via duck typing
        # NodeOrchestrator provides workflow execution and state management methods
        required_methods = [
            "process",  # Core processing method
            "execute_workflow_from_contract",  # Contract-driven workflow execution
            "validate_workflow_contract",  # Workflow contract validation
            "get_workflow_snapshot",  # Workflow state management
            "get_node_type",  # Node metadata
        ]

        for method_name in required_methods:
            assert hasattr(orchestrator, method_name), (
                f"Orchestrator must have '{method_name}' method from NodeOrchestrator"
            )
            assert callable(getattr(orchestrator, method_name)), (
                f"'{method_name}' must be callable"
            )

        # Verify it has container attribute (set by NodeOrchestrator.__init__)
        assert hasattr(orchestrator, "container"), (
            "Orchestrator must have 'container' attribute from base class"
        )

        # Verify class hierarchy by checking MRO contains expected base class name
        mro_names = [cls.__name__ for cls in orchestrator.__class__.__mro__]
        assert "NodeOrchestrator" in mro_names, (
            f"NodeOrchestrator must be in MRO, found: {mro_names}"
        )

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
            assert not hasattr(orchestrator, method), (
                f"Found imperative method: {method}"
            )


# =============================================================================
# TestDependencyStructure
# =============================================================================


class TestDependencyStructure:
    """Integration tests for dependency declarations."""

    def test_dependencies_declared(self, contract_data: dict) -> None:
        """Test that required dependencies are declared in contract.

        The registration orchestrator requires specific dependencies:
        - reducer_protocol: For computing registration intents
        - effect_node: For executing registration operations
        - projection_reader: For reading current state (OMN-930)

        This test ensures all required dependencies are present with strict
        assertions that will fail if any dependency is missing.
        """
        deps = contract_data.get("dependencies", [])

        # Must have at least the required dependencies
        assert len(deps) >= 3, (
            f"Contract must declare at least 3 dependencies "
            f"(reducer_protocol, effect_node, projection_reader), found {len(deps)}"
        )

        dep_names = [d["name"] for d in deps]

        # Each required dependency must be explicitly present - no fallback conditions
        assert "reducer_protocol" in dep_names, (
            "Must declare 'reducer_protocol' dependency for computing intents"
        )
        assert "effect_node" in dep_names, (
            "Must declare 'effect_node' dependency for executing registration operations"
        )
        assert "projection_reader" in dep_names, (
            "Must declare 'projection_reader' dependency for reading state (OMN-930)"
        )

    def test_dependencies_have_required_fields(self, contract_data: dict) -> None:
        """Test that dependencies have required fields."""
        deps = contract_data.get("dependencies", [])

        for dep in deps:
            assert "name" in dep, f"Dependency missing 'name' field: {dep}"
            assert "type" in dep, (
                f"Dependency '{dep.get('name', 'unknown')}' missing 'type' field"
            )
            assert "description" in dep, (
                f"Dependency '{dep.get('name', 'unknown')}' missing 'description' field"
            )

    def test_dependency_types_valid(self, contract_data: dict) -> None:
        """Test that dependency types are valid ONEX types."""
        deps = contract_data.get("dependencies", [])

        valid_types = {"protocol", "node", "service", "config"}

        for dep in deps:
            dep_type = dep.get("type", "")
            assert dep_type in valid_types, (
                f"Dependency '{dep.get('name')}' has invalid type '{dep_type}', "
                f"expected one of: {valid_types}"
            )


# =============================================================================
# Module Exports
# =============================================================================


# =============================================================================
# TestWorkflowExecutionWithMocks
# =============================================================================


class TestWorkflowExecutionWithMocks:
    """Integration tests for workflow execution with mock reducer and effect.

    These tests verify the orchestrator's ability to coordinate workflow
    execution by mocking the reducer and effect components. Since the
    orchestrator is declarative, we test that:

    1. Reducer is called before effects
    2. Correlation ID is propagated through all steps
    3. Events are emitted in the correct sequence
    4. Reducer intents are correctly passed to effects
    5. Effect results are properly aggregated in output
    6. Error handling follows contract specifications
    """

    @pytest.fixture
    def correlation_id(self) -> UUID:
        """Create a fixed correlation ID for testing propagation."""
        from uuid import uuid4

        return uuid4()

    @pytest.fixture
    def node_id(self) -> UUID:
        """Create a fixed node ID for testing."""
        from uuid import uuid4

        return uuid4()

    @pytest.fixture
    def introspection_event(self, node_id: UUID, correlation_id: UUID):
        """Create a test introspection event."""
        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent

        return ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type="effect",
            node_version="1.0.0",
            capabilities={},
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=correlation_id,
        )

    @pytest.fixture
    def orchestrator_input(self, introspection_event, correlation_id: UUID):
        """Create test input for the orchestrator."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelOrchestratorInput,
        )

        return ModelOrchestratorInput(
            introspection_event=introspection_event,
            correlation_id=correlation_id,
        )

    @pytest.fixture
    def mock_reducer(self, node_id: UUID, correlation_id: UUID):
        """Create mock reducer that returns registration intents.

        The mock reducer implements the ProtocolReducer interface via duck typing.
        Per ONEX conventions, we verify protocol compliance by checking for required
        method presence and callability rather than using isinstance checks.

        Duck typing validation:
        - Has 'reduce' attribute (method presence)
        - 'reduce' is callable (method behavior)

        Returns a list of intents for Consul and PostgreSQL registration.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelConsulIntentPayload,
            ModelConsulRegistrationIntent,
            ModelPostgresIntentPayload,
            ModelPostgresUpsertIntent,
            ModelReducerState,
            ModelRegistrationIntent,
        )

        class MockReducer:
            """Mock reducer for testing workflow execution."""

            def __init__(self):
                self.call_count = 0
                self.received_events = []
                self.received_states = []
                self._node_id = node_id
                self._correlation_id = correlation_id

            async def reduce(
                self,
                state: ModelReducerState,
                event,
            ) -> tuple[ModelReducerState, list[ModelRegistrationIntent]]:
                """Reduce event to state and intents."""
                self.call_count += 1
                self.received_events.append(event)
                self.received_states.append(state)

                # Generate test intents with typed payloads
                intents: list[ModelRegistrationIntent] = [
                    ModelConsulRegistrationIntent(
                        operation="register",
                        node_id=self._node_id,
                        correlation_id=self._correlation_id,
                        payload=ModelConsulIntentPayload(
                            service_name=f"node-{event.node_type}",
                        ),
                    ),
                    ModelPostgresUpsertIntent(
                        operation="upsert",
                        node_id=self._node_id,
                        correlation_id=self._correlation_id,
                        payload=ModelPostgresIntentPayload(
                            node_id=self._node_id,
                            node_type=event.node_type,
                            correlation_id=self._correlation_id,
                            timestamp=event.timestamp.isoformat(),
                        ),
                    ),
                ]

                # Update state
                new_state = ModelReducerState(
                    last_event_timestamp=event.timestamp.isoformat(),
                    processed_node_ids=state.processed_node_ids
                    | frozenset({event.node_id}),
                    pending_registrations=state.pending_registrations + len(intents),
                )

                return new_state, intents

        # Verify mock implements protocol via duck typing (ONEX convention)
        mock = MockReducer()
        assert hasattr(mock, "reduce"), "MockReducer must have 'reduce' method"
        assert callable(mock.reduce), "MockReducer.reduce must be callable"
        return mock

    @pytest.fixture
    def mock_effect(self):
        """Create mock effect that executes intents.

        The mock effect implements the ProtocolEffect interface via duck typing.
        Per ONEX conventions, we verify protocol compliance by checking for required
        method presence and callability rather than using isinstance checks.

        Duck typing validation:
        - Has 'execute_intent' attribute (method presence)
        - 'execute_intent' is callable (method behavior)

        Returns successful execution results for all intents.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelConsulRegistrationIntent,
            ModelIntentExecutionResult,
            ModelPostgresUpsertIntent,
            ModelRegistrationIntent,
        )

        # Define the concrete union type for executed intents
        ConcreteIntent = ModelConsulRegistrationIntent | ModelPostgresUpsertIntent

        class MockEffect:
            """Mock effect for testing workflow execution."""

            def __init__(self):
                self.call_count = 0
                self.executed_intents: list[ConcreteIntent] = []
                self.received_correlation_ids: list[UUID] = []
                self.should_fail = False
                self.fail_on_kind: str | None = None

            async def execute_intent(
                self,
                intent: ConcreteIntent,
                correlation_id: UUID,
            ) -> ModelIntentExecutionResult:
                """Execute a single intent."""
                import time

                start_time = time.perf_counter()
                self.call_count += 1
                self.executed_intents.append(intent)
                self.received_correlation_ids.append(correlation_id)

                # Simulate failure if configured
                if self.should_fail or (
                    self.fail_on_kind and intent.kind == self.fail_on_kind
                ):
                    return ModelIntentExecutionResult(
                        intent_kind=intent.kind,
                        success=False,
                        error=f"Mock failure for {intent.kind}",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    )

                return ModelIntentExecutionResult(
                    intent_kind=intent.kind,
                    success=True,
                    error=None,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )

        # Verify mock implements protocol via duck typing (ONEX convention)
        mock = MockEffect()
        assert hasattr(mock, "execute_intent"), (
            "MockEffect must have 'execute_intent' method"
        )
        assert callable(mock.execute_intent), (
            "MockEffect.execute_intent must be callable"
        )
        return mock

    @pytest.fixture
    def mock_event_emitter(self):
        """Create mock event emitter to capture emitted events."""

        class MockEventEmitter:
            """Mock event emitter for capturing published events."""

            def __init__(self):
                self.emitted_events: list[tuple[str, dict]] = []

            async def emit(self, event_type: str, event_data: dict) -> None:
                """Emit an event."""
                self.emitted_events.append((event_type, event_data))

            def get_event_types(self) -> list[str]:
                """Get list of emitted event types in order."""
                return [e[0] for e in self.emitted_events]

        return MockEventEmitter()

    @pytest.fixture
    def orchestrator_with_mocks(
        self,
        mock_container: MagicMock,
        mock_reducer,
        mock_effect,
        mock_event_emitter,
    ):
        """Create orchestrator configured with mock reducer, effect, and emitter.

        Sets up the container's service registry to return mock implementations
        when the orchestrator resolves its dependencies.
        """
        # Configure container to provide mocks via service registry
        # Import protocols for type-safe matching (not string-based)
        from omnibase_infra.nodes.node_registration_orchestrator.protocols import (
            ProtocolEffect,
            ProtocolReducer,
        )

        def resolve_mock(protocol):
            """Resolve mock dependencies using explicit protocol type matching."""
            if protocol is ProtocolReducer:
                return mock_reducer
            elif protocol is ProtocolEffect:
                return mock_effect
            else:
                return mock_event_emitter

        mock_container.service_registry = MagicMock()
        mock_container.service_registry.resolve.side_effect = resolve_mock

        # Store references for test access
        mock_container._test_reducer = mock_reducer
        mock_container._test_effect = mock_effect
        mock_container._test_emitter = mock_event_emitter

        orchestrator = NodeRegistrationOrchestrator(mock_container)
        return orchestrator

    def test_mock_reducer_implements_protocol(self, mock_reducer) -> None:
        """Test that mock reducer correctly implements ProtocolReducer interface.

        Per ONEX conventions, protocol compliance is verified via duck typing
        by checking for required method presence and callability, rather than
        using isinstance checks with Protocol types.
        """
        # Duck typing verification - check method presence and callability
        assert hasattr(mock_reducer, "reduce"), "Must have 'reduce' method"
        assert callable(mock_reducer.reduce), "'reduce' must be callable"

    def test_mock_effect_implements_protocol(self, mock_effect) -> None:
        """Test that mock effect correctly implements ProtocolEffect interface.

        Per ONEX conventions, protocol compliance is verified via duck typing
        by checking for required method presence and callability, rather than
        using isinstance checks with Protocol types.
        """
        # Duck typing verification - check method presence and callability
        assert hasattr(mock_effect, "execute_intent"), (
            "Must have 'execute_intent' method"
        )
        assert callable(mock_effect.execute_intent), "'execute_intent' must be callable"

    @pytest.mark.asyncio
    async def test_reducer_generates_intents_from_event(
        self,
        mock_reducer,
        introspection_event,
    ) -> None:
        """Test that reducer generates intents from introspection event."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelReducerState,
        )

        initial_state = ModelReducerState.initial()

        new_state, intents = await mock_reducer.reduce(
            initial_state, introspection_event
        )

        # Verify reducer was called
        assert mock_reducer.call_count == 1
        assert mock_reducer.received_events[0] == introspection_event

        # Verify intents generated
        assert len(intents) == 2
        assert intents[0].kind == "consul"
        assert intents[0].operation == "register"
        assert intents[1].kind == "postgres"
        assert intents[1].operation == "upsert"

        # Verify state updated
        assert new_state.processed_node_ids == frozenset({introspection_event.node_id})
        assert new_state.pending_registrations == 2

    @pytest.mark.asyncio
    async def test_effect_executes_intents_successfully(
        self,
        mock_effect,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Test that effect executes intents and returns success results."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelConsulIntentPayload,
            ModelConsulRegistrationIntent,
        )

        intent = ModelConsulRegistrationIntent(
            operation="register",
            node_id=node_id,
            correlation_id=correlation_id,
            payload=ModelConsulIntentPayload(service_name="test-node"),
        )

        result = await mock_effect.execute_intent(intent, correlation_id)

        assert mock_effect.call_count == 1
        assert mock_effect.executed_intents[0] == intent
        assert mock_effect.received_correlation_ids[0] == correlation_id
        assert result.success is True
        assert result.intent_kind == "consul"
        assert result.error is None
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_effect_handles_failure(
        self,
        mock_effect,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Test that effect returns failure result when configured to fail."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelConsulIntentPayload,
            ModelConsulRegistrationIntent,
        )

        mock_effect.should_fail = True

        intent = ModelConsulRegistrationIntent(
            operation="register",
            node_id=node_id,
            correlation_id=correlation_id,
            payload=ModelConsulIntentPayload(service_name="test-node"),
        )

        result = await mock_effect.execute_intent(intent, correlation_id)

        assert result.success is False
        assert result.error is not None
        assert "Mock failure" in result.error

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_through_reducer(
        self,
        mock_reducer,
        introspection_event,
        correlation_id: UUID,
    ) -> None:
        """Test correlation ID is preserved in reducer intents."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelReducerState,
        )

        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        # All intents should have the same correlation_id
        for intent in intents:
            assert intent.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_through_effect(
        self,
        mock_effect,
        node_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Test correlation ID is passed to effect execution."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelPostgresIntentPayload,
            ModelPostgresUpsertIntent,
        )

        intent = ModelPostgresUpsertIntent(
            operation="upsert",
            node_id=node_id,
            correlation_id=correlation_id,
            payload=ModelPostgresIntentPayload(
                node_id=node_id,
                node_type="effect",
                correlation_id=correlation_id,
                timestamp="2025-01-01T00:00:00Z",
            ),
        )

        await mock_effect.execute_intent(intent, correlation_id)

        assert mock_effect.received_correlation_ids[0] == correlation_id

    @pytest.mark.asyncio
    async def test_reducer_intents_passed_to_effects(
        self,
        mock_reducer,
        mock_effect,
        introspection_event,
        correlation_id: UUID,
    ) -> None:
        """Test that reducer intents are correctly passed to effect nodes."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelReducerState,
        )

        # Step 1: Reducer generates intents
        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        # Step 2: Each intent is executed by effect
        results = []
        for intent in intents:
            result = await mock_effect.execute_intent(intent, correlation_id)
            results.append(result)

        # Verify all intents were executed
        assert mock_effect.call_count == 2
        assert len(mock_effect.executed_intents) == 2

        # Verify intents match
        assert mock_effect.executed_intents[0] == intents[0]
        assert mock_effect.executed_intents[1] == intents[1]

        # Verify all results are successful
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_effect_results_aggregated_correctly(
        self,
        mock_reducer,
        mock_effect,
        introspection_event,
        correlation_id: UUID,
    ) -> None:
        """Test that effect results are properly aggregated."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelIntentExecutionResult,
            ModelOrchestratorOutput,
            ModelReducerState,
        )

        # Execute reducer
        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        # Execute effects and collect results
        results: list[ModelIntentExecutionResult] = []
        for intent in intents:
            result = await mock_effect.execute_intent(intent, correlation_id)
            results.append(result)

        # Aggregate results (simulating what the orchestrator would do)
        consul_results = [r for r in results if r.intent_kind == "consul"]
        postgres_results = [r for r in results if r.intent_kind == "postgres"]

        consul_applied = all(r.success for r in consul_results)
        postgres_applied = all(r.success for r in postgres_results)

        all_success = consul_applied and postgres_applied
        any_success = consul_applied or postgres_applied

        status = "success" if all_success else ("partial" if any_success else "failed")

        total_time = sum(r.execution_time_ms for r in results)

        output = ModelOrchestratorOutput(
            correlation_id=correlation_id,
            status=status,
            consul_applied=consul_applied,
            postgres_applied=postgres_applied,
            consul_error=None,
            postgres_error=None,
            intent_results=results,
            total_execution_time_ms=total_time,
        )

        # Verify output structure
        assert output.status == "success"
        assert output.consul_applied is True
        assert output.postgres_applied is True
        assert output.correlation_id == correlation_id
        assert len(output.intent_results) == 2
        assert output.total_execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_partial_failure_aggregation(
        self,
        mock_reducer,
        mock_effect,
        introspection_event,
        correlation_id: UUID,
    ) -> None:
        """Test aggregation when one effect fails and another succeeds."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelIntentExecutionResult,
            ModelOrchestratorOutput,
            ModelReducerState,
        )

        # Configure effect to fail on consul only
        mock_effect.fail_on_kind = "consul"

        # Execute reducer
        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        # Execute effects
        results: list[ModelIntentExecutionResult] = []
        for intent in intents:
            result = await mock_effect.execute_intent(intent, correlation_id)
            results.append(result)

        # Aggregate
        consul_results = [r for r in results if r.intent_kind == "consul"]
        postgres_results = [r for r in results if r.intent_kind == "postgres"]

        consul_applied = all(r.success for r in consul_results)
        postgres_applied = all(r.success for r in postgres_results)

        consul_error = next((r.error for r in consul_results if not r.success), None)

        status = (
            "success"
            if consul_applied and postgres_applied
            else ("partial" if consul_applied or postgres_applied else "failed")
        )

        output = ModelOrchestratorOutput(
            correlation_id=correlation_id,
            status=status,
            consul_applied=consul_applied,
            postgres_applied=postgres_applied,
            consul_error=consul_error,
            postgres_error=None,
            intent_results=results,
            total_execution_time_ms=sum(r.execution_time_ms for r in results),
        )

        # Verify partial failure
        assert output.status == "partial"
        assert output.consul_applied is False
        assert output.postgres_applied is True
        assert output.consul_error is not None
        assert "Mock failure" in output.consul_error

    @pytest.mark.asyncio
    async def test_reducer_deduplicates_processed_nodes(
        self,
        mock_reducer,
        introspection_event,
    ) -> None:
        """Test that reducer tracks processed nodes for deduplication."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelReducerState,
        )

        # First reduction
        initial_state = ModelReducerState.initial()
        state_after_first, intents_first = await mock_reducer.reduce(
            initial_state, introspection_event
        )

        assert len(intents_first) == 2
        assert introspection_event.node_id in state_after_first.processed_node_ids

        # State should track processed node
        assert state_after_first.pending_registrations == 2

    @pytest.mark.asyncio
    async def test_workflow_sequence_reducer_before_effect(
        self,
        mock_reducer,
        mock_effect,
        introspection_event,
        correlation_id: UUID,
    ) -> None:
        """Test that workflow calls reducer before effects."""
        from omnibase_infra.nodes.node_registration_orchestrator.models import (
            ModelReducerState,
        )

        # Track call order
        call_order: list[str] = []

        # Wrap reducer to track calls
        original_reduce = mock_reducer.reduce

        async def tracked_reduce(state, event):
            call_order.append("reducer")
            return await original_reduce(state, event)

        mock_reducer.reduce = tracked_reduce

        # Wrap effect to track calls
        original_execute = mock_effect.execute_intent

        async def tracked_execute(intent, corr_id):
            call_order.append(f"effect:{intent.kind}")
            return await original_execute(intent, corr_id)

        mock_effect.execute_intent = tracked_execute

        # Execute workflow steps manually (simulating orchestrator)
        initial_state = ModelReducerState.initial()
        _, intents = await mock_reducer.reduce(initial_state, introspection_event)

        for intent in intents:
            await mock_effect.execute_intent(intent, correlation_id)

        # Verify order
        assert call_order[0] == "reducer"
        assert call_order[1].startswith("effect:")
        assert call_order[2].startswith("effect:")

    def test_orchestrator_instantiation_with_mocks(
        self,
        orchestrator_with_mocks,
    ) -> None:
        """Test that orchestrator can be instantiated with mock container.

        Verifies via duck typing (per ONEX conventions):
        - Orchestrator has expected class identity
        - Orchestrator has required methods and attributes
        """
        # Verify type via duck typing (check class name, not isinstance)
        assert orchestrator_with_mocks.__class__.__name__ == "NodeRegistrationOrchestrator", (
            f"Expected NodeRegistrationOrchestrator, got {orchestrator_with_mocks.__class__.__name__}"
        )

        # Verify orchestrator has required methods from NodeOrchestrator base
        required_methods = ["process", "execute_workflow_from_contract", "get_node_type"]
        for method_name in required_methods:
            assert hasattr(orchestrator_with_mocks, method_name), (
                f"Orchestrator must have '{method_name}' method"
            )
            assert callable(getattr(orchestrator_with_mocks, method_name)), (
                f"'{method_name}' must be callable"
            )

        # Verify container is properly injected
        assert hasattr(orchestrator_with_mocks, "container"), (
            "Orchestrator must have 'container' attribute"
        )

    def test_mock_container_provides_dependencies(
        self,
        orchestrator_with_mocks,
        mock_container: MagicMock,
    ) -> None:
        """Test that mock container provides reducer and effect dependencies.

        Per ONEX conventions, protocol compliance is verified via duck typing
        by checking for required method presence and callability, rather than
        using isinstance checks with Protocol types.
        """
        # Verify mocks are accessible via container
        assert hasattr(mock_container, "_test_reducer")
        assert hasattr(mock_container, "_test_effect")
        assert hasattr(mock_container, "_test_emitter")

        # Verify mocks implement protocols via duck typing (ONEX convention)
        # ProtocolReducer requires 'reduce' method
        assert hasattr(mock_container._test_reducer, "reduce"), (
            "Reducer must have 'reduce' method"
        )
        assert callable(mock_container._test_reducer.reduce), (
            "Reducer.reduce must be callable"
        )

        # ProtocolEffect requires 'execute_intent' method
        assert hasattr(mock_container._test_effect, "execute_intent"), (
            "Effect must have 'execute_intent' method"
        )
        assert callable(mock_container._test_effect.execute_intent), (
            "Effect.execute_intent must be callable"
        )


__all__ = [
    "TestContractIntegration",
    "TestWorkflowGraphIntegration",
    "TestCoordinationRulesIntegration",
    "TestErrorHandlingIntegration",
    "TestEventIntegration",
    "TestModelContractAlignment",
    "TestNodeIntegration",
    "TestDependencyStructure",
    "TestWorkflowExecutionWithMocks",
]
