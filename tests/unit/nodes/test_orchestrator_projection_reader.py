# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for projection reader interactions (OMN-952).

These tests verify that the NodeRegistrationOrchestrator uses projection
reader for state decisions rather than direct topic scanning. Per OMN-930,
state should be read through ProtocolProjectionReader, not by consuming
Kafka topics directly.

The projection reader pattern ensures:
1. State decisions are based on materialized projections (consistent views)
2. No direct topic consumption for state reading (avoids race conditions)
3. Proper dependency injection of projection reader protocol
4. Clear dependency chain: receive_introspection -> read_projection -> evaluate_timeout

Related tickets:
- OMN-930: Implemented ProtocolProjectionReader in omnibase_spi
- OMN-952: Comprehensive orchestrator tests including projection reader validation
- OMN-973: Time injection for timeout evaluation (depends on projection state)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def contract_path() -> Path:
    """Return path to contract.yaml.

    Raises:
        pytest.skip: If contract file doesn't exist (allows tests to be skipped gracefully).
    """
    path = Path("src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml")
    if not path.exists():
        pytest.skip(f"Contract file not found: {path}")
    return path


@pytest.fixture
def contract_data(contract_path: Path) -> dict:
    """Load and return contract.yaml as dict.

    Raises:
        pytest.skip: If contract file doesn't exist (allows tests to be skipped gracefully).
        pytest.fail: If contract file contains invalid YAML.
    """
    if not contract_path.exists():
        pytest.skip(f"Contract file not found: {contract_path}")

    with open(contract_path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in contract file: {e}")


@pytest.fixture
def execution_graph_nodes(contract_data: dict) -> list[dict]:
    """Extract execution graph nodes from contract data.

    Returns:
        List of node dictionaries from the execution graph.

    Raises:
        pytest.fail: If execution_graph.nodes is not found in contract.
    """
    try:
        return contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]
    except KeyError as e:
        pytest.fail(f"Missing key in contract structure: {e}")


@pytest.fixture
def dependencies(contract_data: dict) -> list[dict]:
    """Extract dependencies from contract data.

    Returns:
        List of dependency dictionaries from the contract.

    Raises:
        pytest.fail: If dependencies section is not found in contract.
    """
    if "dependencies" not in contract_data:
        pytest.fail("Missing 'dependencies' section in contract")
    return contract_data["dependencies"]


# =============================================================================
# TestProjectionReaderDependency
# =============================================================================


class TestProjectionReaderDependency:
    """Tests verifying projection_reader is properly declared as a dependency."""

    def test_contract_declares_projection_reader_dependency(
        self, dependencies: list[dict]
    ) -> None:
        """Test that projection_reader is declared in dependencies section.

        The orchestrator must declare projection_reader as a dependency to
        enable dependency injection of the projection reading capability.
        This ensures the orchestrator can access registration state through
        projections rather than direct topic consumption.

        Acceptance criteria (OMN-952):
        - dependencies section contains an entry with name="projection_reader"
        """
        dependency_names = {dep["name"] for dep in dependencies}

        assert "projection_reader" in dependency_names, (
            "projection_reader must be declared as a dependency. "
            f"Found dependencies: {dependency_names}"
        )

    def test_projection_reader_protocol_specified(
        self, dependencies: list[dict]
    ) -> None:
        """Test that projection_reader dependency specifies ProtocolProjectionReader.

        The projection_reader dependency must reference ProtocolProjectionReader
        to ensure type-safe protocol-based dependency injection. This protocol
        is defined in omnibase_spi and provides the contract for reading
        materialized projections.

        Acceptance criteria (OMN-952):
        - projection_reader dependency has type="protocol"
        """
        projection_reader_dep = next(
            (dep for dep in dependencies if dep["name"] == "projection_reader"),
            None,
        )

        assert projection_reader_dep is not None, (
            "projection_reader dependency not found in dependencies list"
        )
        assert projection_reader_dep.get("type") == "protocol", (
            f"projection_reader must have type='protocol', "
            f"found: {projection_reader_dep.get('type')}"
        )

    def test_projection_reader_module_is_spi(self, dependencies: list[dict]) -> None:
        """Test that projection_reader module is omnibase_spi.protocols.

        The ProtocolProjectionReader protocol must be sourced from omnibase_spi
        to ensure proper separation of concerns between SPI (protocol definitions)
        and infrastructure (implementations).

        Acceptance criteria (OMN-952):
        - projection_reader dependency has module="omnibase_spi.protocols"
        """
        projection_reader_dep = next(
            (dep for dep in dependencies if dep["name"] == "projection_reader"),
            None,
        )

        assert projection_reader_dep is not None, (
            "projection_reader dependency not found in dependencies list"
        )
        assert projection_reader_dep.get("module") == "omnibase_spi.protocols", (
            f"projection_reader must have module='omnibase_spi.protocols', "
            f"found: {projection_reader_dep.get('module')}"
        )


# =============================================================================
# TestReadProjectionStep
# =============================================================================


class TestReadProjectionStep:
    """Tests verifying read_projection step in execution graph."""

    def test_read_projection_step_in_execution_graph(
        self, execution_graph_nodes: list[dict]
    ) -> None:
        """Test that read_projection step exists in execution graph.

        The execution graph must include a read_projection step that reads
        the current registration state from the projection. This step provides
        the state context needed for timeout evaluation and decision making.

        Acceptance criteria (OMN-952):
        - execution_graph.nodes contains a node with node_id="read_projection"
        """
        node_ids = {node["node_id"] for node in execution_graph_nodes}

        assert "read_projection" in node_ids, (
            "read_projection step must exist in execution graph. "
            f"Found nodes: {node_ids}"
        )

    def test_read_projection_depends_on_receive_introspection(
        self, execution_graph_nodes: list[dict]
    ) -> None:
        """Test that read_projection depends on receive_introspection.

        The dependency chain must be:
        receive_introspection -> read_projection -> evaluate_timeout

        This ensures projection reading happens after receiving the trigger
        event but before making timeout/state decisions.

        Acceptance criteria (OMN-952):
        - read_projection node has depends_on including "receive_introspection"
        """
        read_projection_node = next(
            (
                node
                for node in execution_graph_nodes
                if node["node_id"] == "read_projection"
            ),
            None,
        )

        assert read_projection_node is not None, (
            "read_projection node not found in execution graph"
        )

        depends_on = read_projection_node.get("depends_on", [])
        assert "receive_introspection" in depends_on, (
            f"read_projection must depend on receive_introspection. "
            f"Found depends_on: {depends_on}"
        )

    def test_read_projection_step_config_has_protocol(
        self, execution_graph_nodes: list[dict]
    ) -> None:
        """Test that read_projection step_config references ProtocolProjectionReader.

        The step configuration must specify the protocol to use for reading
        projections. This enables the workflow executor to resolve the correct
        implementation at runtime.

        Acceptance criteria (OMN-952):
        - read_projection step_config contains protocol="ProtocolProjectionReader"
        """
        read_projection_node = next(
            (
                node
                for node in execution_graph_nodes
                if node["node_id"] == "read_projection"
            ),
            None,
        )

        assert read_projection_node is not None, (
            "read_projection node not found in execution graph"
        )

        step_config = read_projection_node.get("step_config", {})
        assert step_config.get("protocol") == "ProtocolProjectionReader", (
            f"read_projection step_config must have protocol='ProtocolProjectionReader'. "
            f"Found step_config: {step_config}"
        )

    def test_read_projection_has_projection_name(
        self, execution_graph_nodes: list[dict]
    ) -> None:
        """Test that read_projection step_config specifies projection name.

        The step configuration must specify which projection to read.
        For registration state, this should be "node_registration_state".

        Acceptance criteria (OMN-952):
        - read_projection step_config contains projection_name
        """
        read_projection_node = next(
            (
                node
                for node in execution_graph_nodes
                if node["node_id"] == "read_projection"
            ),
            None,
        )

        assert read_projection_node is not None, (
            "read_projection node not found in execution graph"
        )

        step_config = read_projection_node.get("step_config", {})
        assert "projection_name" in step_config, (
            f"read_projection step_config must specify projection_name. "
            f"Found step_config: {step_config}"
        )
        assert step_config["projection_name"] == "node_registration_state", (
            f"projection_name should be 'node_registration_state'. "
            f"Found: {step_config['projection_name']}"
        )


# =============================================================================
# TestProjectionConfiguration
# =============================================================================


class TestProjectionConfiguration:
    """Tests verifying projection_reader configuration in contract."""

    def test_projection_name_defined(self, contract_data: dict) -> None:
        """Test that projection_reader.projections contains node_registration_state.

        The contract must define which projections the orchestrator can read.
        This explicit declaration enables validation and documentation of
        the projection dependencies.

        Acceptance criteria (OMN-952):
        - projection_reader.projections includes an entry with
          name="node_registration_state"
        """
        assert "projection_reader" in contract_data, (
            "projection_reader section must exist in contract"
        )

        projection_reader = contract_data["projection_reader"]
        assert "projections" in projection_reader, (
            "projection_reader must have 'projections' list"
        )

        projections = projection_reader["projections"]
        projection_names = {p["name"] for p in projections}

        assert "node_registration_state" in projection_names, (
            f"projection_reader.projections must include 'node_registration_state'. "
            f"Found projections: {projection_names}"
        )


# =============================================================================
# TestNoTopicScanning
# =============================================================================


class TestNoTopicScanning:
    """Tests verifying no direct topic scanning for state reading."""

    def test_no_topic_scanning_for_state(self, contract_data: dict) -> None:
        """Test that contract does not use Kafka consumer patterns for state reading.

        The orchestrator should read state through projection reader, not by
        consuming Kafka topics directly. Direct topic consumption for state
        reading can cause:
        - Race conditions (reading uncommitted state)
        - Inconsistent views (partial replay)
        - Performance issues (re-reading entire topic)

        This test verifies the contract does not contain patterns that would
        indicate direct state consumption from topics.

        Acceptance criteria (OMN-952):
        - No consumed_events entry with purpose of "state reading"
        - No execution graph node that consumes topics for state lookup
        - State access is exclusively through projection_reader
        """
        # Check consumed_events for any state-reading patterns
        consumed_events = contract_data.get("consumed_events", [])

        state_reading_keywords = [
            "state",
            "projection",
            "current_state",
            "registration_state",
        ]

        for event in consumed_events:
            event_type = event.get("event_type", "").lower()
            description = event.get("description", "").lower()

            # Ensure no consumed event is for reading state
            for keyword in state_reading_keywords:
                if keyword in description and "read" in description:
                    pytest.fail(
                        f"Found potential state-reading event consumption: "
                        f"event_type={event_type}, description={description}. "
                        f"State should be read through projection_reader, not topic consumption."
                    )

        # Verify no execution graph node consumes topics for state
        execution_graph = (
            contract_data.get("workflow_coordination", {})
            .get("workflow_definition", {})
            .get("execution_graph", {})
        )
        nodes = execution_graph.get("nodes", [])

        for node in nodes:
            node_id = node.get("node_id", "")
            step_config = node.get("step_config", {})

            # Skip the legitimate read_projection node
            if node_id == "read_projection":
                continue

            # Check for patterns that suggest topic-based state reading
            if (
                "consume" in str(step_config).lower()
                and "state" in str(step_config).lower()
            ):
                pytest.fail(
                    f"Node '{node_id}' appears to consume topics for state. "
                    f"State should be read through projection_reader. "
                    f"step_config: {step_config}"
                )


# =============================================================================
# TestProjectionBackedDecisionFlow
# =============================================================================


class TestProjectionBackedDecisionFlow:
    """Tests verifying decision flow is backed by projection reading."""

    def test_projection_backed_decision_flow(
        self, execution_graph_nodes: list[dict]
    ) -> None:
        """Test that evaluate_timeout depends on read_projection.

        The decision flow must be:
        receive_introspection -> read_projection -> evaluate_timeout

        This ensures timeout evaluation uses the current projection state
        rather than stale or inconsistent data. The evaluate_timeout node
        should never run before the projection is read.

        Acceptance criteria (OMN-952):
        - evaluate_timeout node has depends_on including "read_projection"
        """
        evaluate_timeout_node = next(
            (
                node
                for node in execution_graph_nodes
                if node["node_id"] == "evaluate_timeout"
            ),
            None,
        )

        assert evaluate_timeout_node is not None, (
            "evaluate_timeout node not found in execution graph"
        )

        depends_on = evaluate_timeout_node.get("depends_on", [])
        assert "read_projection" in depends_on, (
            f"evaluate_timeout must depend on read_projection to ensure "
            f"decision is based on current projection state. "
            f"Found depends_on: {depends_on}"
        )

    def test_full_dependency_chain_exists(
        self, execution_graph_nodes: list[dict]
    ) -> None:
        """Test that the full dependency chain is correctly defined.

        The dependency chain should be:
        receive_introspection (no deps)
            -> read_projection (depends on receive_introspection)
                -> evaluate_timeout (depends on read_projection)
                    -> compute_intents (depends on evaluate_timeout)

        This ensures a clear, linear flow for state reading and decision making.

        Acceptance criteria (OMN-952):
        - Complete dependency chain from receive to decision to action
        """
        # Build a map of node_id -> depends_on
        node_deps = {
            node["node_id"]: node.get("depends_on", [])
            for node in execution_graph_nodes
        }

        # Verify receive_introspection has no dependencies (it's the entry point)
        assert node_deps.get("receive_introspection", []) == [], (
            "receive_introspection should have no dependencies"
        )

        # Verify read_projection depends on receive_introspection
        assert "receive_introspection" in node_deps.get("read_projection", []), (
            "read_projection must depend on receive_introspection"
        )

        # Verify evaluate_timeout depends on read_projection
        assert "read_projection" in node_deps.get("evaluate_timeout", []), (
            "evaluate_timeout must depend on read_projection"
        )

        # Verify compute_intents depends on evaluate_timeout
        assert "evaluate_timeout" in node_deps.get("compute_intents", []), (
            "compute_intents must depend on evaluate_timeout"
        )
