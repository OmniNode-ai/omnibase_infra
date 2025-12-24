# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests verifying orchestrator performs no I/O (all delegated to effect nodes).

This module validates the ONEX architectural principle that orchestrators are
pure coordinators that delegate all I/O operations to effect nodes. This is
a critical acceptance criterion from OMN-952.

Architectural Principles Tested:
    1. No I/O library imports in orchestrator module
    2. No direct network/database calls in orchestrator methods
    3. All I/O delegated through ProtocolEffect protocol
    4. Orchestrator is a pure workflow coordinator
    5. Contract defines I/O operations only in effect-type nodes

Related:
    - OMN-952: Comprehensive orchestrator tests
    - CLAUDE.md: ONEX 4-Node Architecture (EFFECT for I/O, ORCHESTRATOR for coordination)
    - protocols.py: ProtocolEffect defines the I/O delegation interface
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import yaml

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants
# =============================================================================

# I/O libraries that should NOT be imported by an orchestrator
IO_LIBRARIES = frozenset(
    {
        # HTTP clients
        "httpx",
        "requests",
        "aiohttp",
        "urllib",
        "urllib3",
        # Database clients
        "psycopg",
        "psycopg2",
        "asyncpg",
        "sqlalchemy",
        "databases",
        # Message queue clients
        "kafka",
        "kafka-python",
        "aiokafka",
        "confluent_kafka",
        # Service discovery / infrastructure
        "consul",
        "python-consul",
        # Secret management
        "hvac",
        # Cache clients
        "redis",
        "aioredis",
        "valkey",
        # gRPC
        "grpc",
        "grpcio",
        # File I/O modules (network-related)
        "paramiko",
        "ftplib",
        "smtplib",
    }
)

# Method name patterns that indicate direct I/O operations
IO_METHOD_PATTERNS = frozenset(
    {
        # HTTP patterns
        "get",
        "post",
        "put",
        "patch",
        "delete",
        "request",
        "fetch",
        # Database patterns
        "execute",
        "query",
        "insert",
        "update",
        "select",
        "connect",
        # Message queue patterns
        "send",
        "publish",
        "produce",
        "consume",
        # File/Network patterns
        "open",
        "read",
        "write",
        "download",
        "upload",
    }
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def orchestrator_module_path() -> Path:
    """Return path to the orchestrator node.py file."""
    path = Path("src/omnibase_infra/nodes/node_registration_orchestrator/node.py")
    if not path.exists():
        pytest.skip(f"Orchestrator file not found: {path}")
    return path


@pytest.fixture
def orchestrator_source(orchestrator_module_path: Path) -> str:
    """Load and return the orchestrator source code."""
    return orchestrator_module_path.read_text(encoding="utf-8")


@pytest.fixture
def orchestrator_ast(orchestrator_source: str) -> ast.Module:
    """Parse and return the orchestrator AST."""
    return ast.parse(orchestrator_source)


@pytest.fixture
def contract_path() -> Path:
    """Return path to contract.yaml."""
    path = Path("src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml")
    if not path.exists():
        pytest.skip(f"Contract file not found: {path}")
    return path


@pytest.fixture
def contract_data(contract_path: Path) -> dict:
    """Load and return contract.yaml as dict."""
    with open(contract_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock ONEX container."""
    container = MagicMock()
    container.config = MagicMock()
    return container


# =============================================================================
# TestOrchestratorNoIOImports
# =============================================================================


class TestOrchestratorNoIOImports:
    """Tests verifying orchestrator has no I/O library imports."""

    def test_orchestrator_has_no_io_imports(self, orchestrator_ast: ast.Module) -> None:
        """Verify node.py does not import I/O libraries.

        The orchestrator should only import:
        - omnibase_core components (base classes, models)
        - typing utilities
        - Local models/protocols

        It should NOT import any I/O libraries like:
        - httpx, requests, aiohttp (HTTP clients)
        - psycopg2, asyncpg (database clients)
        - kafka, aiokafka (message queues)
        - consul, hvac (infrastructure)
        """
        imported_modules: set[str] = set()

        for node in ast.walk(orchestrator_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Get the top-level module name
                    top_level = alias.name.split(".")[0]
                    imported_modules.add(top_level)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Get the top-level module name
                    top_level = node.module.split(".")[0]
                    imported_modules.add(top_level)

        # Check for I/O library imports
        io_imports_found = imported_modules & IO_LIBRARIES

        assert not io_imports_found, (
            f"Orchestrator should not import I/O libraries.\n"
            f"Found I/O imports: {sorted(io_imports_found)}\n"
            f"All I/O operations must be delegated to effect nodes via ProtocolEffect."
        )

    def test_no_socket_or_network_imports(self, orchestrator_ast: ast.Module) -> None:
        """Verify no low-level socket or network imports."""
        network_modules = {"socket", "ssl", "http", "http.client"}
        imported_modules: set[str] = set()

        for node in ast.walk(orchestrator_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)

        network_imports_found = imported_modules & network_modules

        assert not network_imports_found, (
            f"Orchestrator should not import network modules.\n"
            f"Found: {sorted(network_imports_found)}\n"
            f"All network I/O must be delegated to effect nodes."
        )


# =============================================================================
# TestOrchestratorNoDirectNetworkCalls
# =============================================================================


class TestOrchestratorNoDirectNetworkCalls:
    """Tests verifying orchestrator has no direct network/I/O calls."""

    def test_orchestrator_has_no_direct_network_calls(
        self, orchestrator_ast: ast.Module
    ) -> None:
        """Verify orchestrator class has no methods that make direct network calls.

        This test inspects all method bodies in the orchestrator class to ensure
        there are no direct calls to I/O methods like:
        - client.get(), client.post() (HTTP)
        - conn.execute(), cursor.query() (database)
        - producer.send(), consumer.poll() (Kafka)
        """
        io_calls_found: list[str] = []

        for node in ast.walk(orchestrator_ast):
            if isinstance(node, ast.ClassDef):
                if node.name == "NodeRegistrationOrchestrator":
                    # Walk through all function definitions in the class
                    for item in ast.walk(node):
                        if isinstance(item, ast.Call):
                            # Check for attribute calls like client.get()
                            if isinstance(item.func, ast.Attribute):
                                method_name = item.func.attr.lower()
                                if method_name in IO_METHOD_PATTERNS:
                                    io_calls_found.append(
                                        f"{item.func.attr}() at line {item.lineno}"
                                    )

        assert not io_calls_found, (
            f"Orchestrator should not make direct I/O calls.\n"
            f"Found potential I/O calls: {io_calls_found}\n"
            f"All I/O must be delegated to effect nodes."
        )

    def test_orchestrator_methods_are_minimal(
        self, mock_container: MagicMock
    ) -> None:
        """Verify orchestrator has minimal methods (pure delegation pattern).

        A pure coordinator orchestrator should have very few methods defined
        directly on the class - most behavior should be inherited from the
        base NodeOrchestrator class.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        # Get methods defined directly on NodeRegistrationOrchestrator
        own_methods = [
            name
            for name in dir(NodeRegistrationOrchestrator)
            if not name.startswith("_")
            and callable(getattr(NodeRegistrationOrchestrator, name, None))
            and name in NodeRegistrationOrchestrator.__dict__
        ]

        # Orchestrator should have NO public methods defined directly
        # All behavior comes from base class
        assert own_methods == [], (
            f"Orchestrator should have no custom public methods.\n"
            f"Found: {own_methods}\n"
            f"A pure coordinator delegates all work to the base class "
            f"which handles workflow execution via contract.yaml."
        )


# =============================================================================
# TestOrchestratorDelegatesToEffectProtocol
# =============================================================================


class TestOrchestratorDelegatesToEffectProtocol:
    """Tests verifying I/O is delegated through ProtocolEffect."""

    def test_orchestrator_delegates_to_effect_protocol(self) -> None:
        """Verify ProtocolEffect is used for all I/O operations.

        The orchestrator's workflow should delegate all I/O to effect nodes
        through the ProtocolEffect interface. This test verifies:
        1. ProtocolEffect is defined in protocols.py
        2. ProtocolEffect has execute_intent() method for I/O delegation
        3. The protocol pattern enforces I/O separation
        """
        from omnibase_infra.nodes.node_registration_orchestrator.protocols import (
            ProtocolEffect,
        )

        # Verify ProtocolEffect exists and has the delegation method
        assert hasattr(ProtocolEffect, "execute_intent"), (
            "ProtocolEffect must have execute_intent() method for I/O delegation"
        )

        # Verify it's a Protocol (runtime_checkable)
        from typing import runtime_checkable, Protocol

        # Check it's marked as runtime_checkable
        assert hasattr(ProtocolEffect, "__protocol_attrs__") or hasattr(
            ProtocolEffect, "__subclasshook__"
        ), "ProtocolEffect should be a proper Protocol"

    def test_protocol_effect_signature_supports_io_delegation(self) -> None:
        """Verify ProtocolEffect.execute_intent() has proper signature for I/O delegation."""
        from omnibase_infra.nodes.node_registration_orchestrator.protocols import (
            ProtocolEffect,
        )

        # Get the execute_intent method signature
        sig = inspect.signature(ProtocolEffect.execute_intent)
        params = list(sig.parameters.keys())

        # Should accept intent and correlation_id for tracing
        assert "intent" in params, (
            "execute_intent must accept 'intent' parameter for I/O operation details"
        )
        assert "correlation_id" in params, (
            "execute_intent must accept 'correlation_id' for distributed tracing"
        )

    def test_reducer_protocol_performs_no_io(self) -> None:
        """Verify ProtocolReducer explicitly forbids I/O.

        Per ONEX architecture, reducers are pure functions that:
        - Take state + event as input
        - Return new state + intents
        - MUST NOT perform I/O operations
        """
        from omnibase_infra.nodes.node_registration_orchestrator.protocols import (
            ProtocolReducer,
        )

        # Check docstring mentions no I/O
        docstring = ProtocolReducer.__doc__ or ""
        assert "MUST NOT perform I/O" in docstring or "Reducer MUST NOT perform I/O" in docstring, (
            "ProtocolReducer docstring must explicitly state reducers perform no I/O"
        )


# =============================================================================
# TestOrchestratorIsPureCoordinator
# =============================================================================


class TestOrchestratorIsPureCoordinator:
    """Tests verifying orchestrator is a pure workflow coordinator."""

    def test_orchestrator_is_pure_coordinator(
        self, orchestrator_source: str, mock_container: MagicMock
    ) -> None:
        """Verify orchestrator only coordinates workflow, doesn't execute I/O.

        A pure coordinator:
        1. Inherits from NodeOrchestrator (which handles workflow execution)
        2. Has minimal code (just __init__)
        3. Relies on contract.yaml for all workflow logic
        4. Does not implement any I/O methods directly
        """
        from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        orchestrator = NodeRegistrationOrchestrator(mock_container)

        # Verify inheritance
        assert isinstance(orchestrator, NodeOrchestrator), (
            "Orchestrator must inherit from NodeOrchestrator for contract-driven workflow"
        )

        # Verify minimal implementation (source code check)
        # Count non-comment, non-docstring, non-blank lines in class body
        class_lines = []
        in_class = False
        in_docstring = False
        docstring_delimiter = None

        for line in orchestrator_source.split("\n"):
            stripped = line.strip()

            if "class NodeRegistrationOrchestrator" in line:
                in_class = True
                continue

            if in_class:
                # End of class (next top-level definition)
                if stripped and not stripped.startswith(" ") and not stripped.startswith("#"):
                    if not stripped.startswith('"""') and not stripped.startswith("'''"):
                        break

                # Skip empty lines and comments
                if not stripped or stripped.startswith("#"):
                    continue

                # Handle docstrings
                if '"""' in stripped or "'''" in stripped:
                    if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                        continue  # Single-line docstring
                    if not in_docstring:
                        in_docstring = True
                        docstring_delimiter = '"""' if '"""' in stripped else "'''"
                        continue
                    elif docstring_delimiter in stripped:
                        in_docstring = False
                        continue

                if in_docstring:
                    continue

                class_lines.append(stripped)

        # Pure coordinator should have very few lines (mainly __init__ and super call)
        # Allow for: def __init__, super().__init__, any type hints
        effective_lines = [
            line
            for line in class_lines
            if line and not line.startswith("def __init__")
        ]

        # The only real line should be the super().__init__ call
        assert len(effective_lines) <= 2, (
            f"Pure coordinator should have minimal code.\n"
            f"Found {len(effective_lines)} code lines: {effective_lines}\n"
            f"Orchestrator should only call super().__init__(container) and rely on "
            f"base class + contract.yaml for all behavior."
        )

    def test_orchestrator_docstring_documents_delegation_pattern(
        self, mock_container: MagicMock
    ) -> None:
        """Verify orchestrator documents that it delegates I/O to effect nodes."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        docstring = NodeRegistrationOrchestrator.__doc__ or ""

        # Should mention the delegation pattern
        delegation_keywords = ["contract", "workflow", "delegate", "effect", "reducer"]
        found_keywords = [kw for kw in delegation_keywords if kw.lower() in docstring.lower()]

        assert len(found_keywords) >= 2, (
            f"Orchestrator docstring should document delegation pattern.\n"
            f"Found keywords: {found_keywords}\n"
            f"Expected at least 2 of: {delegation_keywords}"
        )


# =============================================================================
# TestContractIOOperationsAreEffectNodes
# =============================================================================


class TestContractIOOperationsAreEffectNodes:
    """Tests verifying contract structure enforces I/O in effect nodes only."""

    def test_contract_io_operations_are_effect_nodes(
        self, contract_data: dict
    ) -> None:
        """Verify all nodes with I/O operations in contract are type 'effect'.

        Per ONEX 4-node architecture:
        - EFFECT nodes: Perform external I/O (Consul, PostgreSQL, Kafka, etc.)
        - COMPUTE nodes: Pure transformations, no I/O
        - REDUCER nodes: State aggregation, no I/O
        - ORCHESTRATOR nodes: Workflow coordination, delegates I/O to effects

        This test ensures the contract correctly marks I/O operations as effect nodes.
        """
        execution_graph = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        # Operations that involve I/O should be effect nodes
        io_operation_keywords = {
            "receive",
            "read",
            "execute",
            "publish",
            "send",
            "fetch",
            "write",
            "register",
            "deregister",
        }

        misclassified_nodes: list[str] = []

        for node in execution_graph:
            node_id = node["node_id"]
            node_type = node["node_type"]
            description = node.get("description", "").lower()

            # Check if node_id or description suggests I/O
            node_id_lower = node_id.lower()
            suggests_io = any(
                keyword in node_id_lower or keyword in description
                for keyword in io_operation_keywords
            )

            # Special cases that are NOT I/O despite keyword matches
            non_io_exceptions = {
                "compute_intents",  # reducer computing intents (pure)
                "aggregate_results",  # compute aggregating results (pure)
                "evaluate_timeout",  # compute evaluating timeout (pure)
            }

            if node_id in non_io_exceptions:
                # These should NOT be effect nodes despite keyword matches
                if node_type == "effect":
                    misclassified_nodes.append(
                        f"{node_id}: marked as 'effect' but should be '{node['node_type']}'"
                    )
                continue

            # I/O operations should be effect nodes
            if suggests_io and node_type != "effect":
                misclassified_nodes.append(
                    f"{node_id}: performs I/O but marked as '{node_type}' instead of 'effect'"
                )

        assert not misclassified_nodes, (
            f"Contract has misclassified nodes:\n"
            + "\n".join(f"  - {msg}" for msg in misclassified_nodes)
            + "\n\nAll I/O operations must use node_type: effect"
        )

    def test_effect_nodes_handle_external_systems(self, contract_data: dict) -> None:
        """Verify effect nodes are the ones interacting with external systems.

        Effect nodes should be the only nodes that:
        - Interact with Consul
        - Interact with PostgreSQL
        - Publish/consume events
        - Read projections
        """
        execution_graph = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        effect_nodes = [n for n in execution_graph if n["node_type"] == "effect"]
        effect_node_ids = {n["node_id"] for n in effect_nodes}

        # Expected effect nodes for external system interaction
        expected_effect_operations = {
            "receive_introspection",  # Event consumption
            "read_projection",  # Projection read
            "execute_consul_registration",  # Consul I/O
            "execute_postgres_registration",  # PostgreSQL I/O
            "publish_outcome",  # Event publishing
        }

        # All expected effect operations should be in effect nodes
        missing = expected_effect_operations - effect_node_ids
        assert not missing, (
            f"Expected these I/O operations to be effect nodes: {missing}\n"
            f"Actual effect nodes: {effect_node_ids}"
        )

    def test_non_effect_nodes_are_pure(self, contract_data: dict) -> None:
        """Verify non-effect nodes (compute, reducer) are pure operations."""
        execution_graph = contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]

        non_effect_nodes = [n for n in execution_graph if n["node_type"] != "effect"]

        # Pure node types
        pure_types = {"compute", "reducer"}

        for node in non_effect_nodes:
            node_type = node["node_type"]
            assert node_type in pure_types, (
                f"Node '{node['node_id']}' has type '{node_type}' which is not a pure type.\n"
                f"Non-effect nodes must be 'compute' or 'reducer'."
            )

    def test_contract_dependencies_reference_effect_node(
        self, contract_data: dict
    ) -> None:
        """Verify contract dependencies include effect_node for I/O delegation."""
        dependencies = contract_data.get("dependencies", [])

        # Find the effect_node dependency
        effect_dep = next(
            (d for d in dependencies if d.get("name") == "effect_node"), None
        )

        assert effect_dep is not None, (
            "Contract must declare 'effect_node' dependency for I/O delegation.\n"
            "The orchestrator delegates all I/O to this effect node."
        )

        assert effect_dep.get("type") == "node", (
            f"effect_node dependency should be type 'node', got: {effect_dep.get('type')}"
        )


# =============================================================================
# TestOrchestratorModuleStructure
# =============================================================================


class TestOrchestratorModuleStructure:
    """Additional structural tests for orchestrator module."""

    def test_module_docstring_documents_no_io_pattern(
        self, orchestrator_source: str
    ) -> None:
        """Verify module docstring documents the no-I/O pattern."""
        # Extract module docstring (first string literal)
        tree = ast.parse(orchestrator_source)
        module_docstring = ast.get_docstring(tree) or ""

        # Should mention key architectural principles
        patterns_to_find = [
            "contract",  # Contract-driven
            "workflow",  # Workflow coordination
        ]

        found = [p for p in patterns_to_find if p.lower() in module_docstring.lower()]

        assert len(found) >= 1, (
            f"Module docstring should document the delegation pattern.\n"
            f"Found: {found}, expected at least 1 of: {patterns_to_find}"
        )

    def test_all_exports_are_minimal(self, orchestrator_source: str) -> None:
        """Verify __all__ exports are minimal (just the orchestrator class)."""
        tree = ast.parse(orchestrator_source)

        all_value = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            all_value = [
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]

        assert all_value is not None, "Module should define __all__ for explicit exports"
        assert all_value == ["NodeRegistrationOrchestrator"], (
            f"Module should only export NodeRegistrationOrchestrator.\n"
            f"Found exports: {all_value}"
        )
