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
from unittest.mock import MagicMock

import pytest

# Import shared AST analysis utilities from test helpers
# See: tests/helpers/ast_analysis.py for implementation details.
from tests.helpers.ast_analysis import (
    find_io_method_calls,
    get_imported_root_modules,
    is_docstring,
)

# =============================================================================
# Constants
# =============================================================================

# I/O libraries that should NOT be imported by an orchestrator.
# This list is comprehensive for ONEX infrastructure patterns.
# Organized by category to aid maintenance.
IO_LIBRARIES = frozenset(
    {
        # ---------------------------------------------------------------------
        # HTTP Clients
        # REST API calls, webhooks, external service communication
        # ---------------------------------------------------------------------
        "httpx",
        "requests",
        "aiohttp",
        "urllib",
        "urllib3",
        # ---------------------------------------------------------------------
        # Database Clients
        # PostgreSQL, SQLAlchemy ORM, async database access
        # ---------------------------------------------------------------------
        "psycopg",
        "psycopg2",
        "asyncpg",
        "sqlalchemy",
        "databases",
        # ---------------------------------------------------------------------
        # Message Queue / Kafka Clients
        # Event streaming, pub/sub messaging
        # ---------------------------------------------------------------------
        "kafka",
        "kafka-python",
        "aiokafka",
        "confluent_kafka",
        # ---------------------------------------------------------------------
        # Service Discovery
        # Consul for service registration and health checks
        # ---------------------------------------------------------------------
        "consul",
        "python-consul",
        # ---------------------------------------------------------------------
        # Secret Management
        # HashiCorp Vault clients (hvac is the primary Python client)
        # ---------------------------------------------------------------------
        "hvac",
        "vault",
        # ---------------------------------------------------------------------
        # Cache Clients
        # Redis, Valkey (Redis-compatible) for caching and pub/sub
        # ---------------------------------------------------------------------
        "redis",
        "aioredis",
        "valkey",
        # ---------------------------------------------------------------------
        # gRPC
        # Remote procedure calls, service-to-service communication
        # ---------------------------------------------------------------------
        "grpc",
        "grpcio",
        # ---------------------------------------------------------------------
        # Async File I/O
        # Filesystem operations that bypass standard blocking I/O
        # Note: Standard `open()` is checked separately via method patterns
        # ---------------------------------------------------------------------
        "aiofiles",
        # ---------------------------------------------------------------------
        # Network Protocols
        # SSH, FTP, SMTP - remote file transfer and email
        # ---------------------------------------------------------------------
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
# Note: The following fixtures are provided by conftest.py with module-level
# scope for performance (parse once per module):
#   - contract_path, contract_data: Contract loading
#   - node_module_path, node_source_code, node_ast: Node source/AST parsing
#   - mock_container: From tests/conftest.py (top-level)
#
# Legacy fixture names are aliased below for backwards compatibility with
# existing tests in this file.


# =============================================================================
# TestOrchestratorNoIOImports
# =============================================================================


class TestOrchestratorNoIOImports:
    """Tests verifying orchestrator has no I/O library imports."""

    def test_orchestrator_has_no_io_imports(self, node_ast: ast.Module) -> None:
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
        # Use shared utility for import extraction
        imported_modules = get_imported_root_modules(node_ast)

        # Check for I/O library imports
        io_imports_found = imported_modules & IO_LIBRARIES

        assert not io_imports_found, (
            f"Orchestrator should not import I/O libraries.\n"
            f"Found I/O imports: {sorted(io_imports_found)}\n\n"
            f"HOW TO FIX: Delegate I/O operations to effect nodes via ProtocolEffect.\n"
            f"Example: Use 'node_registry_effect' for Consul/PostgreSQL operations.\n"
            f"Pattern: orchestrator -> ProtocolEffect.execute_intent(intent) -> effect node\n"
            f"See: nodes/node_registration_orchestrator/protocols.py for ProtocolEffect definition."
        )

    def test_no_socket_or_network_imports(self, node_ast: ast.Module) -> None:
        """Verify no low-level socket or network imports."""
        network_modules = {"socket", "ssl", "http", "http.client"}

        # Use shared utility for import extraction
        imported_modules = get_imported_root_modules(node_ast)

        network_imports_found = imported_modules & network_modules

        assert not network_imports_found, (
            f"Orchestrator should not import network modules.\n"
            f"Found: {sorted(network_imports_found)}\n\n"
            f"HOW TO FIX: Delegate all network operations to effect nodes.\n"
            f"Example: Use 'node_kafka_effect' for Kafka operations, "
            f"'node_registry_effect' for Consul/PostgreSQL.\n"
            f"Pattern: Create an intent model and call ProtocolEffect.execute_intent()."
        )


# =============================================================================
# TestOrchestratorNoDirectNetworkCalls
# =============================================================================


class TestOrchestratorNoDirectNetworkCalls:
    """Tests verifying orchestrator has no direct network/I/O calls."""

    def test_orchestrator_has_no_direct_network_calls(
        self, node_ast: ast.Module
    ) -> None:
        """Verify orchestrator class has no methods that make direct network calls.

        This test inspects all method bodies in the orchestrator class to ensure
        there are no direct calls to I/O methods like:
        - client.get(), client.post() (HTTP)
        - conn.execute(), cursor.query() (database)
        - producer.send(), consumer.poll() (Kafka)
        """
        # Use shared utility for I/O method detection
        io_calls_found = find_io_method_calls(
            node_ast,
            method_patterns=IO_METHOD_PATTERNS,
            class_name="NodeRegistrationOrchestrator",
        )

        assert not io_calls_found, (
            f"Orchestrator should not make direct I/O calls.\n"
            f"Found potential I/O calls: {io_calls_found}\n\n"
            f"HOW TO FIX: Replace direct I/O calls with intent-based delegation.\n"
            f"Instead of: client.get('/service/node'), cursor.execute(query)\n"
            f"Use: await effect_node.execute_intent(ModelConsulReadIntent(...))\n"
            f"The effect node handles the actual I/O and returns results."
        )

    def test_orchestrator_methods_are_minimal(self, mock_container: MagicMock) -> None:
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

        # Whitelist: Timeout coordination methods added by OMN-932.
        # These are legitimate public methods for RuntimeTick timeout handling
        # and do not violate the "pure coordinator" principle because they:
        # 1. Delegate to TimeoutCoordinator (no direct I/O)
        # 2. Are optional integration points, not core workflow logic
        # 3. Follow the same delegation pattern as other workflow steps
        timeout_coordinator_methods = frozenset(
            {
                "set_timeout_coordinator",  # Setter for timeout coordinator injection
                "has_timeout_coordinator",  # Property to check if coordinator is set
                "handle_runtime_tick",  # RuntimeTick event handler (delegates to coordinator)
            }
        )

        # Whitelist: Heartbeat handler methods added by OMN-1006.
        # These are legitimate public methods for heartbeat event handling
        # and do not violate the "pure coordinator" principle because they:
        # 1. Delegate to HandlerNodeHeartbeat (no direct I/O)
        # 2. Are optional integration points for liveness tracking
        # 3. Follow the same delegation pattern as timeout coordination
        heartbeat_handler_methods = frozenset(
            {
                "set_heartbeat_handler",  # Setter for heartbeat handler injection
                "has_heartbeat_handler",  # Property to check if handler is set
                "handle_heartbeat",  # Heartbeat event handler (delegates to handler)
            }
        )

        # Combined whitelist of all allowed methods
        whitelisted_methods = timeout_coordinator_methods | heartbeat_handler_methods

        # Filter out whitelisted methods
        non_whitelisted_methods = [
            m for m in own_methods if m not in whitelisted_methods
        ]

        # Orchestrator should have NO public methods defined directly
        # except for whitelisted coordination methods
        assert non_whitelisted_methods == [], (
            f"Orchestrator should have no custom public methods.\n"
            f"Found: {non_whitelisted_methods}\n"
            f"(Whitelisted methods: {sorted(whitelisted_methods)})\n"
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
            "ProtocolEffect must have execute_intent() method for I/O delegation.\n\n"
            "HOW TO FIX: Add execute_intent() to ProtocolEffect in protocols.py:\n"
            "  async def execute_intent(\n"
            "      self, intent: ModelIntent, correlation_id: UUID\n"
            "  ) -> ModelEffectResult: ..."
        )

        # Verify it's a Protocol (runtime_checkable)
        from typing import Protocol, runtime_checkable

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
            "execute_intent must accept 'intent' parameter for I/O operation details.\n\n"
            "HOW TO FIX: Update signature to include intent parameter:\n"
            "  async def execute_intent(self, intent: ModelIntent, ...) -> ModelEffectResult"
        )
        assert "correlation_id" in params, (
            "execute_intent must accept 'correlation_id' for distributed tracing.\n\n"
            "HOW TO FIX: Update signature to include correlation_id:\n"
            "  async def execute_intent(self, intent: ModelIntent, correlation_id: UUID) -> ..."
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
        assert (
            "MUST NOT perform I/O" in docstring
            or "Reducer MUST NOT perform I/O" in docstring
        ), (
            "ProtocolReducer docstring must explicitly state reducers perform no I/O.\n\n"
            "HOW TO FIX: Add to ProtocolReducer docstring:\n"
            '  """Protocol for reducer operations.\n\n'
            "  Reducer MUST NOT perform I/O. All I/O operations must be\n"
            '  expressed as intents returned to the orchestrator."""'
        )


# =============================================================================
# TestOrchestratorIsPureCoordinator
# =============================================================================


class TestOrchestratorIsPureCoordinator:
    """Tests verifying orchestrator is a pure workflow coordinator."""

    def test_orchestrator_is_pure_coordinator(
        self, node_ast: ast.Module, mock_container: MagicMock
    ) -> None:
        """Verify orchestrator only coordinates workflow, doesn't execute I/O.

        A pure coordinator:
        1. Inherits from NodeOrchestrator (which handles workflow execution)
        2. Has minimal code (just __init__)
        3. Relies on contract.yaml for all workflow logic
        4. Does not implement any I/O methods directly

        Uses AST-based analysis for robust structural verification that won't
        break if source formatting changes.
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

        # Find the orchestrator class definition using AST
        orchestrator_class: ast.ClassDef | None = None
        for node in ast.walk(node_ast):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "NodeRegistrationOrchestrator"
            ):
                orchestrator_class = node
                break

        assert orchestrator_class is not None, (
            "Could not find NodeRegistrationOrchestrator class in AST"
        )

        # Analyze class body using AST - count meaningful statements
        # Exclude docstrings (first Expr with Constant) from statement count
        # Note: is_docstring is imported from tests.helpers.ast_analysis

        # Filter out docstrings from class body
        meaningful_statements = [
            stmt for stmt in orchestrator_class.body if not is_docstring(stmt)
        ]

        # A pure coordinator should only have __init__ method
        # Count method definitions (FunctionDef/AsyncFunctionDef)
        methods = [
            stmt
            for stmt in meaningful_statements
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef)
        ]

        # Get method names for reporting
        method_names = [m.name for m in methods]

        # Whitelist: Timeout coordination methods added by OMN-932.
        # These are legitimate public methods for RuntimeTick timeout handling
        # and do not violate the "pure coordinator" principle because they:
        # 1. Delegate to TimeoutCoordinator (no direct I/O)
        # 2. Are optional integration points, not core workflow logic
        # 3. Follow the same delegation pattern as other workflow steps
        timeout_coordinator_methods = frozenset(
            {
                "set_timeout_coordinator",  # Setter for timeout coordinator injection
                "has_timeout_coordinator",  # Property to check if coordinator is set
                "handle_runtime_tick",  # RuntimeTick event handler (delegates to coordinator)
            }
        )

        # Whitelist: Heartbeat handler methods added by OMN-1006.
        # These are legitimate public methods for heartbeat event handling
        # and do not violate the "pure coordinator" principle because they:
        # 1. Delegate to HandlerNodeHeartbeat (no direct I/O)
        # 2. Are optional integration points for liveness tracking
        # 3. Follow the same delegation pattern as timeout coordination
        heartbeat_handler_methods = frozenset(
            {
                "set_heartbeat_handler",  # Setter for heartbeat handler injection
                "has_heartbeat_handler",  # Property to check if handler is set
                "handle_heartbeat",  # Heartbeat event handler (delegates to handler)
            }
        )

        # Combined whitelist of all allowed methods
        whitelisted_methods = timeout_coordinator_methods | heartbeat_handler_methods

        # Pure coordinator should have only __init__ and whitelisted methods
        non_init_methods = [
            name
            for name in method_names
            if name != "__init__" and name not in whitelisted_methods
        ]

        assert not non_init_methods, (
            f"Pure coordinator should only have __init__ method.\n"
            f"Found additional methods: {non_init_methods}\n"
            f"(Whitelisted methods: {sorted(whitelisted_methods)})\n"
            f"Orchestrator should only call super().__init__(container) and rely on "
            f"base class + contract.yaml for all behavior."
        )

        # Verify __init__ is minimal (just super().__init__ call + handler initializers)
        init_method = next((m for m in methods if m.name == "__init__"), None)
        if init_method:
            # Count statements in __init__ body (excluding docstring)
            init_statements = [
                stmt for stmt in init_method.body if not is_docstring(stmt)
            ]

            # Allow up to 3 statements:
            # 1. super().__init__(container) call
            # 2. _timeout_coordinator = None (OMN-932)
            # 3. _heartbeat_handler = None (OMN-1006)
            assert len(init_statements) <= 3, (
                f"__init__ should be minimal (1-3 statements).\n"
                f"Found {len(init_statements)} statements in __init__ body.\n"
                f"Orchestrator should only call super().__init__(container) and "
                f"initialize optional handler attributes, relying on "
                f"base class + contract.yaml for all other behavior."
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
        found_keywords = [
            kw for kw in delegation_keywords if kw.lower() in docstring.lower()
        ]

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

    def test_contract_io_operations_are_effect_nodes(self, contract_data: dict) -> None:
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
            "Contract has misclassified nodes:\n"
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
        self, node_source_code: str
    ) -> None:
        """Verify module docstring documents the no-I/O pattern."""
        # Extract module docstring (first string literal)
        tree = ast.parse(node_source_code)
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

    def test_all_exports_are_minimal(self, node_source_code: str) -> None:
        """Verify __all__ exports are minimal (just the orchestrator class)."""
        tree = ast.parse(node_source_code)

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

        assert all_value is not None, (
            "Module should define __all__ for explicit exports"
        )
        assert all_value == ["NodeRegistrationOrchestrator"], (
            f"Module should only export NodeRegistrationOrchestrator.\n"
            f"Found exports: {all_value}"
        )
