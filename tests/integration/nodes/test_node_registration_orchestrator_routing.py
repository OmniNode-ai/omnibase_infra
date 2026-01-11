# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for contract-driven handler routing.

These tests verify that the NodeRegistrationOrchestrator's handler_routing
configuration in contract.yaml is correct, that all referenced handlers
and event models are importable, and that runtime routing works correctly.

Test Categories (Contract Validation):
    - TestHandlerRoutingContract: Contract handler_routing section validation
    - TestHandlerRoutingMappings: Event-to-handler mapping correctness
    - TestHandlerRoutingModuleImports: Module import verification
    - TestHandlerRoutingModulePaths: Module path convention checks
    - TestHandlerRoutingOutputEvents: Handler output_events configuration
    - TestHandlerDependencies: Handler dependency configuration

Test Categories (Runtime Routing):
    - TestOrchestratorInstantiation: Orchestrator creation tests
    - TestHandlerRoutingInitialization: Deferred initialization tests
    - TestRouteToHandlers: route_to_handlers() method verification
    - TestValidateHandlerRouting: validate_handler_routing() method tests
    - TestHandlerRoutingContractCodeConsistency: Contract/code alignment

The handler_routing section defines:
    - routing_strategy: "payload_type_match" - Route based on event model type
    - handlers: List of event-to-handler mappings with:
        - event_model: {name, module} - The event model class to match
        - handler: {name, module} - The handler class to invoke
        - output_events: List of event types the handler may emit

Handler ID Mapping:
    - handler-node-introspected -> ModelNodeIntrospectionEvent
    - handler-runtime-tick -> ModelRuntimeTick
    - handler-node-registration-acked -> ModelNodeRegistrationAcked
    - handler-node-heartbeat -> ModelNodeHeartbeatEvent (requires projector)

Running Tests:
    # Run all handler routing tests:
    pytest tests/integration/nodes/test_node_registration_orchestrator_routing.py

    # Run with verbose output:
    pytest tests/integration/nodes/test_node_registration_orchestrator_routing.py -v

    # Run specific test class:
    pytest tests/integration/nodes/test_node_registration_orchestrator_routing.py::TestHandlerRoutingContract

    # Run only runtime routing tests:
    pytest tests/integration/nodes/test_node_registration_orchestrator_routing.py -k "Route or Validate or Initialization"
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.node import (
        NodeRegistrationOrchestrator,
    )


# =============================================================================
# TestHandlerRoutingContract
# =============================================================================


class TestHandlerRoutingContract:
    """Integration tests for contract-driven handler routing configuration.

    These tests verify that the handler_routing section in contract.yaml
    is properly structured and contains all required fields.
    """

    def test_handler_routing_section_exists(self, contract_data: dict) -> None:
        """Verify handler_routing section exists in contract."""
        assert "handler_routing" in contract_data, (
            "Contract must have 'handler_routing' section for declarative routing"
        )

    def test_routing_strategy_is_payload_type_match(self, contract_data: dict) -> None:
        """Verify routing_strategy is 'payload_type_match'.

        The payload_type_match strategy routes events based on the payload
        model class type, enabling type-safe event-to-handler mapping.
        """
        handler_routing = contract_data.get("handler_routing", {})

        assert "routing_strategy" in handler_routing, (
            "handler_routing must have 'routing_strategy' field"
        )
        assert handler_routing["routing_strategy"] == "payload_type_match", (
            f"routing_strategy should be 'payload_type_match', "
            f"got '{handler_routing['routing_strategy']}'"
        )

    def test_handlers_section_exists(self, contract_data: dict) -> None:
        """Verify handlers section exists and is a non-empty list."""
        handler_routing = contract_data.get("handler_routing", {})

        assert "handlers" in handler_routing, (
            "handler_routing must have 'handlers' section"
        )
        assert isinstance(handler_routing["handlers"], list), "handlers must be a list"
        assert len(handler_routing["handlers"]) > 0, "handlers list must not be empty"

    def test_handlers_have_required_fields(self, contract_data: dict) -> None:
        """Verify each handler entry has required event_model and handler fields."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        for i, handler_entry in enumerate(handlers):
            # Verify event_model exists and has required subfields
            assert "event_model" in handler_entry, (
                f"Handler entry {i} missing 'event_model' field"
            )
            event_model = handler_entry["event_model"]
            assert "name" in event_model, (
                f"Handler entry {i} event_model missing 'name' field"
            )
            assert "module" in event_model, (
                f"Handler entry {i} event_model missing 'module' field"
            )

            # Verify handler exists and has required subfields
            assert "handler" in handler_entry, (
                f"Handler entry {i} missing 'handler' field"
            )
            handler = handler_entry["handler"]
            assert "name" in handler, f"Handler entry {i} handler missing 'name' field"
            assert "module" in handler, (
                f"Handler entry {i} handler missing 'module' field"
            )

    def test_expected_handler_count(self, contract_data: dict) -> None:
        """Verify contract defines exactly 3 handlers.

        The registration orchestrator routes:
        1. ModelNodeIntrospectionEvent -> HandlerNodeIntrospected
        2. ModelRuntimeTick -> HandlerRuntimeTick
        3. ModelNodeRegistrationAcked -> HandlerNodeRegistrationAcked
        """
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        assert len(handlers) == 3, (
            f"Expected exactly 3 handler entries, found {len(handlers)}. "
            f"Events: {[h.get('event_model', {}).get('name', 'unknown') for h in handlers]}"
        )

    def test_expected_event_model_names(self, contract_data: dict) -> None:
        """Verify contract maps the expected event model names."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        expected_event_models = {
            "ModelNodeIntrospectionEvent",
            "ModelRuntimeTick",
            "ModelNodeRegistrationAcked",
        }

        actual_event_models = {
            h["event_model"]["name"]
            for h in handlers
            if "event_model" in h and "name" in h["event_model"]
        }

        assert expected_event_models == actual_event_models, (
            f"Event model mismatch.\n"
            f"Missing: {expected_event_models - actual_event_models}\n"
            f"Extra: {actual_event_models - expected_event_models}"
        )

    def test_expected_handler_names(self, contract_data: dict) -> None:
        """Verify contract maps to the expected handler class names."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        expected_handlers = {
            "HandlerNodeIntrospected",
            "HandlerRuntimeTick",
            "HandlerNodeRegistrationAcked",
        }

        actual_handlers = {
            h["handler"]["name"]
            for h in handlers
            if "handler" in h and "name" in h["handler"]
        }

        assert expected_handlers == actual_handlers, (
            f"Handler name mismatch.\n"
            f"Missing: {expected_handlers - actual_handlers}\n"
            f"Extra: {actual_handlers - expected_handlers}"
        )


# =============================================================================
# TestHandlerRoutingMappings
# =============================================================================


class TestHandlerRoutingMappings:
    """Integration tests for event-to-handler mapping correctness.

    These tests verify that each event model is mapped to the correct
    handler class according to the contract specification.
    """

    def test_introspection_event_maps_to_correct_handler(
        self, contract_data: dict
    ) -> None:
        """Verify ModelNodeIntrospectionEvent maps to HandlerNodeIntrospected."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        # Find the handler entry for ModelNodeIntrospectionEvent
        introspection_handler = None
        for handler_entry in handlers:
            if (
                handler_entry.get("event_model", {}).get("name")
                == "ModelNodeIntrospectionEvent"
            ):
                introspection_handler = handler_entry
                break

        assert introspection_handler is not None, (
            "No handler mapping found for ModelNodeIntrospectionEvent"
        )
        assert introspection_handler["handler"]["name"] == "HandlerNodeIntrospected", (
            f"ModelNodeIntrospectionEvent should map to HandlerNodeIntrospected, "
            f"got '{introspection_handler['handler']['name']}'"
        )

    def test_runtime_tick_maps_to_correct_handler(self, contract_data: dict) -> None:
        """Verify ModelRuntimeTick maps to HandlerRuntimeTick."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        # Find the handler entry for ModelRuntimeTick
        tick_handler = None
        for handler_entry in handlers:
            if handler_entry.get("event_model", {}).get("name") == "ModelRuntimeTick":
                tick_handler = handler_entry
                break

        assert tick_handler is not None, "No handler mapping found for ModelRuntimeTick"
        assert tick_handler["handler"]["name"] == "HandlerRuntimeTick", (
            f"ModelRuntimeTick should map to HandlerRuntimeTick, "
            f"got '{tick_handler['handler']['name']}'"
        )

    def test_registration_acked_maps_to_correct_handler(
        self, contract_data: dict
    ) -> None:
        """Verify ModelNodeRegistrationAcked maps to HandlerNodeRegistrationAcked."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        # Find the handler entry for ModelNodeRegistrationAcked
        acked_handler = None
        for handler_entry in handlers:
            if (
                handler_entry.get("event_model", {}).get("name")
                == "ModelNodeRegistrationAcked"
            ):
                acked_handler = handler_entry
                break

        assert acked_handler is not None, (
            "No handler mapping found for ModelNodeRegistrationAcked"
        )
        assert acked_handler["handler"]["name"] == "HandlerNodeRegistrationAcked", (
            f"ModelNodeRegistrationAcked should map to HandlerNodeRegistrationAcked, "
            f"got '{acked_handler['handler']['name']}'"
        )


# =============================================================================
# TestHandlerRoutingModuleImports
# =============================================================================


class TestHandlerRoutingModuleImports:
    """Integration tests for module import verification.

    These tests verify that all module paths specified in the contract
    are valid and the referenced classes exist.
    """

    def test_all_event_model_modules_importable(self, contract_data: dict) -> None:
        """Verify all event model modules can be imported."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        for handler_entry in handlers:
            event_model = handler_entry.get("event_model", {})
            module_path = event_model.get("module")
            class_name = event_model.get("name")

            assert module_path is not None, (
                f"Event model {class_name} missing module path"
            )

            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                pytest.fail(f"Failed to import event model module '{module_path}': {e}")

            assert hasattr(module, class_name), (
                f"Module '{module_path}' does not have class '{class_name}'"
            )

    def test_all_handler_modules_importable(self, contract_data: dict) -> None:
        """Verify all handler modules can be imported."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        for handler_entry in handlers:
            handler = handler_entry.get("handler", {})
            module_path = handler.get("module")
            class_name = handler.get("name")

            assert module_path is not None, f"Handler {class_name} missing module path"

            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                pytest.fail(f"Failed to import handler module '{module_path}': {e}")

            assert hasattr(module, class_name), (
                f"Module '{module_path}' does not have class '{class_name}'"
            )

    def test_event_models_are_pydantic_models(self, contract_data: dict) -> None:
        """Verify all event model classes are Pydantic models."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        for handler_entry in handlers:
            event_model = handler_entry.get("event_model", {})
            module_path = event_model.get("module")
            class_name = event_model.get("name")

            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            # Verify it's a Pydantic model via duck typing
            assert hasattr(model_class, "model_fields"), (
                f"Event model '{class_name}' must be a Pydantic model "
                f"(missing 'model_fields' attribute)"
            )
            assert hasattr(model_class, "model_validate"), (
                f"Event model '{class_name}' must be a Pydantic model "
                f"(missing 'model_validate' method)"
            )

    def test_handlers_are_classes(self, contract_data: dict) -> None:
        """Verify all handler classes are actual classes (not functions)."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        for handler_entry in handlers:
            handler = handler_entry.get("handler", {})
            module_path = handler.get("module")
            class_name = handler.get("name")

            module = importlib.import_module(module_path)
            handler_class = getattr(module, class_name)

            assert isinstance(handler_class, type), (
                f"Handler '{class_name}' must be a class, "
                f"got {type(handler_class).__name__}"
            )


# =============================================================================
# TestHandlerRoutingModulePaths
# =============================================================================


class TestHandlerRoutingModulePaths:
    """Integration tests for module path correctness.

    These tests verify that module paths follow expected conventions
    and point to the correct locations in the codebase.
    """

    def test_event_model_module_paths_follow_convention(
        self, contract_data: dict
    ) -> None:
        """Verify event model module paths follow omnibase_infra conventions."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        expected_prefixes = {
            "ModelNodeIntrospectionEvent": "omnibase_infra.models.registration",
            "ModelRuntimeTick": "omnibase_infra.runtime.models",
            "ModelNodeRegistrationAcked": "omnibase_infra.models.registration.commands",
        }

        for handler_entry in handlers:
            event_model = handler_entry.get("event_model", {})
            class_name = event_model.get("name")
            module_path = event_model.get("module", "")

            expected_prefix = expected_prefixes.get(class_name)
            if expected_prefix is not None:
                assert module_path.startswith(expected_prefix), (
                    f"Event model '{class_name}' module path should start with "
                    f"'{expected_prefix}', got '{module_path}'"
                )

    def test_handler_module_paths_point_to_handlers_package(
        self, contract_data: dict
    ) -> None:
        """Verify all handler module paths point to the handlers package."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        expected_prefix = "omnibase_infra.nodes.node_registration_orchestrator.handlers"

        for handler_entry in handlers:
            handler = handler_entry.get("handler", {})
            class_name = handler.get("name")
            module_path = handler.get("module", "")

            assert module_path.startswith(expected_prefix), (
                f"Handler '{class_name}' module path should start with "
                f"'{expected_prefix}', got '{module_path}'"
            )


# =============================================================================
# TestHandlerRoutingOutputEvents
# =============================================================================


class TestHandlerRoutingOutputEvents:
    """Integration tests for handler output_events configuration.

    These tests verify that each handler declares its output events
    and that the declarations match the expected event types.
    """

    def test_handlers_have_output_events(self, contract_data: dict) -> None:
        """Verify each handler entry has output_events field."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        for handler_entry in handlers:
            handler_name = handler_entry.get("handler", {}).get("name", "unknown")
            assert "output_events" in handler_entry, (
                f"Handler '{handler_name}' missing 'output_events' field"
            )
            assert isinstance(handler_entry["output_events"], list), (
                f"Handler '{handler_name}' output_events must be a list"
            )

    def test_introspection_handler_output_events(self, contract_data: dict) -> None:
        """Verify HandlerNodeIntrospected declares expected output events."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        # Find the introspection handler
        for handler_entry in handlers:
            if (
                handler_entry.get("handler", {}).get("name")
                == "HandlerNodeIntrospected"
            ):
                output_events = handler_entry.get("output_events", [])
                assert "ModelNodeRegistrationInitiated" in output_events, (
                    "HandlerNodeIntrospected should emit ModelNodeRegistrationInitiated"
                )
                break
        else:
            pytest.fail("HandlerNodeIntrospected not found in handlers")

    def test_runtime_tick_handler_output_events(self, contract_data: dict) -> None:
        """Verify HandlerRuntimeTick declares expected output events."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        # Find the runtime tick handler
        for handler_entry in handlers:
            if handler_entry.get("handler", {}).get("name") == "HandlerRuntimeTick":
                output_events = handler_entry.get("output_events", [])
                expected_events = {
                    "ModelNodeRegistrationAckTimedOut",
                    "ModelNodeLivenessExpired",
                }
                actual_events = set(output_events)
                assert expected_events <= actual_events, (
                    f"HandlerRuntimeTick missing expected output events. "
                    f"Missing: {expected_events - actual_events}"
                )
                break
        else:
            pytest.fail("HandlerRuntimeTick not found in handlers")

    def test_registration_acked_handler_output_events(
        self, contract_data: dict
    ) -> None:
        """Verify HandlerNodeRegistrationAcked declares expected output events."""
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        # Find the registration acked handler
        for handler_entry in handlers:
            if (
                handler_entry.get("handler", {}).get("name")
                == "HandlerNodeRegistrationAcked"
            ):
                output_events = handler_entry.get("output_events", [])
                expected_events = {
                    "ModelNodeRegistrationAckReceived",
                    "ModelNodeBecameActive",
                }
                actual_events = set(output_events)
                assert expected_events <= actual_events, (
                    f"HandlerNodeRegistrationAcked missing expected output events. "
                    f"Missing: {expected_events - actual_events}"
                )
                break
        else:
            pytest.fail("HandlerNodeRegistrationAcked not found in handlers")


# =============================================================================
# TestHandlerDependencies
# =============================================================================


class TestHandlerDependencies:
    """Integration tests for handler dependency configuration.

    These tests verify that the handler_dependencies section is properly
    configured for shared dependencies like projection_reader.
    """

    def test_handler_dependencies_section_exists(self, contract_data: dict) -> None:
        """Verify handler_dependencies section exists."""
        handler_routing = contract_data.get("handler_routing", {})

        assert "handler_dependencies" in handler_routing, (
            "handler_routing should have 'handler_dependencies' section"
        )

    def test_projection_reader_dependency_configured(self, contract_data: dict) -> None:
        """Verify projection_reader dependency is properly configured."""
        handler_routing = contract_data.get("handler_routing", {})
        handler_deps = handler_routing.get("handler_dependencies", {})

        assert "projection_reader" in handler_deps, (
            "handler_dependencies should have 'projection_reader' configuration"
        )

        projection_reader = handler_deps["projection_reader"]

        assert projection_reader.get("protocol") == "ProtocolProjectionReader", (
            f"projection_reader protocol should be 'ProtocolProjectionReader', "
            f"got '{projection_reader.get('protocol')}'"
        )
        assert projection_reader.get("shared") is True, (
            "projection_reader should be shared across handlers"
        )

    def test_projection_reader_implementation_importable(
        self, contract_data: dict
    ) -> None:
        """Verify projection_reader implementation module is importable."""
        handler_routing = contract_data.get("handler_routing", {})
        handler_deps = handler_routing.get("handler_dependencies", {})
        projection_reader = handler_deps.get("projection_reader", {})

        module_path = projection_reader.get("module")
        impl_name = projection_reader.get("implementation")

        if module_path and impl_name:
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                pytest.fail(
                    f"Failed to import projection_reader module '{module_path}': {e}"
                )

            assert hasattr(module, impl_name), (
                f"Module '{module_path}' does not have class '{impl_name}'"
            )


# =============================================================================
# TestOrchestratorInstantiation
# =============================================================================


class TestOrchestratorInstantiation:
    """Integration tests for NodeRegistrationOrchestrator instantiation.

    These tests verify that the orchestrator can be created with proper
    dependency injection via ModelONEXContainer.
    """

    @pytest.fixture
    def mock_container(self) -> MagicMock:
        """Create a mock ONEX container for testing.

        Returns:
            MagicMock configured with minimal container.config attribute.
        """
        from unittest.mock import MagicMock

        container = MagicMock()
        container.config = MagicMock()
        return container

    @pytest.fixture
    def mock_projection_reader(self) -> MagicMock:
        """Create a mock projection reader for testing.

        Returns:
            MagicMock configured as a ProjectionReaderRegistration.
        """
        from unittest.mock import AsyncMock, MagicMock

        reader = MagicMock()
        reader.get_entity_state = AsyncMock(return_value=None)
        reader.list_entities_by_status = AsyncMock(return_value=[])
        return reader

    def test_orchestrator_instantiation_without_projection_reader(
        self, mock_container: MagicMock
    ) -> None:
        """Verify orchestrator can be instantiated without projection_reader.

        When projection_reader is None, handler routing is deferred until
        initialize_handler_routing() is called.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        orchestrator = NodeRegistrationOrchestrator(mock_container)

        assert orchestrator is not None
        # Handler routing should NOT be initialized yet
        assert not orchestrator.is_routing_initialized

    def test_orchestrator_instantiation_with_projection_reader(
        self, mock_container: MagicMock, mock_projection_reader: MagicMock
    ) -> None:
        """Verify orchestrator can be instantiated with projection_reader.

        When projection_reader is provided, handler routing is initialized
        immediately during construction.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        orchestrator = NodeRegistrationOrchestrator(
            mock_container,
            projection_reader=mock_projection_reader,
        )

        assert orchestrator is not None
        # Handler routing should be initialized
        assert orchestrator.is_routing_initialized


# =============================================================================
# TestHandlerRoutingInitialization
# =============================================================================


class TestHandlerRoutingInitialization:
    """Integration tests for handler routing initialization.

    These tests verify that handler routing can be initialized both at
    construction time and via deferred initialization.
    """

    @pytest.fixture
    def mock_container(self) -> MagicMock:
        """Create a mock ONEX container for testing."""
        from unittest.mock import MagicMock

        container = MagicMock()
        container.config = MagicMock()
        return container

    @pytest.fixture
    def mock_projection_reader(self) -> MagicMock:
        """Create a mock projection reader for testing."""
        from unittest.mock import AsyncMock, MagicMock

        reader = MagicMock()
        reader.get_entity_state = AsyncMock(return_value=None)
        reader.list_entities_by_status = AsyncMock(return_value=[])
        return reader

    def test_deferred_handler_routing_initialization(
        self, mock_container: MagicMock, mock_projection_reader: MagicMock
    ) -> None:
        """Verify handler routing can be initialized via deferred method.

        This tests the initialize_handler_routing() method which is called
        after construction when projection_reader is not available at init.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        # Create without projection_reader
        orchestrator = NodeRegistrationOrchestrator(mock_container)
        assert not orchestrator.is_routing_initialized

        # Initialize routing later
        orchestrator.initialize_handler_routing(mock_projection_reader)
        assert orchestrator.is_routing_initialized

    def test_double_initialization_raises_error(
        self, mock_container: MagicMock, mock_projection_reader: MagicMock
    ) -> None:
        """Verify double initialization raises RuntimeError.

        Calling initialize_handler_routing() twice should fail.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        # Create with projection_reader (initializes immediately)
        orchestrator = NodeRegistrationOrchestrator(
            mock_container, projection_reader=mock_projection_reader
        )
        assert orchestrator.is_routing_initialized

        # Attempt to initialize again should raise
        with pytest.raises(RuntimeError, match="already initialized"):
            orchestrator.initialize_handler_routing(mock_projection_reader)

    def test_routing_strategy_is_payload_type_match(
        self, mock_container: MagicMock, mock_projection_reader: MagicMock
    ) -> None:
        """Verify routing strategy is set to payload_type_match.

        The contract.yaml specifies 'payload_type_match' as the routing strategy.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        orchestrator = NodeRegistrationOrchestrator(
            mock_container, projection_reader=mock_projection_reader
        )

        assert orchestrator.routing_strategy == "payload_type_match"


# =============================================================================
# TestRouteToHandlers
# =============================================================================


class TestRouteToHandlers:
    """Integration tests for route_to_handlers() method.

    These tests verify that events are routed to the correct handlers
    based on the payload type matching strategy.
    """

    @pytest.fixture
    def mock_container(self) -> MagicMock:
        """Create a mock ONEX container for testing."""
        from unittest.mock import MagicMock

        container = MagicMock()
        container.config = MagicMock()
        return container

    @pytest.fixture
    def mock_projection_reader(self) -> MagicMock:
        """Create a mock projection reader for testing."""
        from unittest.mock import AsyncMock, MagicMock

        reader = MagicMock()
        reader.get_entity_state = AsyncMock(return_value=None)
        reader.list_entities_by_status = AsyncMock(return_value=[])
        return reader

    @pytest.fixture
    def initialized_orchestrator(
        self, mock_container: MagicMock, mock_projection_reader: MagicMock
    ) -> NodeRegistrationOrchestrator:
        """Create an orchestrator with handler routing initialized."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        return NodeRegistrationOrchestrator(
            mock_container, projection_reader=mock_projection_reader
        )

    def test_route_introspection_event_to_handler(
        self, initialized_orchestrator: NodeRegistrationOrchestrator
    ) -> None:
        """Verify ModelNodeIntrospectionEvent routes to handler-node-introspected."""
        from omnibase_core.enums import EnumMessageCategory

        handlers = initialized_orchestrator.route_to_handlers(
            routing_key="ModelNodeIntrospectionEvent",
            category=EnumMessageCategory.EVENT,
        )

        assert len(handlers) == 1
        assert handlers[0].handler_id == "handler-node-introspected"

    def test_route_runtime_tick_to_handler(
        self, initialized_orchestrator: NodeRegistrationOrchestrator
    ) -> None:
        """Verify ModelRuntimeTick routes to handler-runtime-tick."""
        from omnibase_core.enums import EnumMessageCategory

        handlers = initialized_orchestrator.route_to_handlers(
            routing_key="ModelRuntimeTick",
            category=EnumMessageCategory.EVENT,
        )

        assert len(handlers) == 1
        assert handlers[0].handler_id == "handler-runtime-tick"

    def test_route_registration_acked_to_handler(
        self, initialized_orchestrator: NodeRegistrationOrchestrator
    ) -> None:
        """Verify ModelNodeRegistrationAcked routes to handler-node-registration-acked."""
        from omnibase_core.enums import EnumMessageCategory

        handlers = initialized_orchestrator.route_to_handlers(
            routing_key="ModelNodeRegistrationAcked",
            category=EnumMessageCategory.COMMAND,
        )

        assert len(handlers) == 1
        assert handlers[0].handler_id == "handler-node-registration-acked"

    def test_route_unknown_event_returns_empty(
        self, initialized_orchestrator: NodeRegistrationOrchestrator
    ) -> None:
        """Verify unknown event types return empty handler list."""
        from omnibase_core.enums import EnumMessageCategory

        handlers = initialized_orchestrator.route_to_handlers(
            routing_key="UnknownEventModel",
            category=EnumMessageCategory.EVENT,
        )

        assert len(handlers) == 0

    def test_routing_table_contains_expected_mappings(
        self, initialized_orchestrator: NodeRegistrationOrchestrator
    ) -> None:
        """Verify routing table contains all expected event-to-handler mappings."""
        routing_table = initialized_orchestrator.get_routing_table()

        # Expected routing keys
        expected_keys = {
            "ModelNodeIntrospectionEvent",
            "ModelRuntimeTick",
            "ModelNodeRegistrationAcked",
        }

        actual_keys = set(routing_table.keys())

        assert expected_keys <= actual_keys, (
            f"Missing routing keys: {expected_keys - actual_keys}"
        )


# =============================================================================
# TestValidateHandlerRouting
# =============================================================================


class TestValidateHandlerRouting:
    """Integration tests for validate_handler_routing() method.

    These tests verify that the handler routing configuration is valid
    and all referenced handlers are registered.
    """

    @pytest.fixture
    def mock_container(self) -> MagicMock:
        """Create a mock ONEX container for testing."""
        from unittest.mock import MagicMock

        container = MagicMock()
        container.config = MagicMock()
        return container

    @pytest.fixture
    def mock_projection_reader(self) -> MagicMock:
        """Create a mock projection reader for testing."""
        from unittest.mock import AsyncMock, MagicMock

        reader = MagicMock()
        reader.get_entity_state = AsyncMock(return_value=None)
        reader.list_entities_by_status = AsyncMock(return_value=[])
        return reader

    @pytest.fixture
    def initialized_orchestrator(
        self, mock_container: MagicMock, mock_projection_reader: MagicMock
    ) -> NodeRegistrationOrchestrator:
        """Create an orchestrator with handler routing initialized."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        return NodeRegistrationOrchestrator(
            mock_container, projection_reader=mock_projection_reader
        )

    def test_validate_handler_routing_returns_expected_errors(
        self, initialized_orchestrator: NodeRegistrationOrchestrator
    ) -> None:
        """Verify validate_handler_routing() returns expected errors.

        The subcontract includes handler-node-heartbeat which is only registered
        when a projector is provided. Since we use None projector, this handler
        is not registered, and validation will report it as missing.

        This is expected behavior - heartbeat handler requires a projector
        for state persistence, so it's conditionally registered.
        """
        errors = initialized_orchestrator.validate_handler_routing()

        # Expect exactly one error about missing heartbeat handler
        assert len(errors) == 1, f"Expected 1 error, got {len(errors)}: {errors}"
        assert "handler-node-heartbeat" in errors[0], (
            f"Expected error about handler-node-heartbeat, got: {errors[0]}"
        )

    def test_validate_before_initialization_raises_error(
        self, mock_container: MagicMock
    ) -> None:
        """Verify validation before initialization raises ModelOnexError.

        When handler routing is not initialized, calling validate_handler_routing()
        raises a ModelOnexError with error code ONEX_CORE_086_INVALID_STATE.
        """
        from omnibase_core.models.errors import ModelOnexError

        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        orchestrator = NodeRegistrationOrchestrator(mock_container)
        assert not orchestrator.is_routing_initialized

        # Validation before initialization raises an exception
        with pytest.raises(ModelOnexError) as exc_info:
            orchestrator.validate_handler_routing()

        assert "not initialized" in str(exc_info.value).lower()


# =============================================================================
# TestHandlerRoutingContractCodeConsistency
# =============================================================================


class TestHandlerRoutingContractCodeConsistency:
    """Integration tests for consistency between contract.yaml and code.

    These tests verify that the handler_routing configuration in contract.yaml
    matches the actual handler IDs used in the registry.
    """

    @pytest.fixture
    def mock_container(self) -> MagicMock:
        """Create a mock ONEX container for testing."""
        from unittest.mock import MagicMock

        container = MagicMock()
        container.config = MagicMock()
        return container

    @pytest.fixture
    def mock_projection_reader(self) -> MagicMock:
        """Create a mock projection reader for testing."""
        from unittest.mock import AsyncMock, MagicMock

        reader = MagicMock()
        reader.get_entity_state = AsyncMock(return_value=None)
        reader.list_entities_by_status = AsyncMock(return_value=[])
        return reader

    def test_contract_event_models_match_routing_table_keys(
        self,
        contract_data: dict,
        mock_container: MagicMock,
        mock_projection_reader: MagicMock,
    ) -> None:
        """Verify contract event models match routing table keys.

        The event model names in contract.yaml should match the routing keys
        used in the orchestrator's routing table.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        orchestrator = NodeRegistrationOrchestrator(
            mock_container, projection_reader=mock_projection_reader
        )

        # Extract event model names from contract
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])
        contract_event_models = {
            h["event_model"]["name"]
            for h in handlers
            if "event_model" in h and "name" in h["event_model"]
        }

        # Get routing table keys from orchestrator
        routing_table = orchestrator.get_routing_table()
        routing_table_keys = set(routing_table.keys())

        # All contract event models should be in routing table
        assert contract_event_models <= routing_table_keys, (
            f"Contract event models not in routing table: "
            f"{contract_event_models - routing_table_keys}"
        )

    def test_handler_ids_are_consistent(
        self,
        mock_container: MagicMock,
        mock_projection_reader: MagicMock,
    ) -> None:
        """Verify handler IDs match between subcontract and registry adapters.

        The handler_key values in _create_handler_routing_subcontract() must
        match the handler_id properties of the registered adapter classes.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
            _create_handler_routing_subcontract,
        )

        # Get expected handler keys from subcontract
        subcontract = _create_handler_routing_subcontract()
        expected_handler_keys = {entry.handler_key for entry in subcontract.handlers}

        # Create orchestrator to get actual handler IDs
        orchestrator = NodeRegistrationOrchestrator(
            mock_container, projection_reader=mock_projection_reader
        )

        # Get routing table and extract handler IDs
        routing_table = orchestrator.get_routing_table()
        actual_handler_ids = set()
        for handler_keys in routing_table.values():
            actual_handler_ids.update(handler_keys)

        # All expected handler keys should be registered
        assert expected_handler_keys <= actual_handler_ids, (
            f"Missing handler IDs: {expected_handler_keys - actual_handler_ids}"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestHandlerRoutingContract",
    "TestHandlerRoutingMappings",
    "TestHandlerRoutingModuleImports",
    "TestHandlerRoutingModulePaths",
    "TestHandlerRoutingOutputEvents",
    "TestHandlerDependencies",
    "TestOrchestratorInstantiation",
    "TestHandlerRoutingInitialization",
    "TestRouteToHandlers",
    "TestValidateHandlerRouting",
    "TestHandlerRoutingContractCodeConsistency",
]
