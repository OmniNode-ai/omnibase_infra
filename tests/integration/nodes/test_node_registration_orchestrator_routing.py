# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for contract-driven handler routing.

These tests verify that the NodeRegistrationOrchestrator's handler_routing
configuration in contract.yaml is correct, that all referenced handlers
and event models are importable, and that the subcontract is properly configured.

Note: The orchestrator is now pure declarative with only container injection.
Handler routing initialization is performed by the runtime, not the orchestrator.
Tests that previously tested runtime routing behavior now validate the
subcontract configuration directly via _create_handler_routing_subcontract().

Test Categories (Contract Validation):
    - TestHandlerRoutingContract: Contract handler_routing section validation
    - TestHandlerRoutingMappings: Event-to-handler mapping correctness
    - TestHandlerRoutingModuleImports: Module import verification
    - TestHandlerRoutingModulePaths: Module path convention checks
    - TestHandlerRoutingOutputEvents: Handler output_events configuration
    - TestHandlerDependencies: Handler dependency configuration

Test Categories (Subcontract/Instantiation):
    - TestOrchestratorInstantiation: Orchestrator creation tests (container-only)
    - TestHandlerRoutingInitialization: Subcontract configuration tests
    - TestRouteToHandlers: Subcontract routing entry verification
    - TestValidateHandlerRouting: Subcontract structure validation
    - TestHandlerRoutingContractCodeConsistency: Contract/subcontract alignment

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
    - handler-node-heartbeat -> ModelNodeHeartbeatEvent

Running Tests:
    # Run all handler routing tests:
    pytest tests/integration/nodes/test_node_registration_orchestrator_routing.py

    # Run with verbose output:
    pytest tests/integration/nodes/test_node_registration_orchestrator_routing.py -v

    # Run specific test class:
    pytest tests/integration/nodes/test_node_registration_orchestrator_routing.py::TestHandlerRoutingContract

    # Run only subcontract tests:
    pytest tests/integration/nodes/test_node_registration_orchestrator_routing.py -k "subcontract"
"""

from __future__ import annotations

import importlib
import re
from unittest.mock import MagicMock

import pytest

# Module-level marker for test discovery/filtering
pytestmark = pytest.mark.integration


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
        """Verify contract defines exactly 5 handlers.

        The registration orchestrator routes:
        1. ModelNodeIntrospectionEvent -> HandlerNodeIntrospected
        2. ModelRuntimeTick -> HandlerRuntimeTick
        3. ModelNodeRegistrationAcked -> HandlerNodeRegistrationAcked
        4. ModelNodeHeartbeatEvent -> HandlerNodeHeartbeat
        5. ModelTopicCatalogQuery -> HandlerTopicCatalogQuery  # OMN-2313
        """
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])

        assert len(handlers) == 6, (
            f"Expected exactly 6 handler entries, found {len(handlers)}. "
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
            "ModelNodeHeartbeatEvent",
            "ModelTopicCatalogQuery",  # OMN-2313
            "ModelTopicCatalogRequest",  # OMN-2923
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
            "HandlerNodeHeartbeat",
            "HandlerTopicCatalogQuery",  # OMN-2313
            "HandlerCatalogRequest",  # OMN-2923
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
            "ModelNodeHeartbeatEvent": "omnibase_infra.models.registration",
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

    Note: The orchestrator is now pure declarative with only container
    injection. Handler routing initialization is done by the runtime,
    not by the orchestrator itself.
    """

    @pytest.fixture
    def mock_container(self) -> MagicMock:
        """Create a mock ONEX container for testing.

        Returns:
            MagicMock configured with minimal container.config attribute.
        """
        container = MagicMock()
        container.config = MagicMock()
        return container

    def test_orchestrator_instantiation(self, mock_container: MagicMock) -> None:
        """Verify orchestrator can be instantiated with container only.

        The orchestrator is pure declarative - it only requires a container.
        Handler routing initialization is handled by the runtime.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        orchestrator = NodeRegistrationOrchestrator(mock_container)

        assert orchestrator is not None

    def test_orchestrator_has_container_attribute(
        self, mock_container: MagicMock
    ) -> None:
        """Verify orchestrator stores container reference.

        The container is used for dependency injection throughout
        the orchestrator lifecycle.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        orchestrator = NodeRegistrationOrchestrator(mock_container)

        # Orchestrator should have access to container (via base class)
        assert orchestrator.container is mock_container


# =============================================================================
# TestHandlerRoutingInitialization
# =============================================================================


class TestHandlerRoutingInitialization:
    """Integration tests for handler routing configuration.

    These tests verify that the handler routing subcontract is properly
    configured. The orchestrator is now pure declarative - handler routing
    initialization is performed by the runtime, not the orchestrator.

    Note: Tests that previously tested deferred initialization or double
    initialization have been removed as the orchestrator no longer owns
    the initialization lifecycle.
    """

    def test_routing_strategy_is_payload_type_match_in_subcontract(self) -> None:
        """Verify routing strategy is set to payload_type_match in subcontract.

        The contract.yaml specifies 'payload_type_match' as the routing strategy.
        This is reflected in the _create_handler_routing_subcontract() function.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        assert subcontract.routing_strategy == "payload_type_match"

    def test_subcontract_has_expected_handlers(self) -> None:
        """Verify subcontract defines expected handler entries.

        The subcontract should include entries for:
        - ModelNodeIntrospectionEvent -> handler-node-introspected
        - ModelRuntimeTick -> handler-runtime-tick
        - ModelNodeRegistrationAcked -> handler-node-registration-acked
        - ModelNodeHeartbeatEvent -> handler-node-heartbeat
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        expected_mappings = {
            "ModelNodeIntrospectionEvent": "handler-node-introspected",
            "ModelRuntimeTick": "handler-runtime-tick",
            "ModelNodeRegistrationAcked": "handler-node-registration-acked",
            "ModelNodeHeartbeatEvent": "handler-node-heartbeat",
            "ModelTopicCatalogQuery": "handler-topic-catalog-query",  # OMN-2313
            "ModelTopicCatalogRequest": "handler-catalog-request",  # OMN-2923
        }

        actual_mappings = {
            entry.routing_key: entry.handler_key for entry in subcontract.handlers
        }

        assert actual_mappings == expected_mappings, (
            f"Handler mapping mismatch.\n"
            f"Expected: {expected_mappings}\n"
            f"Actual: {actual_mappings}"
        )

    def test_subcontract_version_is_set(self) -> None:
        """Verify subcontract has a valid version."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        assert subcontract.version is not None
        assert subcontract.version.major >= 1


# =============================================================================
# TestRouteToHandlers
# =============================================================================


class TestRouteToHandlers:
    """Integration tests for handler routing configuration.

    These tests verify that the handler routing configuration is correct
    by examining the subcontract. The actual route_to_handlers() method
    is provided by MixinHandlerRouting and is initialized by the runtime.

    Note: Tests for actual routing behavior (route_to_handlers method)
    require runtime initialization which is no longer owned by the
    orchestrator. These tests now validate the subcontract configuration.
    """

    def test_subcontract_routes_introspection_event(self) -> None:
        """Verify ModelNodeIntrospectionEvent maps to handler-node-introspected."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        # Find the entry for ModelNodeIntrospectionEvent
        entry = next(
            (
                e
                for e in subcontract.handlers
                if e.routing_key == "ModelNodeIntrospectionEvent"
            ),
            None,
        )

        assert entry is not None, "No routing entry for ModelNodeIntrospectionEvent"
        assert entry.handler_key == "handler-node-introspected"

    def test_subcontract_routes_runtime_tick(self) -> None:
        """Verify ModelRuntimeTick maps to handler-runtime-tick."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        entry = next(
            (e for e in subcontract.handlers if e.routing_key == "ModelRuntimeTick"),
            None,
        )

        assert entry is not None, "No routing entry for ModelRuntimeTick"
        assert entry.handler_key == "handler-runtime-tick"

    def test_subcontract_routes_registration_acked(self) -> None:
        """Verify ModelNodeRegistrationAcked maps to handler-node-registration-acked."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        entry = next(
            (
                e
                for e in subcontract.handlers
                if e.routing_key == "ModelNodeRegistrationAcked"
            ),
            None,
        )

        assert entry is not None, "No routing entry for ModelNodeRegistrationAcked"
        assert entry.handler_key == "handler-node-registration-acked"

    def test_subcontract_routes_heartbeat_event(self) -> None:
        """Verify ModelNodeHeartbeatEvent maps to handler-node-heartbeat."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        entry = next(
            (
                e
                for e in subcontract.handlers
                if e.routing_key == "ModelNodeHeartbeatEvent"
            ),
            None,
        )

        assert entry is not None, "No routing entry for ModelNodeHeartbeatEvent"
        assert entry.handler_key == "handler-node-heartbeat"

    def test_subcontract_contains_expected_routing_keys(self) -> None:
        """Verify subcontract contains all expected event-to-handler mappings."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        # Expected routing keys
        expected_keys = {
            "ModelNodeIntrospectionEvent",
            "ModelRuntimeTick",
            "ModelNodeRegistrationAcked",
            "ModelNodeHeartbeatEvent",
            "ModelTopicCatalogQuery",  # OMN-2313
            "ModelTopicCatalogRequest",  # OMN-2923
        }

        actual_keys = {entry.routing_key for entry in subcontract.handlers}

        assert expected_keys == actual_keys, (
            f"Routing key mismatch.\n"
            f"Missing: {expected_keys - actual_keys}\n"
            f"Extra: {actual_keys - expected_keys}"
        )


# =============================================================================
# TestValidateHandlerRouting
# =============================================================================


class TestValidateHandlerRouting:
    """Integration tests for handler routing subcontract validation.

    These tests verify that the handler routing subcontract configuration
    is structurally valid. The actual validate_handler_routing() method
    requires runtime initialization which is now handled externally.

    Note: Tests that validated runtime state (validate_handler_routing method)
    have been replaced with subcontract structure validation tests.
    """

    def test_subcontract_handlers_have_valid_routing_keys(self) -> None:
        """Verify all handler entries have non-empty routing keys."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        for entry in subcontract.handlers:
            assert entry.routing_key, f"Handler entry has empty routing_key: {entry}"
            assert isinstance(entry.routing_key, str), (
                f"routing_key must be a string: {entry.routing_key}"
            )

    def test_subcontract_handlers_have_valid_handler_keys(self) -> None:
        """Verify all handler entries have non-empty handler keys."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        for entry in subcontract.handlers:
            assert entry.handler_key, f"Handler entry has empty handler_key: {entry}"
            assert isinstance(entry.handler_key, str), (
                f"handler_key must be a string: {entry.handler_key}"
            )

    def test_subcontract_handler_keys_follow_naming_convention(self) -> None:
        """Verify handler keys follow the 'handler-<name>' convention."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        for entry in subcontract.handlers:
            assert entry.handler_key.startswith("handler-"), (
                f"Handler key should start with 'handler-': {entry.handler_key}"
            )

    def test_subcontract_routing_keys_follow_model_naming_convention(self) -> None:
        """Verify routing keys follow the 'Model<Name>' convention."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        subcontract = _create_handler_routing_subcontract()

        for entry in subcontract.handlers:
            assert entry.routing_key.startswith("Model"), (
                f"Routing key should start with 'Model': {entry.routing_key}"
            )


# =============================================================================
# TestHandlerRoutingContractCodeConsistency
# =============================================================================


class TestHandlerRoutingContractCodeConsistency:
    """Integration tests for consistency between contract.yaml and code.

    These tests verify that the handler_routing configuration in contract.yaml
    matches the subcontract configuration in the node module.
    """

    def test_contract_event_models_match_subcontract_routing_keys(
        self,
        contract_data: dict,
    ) -> None:
        """Verify contract event models match subcontract routing keys.

        The event model names in contract.yaml should match the routing keys
        defined in _create_handler_routing_subcontract().
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        # Extract event model names from contract
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])
        contract_event_models = {
            h["event_model"]["name"]
            for h in handlers
            if "event_model" in h and "name" in h["event_model"]
        }

        # Get routing keys from subcontract
        subcontract = _create_handler_routing_subcontract()
        subcontract_routing_keys = {entry.routing_key for entry in subcontract.handlers}

        # INTENTIONAL: Use subset check (<=) instead of equality (==).
        #
        # Why subset instead of equality:
        # - Contract defines the MINIMUM required handlers (always present)
        # - Subcontract may include ADDITIONAL conditional handlers
        #
        # Example of intentional divergence:
        # - The heartbeat handler is only registered when a projector is available
        # - It appears in subcontract but not in the base contract
        #
        # This design allows:
        # - Contract to be stable (all entries are required and always active)
        # - Subcontract to be flexible (can include optional/conditional handlers)
        assert contract_event_models <= subcontract_routing_keys, (
            f"Contract event models not in subcontract: "
            f"{contract_event_models - subcontract_routing_keys}"
        )

    def test_handler_ids_are_consistent(self) -> None:
        """Verify handler IDs in subcontract follow expected naming.

        The handler_key values in _create_handler_routing_subcontract() must
        follow the 'handler-<name>' naming convention.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        # Get expected handler keys from subcontract
        subcontract = _create_handler_routing_subcontract()
        handler_keys = {entry.handler_key for entry in subcontract.handlers}

        # Expected handler IDs based on contract
        expected_handler_ids = {
            "handler-node-introspected",
            "handler-runtime-tick",
            "handler-node-registration-acked",
            "handler-node-heartbeat",
            "handler-topic-catalog-query",  # OMN-2313
            "handler-catalog-request",  # OMN-2923
        }

        assert handler_keys == expected_handler_ids, (
            f"Handler ID mismatch.\n"
            f"Missing: {expected_handler_ids - handler_keys}\n"
            f"Extra: {handler_keys - expected_handler_ids}"
        )

    def test_contract_handler_names_map_to_subcontract_handler_keys(
        self,
        contract_data: dict,
    ) -> None:
        """Verify contract handler class names can be mapped to subcontract keys.

        The handler class names in contract.yaml should correspond to the
        handler_key values in the subcontract via a consistent naming convention.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        # Extract handler class names from contract
        handler_routing = contract_data.get("handler_routing", {})
        handlers = handler_routing.get("handlers", [])
        contract_handler_names = {
            h["handler"]["name"]
            for h in handlers
            if "handler" in h and "name" in h["handler"]
        }

        # Get handler keys from subcontract
        subcontract = _create_handler_routing_subcontract()
        subcontract_handler_keys = {entry.handler_key for entry in subcontract.handlers}

        # Contract handler names should correspond to subcontract keys
        # (e.g., HandlerNodeIntrospected -> handler-node-introspected)
        for handler_name in contract_handler_names:
            # Convert CamelCase to kebab-case
            kebab_name = re.sub(r"(?<!^)(?=[A-Z])", "-", handler_name).lower()
            assert kebab_name in subcontract_handler_keys, (
                f"Handler '{handler_name}' (as '{kebab_name}') not found in "
                f"subcontract keys: {subcontract_handler_keys}"
            )


# =============================================================================
# Module Exports
# =============================================================================

# NOTE: __all__ must include ALL test classes defined in this module.
# When adding a new test class, add it here to ensure proper discovery
# and documentation. The list is ordered by test category:
# 1. Contract validation tests (TestHandlerRouting*)
# 2. Subcontract/instantiation tests (TestOrchestrator*, TestRoute*, TestValidate*)
# 3. Consistency tests (TestHandlerRoutingContractCodeConsistency)
__all__ = [
    # Contract validation tests
    "TestHandlerRoutingContract",
    "TestHandlerRoutingMappings",
    "TestHandlerRoutingModuleImports",
    "TestHandlerRoutingModulePaths",
    "TestHandlerRoutingOutputEvents",
    "TestHandlerDependencies",
    # Subcontract/instantiation tests
    "TestOrchestratorInstantiation",
    "TestHandlerRoutingInitialization",
    "TestRouteToHandlers",
    "TestValidateHandlerRouting",
    # Consistency tests
    "TestHandlerRoutingContractCodeConsistency",
]
