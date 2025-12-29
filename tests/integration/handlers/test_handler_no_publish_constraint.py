# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests proving handler no-publish constraint is enforced.

This module validates that handlers cannot directly access the event bus.
Handlers return results/events to the caller; they do NOT publish.

This is a structural constraint enforced by dependency injection patterns:
- Handlers receive only domain-specific dependencies (readers, projectors, etc.)
- No bus, dispatcher, or event publisher is injected
- Handlers return data structures; callers decide what to publish

Related Tickets:
    - OMN-1094: Test Coverage - Existing No-Publish Constraint
    - OMN-888: Registration Orchestrator (established pattern)

Test Strategy:
    These tests use introspection to prove that handlers CANNOT access
    the event bus because:
    1. No bus dependency is accepted in their constructor signatures
    2. No bus-related attributes exist on handler instances
    3. Return types are data structures, not publish operations
"""

from __future__ import annotations

import inspect
from typing import get_type_hints
from unittest.mock import MagicMock

import pytest

from omnibase_infra.handlers.handler_http import HttpRestHandler
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)

# ============================================================================
# Test HttpRestHandler Bus Isolation
# ============================================================================


class TestHttpRestHandlerBusIsolation:
    """Prove HttpRestHandler cannot access the event bus structurally."""

    def test_constructor_takes_no_parameters(self) -> None:
        """HttpRestHandler.__init__ takes only self - no dependency injection."""
        sig = inspect.signature(HttpRestHandler.__init__)
        params = list(sig.parameters.keys())

        # Only 'self' parameter
        assert params == ["self"], (
            f"HttpRestHandler.__init__ should take only 'self', "
            f"but has parameters: {params}"
        )

    def test_no_bus_attribute_after_instantiation(self) -> None:
        """Handler instance has no bus-related attributes."""
        handler = HttpRestHandler()

        # Check for common bus/event publishing attribute names
        forbidden_attrs = [
            "bus",
            "event_bus",
            "message_bus",
            "dispatcher",
            "publisher",
            "event_publisher",
            "kafka",
            "kafka_producer",
            "producer",
        ]

        for attr in forbidden_attrs:
            assert not hasattr(handler, attr), (
                f"HttpRestHandler should not have '{attr}' attribute - "
                f"handlers must not have bus access"
            )
            assert not hasattr(handler, f"_{attr}"), (
                f"HttpRestHandler should not have '_{attr}' attribute - "
                f"handlers must not have bus access"
            )

    def test_execute_returns_model_handler_output(self) -> None:
        """execute() returns ModelHandlerOutput, not a publish action."""
        sig = inspect.signature(HttpRestHandler.execute)

        # Check return annotation
        return_annotation = sig.return_annotation
        assert return_annotation != inspect.Signature.empty, (
            "execute() should have a return type annotation"
        )

        # The return type should be ModelHandlerOutput, not None/void
        annotation_str = str(return_annotation)
        assert "ModelHandlerOutput" in annotation_str, (
            f"execute() should return ModelHandlerOutput, got: {annotation_str}"
        )

    def test_no_publish_methods_exist(self) -> None:
        """Handler has no publish/emit/dispatch methods."""
        handler = HttpRestHandler()

        forbidden_methods = [
            "publish",
            "emit",
            "dispatch",
            "send_event",
            "send_message",
            "produce",
        ]

        for method_name in forbidden_methods:
            assert not hasattr(handler, method_name), (
                f"HttpRestHandler should not have '{method_name}' method - "
                f"handlers must not publish directly"
            )

    def test_handler_has_no_messaging_infrastructure_attributes(self) -> None:
        """Handler has no messaging/bus-related internal state."""
        handler = HttpRestHandler()

        internal_attrs = [attr for attr in dir(handler) if attr.startswith("_")]
        handler_attrs = [attr for attr in internal_attrs if not attr.startswith("__")]

        # Check that no internal attributes are bus/messaging-related
        # Using specific patterns to avoid false positives (e.g., _event_loop, _message_format)
        bus_keywords = [
            "bus",  # General bus pattern
            "kafka",  # Kafka-specific
            "dispatch",  # Dispatcher pattern
            "publish",  # Publishing pattern
            "producer",  # Producer pattern
            "event_bus",  # Specific event bus
            "event_publisher",  # Specific event publisher
            "message_bus",  # Specific message bus
            "message_broker",  # Specific message broker
        ]
        for attr in handler_attrs:
            if callable(getattr(handler, attr, None)):
                continue
            attr_lower = attr.lower()
            for keyword in bus_keywords:
                assert keyword not in attr_lower, (
                    f"Found bus-related attribute '{attr}' - "
                    f"handler must not have messaging infrastructure"
                )


# ============================================================================
# Test HandlerNodeIntrospected Bus Isolation
# ============================================================================


class TestHandlerNodeIntrospectedBusIsolation:
    """Prove HandlerNodeIntrospected cannot access the event bus structurally."""

    def test_constructor_has_no_bus_parameter(self) -> None:
        """Constructor accepts only domain dependencies, not bus/dispatcher."""
        sig = inspect.signature(HandlerNodeIntrospected.__init__)
        params = list(sig.parameters.keys())

        # Expected parameters (domain-specific, not messaging)
        expected = {
            "self",
            "projection_reader",
            "projector",
            "ack_timeout_seconds",
            "consul_handler",
        }
        actual = set(params)

        assert actual == expected, (
            f"HandlerNodeIntrospected.__init__ has unexpected parameters.\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )

        # Explicitly verify no bus-related parameters
        forbidden_param_names = [
            "bus",
            "event_bus",
            "message_bus",
            "dispatcher",
            "publisher",
            "kafka",
            "producer",
        ]

        for forbidden in forbidden_param_names:
            assert forbidden not in params, (
                f"HandlerNodeIntrospected should not accept '{forbidden}' parameter - "
                f"handlers must not have bus access"
            )

    def test_no_bus_attribute_after_instantiation(self) -> None:
        """Handler instance has no bus-related attributes."""
        # Create with mock dependencies
        mock_reader = MagicMock()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        # Check for common bus/event publishing attribute names
        forbidden_attrs = [
            "bus",
            "event_bus",
            "message_bus",
            "dispatcher",
            "publisher",
            "event_publisher",
            "kafka",
            "kafka_producer",
            "producer",
        ]

        for attr in forbidden_attrs:
            assert not hasattr(handler, attr), (
                f"HandlerNodeIntrospected should not have '{attr}' attribute - "
                f"handlers must not have bus access"
            )
            assert not hasattr(handler, f"_{attr}"), (
                f"HandlerNodeIntrospected should not have '_{attr}' attribute - "
                f"handlers must not have bus access"
            )

    def test_handle_returns_list_of_events(self) -> None:
        """handle() returns list[BaseModel], not a publish action.

        The handler RETURNS events for the orchestrator to publish.
        It does NOT publish them directly.
        """
        sig = inspect.signature(HandlerNodeIntrospected.handle)

        # Check return annotation
        return_annotation = sig.return_annotation
        assert return_annotation != inspect.Signature.empty, (
            "handle() should have a return type annotation"
        )

        # The return type should be list[BaseModel]
        annotation_str = str(return_annotation)
        assert "list" in annotation_str.lower(), (
            f"handle() should return a list, got: {annotation_str}"
        )

    def test_no_publish_methods_exist(self) -> None:
        """Handler has no publish/emit/dispatch methods."""
        mock_reader = MagicMock()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        forbidden_methods = [
            "publish",
            "emit",
            "dispatch",
            "send_event",
            "send_message",
            "produce",
        ]

        for method_name in forbidden_methods:
            assert not hasattr(handler, method_name), (
                f"HandlerNodeIntrospected should not have '{method_name}' method - "
                f"handlers must not publish directly"
            )

    def test_only_domain_dependencies_stored(self) -> None:
        """Handler only stores domain-specific dependencies, not bus."""
        mock_reader = MagicMock()
        mock_projector = MagicMock()
        mock_consul = MagicMock()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
            consul_handler=mock_consul,
            ack_timeout_seconds=60.0,
        )

        # Verify stored attributes are domain-specific
        assert handler._projection_reader is mock_reader
        assert handler._projector is mock_projector
        assert handler._consul_handler is mock_consul
        assert handler._ack_timeout_seconds == 60.0

        # Verify no bus-related attributes exist
        all_attrs = dir(handler)
        bus_keywords = ["bus", "kafka", "dispatch", "publish", "producer"]

        for attr in all_attrs:
            attr_lower = attr.lower()
            for keyword in bus_keywords:
                assert keyword not in attr_lower or attr.startswith("__"), (
                    f"Unexpected bus-related attribute '{attr}' found - "
                    f"handler should only have domain dependencies"
                )


# ============================================================================
# Cross-Handler Verification
# ============================================================================


class TestHandlerNoPublishConstraintCrossValidation:
    """Cross-validate the no-publish constraint across handler types."""

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs"),
        [
            (HttpRestHandler, {}),
            (HandlerNodeIntrospected, {"projection_reader": MagicMock()}),
        ],
    )
    def test_handler_has_no_async_context_bus_access(
        self,
        handler_class: type,
        init_kwargs: dict,
    ) -> None:
        """Handlers don't use async context managers for bus access.

        The pattern `async with self.bus:` or similar is forbidden.
        This test verifies no such pattern exists in the handler methods.
        """
        handler = handler_class(**init_kwargs)

        # Check all async methods for bus-related patterns
        for name in dir(handler):
            if name.startswith("_"):
                continue

            method = getattr(handler, name, None)
            if not callable(method):
                continue

            # Get method source if available
            try:
                source = inspect.getsource(method)
            except (TypeError, OSError):
                continue

            # Check for forbidden patterns in source
            # Note: This is a defensive check; the primary enforcement is through
            # dependency injection (no bus is injected into handlers). This catches
            # common direct-access patterns but won't detect all indirect access.
            forbidden_patterns = [
                "async with self.bus",
                "async with self._bus",
                "async with self.dispatcher",
                "async with self._dispatcher",
                "await self.bus.publish",
                "await self._bus.publish",
                "await self.publish(",
                "await self.emit(",
            ]

            for pattern in forbidden_patterns:
                assert pattern not in source, (
                    f"Handler {handler_class.__name__}.{name} contains "
                    f"forbidden pattern '{pattern}' - handlers must not access bus"
                )

    def test_http_handler_execute_signature_matches_pattern(self) -> None:
        """HttpRestHandler.execute follows the handler pattern.

        Pattern: async def execute(envelope) -> ModelHandlerOutput
        NOT: async def execute(envelope, bus) -> None
        """
        sig = inspect.signature(HttpRestHandler.execute)
        params = list(sig.parameters.keys())

        # Should be (self, envelope) - no bus parameter
        assert params == ["self", "envelope"], (
            f"execute() should take (self, envelope), got: {params}"
        )

    def test_introspection_handler_handle_signature_matches_pattern(self) -> None:
        """HandlerNodeIntrospected.handle follows the handler pattern.

        Pattern: async def handle(event, now, correlation_id) -> list[BaseModel]
        NOT: async def handle(event, bus) -> None
        """
        sig = inspect.signature(HandlerNodeIntrospected.handle)
        params = list(sig.parameters.keys())

        # Should be (self, event, now, correlation_id) - no bus parameter
        expected = ["self", "event", "now", "correlation_id"]
        assert params == expected, f"handle() should take {expected}, got: {params}"


# ============================================================================
# Handler Protocol Compliance
# ============================================================================


class TestHandlerProtocolCompliance:
    """Verify handlers implement the ProtocolHandler protocol interface.

    These tests ensure that handlers conform to the ProtocolHandler protocol
    from omnibase_spi, which defines the expected interface for all handlers.

    Handler Types:
        - Protocol handlers (HttpRestHandler): Full protocol implementation with
          handler_type property, execute() method, and describe() method.
        - Domain handlers (HandlerNodeIntrospected): Domain-specific handlers that
          implement handle() method for event processing. These may not implement
          all ProtocolHandler members.
    """

    def test_http_rest_handler_has_handler_type_property(self) -> None:
        """HttpRestHandler must expose handler_type property per ProtocolHandler."""
        handler = HttpRestHandler()

        # handler_type should be a property or attribute
        assert hasattr(handler, "handler_type"), (
            "HttpRestHandler must have 'handler_type' property "
            "per ProtocolHandler protocol"
        )

        # handler_type should return a string
        handler_type = handler.handler_type
        assert isinstance(handler_type, str), (
            f"HttpRestHandler.handler_type must return str, "
            f"got {type(handler_type).__name__}"
        )

        # handler_type should be non-empty
        assert handler_type, "HttpRestHandler.handler_type must not be empty"

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs"),
        [
            (HttpRestHandler, {}),
            (HandlerNodeIntrospected, {"projection_reader": MagicMock()}),
        ],
    )
    def test_handler_has_execute_method(
        self,
        handler_class: type,
        init_kwargs: dict,
    ) -> None:
        """Handlers must have an execute or handle method."""
        handler = handler_class(**init_kwargs)

        # Should have execute (for HttpRestHandler) or handle (for domain handlers)
        has_execute = hasattr(handler, "execute") and callable(handler.execute)
        has_handle = hasattr(handler, "handle") and callable(handler.handle)

        assert has_execute or has_handle, (
            f"{handler_class.__name__} must have 'execute' or 'handle' method "
            f"per ProtocolHandler protocol"
        )

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs"),
        [
            (HttpRestHandler, {}),
            (HandlerNodeIntrospected, {"projection_reader": MagicMock()}),
        ],
    )
    def test_handler_has_describe_method(
        self,
        handler_class: type,
        init_kwargs: dict,
    ) -> None:
        """Handlers may have describe method for introspection.

        Protocol handlers (HttpRestHandler) implement full ProtocolHandler interface
        including describe(). Domain handlers (HandlerNodeIntrospected) may omit
        describe() as they implement domain-specific handle() instead of execute().
        """
        handler = handler_class(**init_kwargs)

        if hasattr(handler, "describe"):
            # Protocol handlers should have callable describe
            assert callable(handler.describe), (
                f"{handler_class.__name__}.describe must be callable"
            )
        # Domain handlers may omit describe() - this is acceptable as they
        # implement handle() instead of the full ProtocolHandler interface


__all__: list[str] = [
    "TestHttpRestHandlerBusIsolation",
    "TestHandlerNodeIntrospectedBusIsolation",
    "TestHandlerNoPublishConstraintCrossValidation",
    "TestHandlerProtocolCompliance",
]
