# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests proving handler no-publish constraint is enforced.

This module validates two critical ONEX handler constraints:

1. **No-Publish Constraint**: Handlers cannot directly access the event bus.
   Handlers return results/events to the caller; they do NOT publish.

2. **Protocol Compliance**: Handlers implement the ProtocolHandler protocol
   from omnibase_spi, ensuring consistent interfaces across all handler types.

The no-publish constraint is enforced by dependency injection patterns:
- Handlers receive only domain-specific dependencies (readers, projectors, etc.)
- No bus, dispatcher, or event publisher is injected
- Handlers return data structures; callers decide what to publish

Protocol compliance uses duck typing (hasattr checks) per ONEX patterns:
- Protocol handlers implement: handler_type, execute(), initialize(), shutdown()
- describe() is optional but recommended for introspection support
- Domain handlers may implement handle() instead of execute()

Related Tickets:
    - OMN-1094: Test Coverage - Existing No-Publish Constraint
    - OMN-888: Registration Orchestrator (established pattern)

Test Strategy:
    These tests use introspection to prove that handlers CANNOT access
    the event bus because:
    1. No bus dependency is accepted in their constructor signatures
    2. No bus-related attributes exist on handler instances
    3. Return types are data structures, not publish operations

    Additionally, protocol compliance tests verify handlers implement
    the expected ProtocolHandler interface using duck typing to support
    structural subtyping without requiring explicit inheritance.
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
    """Prove HttpRestHandler cannot access the event bus structurally.

    These tests verify that HttpRestHandler is architecturally isolated from
    the event bus infrastructure. The isolation is enforced through:

    1. Constructor signature analysis - no bus-related parameters accepted
    2. Instance attribute inspection - no bus-related attributes present
    3. Method signature validation - returns data, not publish actions
    4. Source code pattern matching - no direct bus access patterns

    This isolation is a core ONEX architectural constraint: handlers process
    requests and return results; orchestrators handle event publishing.
    """

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
    """Prove HandlerNodeIntrospected cannot access the event bus structurally.

    HandlerNodeIntrospected is a domain-specific handler for processing node
    introspection events. These tests verify it maintains bus isolation through:

    1. Constructor parameter validation - only domain dependencies accepted
    2. Instance attribute inspection - no bus-related attributes present
    3. Method signature validation - handle() returns list of events
    4. Stored dependency verification - only domain dependencies stored

    Unlike protocol handlers (HttpRestHandler), domain handlers use handle()
    instead of execute(), but the no-publish constraint applies equally.
    """

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
    """Cross-validate the no-publish constraint across handler types.

    These tests apply parametrized validation across multiple handler types
    to ensure the no-publish constraint is consistently enforced. Tests use:

    1. Source code analysis - detect forbidden bus access patterns
    2. Method signature validation - verify handler patterns are followed

    This cross-cutting validation catches constraint violations that might
    slip through type-specific tests by applying uniform checks.
    """

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
        init_kwargs: dict[str, object],
    ) -> None:
        """Handlers don't use async context managers for bus access.

        Verifies handlers don't contain bus access patterns in their source code.
        The pattern `async with self.bus:` or similar is forbidden.

        Note: This is a defensive check; the primary enforcement is through
        dependency injection (no bus is injected into handlers). This catches
        common direct-access patterns but won't detect all indirect access.

        Args:
            handler_class: Handler class to test
            init_kwargs: Keyword arguments for handler instantiation
        """
        handler = handler_class(**init_kwargs)

        for name in dir(handler):
            if name.startswith("_"):
                continue

            method = getattr(handler, name, None)
            if not callable(method):
                continue

            try:
                source = inspect.getsource(method)
            except (TypeError, OSError):
                continue

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
        assert params == [
            "self",
            "envelope",
        ], f"execute() should take (self, envelope), got: {params}"

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

    Per ONEX patterns, we use duck typing (hasattr checks) rather than isinstance
    to support structural subtyping. This allows handlers to implement the
    protocol without explicit inheritance, following Python's protocol pattern.

    ProtocolHandler Interface (from omnibase_spi):
        Required Members:
            - handler_type (property): Returns EnumHandlerType or compatible value
            - execute(envelope) (method): Async method for executing operations

        Optional Members:
            - initialize(config) (method): Async initialization with configuration
            - shutdown() (method): Async cleanup/resource release
            - describe() (method): Returns handler metadata for introspection

    Handler Types:
        - Protocol handlers (HttpRestHandler): Full protocol implementation with
          handler_type property, execute() method, initialize/shutdown lifecycle,
          and describe() introspection method.
        - Domain handlers (HandlerNodeIntrospected): Domain-specific handlers that
          implement handle() method for event processing. These may not implement
          all ProtocolHandler members as they serve different architectural roles.

    Test Strategy:
        Duck typing verification using hasattr() to check for required interface
        members without requiring explicit protocol inheritance. This aligns with
        ONEX's structural subtyping approach and Python's Protocol pattern.
    """

    def test_http_rest_handler_implements_protocol_handler_interface(self) -> None:
        """Verify HttpRestHandler implements ProtocolHandler using duck typing.

        This comprehensive test verifies that HttpRestHandler implements all
        required and optional members of the ProtocolHandler protocol from
        omnibase_spi. Per ONEX patterns, we use hasattr() for duck typing
        rather than isinstance() to support structural subtyping.

        Required Protocol Members Verified:
            - handler_type: Property returning handler type identifier
            - execute: Async method for processing envelopes

        Optional Protocol Members Verified:
            - initialize: Async method for handler initialization
            - shutdown: Async method for cleanup
            - describe: Method for returning handler metadata
        """
        handler = HttpRestHandler()

        # =====================================================================
        # Required: handler_type property
        # =====================================================================
        assert hasattr(handler, "handler_type"), (
            "HttpRestHandler must have 'handler_type' property per ProtocolHandler"
        )

        # handler_type must be accessible (not raise on access)
        handler_type = handler.handler_type
        assert handler_type is not None, "HttpRestHandler.handler_type must not be None"

        # handler_type.value should be a non-empty string (EnumHandlerType pattern)
        if hasattr(handler_type, "value"):
            assert isinstance(handler_type.value, str), (
                f"HttpRestHandler.handler_type.value must be str, "
                f"got {type(handler_type.value).__name__}"
            )
            assert handler_type.value, (
                "HttpRestHandler.handler_type.value must not be empty"
            )

        # =====================================================================
        # Required: execute method
        # =====================================================================
        assert hasattr(handler, "execute"), (
            "HttpRestHandler must have 'execute' method per ProtocolHandler"
        )
        assert callable(handler.execute), "HttpRestHandler.execute must be callable"

        # Verify execute signature takes envelope parameter
        sig = inspect.signature(handler.execute)
        param_names = list(sig.parameters.keys())
        assert "envelope" in param_names, (
            f"HttpRestHandler.execute must accept 'envelope' parameter, "
            f"has parameters: {param_names}"
        )

        # =====================================================================
        # Optional: initialize method (recommended for protocol handlers)
        # =====================================================================
        assert hasattr(handler, "initialize"), (
            "HttpRestHandler should have 'initialize' method for lifecycle management"
        )
        if hasattr(handler, "initialize"):
            assert callable(handler.initialize), (
                "HttpRestHandler.initialize must be callable"
            )

        # =====================================================================
        # Optional: shutdown method (recommended for protocol handlers)
        # =====================================================================
        assert hasattr(handler, "shutdown"), (
            "HttpRestHandler should have 'shutdown' method for cleanup"
        )
        if hasattr(handler, "shutdown"):
            assert callable(handler.shutdown), (
                "HttpRestHandler.shutdown must be callable"
            )

        # =====================================================================
        # Optional: describe method (recommended for introspection)
        # =====================================================================
        assert hasattr(handler, "describe"), (
            "HttpRestHandler should have 'describe' method for introspection"
        )
        if hasattr(handler, "describe"):
            assert callable(handler.describe), (
                "HttpRestHandler.describe must be callable"
            )
            # describe() should return a dict (metadata)
            description = handler.describe()
            assert isinstance(description, dict), (
                f"HttpRestHandler.describe() must return dict, "
                f"got {type(description).__name__}"
            )

    def test_http_rest_handler_has_handler_type_property(self) -> None:
        """HttpRestHandler must expose handler_type property per ProtocolHandler.

        This test specifically validates the handler_type property returns a
        string-compatible value, which is the primary identifier used for
        handler routing in the runtime host.
        """
        handler = HttpRestHandler()

        # handler_type should be a property or attribute
        assert hasattr(handler, "handler_type"), (
            "HttpRestHandler must have 'handler_type' property "
            "per ProtocolHandler protocol"
        )

        # handler_type should return a string-compatible value
        handler_type = handler.handler_type
        # Check for EnumHandlerType pattern (has .value) or direct string
        if hasattr(handler_type, "value"):
            assert isinstance(handler_type.value, str), (
                f"HttpRestHandler.handler_type.value must be str, "
                f"got {type(handler_type.value).__name__}"
            )
        else:
            assert isinstance(handler_type, str), (
                f"HttpRestHandler.handler_type must return str, "
                f"got {type(handler_type).__name__}"
            )

        # handler_type should be non-empty
        type_value = (
            handler_type.value if hasattr(handler_type, "value") else handler_type
        )
        assert type_value, "HttpRestHandler.handler_type must not be empty"

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
        init_kwargs: dict[str, object],
    ) -> None:
        """Handlers must have an execute or handle method.

        Protocol handlers (like HttpRestHandler) implement execute() for
        envelope-based operations. Domain handlers (like HandlerNodeIntrospected)
        may implement handle() for event-specific processing.

        Both patterns satisfy the handler contract requirement of having a
        callable entry point for processing requests.

        Args:
            handler_class: Handler class to test
            init_kwargs: Keyword arguments for handler instantiation
        """
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
        init_kwargs: dict[str, object],
    ) -> None:
        """Handlers may have describe method for introspection.

        The describe() method is optional per ProtocolHandler but recommended
        for protocol handlers that need to expose metadata for runtime discovery
        and introspection capabilities.

        Protocol handlers (HttpRestHandler) implement full ProtocolHandler interface
        including describe(). Domain handlers (HandlerNodeIntrospected) may omit
        describe() as they implement domain-specific handle() instead of execute().

        Args:
            handler_class: Handler class to test
            init_kwargs: Keyword arguments for handler instantiation
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
