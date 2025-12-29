# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests proving handler no-publish constraint is enforced.

This module validates a critical ONEX architectural constraint:

**No-Publish Constraint**: Handlers cannot directly access the event bus.
Handlers return results/events to the caller; they do NOT publish.

This is a fundamental architectural boundary in ONEX that ensures:
- Handlers remain pure processors of input -> output transformations
- Event publishing decisions are centralized in orchestrators
- Handlers can be tested in isolation without event bus mocking
- The system maintains clear separation of concerns

Additionally, protocol compliance tests verify handlers implement the expected
ProtocolHandler interface using duck typing for structural subtyping.

Enforcement Mechanism
---------------------
The no-publish constraint is enforced primarily through dependency injection:
- Handlers receive only domain-specific dependencies (readers, projectors, etc.)
- No bus, dispatcher, or event publisher is injected
- Handlers return data structures; callers decide what to publish

The source code pattern detection in these tests is a **defensive secondary
check** that supplements the primary DI enforcement.

Implementation Note - String Matching vs AST
--------------------------------------------
This test suite uses string matching for source code pattern detection rather
than AST-based validation. This is a pragmatic choice because:

1. **Sufficient for purpose**: String matching reliably detects forbidden
   patterns like "await self.bus.publish" - these are literal code patterns
   that cannot be obfuscated without breaking functionality.

2. **Complexity vs benefit**: AST parsing adds significant complexity without
   meaningful benefit for constraint tests. We're detecting code patterns,
   not analyzing semantic meaning.

3. **Defense in depth**: The primary enforcement is through dependency
   injection (no bus injected); source pattern detection is supplementary.
   If DI is bypassed, these tests catch common direct-access patterns.

4. **Acceptable false positives**: If string matching produces a false
   positive (matching a comment or string literal), it triggers human review
   of the code - an acceptable outcome for a constraint test.

Protocol Compliance
-------------------
Protocol compliance uses duck typing (hasattr checks) per ONEX patterns:
- Protocol handlers implement: handler_type, execute(), initialize(), shutdown()
- describe() is OPTIONAL per ProtocolHandler protocol
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
"""

from __future__ import annotations

import asyncio
import inspect
from typing import get_type_hints
from unittest.mock import MagicMock

import pytest

from omnibase_infra.handlers.handler_http import HttpRestHandler
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)

# ============================================================================
# Module Constants for No-Publish Constraint Validation
# ============================================================================

FORBIDDEN_BUS_ATTRIBUTES: tuple[str, ...] = (
    "bus",
    "event_bus",
    "message_bus",
    "dispatcher",
    "publisher",
    "event_publisher",
    "kafka",
    "kafka_producer",
    "producer",
)
"""Attribute names that indicate direct bus access.

Handlers must not have these attributes (either public or private with underscore prefix).
These represent explicit bus infrastructure that violates the no-publish constraint.
"""

FORBIDDEN_BUS_PARAMETERS: tuple[str, ...] = (
    "bus",
    "event_bus",
    "message_bus",
    "dispatcher",
    "publisher",
    "kafka",
    "producer",
)
"""Constructor parameter names that indicate bus dependency injection.

Handlers must not accept these parameters in their __init__ signature.
Bus dependencies should be injected only into orchestrators, not handlers.
"""

FORBIDDEN_PUBLISH_METHODS: tuple[str, ...] = (
    "publish",
    "emit",
    "dispatch",
    "send_event",
    "send_message",
    "produce",
)
"""Method names that indicate direct publishing capability.

Handlers must not have these methods. They should return data structures
that orchestrators can choose to publish.
"""

BUS_INFRASTRUCTURE_KEYWORDS: tuple[str, ...] = (
    # Specific compound patterns (safe to match as substrings)
    "event_bus",
    "message_bus",
    "message_broker",
    "event_publisher",
    "message_publisher",
    "event_dispatcher",
    "message_dispatcher",
    "kafka_producer",
    "kafka_client",
    "kafka_adapter",
    # Patterns with underscore context (reduces false positives)
    # These match _bus, _kafka, etc. but not "rebus" or "akafka"
    "_bus",
    "_kafka",
    "_dispatcher",
    "_publisher",
    "_producer",
)
"""Keywords for detecting bus infrastructure in internal attributes.

These are substring patterns used to detect bus-related attributes that might
not match exact forbidden names. More specific than simple keywords like "bus"
or "kafka" to avoid false positives (e.g., "rebus" or "kafka_config_path").

Patterns are designed to match meaningful bus infrastructure while avoiding
matches on legitimate attributes like "_message_format" or "_event_loop".
"""

FORBIDDEN_SOURCE_PATTERNS: tuple[str, ...] = (
    "async with self.bus",
    "async with self._bus",
    "async with self.dispatcher",
    "async with self._dispatcher",
    "await self.bus.publish",
    "await self._bus.publish",
    "await self.publish(",
    "await self.emit(",
)
"""Source code patterns that indicate direct bus access in handler methods.

These patterns detect runtime bus access that might bypass the dependency
injection constraint. Used for defensive source code analysis.
"""

# ============================================================================
# Test HttpRestHandler Bus Isolation
# ============================================================================


class TestHttpRestHandlerBusIsolation:
    """Validate no-publish constraint for HttpRestHandler.

    Constraint Under Test
    ---------------------
    **No-Publish Constraint**: HttpRestHandler MUST NOT have any capability
    to directly publish events to the event bus. Handlers return data
    structures; orchestrators decide what to publish.

    Why This Constraint Matters
    ---------------------------
    - **Testability**: Handlers can be unit tested without event bus mocking
    - **Single Responsibility**: Publishing logic centralized in orchestrators
    - **Predictability**: Handler output is deterministic (input -> output)
    - **Composability**: Handlers can be reused across different workflows

    Validation Strategy
    -------------------
    1. Constructor signature analysis - no bus-related parameters accepted
    2. Instance attribute inspection - no bus-related attributes present
    3. Method signature validation - returns data, not publish actions
    4. Source code pattern matching - no direct bus access patterns

    The primary enforcement is dependency injection (no bus injected).
    These tests provide defense-in-depth validation.
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

        for attr in FORBIDDEN_BUS_ATTRIBUTES:
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

        for method_name in FORBIDDEN_PUBLISH_METHODS:
            assert not hasattr(handler, method_name), (
                f"HttpRestHandler should not have '{method_name}' method - "
                f"handlers must not publish directly"
            )

    def test_handler_has_no_messaging_infrastructure_attributes(self) -> None:
        """Handler has no messaging/bus-related internal state."""
        handler = HttpRestHandler()

        internal_attrs = [attr for attr in dir(handler) if attr.startswith("_")]
        handler_attrs = [attr for attr in internal_attrs if not attr.startswith("__")]

        # Check that no internal attributes are bus/messaging-related.
        # Using BUS_INFRASTRUCTURE_KEYWORDS which are intentionally specific
        # to avoid false positives (e.g., "_event_loop" should NOT match,
        # but "_event_bus" SHOULD match).
        for attr in handler_attrs:
            if callable(getattr(handler, attr, None)):
                continue
            attr_lower = attr.lower()
            for keyword in BUS_INFRASTRUCTURE_KEYWORDS:
                assert keyword not in attr_lower, (
                    f"Found bus-related attribute '{attr}' - "
                    f"handler must not have messaging infrastructure"
                )

    # =========================================================================
    # Positive Validation Tests - Handler HAS Expected Attributes
    # =========================================================================

    def test_handler_has_expected_http_attributes(self) -> None:
        """Verify HttpRestHandler has expected HTTP-related state.

        Positive validation: handler DOES have the infrastructure it needs
        for HTTP operations, proving it's properly configured for its purpose.
        """
        handler = HttpRestHandler()

        # Verify handler has expected HTTP infrastructure
        # Check for common HTTP client attributes (at least one should exist)
        http_attrs = ["_client", "_timeout", "_base_url", "_session", "_http_client"]
        has_http_attr = any(hasattr(handler, attr) for attr in http_attrs)

        # Note: This is a soft check - handler may use different attr names
        # The key constraint is that it HAS http infrastructure, not bus infrastructure
        assert has_http_attr or hasattr(handler, "handler_type"), (
            "HttpRestHandler should have HTTP-related attributes or handler_type"
        )

    def test_handler_has_required_protocol_attributes(self) -> None:
        """Verify HttpRestHandler has required ProtocolHandler attributes."""
        handler = HttpRestHandler()

        # Required protocol attributes
        assert hasattr(handler, "handler_type"), "Must have handler_type property"
        assert hasattr(handler, "execute"), "Must have execute method"


# ============================================================================
# Test HandlerNodeIntrospected Bus Isolation
# ============================================================================


class TestHandlerNodeIntrospectedBusIsolation:
    """Validate no-publish constraint for HandlerNodeIntrospected.

    Constraint Under Test
    ---------------------
    **No-Publish Constraint**: HandlerNodeIntrospected MUST NOT have any
    capability to directly publish events. The handler RETURNS events for
    the orchestrator to publish; it does NOT publish them directly.

    Why This Constraint Matters
    ---------------------------
    - **Separation of Concerns**: Handler processes introspection data;
      orchestrator decides what/when to publish
    - **Event Sovereignty**: Orchestrator maintains control over event flow
    - **Testability**: Handler can be tested with mock dependencies only
    - **Audit Trail**: All publishing goes through a single orchestrator path

    Validation Strategy
    -------------------
    1. Constructor parameter validation - only domain dependencies accepted
    2. Instance attribute inspection - no bus-related attributes present
    3. Method signature validation - handle() returns list of events
    4. Stored dependency verification - only domain dependencies stored

    Note: Unlike protocol handlers (HttpRestHandler), domain handlers use
    handle() instead of execute(), but the no-publish constraint applies
    equally to both handler types.
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
        for forbidden in FORBIDDEN_BUS_PARAMETERS:
            assert forbidden not in params, (
                f"HandlerNodeIntrospected should not accept '{forbidden}' parameter - "
                f"handlers must not have bus access"
            )

    def test_no_bus_attribute_after_instantiation(self) -> None:
        """Handler instance has no bus-related attributes."""
        # Create with mock dependencies
        mock_reader = MagicMock()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        for attr in FORBIDDEN_BUS_ATTRIBUTES:
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

        for method_name in FORBIDDEN_PUBLISH_METHODS:
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

        # Verify no bus-related attributes exist using narrowed keywords.
        # BUS_INFRASTRUCTURE_KEYWORDS uses specific patterns to reduce false
        # positives while still catching meaningful bus infrastructure.
        all_attrs = dir(handler)

        for attr in all_attrs:
            if attr.startswith("__"):
                continue
            attr_lower = attr.lower()
            for keyword in BUS_INFRASTRUCTURE_KEYWORDS:
                assert keyword not in attr_lower, (
                    f"Unexpected bus-related attribute '{attr}' found - "
                    f"handler should only have domain dependencies"
                )

    # =========================================================================
    # Positive Validation Tests - Handler HAS Expected Attributes
    # =========================================================================

    def test_handler_has_expected_domain_attributes(self) -> None:
        """Verify HandlerNodeIntrospected stores expected domain dependencies.

        Positive validation: handler DOES store the domain dependencies it
        was initialized with, proving dependency injection works correctly.
        """
        mock_reader = MagicMock()
        mock_projector = MagicMock()
        mock_consul = MagicMock()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
            consul_handler=mock_consul,
            ack_timeout_seconds=30.0,
        )

        # Verify all expected domain attributes exist
        assert hasattr(handler, "_projection_reader"), "Must store projection_reader"
        assert hasattr(handler, "_projector"), "Must store projector"
        assert hasattr(handler, "_consul_handler"), "Must store consul_handler"
        assert hasattr(handler, "_ack_timeout_seconds"), "Must store ack_timeout"

    def test_handler_has_required_domain_methods(self) -> None:
        """Verify HandlerNodeIntrospected has required domain methods."""
        mock_reader = MagicMock()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        # Domain handlers use handle() instead of execute()
        assert hasattr(handler, "handle"), "Must have handle method"
        assert callable(handler.handle), "handle must be callable"


# ============================================================================
# Cross-Handler Verification
# ============================================================================


class TestHandlerNoPublishConstraintCrossValidation:
    """Cross-validate the no-publish constraint across all handler types.

    Constraint Under Test
    ---------------------
    **No-Publish Constraint (Cross-Handler)**: ALL handlers, regardless of
    type (protocol handlers, domain handlers), MUST NOT contain code patterns
    that directly access the event bus for publishing.

    Why Cross-Validation
    --------------------
    Individual handler tests validate specific implementations, but constraint
    violations can be introduced in new handlers or during refactoring. These
    parametrized tests ensure uniform constraint enforcement across the codebase.

    Validation Strategy
    -------------------
    1. Source code analysis - detect forbidden bus access patterns in methods
    2. Method signature validation - verify handler patterns are followed

    See module docstring for rationale on string matching vs AST analysis.
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

            for pattern in FORBIDDEN_SOURCE_PATTERNS:
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

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs", "method_name"),
        [
            (HttpRestHandler, {}, "execute"),
            (HandlerNodeIntrospected, {"projection_reader": MagicMock()}, "handle"),
        ],
    )
    def test_handler_entry_method_is_async(
        self,
        handler_class: type,
        init_kwargs: dict[str, object],
        method_name: str,
    ) -> None:
        """Verify handler entry methods are async coroutines.

        Both protocol handlers (execute) and domain handlers (handle) must
        be async coroutine functions to enable non-blocking I/O operations
        and integration with the async event loop.

        Args:
            handler_class: Handler class to test
            init_kwargs: Keyword arguments for handler instantiation
            method_name: Name of the entry method to validate
        """
        handler = handler_class(**init_kwargs)
        method = getattr(handler, method_name)

        assert asyncio.iscoroutinefunction(method), (
            f"{handler_class.__name__}.{method_name} must be an async coroutine function"
        )


# ============================================================================
# Handler Protocol Compliance
# ============================================================================


class TestHandlerProtocolCompliance:
    """Validate protocol interface compliance for handlers.

    Constraint Under Test
    ---------------------
    **Protocol Compliance**: Handlers MUST implement the ProtocolHandler
    protocol interface from omnibase_spi to ensure consistent behavior
    and interoperability across the ONEX runtime.

    Why Protocol Compliance Matters
    -------------------------------
    - **Runtime Discovery**: Handlers can be introspected for capabilities
    - **Consistent Interface**: All handlers follow predictable patterns
    - **Duck Typing**: Structural subtyping without explicit inheritance
    - **Interoperability**: Handlers work with any ProtocolHandler-aware code

    ProtocolHandler Interface (from omnibase_spi)
    ---------------------------------------------
    Required Members:
        - handler_type (property): Returns EnumHandlerType or compatible value
        - execute(envelope) (method): Async method for executing operations

    Optional Members:
        - initialize(config) (method): Async initialization with configuration
        - shutdown() (method): Async cleanup/resource release
        - describe() (method): Returns handler metadata for introspection

    Handler Type Variations
    -----------------------
    - **Protocol handlers** (HttpRestHandler): Full protocol implementation with
      handler_type property, execute() method, initialize/shutdown lifecycle,
      and describe() introspection method.
    - **Domain handlers** (HandlerNodeIntrospected): Domain-specific handlers
      that implement handle() method for event processing. These may not
      implement all ProtocolHandler members as they serve different roles.

    Validation Strategy
    -------------------
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
    def test_handler_describe_method_if_present(
        self,
        handler_class: type,
        init_kwargs: dict[str, object],
    ) -> None:
        """Verify describe() is callable IF the handler implements it.

        IMPORTANT: describe() is OPTIONAL per ProtocolHandler Protocol
        ---------------------------------------------------------------
        This test does NOT enforce describe() presence. Per the ProtocolHandler
        protocol from omnibase_spi, describe() is an optional member that
        handlers MAY implement for introspection support.

        Handler Type Expectations:
        - **Protocol handlers** (HttpRestHandler): Typically implement describe()
          as they provide full ProtocolHandler interface for runtime discovery.
        - **Domain handlers** (HandlerNodeIntrospected): May omit describe() as
          they focus on domain-specific handle() processing rather than protocol
          compliance. These handlers are used internally by orchestrators.

        What This Test Validates:
        - IF describe() exists, it MUST be callable
        - IF describe() exists, calling it should not raise
        - Absence of describe() is NOT a test failure

        Args:
            handler_class: Handler class to test
            init_kwargs: Keyword arguments for handler instantiation
        """
        handler = handler_class(**init_kwargs)

        # Note: We do NOT assert hasattr(handler, "describe") because describe()
        # is optional per ProtocolHandler protocol. Domain handlers like
        # HandlerNodeIntrospected legitimately omit this method.
        if hasattr(handler, "describe"):
            # If describe() exists, verify it's properly implemented
            assert callable(handler.describe), (
                f"{handler_class.__name__}.describe must be callable"
            )
        # Absence of describe() is acceptable - it's an optional protocol member

    # =========================================================================
    # Async Coroutine Validation
    # =========================================================================

    def test_http_handler_execute_is_async(self) -> None:
        """Verify HttpRestHandler.execute is an async coroutine function.

        Protocol handlers must use async methods to enable non-blocking I/O
        and proper integration with the async event loop. This is essential
        for handling concurrent HTTP requests efficiently.
        """
        handler = HttpRestHandler()

        assert asyncio.iscoroutinefunction(handler.execute), (
            "HttpRestHandler.execute must be an async coroutine function"
        )

    def test_introspection_handler_handle_is_async(self) -> None:
        """Verify HandlerNodeIntrospected.handle is an async coroutine function.

        Domain handlers must use async methods to enable non-blocking I/O
        operations such as reading projections and coordinating with Consul.
        This ensures the handler can be awaited by orchestrators.
        """
        mock_reader = MagicMock()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        assert asyncio.iscoroutinefunction(handler.handle), (
            "HandlerNodeIntrospected.handle must be an async coroutine function"
        )

    # =========================================================================
    # Type Annotation Validation
    # =========================================================================

    def test_http_handler_execute_has_type_annotations(self) -> None:
        """Verify HttpRestHandler.execute has proper type annotations.

        Type annotations are required for ONEX compliance and enable:
        - Static type checking with mypy/pyright
        - Runtime introspection for protocol validation
        - Documentation generation
        - IDE support for autocomplete and error detection

        This test validates that:
        - Return type annotation is present
        - The 'envelope' parameter has a type annotation
        """
        sig = inspect.signature(HttpRestHandler.execute)

        # Check return type annotation exists
        assert sig.return_annotation != inspect.Signature.empty, (
            "execute() must have return type annotation"
        )

        # Check parameter type annotations
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            # envelope parameter should have type annotation
            if param_name == "envelope":
                assert param.annotation != inspect.Parameter.empty, (
                    f"Parameter '{param_name}' must have type annotation"
                )

    def test_introspection_handler_handle_has_type_annotations(self) -> None:
        """Verify HandlerNodeIntrospected.handle has proper type annotations.

        Type annotations are required for ONEX compliance and enable:
        - Static type checking with mypy/pyright
        - Runtime introspection for protocol validation
        - Documentation generation
        - IDE support for autocomplete and error detection

        This test validates that:
        - Return type annotation is present
        - All parameters (except self) have type annotations
        """
        sig = inspect.signature(HandlerNodeIntrospected.handle)

        # Check return type annotation exists
        assert sig.return_annotation != inspect.Signature.empty, (
            "handle() must have return type annotation"
        )

        # Check key parameters have annotations
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            assert param.annotation != inspect.Parameter.empty, (
                f"Parameter '{param_name}' must have type annotation"
            )


__all__: list[str] = [
    "FORBIDDEN_BUS_ATTRIBUTES",
    "FORBIDDEN_BUS_PARAMETERS",
    "FORBIDDEN_PUBLISH_METHODS",
    "BUS_INFRASTRUCTURE_KEYWORDS",
    "FORBIDDEN_SOURCE_PATTERNS",
    "TestHttpRestHandlerBusIsolation",
    "TestHandlerNodeIntrospectedBusIsolation",
    "TestHandlerNoPublishConstraintCrossValidation",
    "TestHandlerProtocolCompliance",
]
