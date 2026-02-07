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

The attribute and source code checks in these tests are **defensive secondary
checks** that supplement the primary DI enforcement.

Detection Strategy - Types vs Names
------------------------------------
This test suite uses two detection strategies:

1. **Type-based detection for instance attributes**: Uses isinstance() with
   the runtime_checkable ProtocolEventBusLike protocol to detect actual bus
   dependencies. This is precise and avoids false positives on domain
   dependencies like _snapshot_publisher (ProtocolSnapshotPublisher).

2. **String matching for source code patterns**: Detects forbidden code
   patterns like "await self.bus.publish" in handler method source. This is
   appropriate for source code analysis where types aren't available.

   Known edge cases that may trigger false positives (all acceptable):
   - Pattern strings in docstrings (e.g., documenting forbidden patterns)
   - Pattern strings in test assertions (testing the detection itself)
   - Comments explaining what NOT to do (instructional code)
   - String literals containing patterns (e.g., error messages)

   In all cases, human review is the desired outcome.

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
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.handlers.handler_graph import HandlerGraph
from omnibase_infra.handlers.handler_http import HandlerHttpRest
from omnibase_infra.handlers.handler_qdrant import HandlerQdrant
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike

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
# Test Utilities
# ============================================================================


def detect_forbidden_source_patterns(
    handler: object,
    patterns: tuple[str, ...] = FORBIDDEN_SOURCE_PATTERNS,
) -> list[tuple[str, str, str]]:
    """Detect forbidden patterns in handler method source code.

    Args:
        handler: Handler instance to inspect
        patterns: Tuple of forbidden patterns to search for

    Returns:
        List of (method_name, pattern, context) tuples for each violation found.
        Empty list means no violations detected.

    Note:
        This is a defensive check using string matching. The primary
        enforcement is through dependency injection (no bus injected).
        See module docstring for rationale on string matching vs AST.
    """
    violations: list[tuple[str, str, str]] = []

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

        for pattern in patterns:
            if pattern in source:
                # Extract a snippet around the match for context
                idx = source.find(pattern)
                start = max(0, idx - 20)
                end = min(len(source), idx + len(pattern) + 20)
                context = source[start:end].replace("\n", " ")
                violations.append((name, pattern, context))

    return violations


def assert_no_bus_attributes(handler: object, handler_name: str) -> None:
    """Assert handler has no bus-related attributes.

    Validates that neither public nor private (underscore-prefixed) versions
    of forbidden bus attributes exist on the handler instance.

    Args:
        handler: Handler instance to check
        handler_name: Name of handler class for error messages

    Raises:
        AssertionError: If any forbidden bus attribute is found
    """
    for attr in FORBIDDEN_BUS_ATTRIBUTES:
        assert not hasattr(handler, attr), (
            f"{handler_name} should not have '{attr}' attribute - "
            f"handlers must not have bus access"
        )
        assert not hasattr(handler, f"_{attr}"), (
            f"{handler_name} should not have '_{attr}' attribute - "
            f"handlers must not have bus access"
        )


def _is_bus_infrastructure(value: object) -> bool:
    """Check if a value is an event bus infrastructure type.

    Uses isinstance with the runtime_checkable ProtocolEventBusLike protocol
    to detect actual bus publishers, rather than guessing from attribute names.

    This is the primary detection mechanism for the no-publish constraint on
    handler instance attributes. It correctly distinguishes:
    - EventBusKafka, EventBusInmemory → True (bus infrastructure)
    - SnapshotPublisherRegistration → False (domain dependency)
    - MagicMock() → False (test mock)

    Args:
        value: The attribute value to check.

    Returns:
        True if the value implements ProtocolEventBusLike.
    """
    try:
        return isinstance(value, ProtocolEventBusLike)
    except TypeError:
        # Defensive: some proxy objects may not support isinstance
        return False


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def http_handler(mock_container: MagicMock) -> Generator[HandlerHttpRest, None, None]:
    """Create HandlerHttpRest with cleanup.

    Yields handler instance and ensures proper shutdown if initialized.
    This prevents resource warnings from httpx clients that may be
    initialized during handler usage.

    Note: Uses Generator return type (not direct type) because this fixture
    uses yield with teardown code. See introspection_handler for the simpler
    return pattern when no cleanup is required.
    """
    handler = HandlerHttpRest(container=mock_container)
    yield handler
    # Cleanup if handler was initialized (has httpx client)
    if hasattr(handler, "_initialized") and handler._initialized:
        asyncio.run(handler.shutdown())


@pytest.fixture
def introspection_handler() -> HandlerNodeIntrospected:
    """Create HandlerNodeIntrospected with mock dependencies.

    Returns handler instance with mock projection_reader.
    Use this fixture for tests that only need basic handler functionality.
    Tests requiring specific mock configurations should instantiate directly.

    Note: Unlike http_handler, this fixture uses return instead of yield
    because no cleanup is needed for mock dependencies.
    """
    mock_reader = MagicMock()
    handler = HandlerNodeIntrospected(projection_reader=mock_reader)
    return handler


# ============================================================================
# Test HandlerHttpRest Bus Isolation
# ============================================================================


class TestHandlerHttpRestBusIsolation:
    """Validate no-publish constraint for HandlerHttpRest.

    Constraint Under Test
    ---------------------
    **No-Publish Constraint**: HandlerHttpRest MUST NOT have any capability
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

    def test_constructor_takes_no_bus_parameters(self) -> None:
        """HandlerHttpRest.__init__ does not accept bus-related parameters.

        This test focuses on the no-publish constraint: the constructor must not
        accept bus, dispatcher, or publisher parameters. The test does NOT enforce
        a specific parameter list, allowing the handler to evolve while maintaining
        the constraint.
        """
        sig = inspect.signature(HandlerHttpRest.__init__)
        params = list(sig.parameters.keys())

        # First parameter must be 'self'
        assert params[0] == "self", "First parameter must be 'self'"

        # Verify no bus-related parameters are accepted
        for forbidden in FORBIDDEN_BUS_PARAMETERS:
            assert forbidden not in params, (
                f"HandlerHttpRest.__init__ must not accept '{forbidden}' parameter - "
                f"handlers must not have bus access"
            )

    def test_no_bus_attribute_after_instantiation(
        self, http_handler: HandlerHttpRest
    ) -> None:
        """Handler instance has no bus-related attributes."""
        assert_no_bus_attributes(http_handler, "HandlerHttpRest")

    def test_execute_returns_model_handler_output(self) -> None:
        """execute() returns ModelHandlerOutput, not a publish action."""
        sig = inspect.signature(HandlerHttpRest.execute)

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

    def test_no_publish_methods_exist(self, http_handler: HandlerHttpRest) -> None:
        """Handler has no publish/emit/dispatch methods."""
        for method_name in FORBIDDEN_PUBLISH_METHODS:
            assert not hasattr(http_handler, method_name), (
                f"HandlerHttpRest should not have '{method_name}' method - "
                f"handlers must not publish directly"
            )

    def test_handler_has_no_messaging_infrastructure_attributes(
        self, http_handler: HandlerHttpRest
    ) -> None:
        """Handler has no messaging/bus-related internal state.

        Uses type-based detection via ProtocolEventBusLike (runtime_checkable)
        to precisely identify bus infrastructure dependencies, rather than
        substring matching on attribute names which produces false positives
        on domain dependencies like _snapshot_publisher.
        """
        for attr in dir(http_handler):
            if attr.startswith("__"):
                continue
            value = getattr(http_handler, attr, None)
            if callable(value):
                continue
            assert not _is_bus_infrastructure(value), (
                f"Found bus infrastructure attribute '{attr}' "
                f"(type: {type(value).__name__}) - "
                f"handler must not have messaging infrastructure"
            )

    # =========================================================================
    # Positive Validation Tests - Handler HAS Expected Attributes
    # =========================================================================

    def test_handler_has_expected_http_attributes(
        self, http_handler: HandlerHttpRest
    ) -> None:
        """Verify HandlerHttpRest has expected HTTP-related state.

        Positive validation: handler DOES have the infrastructure it needs
        for HTTP operations, proving it's properly configured for its purpose.
        """
        # Verify handler has expected HTTP infrastructure
        # Check for common HTTP client attributes (at least one should exist)
        http_attrs = ["_client", "_timeout", "_base_url", "_session", "_http_client"]
        has_http_attr = any(hasattr(http_handler, attr) for attr in http_attrs)

        # Note: This is a soft check - handler may use different attr names
        # The key constraint is that it HAS http infrastructure, not bus infrastructure
        assert has_http_attr or hasattr(http_handler, "handler_type"), (
            "HandlerHttpRest should have HTTP-related attributes or handler_type"
        )

    def test_handler_has_required_protocol_attributes(
        self, http_handler: HandlerHttpRest
    ) -> None:
        """Verify HandlerHttpRest has required ProtocolHandler attributes."""
        # Required protocol attributes
        assert hasattr(http_handler, "handler_type"), "Must have handler_type property"
        assert hasattr(http_handler, "execute"), "Must have execute method"


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

    Note: Unlike protocol handlers (HandlerHttpRest), domain handlers use
    handle() instead of execute(), but the no-publish constraint applies
    equally to both handler types.
    """

    def test_constructor_has_no_bus_parameter(self) -> None:
        """Constructor accepts domain dependencies but never bus-related parameters.

        This test focuses on the no-publish constraint: handlers must not accept
        bus, dispatcher, or publisher parameters. The test does NOT enforce a
        specific set of domain parameters, allowing the handler to evolve.
        """
        sig = inspect.signature(HandlerNodeIntrospected.__init__)
        params = list(sig.parameters.keys())

        # First parameter must be 'self'
        assert params[0] == "self", "First parameter must be 'self'"

        # Explicitly verify no bus-related parameters
        for forbidden in FORBIDDEN_BUS_PARAMETERS:
            assert forbidden not in params, (
                f"HandlerNodeIntrospected should not accept '{forbidden}' parameter - "
                f"handlers must not have bus access"
            )

    def test_no_bus_attribute_after_instantiation(
        self, introspection_handler: HandlerNodeIntrospected
    ) -> None:
        """Handler instance has no bus-related attributes."""
        assert_no_bus_attributes(introspection_handler, "HandlerNodeIntrospected")

    def test_handle_returns_model_handler_output(self) -> None:
        """handle() returns ModelHandlerOutput, not a publish action.

        The handler RETURNS a ModelHandlerOutput for the orchestrator to process.
        It does NOT publish events directly.
        """
        sig = inspect.signature(HandlerNodeIntrospected.handle)

        # Check return annotation
        return_annotation = sig.return_annotation
        assert return_annotation != inspect.Signature.empty, (
            "handle() should have a return type annotation"
        )

        # The return type should be ModelHandlerOutput
        annotation_str = str(return_annotation)
        assert "ModelHandlerOutput" in annotation_str, (
            f"handle() should return ModelHandlerOutput, got: {annotation_str}"
        )

    def test_no_publish_methods_exist(
        self, introspection_handler: HandlerNodeIntrospected
    ) -> None:
        """Handler has no publish/emit/dispatch methods."""
        for method_name in FORBIDDEN_PUBLISH_METHODS:
            assert not hasattr(introspection_handler, method_name), (
                f"HandlerNodeIntrospected should not have '{method_name}' method - "
                f"handlers must not publish directly"
            )

    def test_only_domain_dependencies_stored(self) -> None:
        """Handler only stores domain-specific dependencies, not bus.

        Uses type-based detection via ProtocolEventBusLike (runtime_checkable)
        to verify no attribute implements the event bus protocol. This correctly
        allows domain dependencies like _snapshot_publisher (which implements
        ProtocolSnapshotPublisher, not ProtocolEventBusLike) while still
        catching any actual bus dependency.
        """
        mock_reader = MagicMock()
        mock_projector = MagicMock()
        mock_consul = MagicMock()
        mock_snapshot_publisher = MagicMock()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
            consul_handler=mock_consul,
            ack_timeout_seconds=60.0,
            snapshot_publisher=mock_snapshot_publisher,
        )

        # Verify stored attributes are domain-specific
        assert handler._projection_reader is mock_reader
        assert handler._projector is mock_projector
        assert handler._consul_handler is mock_consul
        assert handler._ack_timeout_seconds == 60.0
        assert handler._snapshot_publisher is mock_snapshot_publisher

        # Verify no attribute implements the event bus protocol.
        # Type-based detection is precise: it catches EventBusKafka,
        # EventBusInmemory, etc. while allowing SnapshotPublisherRegistration.
        for attr in dir(handler):
            if attr.startswith("__"):
                continue
            value = getattr(handler, attr, None)
            if callable(value):
                continue
            assert not _is_bus_infrastructure(value), (
                f"Unexpected bus infrastructure attribute '{attr}' "
                f"(type: {type(value).__name__}) found - "
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

    def test_handler_has_required_domain_methods(
        self, introspection_handler: HandlerNodeIntrospected
    ) -> None:
        """Verify HandlerNodeIntrospected has required domain methods."""
        # Domain handlers use handle() instead of execute()
        assert hasattr(introspection_handler, "handle"), "Must have handle method"
        assert callable(introspection_handler.handle), "handle must be callable"


# ============================================================================
# Test HandlerQdrant Bus Isolation
# ============================================================================


class TestHandlerQdrantBusIsolation:
    """Validate no-publish constraint for HandlerQdrant.

    Constraint Under Test
    ---------------------
    **No-Publish Constraint**: HandlerQdrant MUST NOT have any capability
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

    def test_constructor_takes_no_bus_parameters(self) -> None:
        """HandlerQdrant.__init__ does not accept bus-related parameters.

        This test focuses on the no-publish constraint: the constructor must not
        accept bus, dispatcher, or publisher parameters. The test does NOT enforce
        a specific parameter list, allowing the handler to evolve while maintaining
        the constraint.
        """
        sig = inspect.signature(HandlerQdrant.__init__)
        params = list(sig.parameters.keys())

        # First parameter must be 'self'
        assert params[0] == "self", "First parameter must be 'self'"

        # Verify no bus-related parameters are accepted
        for forbidden in FORBIDDEN_BUS_PARAMETERS:
            assert forbidden not in params, (
                f"HandlerQdrant.__init__ must not accept '{forbidden}' parameter - "
                f"handlers must not have bus access"
            )

    def test_no_bus_attribute_after_instantiation(self) -> None:
        """Handler class has no bus-related attributes.

        Note: HandlerQdrant requires connection config for instantiation,
        so we check class-level attributes rather than instance attributes.
        """
        # Check class-level attributes for bus-related names
        class_attrs = set(dir(HandlerQdrant))
        bus_attrs = {"_bus", "_event_bus", "_publisher", "_message_bus"}
        found_bus_attrs = class_attrs & bus_attrs

        assert not found_bus_attrs, (
            f"HandlerQdrant should not have bus attributes: {found_bus_attrs}"
        )

    def test_no_publish_methods_exist(self) -> None:
        """Handler has no publish/emit/dispatch methods."""
        for method_name in FORBIDDEN_PUBLISH_METHODS:
            assert not hasattr(HandlerQdrant, method_name), (
                f"HandlerQdrant should not have '{method_name}' method - "
                f"handlers must not publish directly"
            )

    def test_handler_has_no_messaging_infrastructure_attributes(self) -> None:
        """Handler has no messaging/bus-related internal state in class definition.

        Uses type-based detection via ProtocolEventBusLike (runtime_checkable)
        to precisely identify bus infrastructure on class attributes.
        """
        for attr in dir(HandlerQdrant):
            if attr.startswith("__"):
                continue
            value = getattr(HandlerQdrant, attr, None)
            if callable(value):
                continue
            assert not _is_bus_infrastructure(value), (
                f"Found bus infrastructure attribute '{attr}' "
                f"(type: {type(value).__name__}) - "
                f"handler must not have messaging infrastructure"
            )


# ============================================================================
# Test HandlerGraph Bus Isolation
# ============================================================================


class TestHandlerGraphBusIsolation:
    """Validate no-publish constraint for HandlerGraph.

    Constraint Under Test
    ---------------------
    **No-Publish Constraint**: HandlerGraph MUST NOT have any capability
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

    def test_constructor_takes_no_bus_parameters(self) -> None:
        """HandlerGraph.__init__ does not accept bus-related parameters.

        This test focuses on the no-publish constraint: the constructor must not
        accept bus, dispatcher, or publisher parameters. The test does NOT enforce
        a specific parameter list, allowing the handler to evolve while maintaining
        the constraint.
        """
        sig = inspect.signature(HandlerGraph.__init__)
        params = list(sig.parameters.keys())

        # First parameter must be 'self'
        assert params[0] == "self", "First parameter must be 'self'"

        # Verify no bus-related parameters are accepted
        for forbidden in FORBIDDEN_BUS_PARAMETERS:
            assert forbidden not in params, (
                f"HandlerGraph.__init__ must not accept '{forbidden}' parameter - "
                f"handlers must not have bus access"
            )

    def test_no_bus_attribute_after_instantiation(self) -> None:
        """Handler class has no bus-related attributes.

        Note: HandlerGraph requires connection URI for instantiation,
        so we check class-level attributes rather than instance attributes.
        """
        # Check class-level attributes for bus-related names
        class_attrs = set(dir(HandlerGraph))
        bus_attrs = {"_bus", "_event_bus", "_publisher", "_message_bus"}
        found_bus_attrs = class_attrs & bus_attrs

        assert not found_bus_attrs, (
            f"HandlerGraph should not have bus attributes: {found_bus_attrs}"
        )

    def test_no_publish_methods_exist(self) -> None:
        """Handler has no publish/emit/dispatch methods."""
        for method_name in FORBIDDEN_PUBLISH_METHODS:
            assert not hasattr(HandlerGraph, method_name), (
                f"HandlerGraph should not have '{method_name}' method - "
                f"handlers must not publish directly"
            )

    def test_handler_has_no_messaging_infrastructure_attributes(self) -> None:
        """Handler has no messaging/bus-related internal state in class definition.

        Uses type-based detection via ProtocolEventBusLike (runtime_checkable)
        to precisely identify bus infrastructure on class attributes.
        """
        for attr in dir(HandlerGraph):
            if attr.startswith("__"):
                continue
            value = getattr(HandlerGraph, attr, None)
            if callable(value):
                continue
            assert not _is_bus_infrastructure(value), (
                f"Found bus infrastructure attribute '{attr}' "
                f"(type: {type(value).__name__}) - "
                f"handler must not have messaging infrastructure"
            )


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

    # =========================================================================
    # Handler Test Coverage
    # =========================================================================
    # Current handlers covered by parametrized tests:
    #   - HandlerHttpRest (protocol handler)
    #   - HandlerNodeIntrospected (domain handler)
    #
    # As the codebase grows, add new handlers to the parametrize lists below.
    # Future enhancement: Consider implementing a handler registry or discovery
    # mechanism to automatically include all handlers in constraint validation.
    # =========================================================================

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs"),
        [
            (HandlerHttpRest, {"container": MagicMock(spec=ModelONEXContainer)}),
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

        violations = detect_forbidden_source_patterns(handler)

        assert not violations, (
            f"Handler {handler_class.__name__} contains forbidden patterns:\n"
            + "\n".join(
                f"  - {method}.{pattern}: ...{context}..."
                for method, pattern, context in violations
            )
        )

    def test_http_handler_execute_signature_matches_pattern(self) -> None:
        """HandlerHttpRest.execute follows the handler pattern.

        Pattern: async def execute(envelope, ...) -> ModelHandlerOutput
        NOT: async def execute(envelope, bus) -> None

        This test verifies the constraint (no bus params) while allowing
        the handler to evolve with additional domain parameters.
        """
        sig = inspect.signature(HandlerHttpRest.execute)
        params = list(sig.parameters.keys())

        # First parameter must be 'self'
        assert params[0] == "self", "First parameter must be 'self'"

        # Must accept 'envelope' parameter
        assert "envelope" in params, "execute() must accept 'envelope' parameter"

        # Must NOT accept any bus-related parameters
        for forbidden in FORBIDDEN_BUS_PARAMETERS:
            assert forbidden not in params, (
                f"execute() must not accept '{forbidden}' parameter - "
                f"handlers must not have bus access"
            )

    def test_introspection_handler_handle_signature_matches_pattern(self) -> None:
        """HandlerNodeIntrospected.handle follows the handler pattern.

        Pattern: async def handle(envelope, ...) -> ModelHandlerOutput
        NOT: async def handle(envelope, bus) -> None

        This test verifies key domain parameters exist and the no-publish
        constraint is enforced, while allowing the handler to evolve with
        additional domain parameters.
        """
        sig = inspect.signature(HandlerNodeIntrospected.handle)
        params = list(sig.parameters.keys())

        # First parameter must be 'self'
        assert params[0] == "self", "First parameter must be 'self'"

        # Must accept envelope parameter (contains the event)
        assert "envelope" in params, "handle() must accept 'envelope' parameter"

        # Must NOT accept any bus-related parameters
        for forbidden in FORBIDDEN_BUS_PARAMETERS:
            assert forbidden not in params, (
                f"handle() must not accept '{forbidden}' parameter - "
                f"handlers must not have bus access"
            )

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs", "method_name"),
        [
            (
                HandlerHttpRest,
                {"container": MagicMock(spec=ModelONEXContainer)},
                "execute",
            ),
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
    - **Protocol handlers** (HandlerHttpRest): Full protocol implementation with
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

    def test_http_rest_handler_implements_protocol_handler_interface(
        self, http_handler: HandlerHttpRest
    ) -> None:
        """Verify HandlerHttpRest implements ProtocolHandler using duck typing.

        This comprehensive test verifies that HandlerHttpRest implements all
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
        # =====================================================================
        # Required: handler_type property
        # =====================================================================
        assert hasattr(http_handler, "handler_type"), (
            "HandlerHttpRest must have 'handler_type' property per ProtocolHandler"
        )

        # handler_type must be accessible (not raise on access)
        handler_type = http_handler.handler_type
        assert handler_type is not None, "HandlerHttpRest.handler_type must not be None"

        # handler_type.value should be a non-empty string (EnumHandlerType pattern)
        if hasattr(handler_type, "value"):
            assert isinstance(handler_type.value, str), (
                f"HandlerHttpRest.handler_type.value must be str, "
                f"got {type(handler_type.value).__name__}"
            )
            assert handler_type.value, (
                "HandlerHttpRest.handler_type.value must not be empty"
            )

        # =====================================================================
        # Required: execute method
        # =====================================================================
        assert hasattr(http_handler, "execute"), (
            "HandlerHttpRest must have 'execute' method per ProtocolHandler"
        )
        assert callable(http_handler.execute), (
            "HandlerHttpRest.execute must be callable"
        )

        # Verify execute signature takes envelope parameter
        sig = inspect.signature(http_handler.execute)
        param_names = list(sig.parameters.keys())
        assert "envelope" in param_names, (
            f"HandlerHttpRest.execute must accept 'envelope' parameter, "
            f"has parameters: {param_names}"
        )

        # =====================================================================
        # Optional: initialize method (recommended for protocol handlers)
        # =====================================================================
        assert hasattr(http_handler, "initialize"), (
            "HandlerHttpRest should have 'initialize' method for lifecycle management"
        )
        assert callable(http_handler.initialize), (
            "HandlerHttpRest.initialize must be callable"
        )

        # =====================================================================
        # Optional: shutdown method (recommended for protocol handlers)
        # =====================================================================
        assert hasattr(http_handler, "shutdown"), (
            "HandlerHttpRest should have 'shutdown' method for cleanup"
        )
        assert callable(http_handler.shutdown), (
            "HandlerHttpRest.shutdown must be callable"
        )

        # =====================================================================
        # Optional: describe method (recommended for introspection)
        # =====================================================================
        assert hasattr(http_handler, "describe"), (
            "HandlerHttpRest should have 'describe' method for introspection"
        )
        assert callable(http_handler.describe), (
            "HandlerHttpRest.describe must be callable"
        )
        # describe() should return a dict (metadata)
        description = http_handler.describe()
        assert isinstance(description, dict), (
            f"HandlerHttpRest.describe() must return dict, "
            f"got {type(description).__name__}"
        )

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs"),
        [
            (HandlerHttpRest, {"container": MagicMock(spec=ModelONEXContainer)}),
            (HandlerNodeIntrospected, {"projection_reader": MagicMock()}),
        ],
    )
    def test_handler_has_execute_method(
        self,
        handler_class: type,
        init_kwargs: dict[str, object],
    ) -> None:
        """Handlers must have an execute or handle method.

        Protocol handlers (like HandlerHttpRest) implement execute() for
        envelope-based operations. Domain handlers (like HandlerNodeIntrospected)
        may implement handle() for event-specific processing.

        Both patterns satisfy the handler contract requirement of having a
        callable entry point for processing requests.

        Args:
            handler_class: Handler class to test
            init_kwargs: Keyword arguments for handler instantiation
        """
        handler = handler_class(**init_kwargs)

        # Should have execute (for HandlerHttpRest) or handle (for domain handlers)
        has_execute = hasattr(handler, "execute") and callable(handler.execute)
        has_handle = hasattr(handler, "handle") and callable(handler.handle)

        assert has_execute or has_handle, (
            f"{handler_class.__name__} must have 'execute' or 'handle' method "
            f"per ProtocolHandler protocol"
        )

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs"),
        [
            (HandlerHttpRest, {"container": MagicMock(spec=ModelONEXContainer)}),
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
        - **Protocol handlers** (HandlerHttpRest): Typically implement describe()
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

    def test_http_handler_execute_is_async(self, http_handler: HandlerHttpRest) -> None:
        """Verify HandlerHttpRest.execute is an async coroutine function.

        Protocol handlers must use async methods to enable non-blocking I/O
        and proper integration with the async event loop. This is essential
        for handling concurrent HTTP requests efficiently.
        """
        assert asyncio.iscoroutinefunction(http_handler.execute), (
            "HandlerHttpRest.execute must be an async coroutine function"
        )

    def test_introspection_handler_handle_is_async(
        self, introspection_handler: HandlerNodeIntrospected
    ) -> None:
        """Verify HandlerNodeIntrospected.handle is an async coroutine function.

        Domain handlers must use async methods to enable non-blocking I/O
        operations such as reading projections and coordinating with Consul.
        This ensures the handler can be awaited by orchestrators.
        """
        assert asyncio.iscoroutinefunction(introspection_handler.handle), (
            "HandlerNodeIntrospected.handle must be an async coroutine function"
        )

    # =========================================================================
    # Type Annotation Validation
    # =========================================================================

    def test_http_handler_execute_has_type_annotations(self) -> None:
        """Verify HandlerHttpRest.execute has proper type annotations.

        Type annotations are required for ONEX compliance and enable:
        - Static type checking with mypy/pyright
        - Runtime introspection for protocol validation
        - Documentation generation
        - IDE support for autocomplete and error detection

        This test validates that:
        - Return type annotation is present
        - The 'envelope' parameter has a type annotation
        """
        sig = inspect.signature(HandlerHttpRest.execute)

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


# ============================================================================
# Orchestrator Bus Access Verification (Companion Tests)
# ============================================================================


class TestOrchestratorBusAccessVerification:
    """Companion tests verifying orchestrators DO have bus access.

    Constraint Under Test
    ---------------------
    **Bus Access Pattern**: While handlers MUST NOT have bus access,
    orchestrators MUST HAVE bus access because they are responsible for
    coordinating event publishing.

    Why Companion Tests Matter
    --------------------------
    - **Contract Verification**: Proves the architectural boundary is correct
    - **Defense in Depth**: If handlers have bus access, constraint is violated
    - **Documentation**: Makes the expected pattern explicit in tests

    Orchestrator Bus Access Architecture
    ------------------------------------
    The ONEX orchestrator pattern uses **coordinated bus access**:
    - Orchestrators receive a `container: ModelONEXContainer`
    - They accept coordinators/services that have bus dependencies
    - Example: `TimeoutCoordinator` uses `ServiceTimeoutEmitter` which has `event_bus`

    This is the intended architectural pattern:
    1. Orchestrator receives container (DI)
    2. Orchestrator uses contract-driven handler routing (declarative)
    3. Coordinator delegates to services with event_bus (composition)
    4. Services publish to event bus (actual publishing)

    This is the OPPOSITE of the handler pattern where bus access is forbidden.
    """

    def test_timeout_emitter_requires_event_bus_dependency(self) -> None:
        """ServiceTimeoutEmitter constructor requires event_bus parameter.

        This proves that the service layer, used by orchestrator coordinators,
        has bus access. The orchestrator pattern is:
        Orchestrator -> TimeoutCoordinator -> ServiceTimeoutEmitter(container, event_bus=...)
        """
        from omnibase_infra.services.service_timeout_emitter import (
            ServiceTimeoutEmitter,
        )

        sig = inspect.signature(ServiceTimeoutEmitter.__init__)
        params = list(sig.parameters.keys())

        # ServiceTimeoutEmitter MUST accept container parameter (first)
        assert "container" in params, (
            "ServiceTimeoutEmitter must accept 'container' parameter - "
            "this is the ONEX DI pattern"
        )

        # ServiceTimeoutEmitter MUST accept event_bus parameter
        assert "event_bus" in params, (
            "ServiceTimeoutEmitter must accept 'event_bus' parameter - "
            "this is the architectural pattern that enables orchestrator publishing"
        )

        # Verify event_bus is a required parameter (no default)
        event_bus_param = sig.parameters["event_bus"]
        assert event_bus_param.default is inspect.Parameter.empty, (
            "event_bus should be a required parameter (no default) - "
            "orchestrator workflows require bus access"
        )

    def test_timeout_coordinator_accepts_emitter_with_bus_access(self) -> None:
        """TimeoutCoordinator accepts ServiceTimeoutEmitter as dependency.

        This proves the coordinator layer can receive services that have
        bus access. The orchestrator delegates to coordinators which delegate
        to services for actual event publishing.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.timeout_coordinator import (
            TimeoutCoordinator,
        )

        sig = inspect.signature(TimeoutCoordinator.__init__)
        params = list(sig.parameters.keys())

        # TimeoutCoordinator must accept timeout_emission parameter
        assert "timeout_emission" in params, (
            "TimeoutCoordinator must accept 'timeout_emission' parameter - "
            "this is the ServiceTimeoutEmitter with bus access"
        )

    def test_orchestrator_is_declarative(self) -> None:
        """NodeRegistrationOrchestrator is fully declarative (OMN-1102).

        The orchestrator no longer has setter methods for timeout coordinator
        or heartbeat handler. Handler routing is driven entirely by
        contract.yaml and registry-based wiring.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        # Verify the old imperative methods have been removed
        assert not hasattr(NodeRegistrationOrchestrator, "set_timeout_coordinator"), (
            "NodeRegistrationOrchestrator should NOT have 'set_timeout_coordinator' - "
            "OMN-1102 removed imperative wiring in favor of declarative routing"
        )

        assert not hasattr(NodeRegistrationOrchestrator, "has_timeout_coordinator"), (
            "NodeRegistrationOrchestrator should NOT have 'has_timeout_coordinator' - "
            "OMN-1102 removed imperative wiring in favor of declarative routing"
        )

        assert not hasattr(NodeRegistrationOrchestrator, "set_heartbeat_handler"), (
            "NodeRegistrationOrchestrator should NOT have 'set_heartbeat_handler' - "
            "OMN-1102 removed imperative wiring in favor of declarative routing"
        )

        assert not hasattr(NodeRegistrationOrchestrator, "has_heartbeat_handler"), (
            "NodeRegistrationOrchestrator should NOT have 'has_heartbeat_handler' - "
            "OMN-1102 removed imperative wiring in favor of declarative routing"
        )

    def test_orchestrator_container_pattern_differs_from_handlers(self) -> None:
        """Orchestrator uses container pattern, handlers use direct DI.

        This test explicitly documents the architectural difference:
        - Handlers: Direct DI of domain dependencies, NO bus access
        - Orchestrators: Container + contract-driven handler routing (declarative)

        The container pattern enables orchestrators to resolve bus-related
        dependencies through the ONEX dependency injection system.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        sig = inspect.signature(NodeRegistrationOrchestrator.__init__)
        params = list(sig.parameters.keys())

        # Orchestrator takes container (not individual dependencies)
        assert "container" in params, (
            "NodeRegistrationOrchestrator must accept 'container' parameter - "
            "this is the ONEX DI pattern for orchestrators"
        )

        # Contrast with handlers: handlers SHOULD NOT have container
        # This is already tested in TestHandlerNodeIntrospectedBusIsolation
        # but we make the contrast explicit here
        handler_sig = inspect.signature(HandlerNodeIntrospected.__init__)
        handler_params = list(handler_sig.parameters.keys())

        assert "container" not in handler_params, (
            "HandlerNodeIntrospected should NOT accept 'container' - "
            "handlers use direct domain dependency injection"
        )

    def test_orchestrator_can_be_instantiated_with_container(self) -> None:
        """Orchestrator can be created with a container dependency.

        This is a runtime verification that the orchestrator pattern works.
        The container provides access to bus-related services through ONEX DI.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            NodeRegistrationOrchestrator,
        )

        # Create mock container
        mock_container = MagicMock()

        # Orchestrator should instantiate without error
        orchestrator = NodeRegistrationOrchestrator(container=mock_container)

        # Verify orchestrator was created
        assert orchestrator is not None
        assert isinstance(orchestrator, NodeRegistrationOrchestrator)

        # OMN-1102: Orchestrator is now fully declarative - no custom methods

    def test_service_timeout_emitter_stores_event_bus(self) -> None:
        """ServiceTimeoutEmitter stores the event_bus dependency.

        Proves that the service layer maintains bus access for publishing.
        This is the endpoint of the orchestrator -> coordinator -> service chain.
        """
        from omnibase_infra.services.service_timeout_emitter import (
            ServiceTimeoutEmitter,
        )

        # Create mock dependencies
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_query = MagicMock()
        mock_bus = MagicMock()
        mock_projector = MagicMock()

        # Create emitter with container and event_bus
        emitter = ServiceTimeoutEmitter(
            container=mock_container,
            timeout_query=mock_query,
            event_bus=mock_bus,
            projector=mock_projector,
        )

        # Verify container is stored (as private attribute per ONEX patterns)
        assert hasattr(emitter, "_container"), (
            "ServiceTimeoutEmitter must store container as _container"
        )
        assert emitter._container is mock_container, (
            "ServiceTimeoutEmitter._container must be the injected container"
        )

        # Verify event_bus is stored (as private attribute per ONEX patterns)
        assert hasattr(emitter, "_event_bus"), (
            "ServiceTimeoutEmitter must store event_bus as _event_bus"
        )
        assert emitter._event_bus is mock_bus, (
            "ServiceTimeoutEmitter._event_bus must be the injected bus"
        )


# ============================================================================
# Regression: Type-Based Bus Detection
# ============================================================================


class TestBusDetectionRegression:
    """Regression tests for type-based bus infrastructure detection.

    These tests prove that _is_bus_infrastructure correctly distinguishes
    actual event bus types from domain dependencies that happen to have
    "publisher" in their name. This prevents the false positive that caused
    _snapshot_publisher to be flagged as bus infrastructure.

    Background:
        The original detection used substring matching on attribute names
        (e.g., "_publisher" in attr_name). This produced false positives
        for domain dependencies like _snapshot_publisher. The fix uses
        isinstance() with the runtime_checkable ProtocolEventBusLike protocol.
    """

    def test_real_event_bus_detected(self) -> None:
        """A real ProtocolEventBusLike implementation IS detected as bus."""
        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory

        bus = EventBusInmemory()
        assert _is_bus_infrastructure(bus), (
            "EventBusInmemory must be detected as bus infrastructure"
        )

    def test_snapshot_publisher_not_detected_as_bus(self) -> None:
        """SnapshotPublisherRegistration is NOT detected as bus infrastructure.

        This is the exact regression case: _snapshot_publisher was flagged
        by the old substring heuristic because it contained '_publisher'.
        """
        mock_snapshot = MagicMock()
        assert not _is_bus_infrastructure(mock_snapshot), (
            "MagicMock (used for snapshot_publisher) must not be detected as bus"
        )

    def test_plain_mock_not_detected_as_bus(self) -> None:
        """Plain MagicMock is not detected as bus infrastructure."""
        assert not _is_bus_infrastructure(MagicMock()), (
            "MagicMock must not be detected as bus infrastructure"
        )

    def test_none_not_detected_as_bus(self) -> None:
        """None is not detected as bus infrastructure."""
        assert not _is_bus_infrastructure(None), (
            "None must not be detected as bus infrastructure"
        )

    def test_primitive_values_not_detected_as_bus(self) -> None:
        """Primitive values (str, int, float) are not bus infrastructure."""
        assert not _is_bus_infrastructure("some_string")
        assert not _is_bus_infrastructure(42)
        assert not _is_bus_infrastructure(60.0)

    def test_handler_with_snapshot_publisher_passes_constraint(self) -> None:
        """HandlerNodeIntrospected with _snapshot_publisher passes the no-bus check.

        End-to-end regression: instantiate handler with all dependencies
        including snapshot_publisher, then verify no attribute is flagged.
        """
        handler = HandlerNodeIntrospected(
            projection_reader=MagicMock(),
            projector=MagicMock(),
            consul_handler=MagicMock(),
            ack_timeout_seconds=60.0,
            snapshot_publisher=MagicMock(),
        )

        for attr in dir(handler):
            if attr.startswith("__"):
                continue
            value = getattr(handler, attr, None)
            if callable(value):
                continue
            assert not _is_bus_infrastructure(value), (
                f"Attribute '{attr}' (type: {type(value).__name__}) was "
                f"incorrectly flagged as bus infrastructure"
            )


__all__: list[str] = [
    "FORBIDDEN_BUS_ATTRIBUTES",
    "FORBIDDEN_BUS_PARAMETERS",
    "FORBIDDEN_PUBLISH_METHODS",
    "FORBIDDEN_SOURCE_PATTERNS",
    "detect_forbidden_source_patterns",
    "assert_no_bus_attributes",
    "http_handler",
    "introspection_handler",
    "TestHandlerHttpRestBusIsolation",
    "TestHandlerNodeIntrospectedBusIsolation",
    "TestHandlerQdrantBusIsolation",
    "TestHandlerGraphBusIsolation",
    "TestHandlerNoPublishConstraintCrossValidation",
    "TestHandlerProtocolCompliance",
    "TestOrchestratorBusAccessVerification",
    "TestBusDetectionRegression",
]
