# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for binding resolution dispatch flow.

Tests the full flow: topic -> dispatch -> binding resolution -> handler execution.

This test module verifies end-to-end behavior of declarative operation bindings
integrated with the MessageDispatchEngine. It covers:

- Full dispatch with binding resolution and materialized envelope
- Original envelope immutability guarantees
- Binding resolution error handling (required field missing)
- Dispatchers without bindings receiving original envelope
- Trace ID preservation in materialized envelopes

Related:
    - OMN-1518: Declarative TopicOperationHandler routing
    - Phase 6: Testing (binding resolution dispatch integration)
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.errors import BindingResolutionError
from omnibase_infra.models.bindings import (
    ModelOperationBindingsSubcontract,
    ModelParsedBinding,
)
from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.runtime.service_message_dispatch_engine import MessageDispatchEngine

# =============================================================================
# Mock Payload Models (Prefix with "Mock" to avoid pytest collection)
# =============================================================================


class MockEventPayload(BaseModel):
    """Mock event payload for binding resolution tests."""

    user_id: str
    operation: str = "test.operation"
    data: str | None = None
    optional: str | None = None


class MockNestedProfilePayload(BaseModel):
    """Mock payload with nested structure."""

    user: dict[str, object]
    operation: str = "test.operation"


class MockSimplePayload(BaseModel):
    """Simple payload without optional fields."""

    user_id: str
    operation: str = "test.operation"


# =============================================================================
# Mock Envelope Model
# =============================================================================


class MockEventEnvelope(BaseModel):
    """Mock envelope for binding resolution dispatch tests.

    Mimics ModelEventEnvelope structure for testing binding resolution.
    """

    correlation_id: UUID
    trace_id: UUID | None = None
    span_id: UUID | None = None  # Required by dispatch engine for tracing
    payload: BaseModel

    @property
    def operation(self) -> str:
        """Extract operation from payload if available."""
        if hasattr(self.payload, "operation"):
            return str(self.payload.operation)
        return "unknown"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def dispatch_engine() -> MessageDispatchEngine:
    """Create a fresh dispatch engine for each test."""
    return MessageDispatchEngine()


@pytest.fixture
def operation_bindings() -> ModelOperationBindingsSubcontract:
    """Create test operation bindings with global and operation-specific bindings."""
    return ModelOperationBindingsSubcontract(
        global_bindings=[
            ModelParsedBinding(
                parameter_name="correlation_id",
                source="envelope",
                path_segments=("correlation_id",),
                required=True,
                original_expression="${envelope.correlation_id}",
            ),
        ],
        bindings={
            "test.operation": [
                ModelParsedBinding(
                    parameter_name="user_id",
                    source="payload",
                    path_segments=("user_id",),
                    required=True,
                    original_expression="${payload.user_id}",
                ),
                ModelParsedBinding(
                    parameter_name="optional_field",
                    source="payload",
                    path_segments=("optional",),
                    required=False,
                    default="default_value",
                    original_expression="${payload.optional}",
                ),
            ],
        },
    )


def create_test_route(
    dispatcher_id: str,
    topic_pattern: str = "*.test.events.*",
) -> ModelDispatchRoute:
    """Create a test route for binding resolution tests."""
    return ModelDispatchRoute(
        route_id=f"route-{dispatcher_id}",
        topic_pattern=topic_pattern,
        message_category=EnumMessageCategory.EVENT,
        dispatcher_id=dispatcher_id,
    )


# =============================================================================
# Integration Tests
# =============================================================================


class TestBindingResolutionDispatchFlow:
    """Integration tests for binding resolution in dispatch flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_dispatch_with_bindings(
        self,
        dispatch_engine: MessageDispatchEngine,
        operation_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """Full flow: register -> dispatch -> binding resolution -> handler.

        This test verifies that:
        1. Dispatcher is registered with operation bindings
        2. Dispatch resolves bindings from envelope/payload
        3. Handler receives materialized envelope with __bindings namespace
        4. Global bindings (correlation_id) and operation bindings (user_id) are resolved
        5. Optional bindings with defaults are applied when field is missing
        """
        received_envelopes: list[object] = []

        async def test_handler(envelope: object, context: object | None = None) -> None:
            """Test handler that captures received envelope."""
            received_envelopes.append(envelope)

        # Register dispatcher with bindings
        dispatch_engine.register_dispatcher(
            dispatcher_id="test-dispatcher",
            dispatcher=test_handler,
            category=EnumMessageCategory.EVENT,
            message_types=None,  # Match all message types
            node_kind=EnumNodeKind.EFFECT,
            operation_bindings=operation_bindings,
        )

        # Register route to connect topic to dispatcher
        dispatch_engine.register_route(create_test_route("test-dispatcher"))

        dispatch_engine.freeze()

        # Create test envelope with required fields
        correlation_id = uuid4()
        trace_id = uuid4()
        test_envelope = MockEventEnvelope(
            correlation_id=correlation_id,
            trace_id=trace_id,
            payload=MockEventPayload(
                user_id="user-123",
                data="test_data",
                operation="test.operation",
            ),
        )

        # Dispatch
        await dispatch_engine.dispatch(
            topic="dev.test.events.v1",
            envelope=test_envelope,
        )

        # Verify handler was called
        assert len(received_envelopes) == 1

        # Verify envelope has __bindings namespace
        received = received_envelopes[0]
        assert isinstance(received, dict), "Materialized envelope should be a dict"
        assert "__bindings" in received, "Materialized envelope should have __bindings"

        # Verify resolved bindings
        bindings = received["__bindings"]
        assert bindings["correlation_id"] == correlation_id, (
            "Global binding should resolve"
        )
        assert bindings["user_id"] == "user-123", "Operation binding should resolve"
        assert bindings["optional_field"] == "default_value", (
            "Default should apply for missing optional"
        )

    @pytest.mark.asyncio
    async def test_original_envelope_unchanged(
        self,
        dispatch_engine: MessageDispatchEngine,
        operation_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """Critical invariant: original envelope is NEVER mutated.

        This test verifies the immutability guarantee:
        1. Original payload object reference is preserved
        2. Original payload contents are unchanged
        3. Materialized envelope is a different instance
        """
        received_envelopes: list[object] = []

        async def test_handler(envelope: object, context: object | None = None) -> None:
            received_envelopes.append(envelope)

        dispatch_engine.register_dispatcher(
            dispatcher_id="test-dispatcher",
            dispatcher=test_handler,
            category=EnumMessageCategory.EVENT,
            message_types=None,
            node_kind=EnumNodeKind.EFFECT,
            operation_bindings=operation_bindings,
        )
        dispatch_engine.register_route(create_test_route("test-dispatcher"))
        dispatch_engine.freeze()

        # Create test envelope and preserve original state
        original_payload = MockEventPayload(
            user_id="user-456",
            data="other",
            operation="test.operation",
        )
        original_payload_dump = original_payload.model_dump()

        correlation_id = uuid4()
        test_envelope = MockEventEnvelope(
            correlation_id=correlation_id,
            payload=original_payload,
        )

        await dispatch_engine.dispatch(
            topic="dev.test.events.v1",
            envelope=test_envelope,
        )

        # Original envelope unchanged
        assert test_envelope.payload is original_payload, (
            "Payload reference should be preserved"
        )
        assert test_envelope.payload.model_dump() == original_payload_dump, (
            "Payload contents should be unchanged"
        )

        # Materialized envelope is different instance
        received = received_envelopes[0]
        assert isinstance(received, dict), "Materialized envelope should be a dict"
        assert received is not test_envelope, "Materialized should be a new object"
        assert "__bindings" in received, "Materialized should have __bindings"

    @pytest.mark.asyncio
    async def test_binding_resolution_failure_captured_in_result(
        self,
        dispatch_engine: MessageDispatchEngine,
    ) -> None:
        """Binding resolution failure is captured in dispatch result.

        When a required binding cannot be resolved (field missing in payload),
        the dispatch engine captures the error in the result rather than raising,
        allowing for resilient processing of multiple dispatchers. The error
        message includes diagnostic context: operation name, parameter name,
        and expression.
        """
        from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus

        # Bindings require a field that won't exist in payload
        strict_bindings = ModelOperationBindingsSubcontract(
            bindings={
                "test.operation": [
                    ModelParsedBinding(
                        parameter_name="missing",
                        source="payload",
                        path_segments=("nonexistent_field",),
                        required=True,
                        original_expression="${payload.nonexistent_field}",
                    ),
                ],
            },
        )

        async def test_handler(envelope: object, context: object | None = None) -> None:
            return None

        dispatch_engine.register_dispatcher(
            dispatcher_id="test-dispatcher",
            dispatcher=test_handler,
            category=EnumMessageCategory.EVENT,
            message_types=None,
            node_kind=EnumNodeKind.EFFECT,
            operation_bindings=strict_bindings,
        )
        dispatch_engine.register_route(create_test_route("test-dispatcher"))
        dispatch_engine.freeze()

        test_envelope = MockEventEnvelope(
            correlation_id=uuid4(),
            payload=MockSimplePayload(
                user_id="user-789",
                operation="test.operation",
            ),
        )

        # Dispatch does not raise - error is captured in result
        result = await dispatch_engine.dispatch(
            topic="dev.test.events.v1",
            envelope=test_envelope,
        )

        # Verify dispatch result indicates error
        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None

        # Verify error message contains diagnostic context
        error_msg = result.error_message
        assert "BindingResolutionError" in error_msg or "binding" in error_msg.lower()
        assert (
            "missing" in error_msg.lower() or "nonexistent_field" in error_msg.lower()
        )

    @pytest.mark.asyncio
    async def test_dispatcher_without_bindings_receives_materialized_dict(
        self,
        dispatch_engine: MessageDispatchEngine,
    ) -> None:
        """Dispatcher without operation_bindings receives materialized dict.

        When no bindings are configured for a dispatcher, the handler still
        receives a materialized dict with empty __bindings namespace for
        consistent API across all dispatchers.
        """
        received_envelopes: list[object] = []

        async def test_handler(envelope: object, context: object | None = None) -> None:
            received_envelopes.append(envelope)

        # Register WITHOUT bindings
        dispatch_engine.register_dispatcher(
            dispatcher_id="test-dispatcher",
            dispatcher=test_handler,
            category=EnumMessageCategory.EVENT,
            message_types=None,
            node_kind=EnumNodeKind.EFFECT,
            # No operation_bindings parameter
        )
        dispatch_engine.register_route(create_test_route("test-dispatcher"))
        dispatch_engine.freeze()

        test_envelope = MockEventEnvelope(
            correlation_id=uuid4(),
            payload=MockSimplePayload(user_id="user-999"),
        )

        await dispatch_engine.dispatch(
            topic="dev.test.events.v1",
            envelope=test_envelope,
        )

        # Handler receives materialized dict (always dict format now)
        received = received_envelopes[0]
        assert isinstance(received, dict), "Materialized envelope should be a dict"
        assert "__bindings" in received, "Should have __bindings namespace"
        assert received["__bindings"] == {}, (
            "Bindings should be empty dict when no bindings configured"
        )
        assert "__debug_original_envelope" in received, (
            "Should have __debug_original_envelope (trace-only)"
        )
        assert received["__debug_original_envelope"] is test_envelope, (
            "Should reference original envelope"
        )

    @pytest.mark.asyncio
    async def test_trace_ids_preserved_in_materialized_envelope(
        self,
        dispatch_engine: MessageDispatchEngine,
        operation_bindings: ModelOperationBindingsSubcontract,
    ) -> None:
        """Materialized envelope preserves original envelope reference for trace IDs.

        The materialized envelope includes a __debug_original_envelope reference that
        allows downstream handlers to access correlation_id, trace_id, and other
        envelope metadata for distributed tracing purposes.

        Note: __debug_original_envelope is trace-only and should NOT be used for
        business logic. It may be removed or changed without notice.
        """
        received_envelopes: list[object] = []

        async def test_handler(envelope: object, context: object | None = None) -> None:
            received_envelopes.append(envelope)

        dispatch_engine.register_dispatcher(
            dispatcher_id="test-dispatcher",
            dispatcher=test_handler,
            category=EnumMessageCategory.EVENT,
            message_types=None,
            node_kind=EnumNodeKind.EFFECT,
            operation_bindings=operation_bindings,
        )
        dispatch_engine.register_route(create_test_route("test-dispatcher"))
        dispatch_engine.freeze()

        correlation_id = uuid4()
        trace_id = uuid4()
        test_envelope = MockEventEnvelope(
            correlation_id=correlation_id,
            trace_id=trace_id,
            payload=MockEventPayload(
                user_id="user-trace",
                operation="test.operation",
            ),
        )

        await dispatch_engine.dispatch(
            topic="dev.test.events.v1",
            envelope=test_envelope,
        )

        # Verify original envelope reference is preserved (trace-only)
        received = received_envelopes[0]
        assert isinstance(received, dict)
        assert "__debug_original_envelope" in received, (
            "Should have __debug_original_envelope (trace-only)"
        )
        assert received["__debug_original_envelope"] is test_envelope, (
            "Reference should be original envelope"
        )
        # NOTE: Accessing trace metadata is valid use of __debug_original_envelope
        assert received["__debug_original_envelope"].correlation_id == correlation_id
        assert received["__debug_original_envelope"].trace_id == trace_id


class TestBindingResolutionEdgeCases:
    """Edge case tests for binding resolution."""

    @pytest.mark.asyncio
    async def test_nested_path_resolution(
        self,
        dispatch_engine: MessageDispatchEngine,
    ) -> None:
        """Test resolution of nested path segments like ${payload.user.profile.name}."""
        nested_bindings = ModelOperationBindingsSubcontract(
            bindings={
                "test.operation": [
                    ModelParsedBinding(
                        parameter_name="user_name",
                        source="payload",
                        path_segments=("user", "profile", "name"),
                        required=True,
                        original_expression="${payload.user.profile.name}",
                    ),
                ],
            },
        )

        received_envelopes: list[object] = []

        async def test_handler(envelope: object, context: object | None = None) -> None:
            received_envelopes.append(envelope)

        dispatch_engine.register_dispatcher(
            dispatcher_id="test-dispatcher",
            dispatcher=test_handler,
            category=EnumMessageCategory.EVENT,
            message_types=None,
            node_kind=EnumNodeKind.EFFECT,
            operation_bindings=nested_bindings,
        )
        dispatch_engine.register_route(create_test_route("test-dispatcher"))
        dispatch_engine.freeze()

        test_envelope = MockEventEnvelope(
            correlation_id=uuid4(),
            payload=MockNestedProfilePayload(
                user={
                    "profile": {
                        "name": "Alice",
                    },
                },
                operation="test.operation",
            ),
        )

        await dispatch_engine.dispatch(
            topic="dev.test.events.v1",
            envelope=test_envelope,
        )

        received = received_envelopes[0]
        assert isinstance(received, dict)
        assert received["__bindings"]["user_name"] == "Alice"

    @pytest.mark.asyncio
    async def test_global_binding_override_by_operation_binding(
        self,
        dispatch_engine: MessageDispatchEngine,
    ) -> None:
        """Operation-specific bindings override global bindings for same parameter."""
        override_bindings = ModelOperationBindingsSubcontract(
            global_bindings=[
                ModelParsedBinding(
                    parameter_name="source",
                    source="envelope",
                    path_segments=("trace_id",),
                    required=False,
                    default="global_default",
                    original_expression="${envelope.trace_id}",
                ),
            ],
            bindings={
                "test.operation": [
                    ModelParsedBinding(
                        parameter_name="source",
                        source="payload",
                        path_segments=("data",),
                        required=False,
                        default="operation_default",
                        original_expression="${payload.data}",
                    ),
                ],
            },
        )

        received_envelopes: list[object] = []

        async def test_handler(envelope: object, context: object | None = None) -> None:
            received_envelopes.append(envelope)

        dispatch_engine.register_dispatcher(
            dispatcher_id="test-dispatcher",
            dispatcher=test_handler,
            category=EnumMessageCategory.EVENT,
            message_types=None,
            node_kind=EnumNodeKind.EFFECT,
            operation_bindings=override_bindings,
        )
        dispatch_engine.register_route(create_test_route("test-dispatcher"))
        dispatch_engine.freeze()

        test_envelope = MockEventEnvelope(
            correlation_id=uuid4(),
            trace_id=uuid4(),
            payload=MockEventPayload(
                user_id="user-override",
                data="from_payload",
                operation="test.operation",
            ),
        )

        await dispatch_engine.dispatch(
            topic="dev.test.events.v1",
            envelope=test_envelope,
        )

        received = received_envelopes[0]
        assert isinstance(received, dict)
        # Operation binding should override global binding
        assert received["__bindings"]["source"] == "from_payload"

    @pytest.mark.asyncio
    async def test_optional_binding_with_none_value(
        self,
        dispatch_engine: MessageDispatchEngine,
    ) -> None:
        """Optional binding with None value should use default."""
        optional_bindings = ModelOperationBindingsSubcontract(
            bindings={
                "test.operation": [
                    ModelParsedBinding(
                        parameter_name="limit",
                        source="payload",
                        path_segments=("optional",),
                        required=False,
                        default=100,
                        original_expression="${payload.optional}",
                    ),
                ],
            },
        )

        received_envelopes: list[object] = []

        async def test_handler(envelope: object, context: object | None = None) -> None:
            received_envelopes.append(envelope)

        dispatch_engine.register_dispatcher(
            dispatcher_id="test-dispatcher",
            dispatcher=test_handler,
            category=EnumMessageCategory.EVENT,
            message_types=None,
            node_kind=EnumNodeKind.EFFECT,
            operation_bindings=optional_bindings,
        )
        dispatch_engine.register_route(create_test_route("test-dispatcher"))
        dispatch_engine.freeze()

        # Payload with None for optional field
        test_envelope = MockEventEnvelope(
            correlation_id=uuid4(),
            payload=MockEventPayload(
                user_id="user-optional",
                optional=None,  # Explicitly None
                operation="test.operation",
            ),
        )

        await dispatch_engine.dispatch(
            topic="dev.test.events.v1",
            envelope=test_envelope,
        )

        received = received_envelopes[0]
        assert isinstance(received, dict)
        # Default should apply when value is None
        assert received["__bindings"]["limit"] == 100
