# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for OperationBindingResolver.

These tests verify the OperationBindingResolver's behavior for resolving
declarative operation bindings from envelopes, payloads, and context.

Test Coverage:
- Required field resolution (payload, envelope, context sources)
- Fail-fast behavior on missing required bindings
- Optional bindings with defaults
- Optional bindings without defaults returning None
- Global bindings applied to all operations
- Operation-specific bindings overriding globals
- Dict envelope traversal (uses .get())
- Pydantic model envelope traversal (uses getattr())
- Nested path traversal (multi-segment paths)
- Result model __bool__ behavior
- Edge cases (empty bindings, no operation match)

Related:
- OMN-1518: Declarative TopicOperationHandler routing
- Phase 6: Testing

.. versionadded:: 0.2.6
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_infra.errors import BindingResolutionError
from omnibase_infra.models.bindings import (
    ModelBindingResolutionResult,
    ModelOperationBindingsSubcontract,
    ModelParsedBinding,
)
from omnibase_infra.runtime.binding_resolver import OperationBindingResolver

# =============================================================================
# Mock Models for Testing
# =============================================================================


class MockNestedData(BaseModel):
    """Nested data for testing deep path traversal."""

    level2: str
    level2_value: int = 100


class MockPayload(BaseModel):
    """Mock payload for testing."""

    user_id: str
    count: int
    nested: MockNestedData | None = None


class MockEnvelope(BaseModel):
    """Mock envelope for testing."""

    correlation_id: str
    payload: MockPayload


class MockContext(BaseModel):
    """Mock context for testing."""

    now_iso: str
    dispatcher_id: str
    correlation_id: str


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def resolver() -> OperationBindingResolver:
    """Create a fresh resolver instance."""
    return OperationBindingResolver()


@pytest.fixture
def mock_envelope() -> MockEnvelope:
    """Create a mock envelope with nested payload."""
    return MockEnvelope(
        correlation_id="test-correlation-123",
        payload=MockPayload(
            user_id="user-456",
            count=42,
            nested=MockNestedData(
                level2="deep_value",
                level2_value=999,
            ),
        ),
    )


@pytest.fixture
def mock_context() -> MockContext:
    """Create a mock context with valid context paths."""
    return MockContext(
        now_iso="2026-01-27T12:00:00Z",
        dispatcher_id="test-dispatcher",
        correlation_id="context-correlation-789",
    )


# =============================================================================
# Test Class: Happy Path - Required Field Resolution
# =============================================================================


class TestRequiredFieldResolution:
    """Tests for successful resolution of required fields."""

    def test_resolve_payload_field(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Required payload field resolves correctly."""
        binding = ModelParsedBinding(
            parameter_name="user",
            source="payload",
            path_segments=("user_id",),
            required=True,
            original_expression="${payload.user_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["user"] == "user-456"
        assert result.resolved_from["user"] == "${payload.user_id}"

    def test_resolve_envelope_field(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Required envelope field resolves correctly."""
        binding = ModelParsedBinding(
            parameter_name="corr_id",
            source="envelope",
            path_segments=("correlation_id",),
            required=True,
            original_expression="${envelope.correlation_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["corr_id"] == "test-correlation-123"
        assert result.resolved_from["corr_id"] == "${envelope.correlation_id}"

    def test_resolve_context_now_iso(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Context now_iso field resolves correctly."""
        binding = ModelParsedBinding(
            parameter_name="timestamp",
            source="context",
            path_segments=("now_iso",),
            required=True,
            original_expression="${context.now_iso}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["timestamp"] == "2026-01-27T12:00:00Z"

    def test_resolve_context_dispatcher_id(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Context dispatcher_id field resolves correctly."""
        binding = ModelParsedBinding(
            parameter_name="dispatcher",
            source="context",
            path_segments=("dispatcher_id",),
            required=True,
            original_expression="${context.dispatcher_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["dispatcher"] == "test-dispatcher"

    def test_resolve_nested_payload_path(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Nested payload paths resolve correctly."""
        binding = ModelParsedBinding(
            parameter_name="deep_value",
            source="payload",
            path_segments=("nested", "level2"),
            required=True,
            original_expression="${payload.nested.level2}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["deep_value"] == "deep_value"

    def test_resolve_multiple_bindings(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Multiple bindings resolve in single call."""
        bindings = [
            ModelParsedBinding(
                parameter_name="user",
                source="payload",
                path_segments=("user_id",),
                required=True,
                original_expression="${payload.user_id}",
            ),
            ModelParsedBinding(
                parameter_name="count",
                source="payload",
                path_segments=("count",),
                required=True,
                original_expression="${payload.count}",
            ),
            ModelParsedBinding(
                parameter_name="corr",
                source="envelope",
                path_segments=("correlation_id",),
                required=True,
                original_expression="${envelope.correlation_id}",
            ),
        ]
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": bindings},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["user"] == "user-456"
        assert result.resolved_parameters["count"] == 42
        assert result.resolved_parameters["corr"] == "test-correlation-123"
        assert len(result.resolved_parameters) == 3


# =============================================================================
# Test Class: Fail-Fast Behavior
# =============================================================================


class TestFailFastBehavior:
    """Tests for fail-fast behavior on missing required bindings."""

    def test_missing_required_payload_field_raises(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Missing required payload field raises BindingResolutionError."""
        binding = ModelParsedBinding(
            parameter_name="missing",
            source="payload",
            path_segments=("nonexistent_field",),
            required=True,
            original_expression="${payload.nonexistent_field}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        with pytest.raises(BindingResolutionError) as exc_info:
            resolver.resolve(
                operation="test.op",
                bindings_subcontract=subcontract,
                envelope=mock_envelope,
                context=mock_context,
            )

        error = exc_info.value
        assert error.parameter_name == "missing"
        assert error.expression == "${payload.nonexistent_field}"
        assert error.operation_name == "test.op"

    def test_missing_required_envelope_field_raises(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Missing required envelope field raises BindingResolutionError."""
        binding = ModelParsedBinding(
            parameter_name="missing",
            source="envelope",
            path_segments=("nonexistent_field",),
            required=True,
            original_expression="${envelope.nonexistent_field}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        with pytest.raises(BindingResolutionError) as exc_info:
            resolver.resolve(
                operation="test.op",
                bindings_subcontract=subcontract,
                envelope=mock_envelope,
                context=mock_context,
            )

        assert exc_info.value.parameter_name == "missing"
        assert exc_info.value.operation_name == "test.op"

    def test_missing_nested_path_segment_raises(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Missing nested path segment raises BindingResolutionError."""
        binding = ModelParsedBinding(
            parameter_name="missing",
            source="payload",
            path_segments=("nested", "nonexistent"),
            required=True,
            original_expression="${payload.nested.nonexistent}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        with pytest.raises(BindingResolutionError):
            resolver.resolve(
                operation="test.op",
                bindings_subcontract=subcontract,
                envelope=mock_envelope,
                context=mock_context,
            )

    def test_null_context_with_required_context_binding_raises(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
    ) -> None:
        """None context with required context binding raises error."""
        binding = ModelParsedBinding(
            parameter_name="timestamp",
            source="context",
            path_segments=("now_iso",),
            required=True,
            original_expression="${context.now_iso}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        with pytest.raises(BindingResolutionError):
            resolver.resolve(
                operation="test.op",
                bindings_subcontract=subcontract,
                envelope=mock_envelope,
                context=None,
            )

    def test_error_includes_correlation_id(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """BindingResolutionError includes correlation_id when provided."""
        correlation_id = uuid4()
        binding = ModelParsedBinding(
            parameter_name="missing",
            source="payload",
            path_segments=("nonexistent",),
            required=True,
            original_expression="${payload.nonexistent}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        with pytest.raises(BindingResolutionError) as exc_info:
            resolver.resolve(
                operation="test.op",
                bindings_subcontract=subcontract,
                envelope=mock_envelope,
                context=mock_context,
                correlation_id=correlation_id,
            )

        # Error should have the correlation_id stored
        assert exc_info.value.binding_correlation_id == correlation_id


# =============================================================================
# Test Class: Optional Binding Behavior
# =============================================================================


class TestOptionalBindingBehavior:
    """Tests for optional binding resolution."""

    def test_optional_with_default_uses_default_when_missing(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Optional binding with default uses default when field missing."""
        binding = ModelParsedBinding(
            parameter_name="optional_param",
            source="payload",
            path_segments=("nonexistent",),
            required=False,
            default="default_value",
            original_expression="${payload.nonexistent}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["optional_param"] == "default_value"

    def test_optional_without_default_returns_none_when_missing(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Optional binding without default returns None when field missing."""
        binding = ModelParsedBinding(
            parameter_name="optional_param",
            source="payload",
            path_segments=("nonexistent",),
            required=False,
            default=None,
            original_expression="${payload.nonexistent}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["optional_param"] is None

    def test_optional_with_default_uses_value_when_present(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Optional binding uses actual value when field is present."""
        binding = ModelParsedBinding(
            parameter_name="user",
            source="payload",
            path_segments=("user_id",),
            required=False,
            default="fallback_user",
            original_expression="${payload.user_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        # Should use actual value, not default
        assert result.resolved_parameters["user"] == "user-456"

    def test_optional_with_int_default(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Optional binding with int default works correctly."""
        binding = ModelParsedBinding(
            parameter_name="limit",
            source="payload",
            path_segments=("nonexistent_limit",),
            required=False,
            default=100,
            original_expression="${payload.nonexistent_limit}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["limit"] == 100

    def test_optional_with_null_context_uses_default(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
    ) -> None:
        """Optional context binding with None context uses default."""
        binding = ModelParsedBinding(
            parameter_name="timestamp",
            source="context",
            path_segments=("now_iso",),
            required=False,
            default="1970-01-01T00:00:00Z",
            original_expression="${context.now_iso}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=None,
        )

        assert result.success
        assert result.resolved_parameters["timestamp"] == "1970-01-01T00:00:00Z"


# =============================================================================
# Test Class: Global Bindings
# =============================================================================


class TestGlobalBindings:
    """Tests for global binding behavior."""

    def test_global_bindings_applied_to_operation(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Global bindings are applied to all operations."""
        global_binding = ModelParsedBinding(
            parameter_name="global_corr",
            source="envelope",
            path_segments=("correlation_id",),
            required=True,
            original_expression="${envelope.correlation_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            global_bindings=[global_binding],
            bindings={"test.op": []},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["global_corr"] == "test-correlation-123"

    def test_operation_bindings_override_global(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Operation-specific bindings override global bindings for same param."""
        global_binding = ModelParsedBinding(
            parameter_name="param",
            source="payload",
            path_segments=("user_id",),
            required=True,
            original_expression="${payload.user_id}",
        )
        op_binding = ModelParsedBinding(
            parameter_name="param",  # Same parameter name
            source="envelope",
            path_segments=("correlation_id",),
            required=True,
            original_expression="${envelope.correlation_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            global_bindings=[global_binding],
            bindings={"test.op": [op_binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        # Should have operation-specific value, not global
        assert result.resolved_parameters["param"] == "test-correlation-123"
        assert result.resolved_from["param"] == "${envelope.correlation_id}"

    def test_global_and_operation_bindings_merge(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Global and operation bindings merge (non-overlapping params)."""
        global_binding = ModelParsedBinding(
            parameter_name="global_param",
            source="envelope",
            path_segments=("correlation_id",),
            required=True,
            original_expression="${envelope.correlation_id}",
        )
        op_binding = ModelParsedBinding(
            parameter_name="op_param",
            source="payload",
            path_segments=("user_id",),
            required=True,
            original_expression="${payload.user_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            global_bindings=[global_binding],
            bindings={"test.op": [op_binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["global_param"] == "test-correlation-123"
        assert result.resolved_parameters["op_param"] == "user-456"
        assert len(result.resolved_parameters) == 2

    def test_global_bindings_without_operation_bindings(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Global bindings work even when no operation-specific bindings exist."""
        global_binding = ModelParsedBinding(
            parameter_name="global_only",
            source="context",
            path_segments=("now_iso",),
            required=True,
            original_expression="${context.now_iso}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            global_bindings=[global_binding],
            bindings={},  # No operation-specific bindings
        )

        result = resolver.resolve(
            operation="any.operation",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["global_only"] == "2026-01-27T12:00:00Z"


# =============================================================================
# Test Class: Dict Envelope Traversal
# =============================================================================


class TestDictEnvelopeTraversal:
    """Tests for dict-based envelope traversal."""

    def test_dict_envelope_payload_resolution(
        self,
        resolver: OperationBindingResolver,
        mock_context: MockContext,
    ) -> None:
        """Dict envelopes use .get() for payload resolution."""
        dict_envelope = {
            "correlation_id": "dict-corr-id",
            "payload": {"user_id": "dict-user"},
        }
        binding = ModelParsedBinding(
            parameter_name="user",
            source="payload",
            path_segments=("user_id",),
            required=True,
            original_expression="${payload.user_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=dict_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["user"] == "dict-user"

    def test_dict_envelope_field_resolution(
        self,
        resolver: OperationBindingResolver,
        mock_context: MockContext,
    ) -> None:
        """Dict envelopes resolve envelope-level fields."""
        dict_envelope = {
            "correlation_id": "dict-corr-id",
            "payload": {"user_id": "dict-user"},
        }
        binding = ModelParsedBinding(
            parameter_name="corr",
            source="envelope",
            path_segments=("correlation_id",),
            required=True,
            original_expression="${envelope.correlation_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=dict_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["corr"] == "dict-corr-id"

    def test_nested_dict_payload_resolution(
        self,
        resolver: OperationBindingResolver,
        mock_context: MockContext,
    ) -> None:
        """Nested dict payloads resolve correctly."""
        dict_envelope = {
            "payload": {
                "user": {
                    "profile": {
                        "email": "test@example.com",
                    },
                },
            },
        }
        binding = ModelParsedBinding(
            parameter_name="email",
            source="payload",
            path_segments=("user", "profile", "email"),
            required=True,
            original_expression="${payload.user.profile.email}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=dict_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["email"] == "test@example.com"

    def test_dict_context_resolution(
        self,
        resolver: OperationBindingResolver,
    ) -> None:
        """Dict context resolves correctly."""
        dict_envelope = {"payload": {}}
        dict_context = {
            "now_iso": "2026-01-01T00:00:00Z",
            "dispatcher_id": "dict-dispatcher",
        }
        binding = ModelParsedBinding(
            parameter_name="timestamp",
            source="context",
            path_segments=("now_iso",),
            required=True,
            original_expression="${context.now_iso}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=dict_envelope,
            context=dict_context,
        )

        assert result.success
        assert result.resolved_parameters["timestamp"] == "2026-01-01T00:00:00Z"


# =============================================================================
# Test Class: Result Model Behavior
# =============================================================================


class TestResultModelBehavior:
    """Tests for ModelBindingResolutionResult behavior."""

    def test_result_bool_true_on_success(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """ModelBindingResolutionResult.__bool__ returns True on success."""
        binding = ModelParsedBinding(
            parameter_name="user",
            source="payload",
            path_segments=("user_id",),
            required=True,
            original_expression="${payload.user_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        # __bool__ should return True
        assert result
        assert bool(result) is True
        assert result.success is True

    def test_result_contains_operation_name(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Result includes operation name for context."""
        binding = ModelParsedBinding(
            parameter_name="user",
            source="payload",
            path_segments=("user_id",),
            required=True,
            original_expression="${payload.user_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"my.custom.operation": [binding]},
        )

        result = resolver.resolve(
            operation="my.custom.operation",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.operation_name == "my.custom.operation"

    def test_result_resolved_from_tracks_expressions(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Result tracks original expressions for debugging."""
        bindings = [
            ModelParsedBinding(
                parameter_name="user",
                source="payload",
                path_segments=("user_id",),
                required=True,
                original_expression="${payload.user_id}",
            ),
            ModelParsedBinding(
                parameter_name="timestamp",
                source="context",
                path_segments=("now_iso",),
                required=True,
                original_expression="${context.now_iso}",
            ),
        ]
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": bindings},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.resolved_from["user"] == "${payload.user_id}"
        assert result.resolved_from["timestamp"] == "${context.now_iso}"

    def test_successful_result_has_no_error(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Successful result has error=None."""
        binding = ModelParsedBinding(
            parameter_name="user",
            source="payload",
            path_segments=("user_id",),
            required=True,
            original_expression="${payload.user_id}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.error is None


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_bindings_for_operation(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Empty bindings for operation returns empty parameters."""
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": []},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters == {}

    def test_operation_not_in_bindings(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Operation not in bindings map returns empty parameters."""
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"other.op": []},
        )

        result = resolver.resolve(
            operation="nonexistent.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters == {}

    def test_empty_subcontract(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Empty subcontract returns empty parameters."""
        subcontract = ModelOperationBindingsSubcontract(
            bindings={},
            global_bindings=None,
        )

        result = resolver.resolve(
            operation="any.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters == {}

    def test_integer_value_resolution(
        self,
        resolver: OperationBindingResolver,
        mock_envelope: MockEnvelope,
        mock_context: MockContext,
    ) -> None:
        """Integer values resolve correctly (not converted to string)."""
        binding = ModelParsedBinding(
            parameter_name="count",
            source="payload",
            path_segments=("count",),
            required=True,
            original_expression="${payload.count}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=mock_envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["count"] == 42
        assert isinstance(result.resolved_parameters["count"], int)

    def test_null_nested_object_returns_none(
        self,
        resolver: OperationBindingResolver,
        mock_context: MockContext,
    ) -> None:
        """Traversing through None object in path returns None."""
        envelope_with_null = MockEnvelope(
            correlation_id="test-corr",
            payload=MockPayload(
                user_id="user-123",
                count=1,
                nested=None,  # Null nested object
            ),
        )
        binding = ModelParsedBinding(
            parameter_name="deep",
            source="payload",
            path_segments=("nested", "level2"),
            required=False,  # Optional to avoid exception
            default="fallback",
            original_expression="${payload.nested.level2}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=envelope_with_null,
            context=mock_context,
        )

        assert result.success
        # Should use fallback since nested is None
        assert result.resolved_parameters["deep"] == "fallback"

    def test_envelope_without_payload_attribute(
        self,
        resolver: OperationBindingResolver,
        mock_context: MockContext,
    ) -> None:
        """Envelope without payload attribute handles gracefully."""

        class EnvelopeWithoutPayload(BaseModel):
            correlation_id: str

        envelope = EnvelopeWithoutPayload(correlation_id="test-corr")
        binding = ModelParsedBinding(
            parameter_name="field",
            source="payload",
            path_segments=("something",),
            required=False,
            default="default",
            original_expression="${payload.something}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"test.op": [binding]},
        )

        result = resolver.resolve(
            operation="test.op",
            bindings_subcontract=subcontract,
            envelope=envelope,
            context=mock_context,
        )

        assert result.success
        assert result.resolved_parameters["field"] == "default"


# =============================================================================
# Test Class: Resolver Instance Behavior
# =============================================================================


class TestResolverInstanceBehavior:
    """Tests for resolver instance behavior and thread-safety."""

    def test_resolver_is_stateless(self) -> None:
        """Resolver is stateless and can be reused."""
        resolver = OperationBindingResolver()

        # First resolution
        envelope1 = {"payload": {"user": "user1"}}
        context1 = {"now_iso": "2026-01-01T00:00:00Z"}
        binding = ModelParsedBinding(
            parameter_name="user",
            source="payload",
            path_segments=("user",),
            required=True,
            original_expression="${payload.user}",
        )
        subcontract = ModelOperationBindingsSubcontract(
            bindings={"op1": [binding]},
        )

        result1 = resolver.resolve(
            operation="op1",
            bindings_subcontract=subcontract,
            envelope=envelope1,
            context=context1,
        )

        # Second resolution with different data
        envelope2 = {"payload": {"user": "user2"}}

        result2 = resolver.resolve(
            operation="op1",
            bindings_subcontract=subcontract,
            envelope=envelope2,
            context=context1,
        )

        # Results should be independent
        assert result1.resolved_parameters["user"] == "user1"
        assert result2.resolved_parameters["user"] == "user2"

    def test_resolver_has_expression_parser(self) -> None:
        """Resolver initializes with expression parser."""
        resolver = OperationBindingResolver()
        assert hasattr(resolver, "_parser")
