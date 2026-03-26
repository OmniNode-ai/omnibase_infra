# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for OMN-518: Enhanced error context with stack traces and suggestions.

Tests cover:
- ModelInfraErrorContext new fields (suggested_resolution, retry_after_seconds, original_error_type)
- ModelInfraErrorContext.from_exception() factory
- Error catalog resolution lookup
- Automatic catalog enrichment in RuntimeHostError
- Stack trace preservation
- Error chaining with original_error_type
"""

from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ErrorResolution,
    ModelInfraErrorContext,
    get_resolution,
)
from omnibase_infra.errors.error_infra import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraProtocolError,
    InfraRateLimitedError,
    InfraRequestRejectedError,
    InfraTimeoutError,
    InfraUnavailableError,
    ProtocolConfigurationError,
    RuntimeHostError,
    SecretResolutionError,
)
from omnibase_infra.models.errors.model_timeout_error_context import (
    ModelTimeoutErrorContext,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ModelInfraErrorContext new fields
# ---------------------------------------------------------------------------


class TestModelInfraErrorContextNewFields:
    """Tests for the new fields added to ModelInfraErrorContext in OMN-518."""

    def test_suggested_resolution_field(self) -> None:
        """Test that suggested_resolution field is accepted and stored."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
            suggested_resolution="Check PostgreSQL is running",
        )
        assert context.suggested_resolution == "Check PostgreSQL is running"

    def test_retry_after_seconds_field(self) -> None:
        """Test that retry_after_seconds field is accepted and stored."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
            retry_after_seconds=30.0,
        )
        assert context.retry_after_seconds == 30.0

    def test_retry_after_seconds_rejects_negative(self) -> None:
        """Test that negative retry_after_seconds raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="retry_after_seconds"):
            ModelInfraErrorContext(
                retry_after_seconds=-1.0,
            )

    def test_original_error_type_field(self) -> None:
        """Test that original_error_type field is accepted and stored."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
            original_error_type="OperationalError",
        )
        assert context.original_error_type == "OperationalError"

    def test_all_new_fields_default_to_none(self) -> None:
        """Test that all new fields default to None for backward compatibility."""
        context = ModelInfraErrorContext()
        assert context.suggested_resolution is None
        assert context.retry_after_seconds is None
        assert context.original_error_type is None

    def test_with_correlation_passes_new_fields(self) -> None:
        """Test that with_correlation forwards new fields correctly."""
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
            suggested_resolution="Check credentials",
            retry_after_seconds=5.0,
        )
        assert context.correlation_id is not None
        assert context.suggested_resolution == "Check credentials"
        assert context.retry_after_seconds == 5.0


# ---------------------------------------------------------------------------
# ModelInfraErrorContext.from_exception()
# ---------------------------------------------------------------------------


class TestModelInfraErrorContextFromException:
    """Tests for the from_exception() factory method."""

    def test_captures_exception_type(self) -> None:
        """Test that from_exception captures the exception class name."""
        try:
            raise ConnectionError("test error")
        except ConnectionError as e:
            context = ModelInfraErrorContext.from_exception(
                e,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="connect",
            )
        assert context.original_error_type == "ConnectionError"
        assert context.transport_type == EnumInfraTransportType.DATABASE
        assert context.correlation_id is not None

    def test_uses_provided_correlation_id(self) -> None:
        """Test that from_exception uses provided correlation_id."""
        cid = uuid4()
        try:
            raise ValueError("bad value")
        except ValueError as e:
            context = ModelInfraErrorContext.from_exception(
                e,
                correlation_id=cid,
            )
        assert context.correlation_id == cid
        assert context.original_error_type == "ValueError"

    def test_auto_generates_correlation_id(self) -> None:
        """Test that from_exception auto-generates correlation_id when None."""
        try:
            raise RuntimeError("oops")
        except RuntimeError as e:
            context = ModelInfraErrorContext.from_exception(e)
        assert context.correlation_id is not None
        assert context.original_error_type == "RuntimeError"


# ---------------------------------------------------------------------------
# Error Catalog
# ---------------------------------------------------------------------------


class TestErrorCatalog:
    """Tests for the error resolution catalog."""

    def test_get_resolution_exact_match(self) -> None:
        """Test exact (error_class, transport_type) match."""
        resolution = get_resolution(
            "InfraConnectionError",
            EnumInfraTransportType.DATABASE,
        )
        assert resolution is not None
        assert "PostgreSQL" in resolution.suggestion
        assert resolution.is_retryable is True
        assert resolution.retry_after_seconds is not None

    def test_get_resolution_fallback_to_none_transport(self) -> None:
        """Test fallback to (error_class, None) when transport has no entry."""
        resolution = get_resolution(
            "InfraConnectionError",
            EnumInfraTransportType.GRPC,  # No specific entry, falls back to None
        )
        assert resolution is not None
        assert resolution.suggestion  # Generic fallback

    def test_get_resolution_no_match(self) -> None:
        """Test that unregistered error class returns None."""
        resolution = get_resolution("NonExistentError")
        assert resolution is None

    def test_all_major_error_classes_have_entries(self) -> None:
        """Verify all major error classes have at least a fallback entry."""
        expected_classes = [
            "InfraConnectionError",
            "InfraTimeoutError",
            "InfraAuthenticationError",
            "InfraUnavailableError",
            "InfraRateLimitedError",
            "ProtocolConfigurationError",
            "SecretResolutionError",
            "InfraRequestRejectedError",
            "InfraProtocolError",
        ]
        for cls_name in expected_classes:
            resolution = get_resolution(cls_name)
            assert resolution is not None, f"Missing catalog entry for {cls_name}"

    def test_error_resolution_is_frozen(self) -> None:
        """Test that ErrorResolution instances are immutable."""
        resolution = ErrorResolution(
            suggestion="test",
            retry_after_seconds=5.0,
            is_retryable=True,
        )
        with pytest.raises(AttributeError):
            resolution.suggestion = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Auto-enrichment in RuntimeHostError
# ---------------------------------------------------------------------------


class TestAutoEnrichment:
    """Tests for automatic catalog enrichment in error constructors."""

    def test_connection_error_gets_catalog_suggestion(self) -> None:
        """Test that InfraConnectionError auto-populates suggested_resolution."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
        )
        err = InfraConnectionError("Failed to connect", context=context)
        assert err.suggested_resolution is not None
        assert "PostgreSQL" in err.suggested_resolution

    def test_explicit_suggestion_overrides_catalog(self) -> None:
        """Test that explicit suggested_resolution takes priority over catalog."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
            suggested_resolution="Custom resolution hint",
        )
        err = InfraConnectionError("Failed to connect", context=context)
        assert err.suggested_resolution == "Custom resolution hint"

    def test_explicit_retry_overrides_catalog(self) -> None:
        """Test that explicit retry_after_seconds takes priority over catalog."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
            retry_after_seconds=99.0,
        )
        err = InfraConnectionError("Failed to connect", context=context)
        assert err.retry_after_seconds == 99.0

    def test_timeout_error_gets_catalog_suggestion(self) -> None:
        """Test that InfraTimeoutError auto-populates from catalog."""
        timeout_ctx = ModelTimeoutErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="execute_query",
            timeout_seconds=30.0,
        )
        err = InfraTimeoutError("Query timed out", context=timeout_ctx)
        assert err.suggested_resolution is not None
        assert (
            "timeout" in err.suggested_resolution.lower()
            or "query" in err.suggested_resolution.lower()
        )

    def test_auth_error_gets_catalog_suggestion(self) -> None:
        """Test that InfraAuthenticationError auto-populates from catalog."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="authenticate",
        )
        err = InfraAuthenticationError("Auth failed", context=context)
        assert err.suggested_resolution is not None

    def test_unavailable_error_gets_catalog_suggestion(self) -> None:
        """Test that InfraUnavailableError auto-populates from catalog."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="produce",
        )
        err = InfraUnavailableError("Service down", context=context)
        assert err.suggested_resolution is not None

    def test_no_context_no_suggestion(self) -> None:
        """Test that errors without context have None suggested_resolution."""
        err = RuntimeHostError("Something failed")
        assert err.suggested_resolution is None

    def test_catalog_retry_after_propagated(self) -> None:
        """Test that catalog retry_after_seconds is propagated to error."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="produce",
        )
        err = InfraConnectionError("Kafka connection failed", context=context)
        assert err.retry_after_seconds is not None
        assert err.retry_after_seconds > 0

    def test_config_error_not_retryable(self) -> None:
        """Test that ProtocolConfigurationError has no retry guidance."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="validate_config",
        )
        err = ProtocolConfigurationError("Bad config", context=context)
        assert err.retry_after_seconds is None

    def test_secret_error_not_retryable(self) -> None:
        """Test that SecretResolutionError has no retry guidance."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.INFISICAL,
            operation="get_secret",
        )
        err = SecretResolutionError("Secret not found", context=context)
        assert err.retry_after_seconds is None


# ---------------------------------------------------------------------------
# Stack trace preservation
# ---------------------------------------------------------------------------


class TestStackTracePreservation:
    """Tests for stack trace capture in infrastructure errors."""

    def test_stack_trace_captured(self) -> None:
        """Test that stack_trace is captured on error construction."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
        )
        err = InfraConnectionError("Failed", context=context)
        assert err.stack_trace
        assert "test_stack_trace_captured" in err.stack_trace

    def test_stack_trace_without_context(self) -> None:
        """Test that stack_trace is captured even without context."""
        err = RuntimeHostError("Something failed")
        assert err.stack_trace
        assert len(err.stack_trace) > 0

    def test_stack_trace_is_string(self) -> None:
        """Test that stack_trace is always a string."""
        err = RuntimeHostError("test")
        assert isinstance(err.stack_trace, str)


# ---------------------------------------------------------------------------
# Error chaining with original_error_type
# ---------------------------------------------------------------------------


class TestErrorChaining:
    """Tests for error chaining with original_error_type preservation."""

    def test_from_exception_chains_correctly(self) -> None:
        """Test full error chaining flow: catch -> context -> raise from."""
        original = ConnectionError("connection refused")
        context = ModelInfraErrorContext.from_exception(
            original,
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
        )

        try:
            raise InfraConnectionError(
                "Failed to connect to PostgreSQL",
                context=context,
            ) from original
        except InfraConnectionError as e:
            # Verify chaining
            assert e.__cause__ is original
            # Verify original_error_type is in structured context
            assert e.model.context.get("original_error_type") == "ConnectionError"
            # Verify suggestion was auto-populated
            assert e.suggested_resolution is not None

    def test_original_error_type_in_model_context(self) -> None:
        """Test that original_error_type appears in the error model context."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="request",
            original_error_type="TimeoutError",
        )
        err = InfraConnectionError("Request failed", context=context)
        assert err.model.context.get("original_error_type") == "TimeoutError"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Tests to ensure existing code continues to work unchanged."""

    def test_existing_context_creation_unchanged(self) -> None:
        """Test that existing ModelInfraErrorContext usage still works."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="request",
            target_name="api",
            correlation_id=uuid4(),
            namespace="prod",
        )
        err = RuntimeHostError("test", context=context)
        assert err.model.context["transport_type"] == EnumInfraTransportType.HTTP

    def test_error_without_context_still_works(self) -> None:
        """Test that errors without context still work as before."""
        err = InfraConnectionError("Connection failed")
        assert err.model.message == "Connection failed"
        assert err.suggested_resolution is None

    def test_rate_limited_preserves_retry_attr(self) -> None:
        """Test that InfraRateLimitedError.retry_after_seconds attr is preserved."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="api_call",
        )
        err = InfraRateLimitedError(
            "Rate limited",
            context=context,
            retry_after_seconds=60.0,
        )
        assert err.retry_after_seconds == 60.0

    def test_request_rejected_preserves_attrs(self) -> None:
        """Test that InfraRequestRejectedError preserves status_code and response_body."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="chat_completion",
        )
        err = InfraRequestRejectedError(
            "Rejected",
            context=context,
            status_code=422,
            response_body='{"error": "bad"}',
        )
        assert err.status_code == 422
        assert err.response_body

    def test_protocol_error_preserves_attrs(self) -> None:
        """Test that InfraProtocolError preserves its custom attributes."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="chat_completion",
        )
        err = InfraProtocolError(
            "Bad format",
            context=context,
            status_code=200,
            content_type="text/html",
            response_body="<html>",
        )
        assert err.status_code == 200
        assert err.content_type == "text/html"
