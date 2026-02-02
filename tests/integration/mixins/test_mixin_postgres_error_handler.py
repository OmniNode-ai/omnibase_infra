# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for MixinPostgresErrorResponse mixin.

This test suite validates the PostgreSQL error handling mixin that provides
standardized exception handling for PostgreSQL persistence operations.

Test Categories:
    - Exception Type Mapping: Verify each exception type maps to correct error code
    - Error Sanitization: Verify error messages are properly sanitized
    - Duration Calculation: Verify duration_ms is correctly calculated
    - Correlation ID Propagation: Verify correlation_id flows through
    - Log Context Merging: Verify log_context is properly merged

Usage:
    pytest tests/integration/mixins/test_mixin_postgres_error_handler.py -v

Related:
    - MixinPostgresErrorResponse: The mixin under test
    - PostgresErrorContext: Context dataclass for error handling
    - EnumPostgresErrorCode: Error code enumeration
    - OMN-1867: Implementation ticket
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType, EnumPostgresErrorCode
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    RepositoryExecutionError,
)
from omnibase_infra.mixins import MixinPostgresErrorResponse, PostgresErrorContext
from omnibase_infra.models.errors import (
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
)
from omnibase_infra.nodes.effects.models.model_backend_result import ModelBackendResult

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Helper Functions for Creating Errors
# =============================================================================


def make_infra_timeout_error(message: str) -> InfraTimeoutError:
    """Create an InfraTimeoutError with required context."""
    ctx = ModelTimeoutErrorContext(
        transport_type=EnumInfraTransportType.DATABASE,
        operation="test_operation",
    )
    return InfraTimeoutError(message, context=ctx)


def make_infra_auth_error(message: str) -> InfraAuthenticationError:
    """Create an InfraAuthenticationError with optional context."""
    ctx = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.DATABASE,
        operation="test_operation",
    )
    return InfraAuthenticationError(message, context=ctx)


def make_infra_connection_error(message: str) -> InfraConnectionError:
    """Create an InfraConnectionError with optional context."""
    ctx = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.DATABASE,
        operation="test_operation",
    )
    return InfraConnectionError(message, context=ctx)


def make_repository_execution_error(message: str) -> RepositoryExecutionError:
    """Create a RepositoryExecutionError."""
    return RepositoryExecutionError(message)


# =============================================================================
# Test Fixtures
# =============================================================================


class TestHandler(MixinPostgresErrorResponse):
    """Test handler class that uses the MixinPostgresErrorResponse mixin."""


@pytest.fixture
def handler() -> TestHandler:
    """Create a test handler instance."""
    return TestHandler()


@pytest.fixture
def correlation_id() -> UUID:
    """Generate a fresh correlation ID for each test."""
    return uuid4()


@pytest.fixture
def start_time() -> float:
    """Capture a start time for duration calculation tests."""
    return time.perf_counter()


# =============================================================================
# Exception Type Mapping Tests
# =============================================================================


class TestExceptionTypeMapping:
    """Test that each exception type maps to the correct error code."""

    def test_timeout_error_returns_timeout_error_code(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify standard TimeoutError returns TIMEOUT_ERROR code."""
        exception = TimeoutError("Operation timed out after 30 seconds")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert isinstance(result, ModelBackendResult)
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR.value
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    def test_infra_timeout_error_returns_timeout_error_code(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify InfraTimeoutError returns TIMEOUT_ERROR code."""
        exception = make_infra_timeout_error(
            "Database query exceeded timeout threshold"
        )

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_query",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert isinstance(result, ModelBackendResult)
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR.value
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    def test_infra_authentication_error_returns_auth_error_code(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify InfraAuthenticationError returns AUTH_ERROR code."""
        exception = make_infra_auth_error("Authentication failed for user")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_connect",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert isinstance(result, ModelBackendResult)
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.AUTH_ERROR.value
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    def test_infra_connection_error_returns_connection_error_code(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify InfraConnectionError returns CONNECTION_ERROR code."""
        exception = make_infra_connection_error("Connection refused to database host")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_connect",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert isinstance(result, ModelBackendResult)
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.CONNECTION_ERROR.value
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    def test_repository_execution_error_with_custom_code(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify RepositoryExecutionError uses provided operation_error_code."""
        exception = make_repository_execution_error("Upsert operation failed")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_upsert",
            correlation_id=correlation_id,
            start_time=start_time,
            operation_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
        )

        result = handler._build_error_response(ctx)

        assert isinstance(result, ModelBackendResult)
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UPSERT_ERROR.value
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    def test_repository_execution_error_without_custom_code_defaults_to_unknown(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify RepositoryExecutionError without operation_error_code uses UNKNOWN_ERROR."""
        exception = make_repository_execution_error("Some repository error occurred")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
            # No operation_error_code provided
        )

        result = handler._build_error_response(ctx)

        assert isinstance(result, ModelBackendResult)
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UNKNOWN_ERROR.value
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    def test_generic_exception_returns_unknown_error_code(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify generic Exception returns UNKNOWN_ERROR code."""
        exception = ValueError("An unexpected value error occurred")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert isinstance(result, ModelBackendResult)
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UNKNOWN_ERROR.value
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id


# =============================================================================
# Parametrized Exception Tests
# =============================================================================


class TestParametrizedExceptionMapping:
    """Parametrized tests for exception type mapping."""

    @pytest.mark.parametrize(
        ("exception_factory", "expected_code"),
        [
            (lambda: TimeoutError("timeout"), EnumPostgresErrorCode.TIMEOUT_ERROR),
            (
                lambda: make_infra_timeout_error("infra timeout"),
                EnumPostgresErrorCode.TIMEOUT_ERROR,
            ),
            (
                lambda: make_infra_auth_error("auth failed"),
                EnumPostgresErrorCode.AUTH_ERROR,
            ),
            (
                lambda: make_infra_connection_error("connection refused"),
                EnumPostgresErrorCode.CONNECTION_ERROR,
            ),
            (lambda: RuntimeError("unexpected"), EnumPostgresErrorCode.UNKNOWN_ERROR),
            (lambda: OSError("os error"), EnumPostgresErrorCode.UNKNOWN_ERROR),
            (lambda: Exception("generic"), EnumPostgresErrorCode.UNKNOWN_ERROR),
        ],
        ids=[
            "TimeoutError",
            "InfraTimeoutError",
            "InfraAuthenticationError",
            "InfraConnectionError",
            "RuntimeError-generic",
            "OSError-generic",
            "Exception-generic",
        ],
    )
    def test_exception_to_error_code_mapping(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
        exception_factory: Callable[[], Exception],
        expected_code: EnumPostgresErrorCode,
    ) -> None:
        """Verify exception types map to expected error codes."""
        exception = exception_factory()
        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert result.error_code == expected_code.value


# =============================================================================
# Error Sanitization Tests
# =============================================================================


class TestErrorSanitization:
    """Test that error messages are properly sanitized."""

    def test_connection_string_is_sanitized(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify connection strings with credentials are sanitized."""
        # This message contains a connection string pattern that should be redacted
        exception = make_infra_connection_error(
            "Failed to connect to postgres://user:password@host:5432/db"
        )

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_connect",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        # Error should be sanitized - password should not appear
        assert result.error is not None
        assert "password" not in result.error.lower()
        # Should contain REDACTED indicator
        assert "REDACTED" in result.error

    def test_password_in_message_is_sanitized(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify password patterns in error messages are sanitized."""
        exception = make_infra_auth_error("Authentication failed: password=mysecret123")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_auth",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert result.error is not None
        assert "mysecret123" not in result.error
        assert "REDACTED" in result.error

    def test_safe_error_message_preserved(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify safe error messages are preserved without redaction."""
        safe_message = "Connection refused"
        exception = make_infra_connection_error(safe_message)

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_connect",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert result.error is not None
        # The sanitize_error_message function includes the exception type
        assert "InfraConnectionError" in result.error
        assert "Connection refused" in result.error

    def test_generic_exception_uses_backend_error_sanitization(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify generic exceptions use sanitize_backend_error function."""
        # Generic exceptions go through sanitize_backend_error which is more aggressive
        exception = ValueError("Some internal error details")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        # sanitize_backend_error returns "postgres operation failed" for unknown patterns
        assert result.error is not None
        assert "postgres operation failed" in result.error


# =============================================================================
# Duration Calculation Tests
# =============================================================================


class TestDurationCalculation:
    """Test that duration_ms is correctly calculated."""

    def test_duration_ms_is_positive(
        self,
        handler: TestHandler,
        correlation_id: UUID,
    ) -> None:
        """Verify duration_ms is calculated and positive."""
        start_time = time.perf_counter()
        # Introduce a small delay to ensure measurable duration
        time.sleep(0.001)  # 1ms

        exception = TimeoutError("Operation timed out")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert result.duration_ms > 0
        # Should be at least 1ms (we slept for 1ms)
        assert result.duration_ms >= 1.0

    def test_duration_ms_reflects_elapsed_time(
        self,
        handler: TestHandler,
        correlation_id: UUID,
    ) -> None:
        """Verify duration_ms accurately reflects elapsed time."""
        start_time = time.perf_counter()
        # Sleep for a known duration
        sleep_duration_ms = 10.0
        time.sleep(sleep_duration_ms / 1000)

        exception = make_infra_connection_error("Connection failed")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        # Duration should be at least the sleep time
        assert result.duration_ms >= sleep_duration_ms * 0.9  # Allow 10% tolerance
        # But not excessively long (under 100ms for this simple operation)
        assert result.duration_ms < 100.0


# =============================================================================
# Correlation ID Tests
# =============================================================================


class TestCorrelationIdPropagation:
    """Test that correlation_id is properly propagated."""

    def test_correlation_id_preserved_in_result(
        self,
        handler: TestHandler,
        start_time: float,
    ) -> None:
        """Verify correlation_id is preserved in the result."""
        expected_correlation_id = uuid4()
        exception = TimeoutError("Operation timed out")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=expected_correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert result.correlation_id == expected_correlation_id

    def test_different_correlation_ids_produce_different_results(
        self,
        handler: TestHandler,
        start_time: float,
    ) -> None:
        """Verify different correlation_ids produce different results."""
        correlation_id_1 = uuid4()
        correlation_id_2 = uuid4()
        exception = TimeoutError("Operation timed out")

        ctx1 = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id_1,
            start_time=start_time,
        )

        ctx2 = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id_2,
            start_time=start_time,
        )

        result1 = handler._build_error_response(ctx1)
        result2 = handler._build_error_response(ctx2)

        assert result1.correlation_id == correlation_id_1
        assert result2.correlation_id == correlation_id_2
        assert result1.correlation_id != result2.correlation_id


# =============================================================================
# Log Context Tests
# =============================================================================


class TestLogContextMerging:
    """Test that log_context is properly merged into base context."""

    def test_log_context_merged_into_logging(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify log_context fields appear in log output."""
        exception = make_infra_connection_error("Connection refused")
        custom_log_context: dict[str, object] = {
            "contract_id": "test-contract-123",
            "operation_type": "upsert",
        }

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
            log_context=custom_log_context,
        )

        with caplog.at_level(logging.WARNING):
            handler._build_error_response(ctx)

        # Verify log was generated (warning level for connection errors)
        assert len(caplog.records) > 0
        log_record = caplog.records[0]

        # Log context should include correlation_id and duration_ms
        assert hasattr(log_record, "correlation_id")
        assert hasattr(log_record, "duration_ms")

        # Custom log context should be merged
        assert hasattr(log_record, "contract_id")
        assert log_record.contract_id == "test-contract-123"
        assert hasattr(log_record, "operation_type")
        assert log_record.operation_type == "upsert"

    def test_empty_log_context_produces_valid_result(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
    ) -> None:
        """Verify empty log_context doesn't break error handling."""
        exception = TimeoutError("Operation timed out")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
            log_context={},  # Empty log context
        )

        result = handler._build_error_response(ctx)

        # Should still produce a valid result
        assert isinstance(result, ModelBackendResult)
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR.value


# =============================================================================
# Logging Level Tests
# =============================================================================


class TestLoggingLevels:
    """Test that different exception types produce appropriate log levels."""

    def test_timeout_error_logs_warning(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify TimeoutError produces warning log (retriable)."""
        exception = TimeoutError("Operation timed out")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        with caplog.at_level(logging.WARNING):
            handler._build_error_response(ctx)

        # Should log at WARNING level
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) > 0
        assert "timed out" in warning_records[0].message

    def test_connection_error_logs_warning(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify InfraConnectionError produces warning log (retriable)."""
        exception = make_infra_connection_error("Connection refused")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        with caplog.at_level(logging.WARNING):
            handler._build_error_response(ctx)

        # Should log at WARNING level
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) > 0
        assert "connection failed" in warning_records[0].message

    def test_auth_error_logs_exception(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify InfraAuthenticationError produces error log (non-retriable)."""
        exception = make_infra_auth_error("Authentication failed")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        with caplog.at_level(logging.ERROR):
            handler._build_error_response(ctx)

        # Should log at ERROR level (logger.exception logs at ERROR)
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) > 0
        assert "authentication failed" in error_records[0].message

    def test_generic_exception_logs_exception(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify generic Exception produces error log (needs investigation)."""
        exception = ValueError("Unexpected error")

        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        with caplog.at_level(logging.ERROR):
            handler._build_error_response(ctx)

        # Should log at ERROR level (logger.exception logs at ERROR)
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) > 0
        assert "unexpected error" in error_records[0].message.lower()


# =============================================================================
# Operation-Specific Error Code Tests
# =============================================================================


class TestOperationSpecificErrorCodes:
    """Test operation-specific error codes for RepositoryExecutionError."""

    @pytest.mark.parametrize(
        ("operation_error_code", "operation_name"),
        [
            (EnumPostgresErrorCode.UPSERT_ERROR, "contract_upsert"),
            (EnumPostgresErrorCode.TOPIC_UPDATE_ERROR, "topic_update"),
            (EnumPostgresErrorCode.MARK_STALE_ERROR, "mark_stale"),
            (EnumPostgresErrorCode.HEARTBEAT_ERROR, "heartbeat_update"),
            (EnumPostgresErrorCode.DEACTIVATE_ERROR, "contract_deactivate"),
            (EnumPostgresErrorCode.CLEANUP_ERROR, "topic_cleanup"),
        ],
        ids=[
            "upsert",
            "topic_update",
            "mark_stale",
            "heartbeat",
            "deactivate",
            "cleanup",
        ],
    )
    def test_repository_error_uses_operation_specific_code(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
        operation_error_code: EnumPostgresErrorCode,
        operation_name: str,
    ) -> None:
        """Verify RepositoryExecutionError uses the provided operation-specific code."""
        exception = make_repository_execution_error(
            f"Failed to execute {operation_name}"
        )

        ctx = PostgresErrorContext(
            exception=exception,
            operation=operation_name,
            correlation_id=correlation_id,
            start_time=start_time,
            operation_error_code=operation_error_code,
        )

        result = handler._build_error_response(ctx)

        assert result.error_code == operation_error_code.value


# =============================================================================
# Backend ID Tests
# =============================================================================


class TestBackendId:
    """Test that backend_id is always set to 'postgres'."""

    @pytest.mark.parametrize(
        "exception_factory",
        [
            lambda: TimeoutError("timeout"),
            lambda: make_infra_timeout_error("infra timeout"),
            lambda: make_infra_auth_error("auth failed"),
            lambda: make_infra_connection_error("connection refused"),
            lambda: make_repository_execution_error("repo error"),
            lambda: ValueError("generic error"),
        ],
        ids=[
            "TimeoutError",
            "InfraTimeoutError",
            "InfraAuthenticationError",
            "InfraConnectionError",
            "RepositoryExecutionError",
            "ValueError",
        ],
    )
    def test_backend_id_is_always_postgres(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
        exception_factory: Callable[[], Exception],
    ) -> None:
        """Verify backend_id is always 'postgres' for all exception types."""
        exception = exception_factory()
        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert result.backend_id == "postgres"


# =============================================================================
# Success Flag Tests
# =============================================================================


class TestSuccessFlag:
    """Test that success is always False for error responses."""

    @pytest.mark.parametrize(
        "exception_factory",
        [
            lambda: TimeoutError("timeout"),
            lambda: make_infra_timeout_error("infra timeout"),
            lambda: make_infra_auth_error("auth failed"),
            lambda: make_infra_connection_error("connection refused"),
            lambda: make_repository_execution_error("repo error"),
            lambda: ValueError("generic error"),
        ],
        ids=[
            "TimeoutError",
            "InfraTimeoutError",
            "InfraAuthenticationError",
            "InfraConnectionError",
            "RepositoryExecutionError",
            "ValueError",
        ],
    )
    def test_success_is_always_false(
        self,
        handler: TestHandler,
        correlation_id: UUID,
        start_time: float,
        exception_factory: Callable[[], Exception],
    ) -> None:
        """Verify success is always False for all exception types."""
        exception = exception_factory()
        ctx = PostgresErrorContext(
            exception=exception,
            operation="test_operation",
            correlation_id=correlation_id,
            start_time=start_time,
        )

        result = handler._build_error_response(ctx)

        assert result.success is False
