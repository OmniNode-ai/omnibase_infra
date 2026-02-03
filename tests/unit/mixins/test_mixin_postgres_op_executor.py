# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for MixinPostgresOpExecutor.

These tests verify the core execution mechanics of the PostgreSQL operation
executor mixin, ensuring consistent behavior across all handlers that use it.

Test Categories:
    - Success path: duration_ms present, correct result structure
    - Error classification: each exception type maps to correct error code
    - Error sanitization: no credentials exposed in error messages
    - Retriability: timeout and connection errors marked retriable
    - Logging: structured log entries with correlation_id

See Also:
    - MixinPostgresOpExecutor: The mixin under test
    - OMN-1857: Extraction ticket for this mixin
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    RepositoryExecutionError,
)
from omnibase_infra.mixins.mixin_postgres_op_executor import MixinPostgresOpExecutor
from omnibase_infra.models.model_backend_result import ModelBackendResult


class ConcreteExecutor(MixinPostgresOpExecutor):
    """Concrete implementation for testing the mixin."""


@pytest.fixture
def executor() -> ConcreteExecutor:
    """Create a concrete executor for testing."""
    return ConcreteExecutor()


@pytest.fixture
def correlation_id():
    """Generate a correlation ID for tests."""
    return uuid4()


@pytest.fixture
def log_context() -> dict[str, Any]:
    """Sample log context for tests."""
    return {"contract_id": "test-contract:1.0.0", "node_name": "test-node"}


class TestSuccessPath:
    """Tests for successful operation execution."""

    @pytest.mark.asyncio
    async def test_returns_success_result(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify successful operation returns success=True."""
        fn = AsyncMock(return_value=None)

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is True
        assert result.error is None
        assert result.error_code is None
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duration_ms_always_present(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify duration_ms is always present in result."""
        fn = AsyncMock(return_value=None)

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.duration_ms is not None
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_duration_ms_present_on_error(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify duration_ms is present even when operation fails."""
        fn = AsyncMock(side_effect=TimeoutError("connection timed out"))

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is False
        assert result.duration_ms is not None
        assert result.duration_ms >= 0


class TestErrorClassification:
    """Tests for exception-to-error-code mapping."""

    @pytest.mark.asyncio
    async def test_timeout_error_classification(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify TimeoutError maps to TIMEOUT_ERROR."""
        fn = AsyncMock(side_effect=TimeoutError("connection timed out"))

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR
        assert EnumPostgresErrorCode.TIMEOUT_ERROR.is_retriable is True

    @pytest.mark.asyncio
    async def test_infra_timeout_error_classification(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify InfraTimeoutError maps to TIMEOUT_ERROR."""
        context = ModelTimeoutErrorContext(
            transport_type="db",
            operation="test",
            timeout_seconds=30.0,
        )
        fn = AsyncMock(
            side_effect=InfraTimeoutError("query timed out", context=context)
        )

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_auth_error_classification(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify InfraAuthenticationError maps to AUTH_ERROR (non-retriable)."""
        context = ModelInfraErrorContext.with_correlation(
            transport_type="db",
            operation="test",
        )
        fn = AsyncMock(
            side_effect=InfraAuthenticationError("invalid credentials", context=context)
        )

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.AUTH_ERROR
        assert EnumPostgresErrorCode.AUTH_ERROR.is_retriable is False

    @pytest.mark.asyncio
    async def test_connection_error_classification(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify InfraConnectionError maps to CONNECTION_ERROR (retriable)."""
        context = ModelInfraErrorContext.with_correlation(
            transport_type="db",
            operation="test",
        )
        fn = AsyncMock(
            side_effect=InfraConnectionError("connection refused", context=context)
        )

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.CONNECTION_ERROR
        assert EnumPostgresErrorCode.CONNECTION_ERROR.is_retriable is True

    @pytest.mark.asyncio
    async def test_repository_execution_error_uses_op_error_code(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify RepositoryExecutionError uses handler-provided error code."""
        context = ModelInfraErrorContext.with_correlation(
            transport_type="db",
            operation="test",
        )
        fn = AsyncMock(
            side_effect=RepositoryExecutionError(
                "constraint violation", context=context
            )
        )

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.HEARTBEAT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.HEARTBEAT_ERROR

    @pytest.mark.asyncio
    async def test_unknown_exception_classification(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify generic Exception maps to UNKNOWN_ERROR (non-retriable)."""
        fn = AsyncMock(side_effect=RuntimeError("unexpected internal error"))

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UNKNOWN_ERROR
        assert EnumPostgresErrorCode.UNKNOWN_ERROR.is_retriable is False


class TestErrorSanitization:
    """Tests for error message sanitization."""

    @pytest.mark.asyncio
    async def test_credentials_not_exposed_in_error(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify credentials are not exposed in sanitized error messages."""
        # Simulate an error with credentials in the message
        fn = AsyncMock(
            side_effect=RuntimeError(
                "connection to postgres://user:secret_password@host:5432/db failed"
            )
        )

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is False
        # The sanitized error should not contain the password
        assert "secret_password" not in result.error

    @pytest.mark.asyncio
    async def test_error_message_present_on_failure(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify error message is present (not empty) on failure."""
        fn = AsyncMock(side_effect=TimeoutError("query timeout after 30s"))

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is False
        assert result.error != ""


class TestCorrelationIdPropagation:
    """Tests for correlation ID handling."""

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_on_success(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify correlation_id is propagated on success."""
        fn = AsyncMock(return_value=None)

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_on_failure(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
    ) -> None:
        """Verify correlation_id is propagated on failure."""
        fn = AsyncMock(side_effect=TimeoutError("timeout"))

        result = await executor._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.correlation_id == correlation_id


class TestAllErrorCodesUsed:
    """Verify all operation error codes can be used with the executor."""

    @pytest.mark.parametrize(
        "op_error_code",
        [
            EnumPostgresErrorCode.UPSERT_ERROR,
            EnumPostgresErrorCode.TOPIC_UPDATE_ERROR,
            EnumPostgresErrorCode.MARK_STALE_ERROR,
            EnumPostgresErrorCode.HEARTBEAT_ERROR,
            EnumPostgresErrorCode.DEACTIVATE_ERROR,
            EnumPostgresErrorCode.CLEANUP_ERROR,
        ],
    )
    @pytest.mark.asyncio
    async def test_all_operation_error_codes_work(
        self,
        executor: ConcreteExecutor,
        correlation_id,
        log_context: dict[str, Any],
        op_error_code: EnumPostgresErrorCode,
    ) -> None:
        """Verify all operation error codes work correctly."""
        context = ModelInfraErrorContext.with_correlation(
            transport_type="db",
            operation="test",
        )
        fn = AsyncMock(
            side_effect=RepositoryExecutionError("operation failed", context=context)
        )

        result = await executor._execute_postgres_op(
            op_error_code=op_error_code,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=fn,
        )

        assert result.success is False
        assert result.error_code == op_error_code
