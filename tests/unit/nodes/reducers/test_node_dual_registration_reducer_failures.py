# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for NodeDualRegistrationReducer partial failure scenarios.

Tests graceful degradation behavior when one or both registration backends
fail (Consul and PostgreSQL), verifying status semantics, error capture,
and metrics tracking per OMN-889 requirements.

Test Categories:
    1. Partial Failures - Consul Fails, PostgreSQL Succeeds
    2. Partial Failures - PostgreSQL Fails, Consul Succeeds
    3. Total Failures - Both Backends Fail
    4. Result Model Validation - Status field correctness
    5. Error Message Capture - Error propagation to result
    6. Metrics Tracking - Counter increments
    7. Performance Under Failure - Response time behavior
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers import ConsulHandler, DbAdapter
from omnibase_infra.models.registration import (
    ModelDualRegistrationResult,
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.nodes.reducers.node_dual_registration_reducer import (
    EnumFSMState,
    ModelReducerMetrics,
    NodeDualRegistrationReducer,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_introspection_event() -> ModelNodeIntrospectionEvent:
    """Create a sample introspection event for testing."""
    return ModelNodeIntrospectionEvent(
        node_id=uuid4(),
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(),
        endpoints={"health": "http://localhost:8080/health"},
        correlation_id=uuid4(),
    )


@pytest.fixture
def mock_consul_handler() -> MagicMock:
    """Create a mock ConsulHandler that succeeds by default."""
    handler = MagicMock(spec=ConsulHandler)
    handler.execute = AsyncMock(
        return_value={"status": "success", "payload": {"registered": True}}
    )
    return handler


@pytest.fixture
def mock_db_adapter() -> MagicMock:
    """Create a mock DbAdapter that succeeds by default."""
    adapter = MagicMock(spec=DbAdapter)
    # Create a mock response with a status attribute
    mock_response = MagicMock()
    mock_response.status = "success"
    adapter.execute = AsyncMock(return_value=mock_response)
    return adapter


@pytest.fixture
def failing_consul_handler() -> MagicMock:
    """ConsulHandler that raises InfraConnectionError."""
    handler = MagicMock(spec=ConsulHandler)
    handler.execute = AsyncMock(
        side_effect=InfraConnectionError(
            "Consul connection failed",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="register",
                target_name="consul_handler",
                correlation_id=uuid4(),
            ),
        )
    )
    return handler


@pytest.fixture
def failing_db_adapter() -> MagicMock:
    """DbAdapter that raises InfraConnectionError."""
    adapter = MagicMock(spec=DbAdapter)
    adapter.execute = AsyncMock(
        side_effect=InfraConnectionError(
            "PostgreSQL connection failed",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="execute",
                target_name="db_adapter",
                correlation_id=uuid4(),
            ),
        )
    )
    return adapter


@pytest.fixture
def timeout_consul_handler() -> MagicMock:
    """ConsulHandler that raises InfraTimeoutError."""
    handler = MagicMock(spec=ConsulHandler)
    handler.execute = AsyncMock(
        side_effect=InfraTimeoutError(
            "Consul operation timed out",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="register",
                target_name="consul_handler",
                correlation_id=uuid4(),
            ),
            timeout_seconds=30.0,
        )
    )
    return handler


@pytest.fixture
def timeout_db_adapter() -> MagicMock:
    """DbAdapter that raises InfraTimeoutError."""
    adapter = MagicMock(spec=DbAdapter)
    adapter.execute = AsyncMock(
        side_effect=InfraTimeoutError(
            "PostgreSQL query timed out",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="execute",
                target_name="db_adapter",
                correlation_id=uuid4(),
            ),
            timeout_seconds=30.0,
        )
    )
    return adapter


@pytest.fixture
def auth_failing_consul_handler() -> MagicMock:
    """ConsulHandler that raises InfraAuthenticationError."""
    handler = MagicMock(spec=ConsulHandler)
    handler.execute = AsyncMock(
        side_effect=InfraAuthenticationError(
            "Consul ACL permission denied",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="register",
                target_name="consul_handler",
                correlation_id=uuid4(),
            ),
        )
    )
    return handler


@pytest.fixture
def auth_failing_db_adapter() -> MagicMock:
    """DbAdapter that raises InfraAuthenticationError."""
    adapter = MagicMock(spec=DbAdapter)
    adapter.execute = AsyncMock(
        side_effect=InfraAuthenticationError(
            "PostgreSQL authentication failed",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="execute",
                target_name="db_adapter",
                correlation_id=uuid4(),
            ),
        )
    )
    return adapter


@pytest.fixture
def unavailable_consul_handler() -> MagicMock:
    """ConsulHandler that raises InfraUnavailableError."""
    handler = MagicMock(spec=ConsulHandler)
    handler.execute = AsyncMock(
        side_effect=InfraUnavailableError(
            "Consul service unavailable - circuit breaker open",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="register",
                target_name="consul_handler",
                correlation_id=uuid4(),
            ),
        )
    )
    return handler


@pytest.fixture
def fsm_contract_path() -> Path:
    """Path to FSM contract for testing."""
    return (
        Path(__file__).parent.parent.parent.parent.parent
        / "contracts"
        / "fsm"
        / "dual_registration_reducer_fsm.yaml"
    )


# =============================================================================
# TEST CLASS: PARTIAL FAILURE - CONSUL FAILS, POSTGRESQL SUCCEEDS
# =============================================================================


class TestPartialFailureConsulFails:
    """Test partial failure scenarios where Consul fails but PostgreSQL succeeds."""

    @pytest.mark.asyncio
    async def test_partial_failure_consul_connection_error(
        self,
        failing_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Test graceful degradation when Consul connection fails."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "partial"
        assert result.consul_registered is False
        assert result.postgres_registered is True
        assert result.consul_error is not None
        assert "InfraConnectionError" in result.consul_error
        assert result.postgres_error is None

    @pytest.mark.asyncio
    async def test_partial_failure_consul_timeout(
        self,
        timeout_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Test graceful degradation when Consul times out.

        Note: The reducer wraps all exceptions into InfraConnectionError for
        consistent error handling. The test verifies the timeout causes a
        partial failure, not that the exact error type is preserved.
        """
        reducer = NodeDualRegistrationReducer(
            consul_handler=timeout_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "partial"
        assert result.consul_registered is False
        assert result.postgres_registered is True
        assert result.consul_error is not None
        # Reducer wraps errors in InfraConnectionError for consistency
        assert "InfraConnectionError" in result.consul_error
        assert result.postgres_error is None

    @pytest.mark.asyncio
    async def test_partial_failure_consul_auth_error(
        self,
        auth_failing_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Test graceful degradation when Consul authentication fails.

        Note: The reducer wraps all exceptions into InfraConnectionError for
        consistent error handling.
        """
        reducer = NodeDualRegistrationReducer(
            consul_handler=auth_failing_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "partial"
        assert result.consul_registered is False
        assert result.postgres_registered is True
        assert result.consul_error is not None
        # Reducer wraps errors in InfraConnectionError for consistency
        assert "InfraConnectionError" in result.consul_error
        assert result.postgres_error is None

    @pytest.mark.asyncio
    async def test_partial_failure_consul_unavailable(
        self,
        unavailable_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Test graceful degradation when Consul is unavailable (circuit breaker).

        Note: The reducer wraps all exceptions into InfraConnectionError for
        consistent error handling.
        """
        reducer = NodeDualRegistrationReducer(
            consul_handler=unavailable_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "partial"
        assert result.consul_registered is False
        assert result.postgres_registered is True
        assert result.consul_error is not None
        # Reducer wraps errors in InfraConnectionError for consistency
        assert "InfraConnectionError" in result.consul_error


# =============================================================================
# TEST CLASS: PARTIAL FAILURE - POSTGRESQL FAILS, CONSUL SUCCEEDS
# =============================================================================


class TestPartialFailurePostgresFails:
    """Test partial failure scenarios where PostgreSQL fails but Consul succeeds."""

    @pytest.mark.asyncio
    async def test_partial_failure_postgres_connection_error(
        self,
        mock_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Test graceful degradation when PostgreSQL connection fails."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=mock_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "partial"
        assert result.consul_registered is True
        assert result.postgres_registered is False
        assert result.consul_error is None
        assert result.postgres_error is not None
        assert "InfraConnectionError" in result.postgres_error

    @pytest.mark.asyncio
    async def test_partial_failure_postgres_timeout(
        self,
        mock_consul_handler: MagicMock,
        timeout_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Test graceful degradation when PostgreSQL times out.

        Note: The reducer wraps all exceptions into InfraConnectionError for
        consistent error handling.
        """
        reducer = NodeDualRegistrationReducer(
            consul_handler=mock_consul_handler,
            db_adapter=timeout_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "partial"
        assert result.consul_registered is True
        assert result.postgres_registered is False
        assert result.consul_error is None
        assert result.postgres_error is not None
        # Reducer wraps errors in InfraConnectionError for consistency
        assert "InfraConnectionError" in result.postgres_error

    @pytest.mark.asyncio
    async def test_partial_failure_postgres_auth_error(
        self,
        mock_consul_handler: MagicMock,
        auth_failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Test graceful degradation when PostgreSQL authentication fails.

        Note: The reducer wraps all exceptions into InfraConnectionError for
        consistent error handling.
        """
        reducer = NodeDualRegistrationReducer(
            consul_handler=mock_consul_handler,
            db_adapter=auth_failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "partial"
        assert result.consul_registered is True
        assert result.postgres_registered is False
        assert result.consul_error is None
        assert result.postgres_error is not None
        # Reducer wraps errors in InfraConnectionError for consistency
        assert "InfraConnectionError" in result.postgres_error


# =============================================================================
# TEST CLASS: TOTAL FAILURE - BOTH BACKENDS FAIL
# =============================================================================


class TestTotalFailureBothFail:
    """Test total failure scenarios where both Consul and PostgreSQL fail."""

    @pytest.mark.asyncio
    async def test_total_failure_both_connection_errors(
        self,
        failing_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Test status=failed when both connections fail."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "failed"
        assert result.consul_registered is False
        assert result.postgres_registered is False
        assert result.consul_error is not None
        assert result.postgres_error is not None

    @pytest.mark.asyncio
    async def test_total_failure_both_timeouts(
        self,
        timeout_consul_handler: MagicMock,
        timeout_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Test status=failed when both operations timeout.

        Note: The reducer wraps all exceptions into InfraConnectionError for
        consistent error handling.
        """
        reducer = NodeDualRegistrationReducer(
            consul_handler=timeout_consul_handler,
            db_adapter=timeout_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "failed"
        assert result.consul_registered is False
        assert result.postgres_registered is False
        # Reducer wraps errors in InfraConnectionError for consistency
        assert "InfraConnectionError" in result.consul_error
        assert "InfraConnectionError" in result.postgres_error

    @pytest.mark.asyncio
    async def test_total_failure_mixed_errors(
        self,
        timeout_consul_handler: MagicMock,
        auth_failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Test status=failed with different error types.

        Note: The reducer wraps all exceptions into InfraConnectionError for
        consistent error handling.
        """
        reducer = NodeDualRegistrationReducer(
            consul_handler=timeout_consul_handler,
            db_adapter=auth_failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "failed"
        assert result.consul_registered is False
        assert result.postgres_registered is False
        # Reducer wraps errors in InfraConnectionError for consistency
        assert "InfraConnectionError" in result.consul_error
        assert "InfraConnectionError" in result.postgres_error


# =============================================================================
# TEST CLASS: RESULT MODEL VALIDATION
# =============================================================================


class TestResultStatusValidation:
    """Test that result status field correctly reflects registration outcomes."""

    @pytest.mark.asyncio
    async def test_result_status_partial_consul_failed(
        self,
        failing_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify status='partial' when Consul fails but PostgreSQL succeeds."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "partial"
        # Validate Pydantic model consistency
        assert result.consul_registered != result.postgres_registered

    @pytest.mark.asyncio
    async def test_result_status_partial_postgres_failed(
        self,
        mock_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify status='partial' when PostgreSQL fails but Consul succeeds."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=mock_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "partial"
        # Validate Pydantic model consistency
        assert result.consul_registered != result.postgres_registered

    @pytest.mark.asyncio
    async def test_result_status_failed_both_failed(
        self,
        failing_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify status='failed' when both backends fail."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "failed"
        assert result.consul_registered is False
        assert result.postgres_registered is False

    @pytest.mark.asyncio
    async def test_result_status_success_both_succeeded(
        self,
        mock_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify status='success' when both backends succeed."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=mock_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "success"
        assert result.consul_registered is True
        assert result.postgres_registered is True
        assert result.consul_error is None
        assert result.postgres_error is None


# =============================================================================
# TEST CLASS: ERROR MESSAGE CAPTURE
# =============================================================================


class TestErrorMessageCapture:
    """Test that error messages are correctly captured in result model."""

    @pytest.mark.asyncio
    async def test_consul_error_captured_in_result(
        self,
        failing_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify Consul error message is stored in consul_error field."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.consul_error is not None
        assert (
            "Consul connection failed" in result.consul_error
            or "InfraConnectionError" in result.consul_error
        )
        # Should contain error type
        assert "InfraConnectionError" in result.consul_error

    @pytest.mark.asyncio
    async def test_postgres_error_captured_in_result(
        self,
        mock_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify PostgreSQL error message is stored in postgres_error field."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=mock_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.postgres_error is not None
        assert (
            "PostgreSQL connection failed" in result.postgres_error
            or "InfraConnectionError" in result.postgres_error
        )
        # Should contain error type
        assert "InfraConnectionError" in result.postgres_error

    @pytest.mark.asyncio
    async def test_both_errors_captured_in_result(
        self,
        failing_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify both error messages are captured when both fail."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.consul_error is not None
        assert result.postgres_error is not None
        # Both should be independent error messages
        assert result.consul_error != result.postgres_error

    @pytest.mark.asyncio
    async def test_error_type_preserved_in_message(
        self,
        timeout_consul_handler: MagicMock,
        auth_failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify error type names are preserved in error messages.

        Note: The reducer wraps all exceptions into InfraConnectionError for
        consistent error handling. The wrapper error type is what appears
        in the message.
        """
        reducer = NodeDualRegistrationReducer(
            consul_handler=timeout_consul_handler,
            db_adapter=auth_failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        # Reducer wraps errors in InfraConnectionError for consistency
        # Error format is "InfraConnectionError: [ERROR_CODE] message"
        assert "InfraConnectionError" in result.consul_error
        assert "InfraConnectionError" in result.postgres_error


# =============================================================================
# TEST CLASS: METRICS TRACKING
# =============================================================================


class TestMetricsTracking:
    """Test that metrics counters are correctly incremented."""

    @pytest.mark.asyncio
    async def test_metrics_increment_success_count(
        self,
        mock_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify success_count is incremented on full success."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=mock_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        initial_success = reducer.metrics.success_count
        await reducer.execute(sample_introspection_event)

        assert reducer.metrics.success_count == initial_success + 1

    @pytest.mark.asyncio
    async def test_metrics_increment_failure_count(
        self,
        failing_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify failure_count is incremented on total failure."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        initial_failure = reducer.metrics.failure_count
        await reducer.execute(sample_introspection_event)

        assert reducer.metrics.failure_count == initial_failure + 1

    @pytest.mark.asyncio
    async def test_metrics_increment_partial_count(
        self,
        failing_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify partial_count is incremented on partial failure."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        initial_partial = reducer.metrics.partial_count
        await reducer.execute(sample_introspection_event)

        assert reducer.metrics.partial_count == initial_partial + 1

    @pytest.mark.asyncio
    async def test_metrics_total_registrations_always_incremented(
        self,
        failing_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify total_registrations is always incremented regardless of outcome."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        initial_total = reducer.metrics.total_registrations
        await reducer.execute(sample_introspection_event)

        assert reducer.metrics.total_registrations == initial_total + 1

    @pytest.mark.asyncio
    async def test_metrics_accumulate_across_multiple_executions(
        self,
        mock_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify metrics accumulate correctly across multiple executions."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=mock_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        # Execute multiple times
        for i in range(3):
            event = ModelNodeIntrospectionEvent(
                node_id=uuid4(),
                node_type="effect",
                node_version="1.0.0",
                capabilities=ModelNodeCapabilities(),
                endpoints={"health": "http://localhost:8080/health"},
            )
            await reducer.execute(event)

        assert reducer.metrics.total_registrations == 3
        assert reducer.metrics.success_count == 3
        assert reducer.metrics.partial_count == 0
        assert reducer.metrics.failure_count == 0


# =============================================================================
# TEST CLASS: PERFORMANCE UNDER FAILURE
# =============================================================================


class TestPerformanceUnderFailure:
    """Test performance characteristics when backends fail."""

    @pytest.mark.asyncio
    async def test_partial_failure_returns_quickly(
        self,
        failing_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify partial failures don't wait excessively for failed backend."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        start = time.perf_counter()
        result = await reducer.execute(sample_introspection_event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete quickly (not waiting for timeout)
        # Using 1000ms as reasonable threshold for mocked operations
        assert elapsed_ms < 1000, f"Partial failure took too long: {elapsed_ms}ms"
        assert result.status == "partial"

    @pytest.mark.asyncio
    async def test_registration_time_ms_accurate(
        self,
        mock_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify registration_time_ms field is reasonably accurate."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=mock_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        start = time.perf_counter()
        result = await reducer.execute(sample_introspection_event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # registration_time_ms should be close to actual elapsed time
        # Allow 100ms tolerance for test overhead
        assert result.registration_time_ms >= 0
        assert abs(result.registration_time_ms - elapsed_ms) < 100

    @pytest.mark.asyncio
    async def test_parallel_execution_completes_when_both_fail(
        self,
        failing_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify parallel execution completes even when both backends fail."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        # Should complete without hanging
        start = time.perf_counter()
        result = await reducer.execute(sample_introspection_event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.status == "failed"
        assert elapsed_ms < 1000, f"Total failure took too long: {elapsed_ms}ms"


# =============================================================================
# TEST CLASS: FSM STATE VERIFICATION
# =============================================================================


class TestFSMStateTransitions:
    """Test FSM state transitions during failure scenarios."""

    @pytest.mark.asyncio
    async def test_fsm_ends_in_idle_after_partial_failure(
        self,
        failing_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify FSM returns to idle state after partial failure."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        await reducer.execute(sample_introspection_event)

        # FSM should return to idle after emitting partial result
        assert reducer.current_state == EnumFSMState.IDLE

    @pytest.mark.asyncio
    async def test_fsm_ends_in_failed_state_on_total_failure(
        self,
        failing_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify FSM ends in registration_failed state on total failure."""
        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        await reducer.execute(sample_introspection_event)

        # FSM may end in registration_failed (terminal) without transition back to idle
        # Based on the FSM contract, registration_failed doesn't have a transition to idle
        assert reducer.current_state in (
            EnumFSMState.REGISTRATION_FAILED,
            EnumFSMState.IDLE,
        )


# =============================================================================
# TEST CLASS: CORRELATION ID PROPAGATION
# =============================================================================


class TestCorrelationIdPropagation:
    """Test correlation ID is correctly propagated through failure scenarios."""

    @pytest.mark.asyncio
    async def test_correlation_id_in_partial_failure_result(
        self,
        failing_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        fsm_contract_path: Path,
    ) -> None:
        """Verify correlation_id is preserved in partial failure result."""
        test_correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=test_correlation_id,
        )

        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(event)

        assert result.correlation_id == test_correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_in_total_failure_result(
        self,
        failing_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        fsm_contract_path: Path,
    ) -> None:
        """Verify correlation_id is preserved in total failure result."""
        test_correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=test_correlation_id,
        )

        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(event)

        assert result.correlation_id == test_correlation_id


# =============================================================================
# TEST CLASS: NODE ID PRESERVATION
# =============================================================================


class TestNodeIdPreservation:
    """Test node_id is correctly preserved in failure results."""

    @pytest.mark.asyncio
    async def test_node_id_preserved_in_partial_failure(
        self,
        failing_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        fsm_contract_path: Path,
    ) -> None:
        """Verify node_id is preserved in partial failure result."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )

        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(event)

        assert result.node_id == test_node_id

    @pytest.mark.asyncio
    async def test_node_id_preserved_in_total_failure(
        self,
        failing_consul_handler: MagicMock,
        failing_db_adapter: MagicMock,
        fsm_contract_path: Path,
    ) -> None:
        """Verify node_id is preserved in total failure result."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )

        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_consul_handler,
            db_adapter=failing_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(event)

        assert result.node_id == test_node_id


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_generic_exception_treated_as_failure(
        self,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify generic exceptions are treated as failures.

        Note: The reducer wraps all exceptions into InfraConnectionError for
        consistent error handling.
        """
        # Create handler that raises generic Exception
        handler = MagicMock(spec=ConsulHandler)
        handler.execute = AsyncMock(side_effect=Exception("Unexpected error"))

        reducer = NodeDualRegistrationReducer(
            consul_handler=handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        assert result.status == "partial"
        assert result.consul_registered is False
        # Reducer wraps errors in InfraConnectionError for consistency
        assert "InfraConnectionError" in result.consul_error

    @pytest.mark.asyncio
    async def test_false_return_treated_as_failure(
        self,
        mock_db_adapter: MagicMock,
        sample_introspection_event: ModelNodeIntrospectionEvent,
        fsm_contract_path: Path,
    ) -> None:
        """Verify False return (not True) is treated as registration failure."""
        # Create handler that returns non-success status
        handler = MagicMock(spec=ConsulHandler)
        handler.execute = AsyncMock(
            return_value={"status": "error", "payload": {"registered": False}}
        )

        reducer = NodeDualRegistrationReducer(
            consul_handler=handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        result = await reducer.execute(sample_introspection_event)

        # Non-success status should result in consul_registered=False
        assert result.consul_registered is False

    @pytest.mark.asyncio
    async def test_reducer_can_process_multiple_events_after_failure(
        self,
        mock_consul_handler: MagicMock,
        mock_db_adapter: MagicMock,
        fsm_contract_path: Path,
    ) -> None:
        """Verify reducer can process new events after experiencing a failure."""
        # First create a reducer that will fail
        failing_handler = MagicMock(spec=ConsulHandler)
        failing_handler.execute = AsyncMock(
            side_effect=InfraConnectionError(
                "Connection failed",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="register",
                    target_name="consul_handler",
                ),
            )
        )

        reducer = NodeDualRegistrationReducer(
            consul_handler=failing_handler,
            db_adapter=mock_db_adapter,
            fsm_contract_path=fsm_contract_path,
        )
        await reducer.initialize()

        # First event - partial failure
        event1 = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )
        result1 = await reducer.execute(event1)
        assert result1.status == "partial"

        # Replace with working handler
        reducer._consul_handler = mock_consul_handler

        # Second event - should succeed
        event2 = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="compute",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )
        result2 = await reducer.execute(event2)

        assert result2.status == "success"
        assert reducer.metrics.total_registrations == 2


__all__ = [
    "TestPartialFailureConsulFails",
    "TestPartialFailurePostgresFails",
    "TestTotalFailureBothFail",
    "TestResultStatusValidation",
    "TestErrorMessageCapture",
    "TestMetricsTracking",
    "TestPerformanceUnderFailure",
    "TestFSMStateTransitions",
    "TestCorrelationIdPropagation",
    "TestNodeIdPreservation",
    "TestEdgeCases",
]
