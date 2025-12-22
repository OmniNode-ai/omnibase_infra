# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Effect circuit breaker integration tests for OMN-954.

This test suite validates circuit breaker behavior at the effect layer level,
testing how infrastructure effects (Consul, PostgreSQL, etc.) integrate with
the MixinAsyncCircuitBreaker for fault tolerance.

Test Cases (G4 Acceptance Criteria):
    1. test_circuit_breaker_transitions_under_failure
       - Start CLOSED -> failures -> OPEN -> timeout -> HALF_OPEN -> success -> CLOSED

    2. test_circuit_breaker_blocks_requests_when_open
       - Open circuit -> attempt operation -> InfraUnavailableError raised
       - Verify no I/O attempted to backend

    3. test_circuit_breaker_per_backend
       - Consul circuit opens, PostgreSQL stays closed
       - Verify independent circuit breaker isolation

    4. test_circuit_breaker_correlation_id_propagation
       - Process intent with specific correlation_id
       - Trigger failure -> verify correlation_id in error context

These tests simulate effect-level circuit breaker integration without
requiring actual infrastructure backends.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraUnavailableError,
    ModelInfraErrorContext,
)
from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    CircuitState,
    MixinAsyncCircuitBreaker,
)


class MockConsulBackend:
    """Mock Consul backend that can be configured to succeed or fail."""

    def __init__(self) -> None:
        self.call_count = 0
        self.should_fail = False
        self.failure_exception: Exception | None = None

    def register_service(self, service_name: str) -> dict[str, str]:
        """Simulate service registration."""
        self.call_count += 1
        if self.should_fail and self.failure_exception:
            raise self.failure_exception
        if self.should_fail:
            raise ConnectionError("Consul connection refused")
        return {"status": "registered", "service": service_name}

    def reset(self) -> None:
        """Reset mock state."""
        self.call_count = 0
        self.should_fail = False
        self.failure_exception = None


class MockPostgresBackend:
    """Mock PostgreSQL backend that can be configured to succeed or fail."""

    def __init__(self) -> None:
        self.call_count = 0
        self.should_fail = False
        self.failure_exception: Exception | None = None

    def execute_query(self, query: str) -> list[dict[str, str]]:
        """Simulate query execution."""
        self.call_count += 1
        if self.should_fail and self.failure_exception:
            raise self.failure_exception
        if self.should_fail:
            raise ConnectionError("PostgreSQL connection refused")
        return [{"result": "success", "query": query}]

    def reset(self) -> None:
        """Reset mock state."""
        self.call_count = 0
        self.should_fail = False
        self.failure_exception = None


class MockConsulEffect(MixinAsyncCircuitBreaker):
    """Mock Consul effect with circuit breaker integration.

    Simulates a real infrastructure effect that uses the circuit breaker
    mixin for fault tolerance, similar to ConsulHandler or VaultAdapter.
    """

    def __init__(
        self,
        backend: MockConsulBackend,
        failure_threshold: int = 3,
        reset_timeout: float = 30.0,
    ) -> None:
        """Initialize effect with circuit breaker."""
        self._backend = backend
        self._initialized = True

        # Initialize circuit breaker using mixin
        self._init_circuit_breaker(
            threshold=failure_threshold,
            reset_timeout=reset_timeout,
            service_name="consul.test",
            transport_type=EnumInfraTransportType.CONSUL,
        )

    async def register_service(
        self,
        service_name: str,
        correlation_id: UUID | None = None,
    ) -> dict[str, str]:
        """Register service with circuit breaker protection.

        This follows the pattern used by ConsulHandler and VaultAdapter.
        """
        if correlation_id is None:
            correlation_id = uuid4()

        # Check circuit breaker before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("register_service", correlation_id)

        try:
            # Simulate async operation by running in executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: self._backend.register_service(service_name)
            )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return result

        except Exception as e:
            # Record failure and potentially open circuit
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("register_service", correlation_id)

            # Raise appropriate infrastructure error
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="register_service",
                target_name="consul.test",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"Consul operation failed: {type(e).__name__}",
                context=context,
            ) from e

    def get_circuit_state(self) -> CircuitState:
        """Get current circuit state (for test assertions)."""
        if self._circuit_breaker_open:
            return CircuitState.OPEN
        return CircuitState.CLOSED

    def get_failure_count(self) -> int:
        """Get current failure count (for test assertions)."""
        return self._circuit_breaker_failures


class MockPostgresEffect(MixinAsyncCircuitBreaker):
    """Mock PostgreSQL effect with independent circuit breaker.

    This simulates a separate infrastructure effect with its own
    circuit breaker, demonstrating per-backend isolation.
    """

    def __init__(
        self,
        backend: MockPostgresBackend,
        failure_threshold: int = 3,
        reset_timeout: float = 30.0,
    ) -> None:
        """Initialize effect with circuit breaker."""
        self._backend = backend
        self._initialized = True

        # Initialize circuit breaker using mixin
        self._init_circuit_breaker(
            threshold=failure_threshold,
            reset_timeout=reset_timeout,
            service_name="postgres.test",
            transport_type=EnumInfraTransportType.DATABASE,
        )

    async def execute_query(
        self,
        query: str,
        correlation_id: UUID | None = None,
    ) -> list[dict[str, str]]:
        """Execute query with circuit breaker protection."""
        if correlation_id is None:
            correlation_id = uuid4()

        # Check circuit breaker before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("execute_query", correlation_id)

        try:
            # Simulate async operation
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: self._backend.execute_query(query)
            )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return result

        except Exception as e:
            # Record failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("execute_query", correlation_id)

            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="execute_query",
                target_name="postgres.test",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"PostgreSQL operation failed: {type(e).__name__}",
                context=context,
            ) from e

    def get_circuit_state(self) -> CircuitState:
        """Get current circuit state."""
        if self._circuit_breaker_open:
            return CircuitState.OPEN
        return CircuitState.CLOSED

    def get_failure_count(self) -> int:
        """Get current failure count."""
        return self._circuit_breaker_failures


@pytest.fixture
def mock_consul_backend() -> MockConsulBackend:
    """Create mock Consul backend."""
    return MockConsulBackend()


@pytest.fixture
def mock_postgres_backend() -> MockPostgresBackend:
    """Create mock PostgreSQL backend."""
    return MockPostgresBackend()


@pytest.fixture
def consul_effect(mock_consul_backend: MockConsulBackend) -> MockConsulEffect:
    """Create Consul effect with low threshold for testing."""
    return MockConsulEffect(
        backend=mock_consul_backend,
        failure_threshold=3,
        reset_timeout=0.1,  # Short timeout for tests
    )


@pytest.fixture
def postgres_effect(mock_postgres_backend: MockPostgresBackend) -> MockPostgresEffect:
    """Create PostgreSQL effect with low threshold for testing."""
    return MockPostgresEffect(
        backend=mock_postgres_backend,
        failure_threshold=3,
        reset_timeout=0.1,
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestEffectCircuitBreakerTransitions:
    """Test suite for circuit breaker state transitions at effect level (G4 test 1)."""

    async def test_circuit_breaker_transitions_under_failure(
        self,
        consul_effect: MockConsulEffect,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test complete circuit breaker state machine transitions.

        G4 Acceptance Criteria Test 1:
        - Start with circuit CLOSED
        - Simulate failures up to threshold
        - Verify circuit OPENS
        - Wait for reset timeout
        - Verify circuit transitions to HALF_OPEN
        - Simulate success
        - Verify circuit CLOSES
        """
        # Phase 1: Start with circuit CLOSED
        assert consul_effect.get_circuit_state() == CircuitState.CLOSED
        assert consul_effect.get_failure_count() == 0

        # Phase 2: Successful operation keeps circuit closed
        result = await consul_effect.register_service("test-service")
        assert result["status"] == "registered"
        assert consul_effect.get_circuit_state() == CircuitState.CLOSED

        # Phase 3: Configure backend to fail
        mock_consul_backend.should_fail = True

        # Phase 4: Simulate failures up to threshold (3 failures to open)
        for i in range(3):
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service(f"failing-service-{i}")
            # After each failure, check state
            if i < 2:
                # Before threshold, circuit stays closed
                assert consul_effect.get_circuit_state() == CircuitState.CLOSED
                assert consul_effect.get_failure_count() == i + 1
            else:
                # At threshold, circuit opens
                assert consul_effect.get_circuit_state() == CircuitState.OPEN

        # Phase 5: Verify circuit is OPEN
        assert consul_effect.get_circuit_state() == CircuitState.OPEN
        assert consul_effect.get_failure_count() >= 3

        # Phase 6: Wait for reset timeout (configured as 0.1s)
        await asyncio.sleep(0.15)

        # Phase 7: Configure backend to succeed
        mock_consul_backend.should_fail = False
        mock_consul_backend.reset()

        # Phase 8: Next operation should transition through HALF_OPEN to CLOSED
        # The check_circuit_breaker will transition OPEN -> HALF_OPEN after timeout
        # A successful operation then transitions HALF_OPEN -> CLOSED
        result = await consul_effect.register_service("recovered-service")
        assert result["status"] == "registered"

        # Phase 9: Verify circuit is CLOSED
        assert consul_effect.get_circuit_state() == CircuitState.CLOSED
        assert consul_effect.get_failure_count() == 0

    async def test_circuit_transitions_half_open_to_open_on_failure(
        self,
        consul_effect: MockConsulEffect,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test HALF_OPEN -> OPEN transition when recovery fails."""
        # Open the circuit
        mock_consul_backend.should_fail = True
        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service("failing-service")

        assert consul_effect.get_circuit_state() == CircuitState.OPEN

        # Wait for reset timeout to transition to HALF_OPEN
        await asyncio.sleep(0.15)

        # Backend still failing - should reopen circuit
        # With threshold=3, we need 3 failures to reopen
        for i in range(3):
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service(f"still-failing-{i}")

        # Circuit should be OPEN again
        assert consul_effect.get_circuit_state() == CircuitState.OPEN


@pytest.mark.unit
@pytest.mark.asyncio
class TestEffectCircuitBreakerBlocking:
    """Test suite for circuit breaker blocking behavior (G4 test 2)."""

    async def test_circuit_breaker_blocks_requests_when_open(
        self,
        consul_effect: MockConsulEffect,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test circuit blocks requests when OPEN and no I/O attempted.

        G4 Acceptance Criteria Test 2:
        - Open the circuit breaker
        - Attempt to process intent
        - Verify InfraUnavailableError raised
        - Verify no I/O attempted to backend
        """
        # Phase 1: Open the circuit with 3 failures
        mock_consul_backend.should_fail = True
        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service("failing-service")

        # Verify circuit is OPEN
        assert consul_effect.get_circuit_state() == CircuitState.OPEN

        # Phase 2: Reset backend call count to track new calls
        initial_call_count = mock_consul_backend.call_count

        # Phase 3: Attempt operation - should be blocked immediately
        with pytest.raises(InfraUnavailableError) as exc_info:
            await consul_effect.register_service("blocked-service")

        # Phase 4: Verify InfraUnavailableError with correct context
        error = exc_info.value
        assert "Circuit breaker is open" in error.message
        assert error.model.context.get("circuit_state") == "open"
        assert "retry_after_seconds" in error.model.context

        # Phase 5: Verify NO backend I/O was attempted
        assert mock_consul_backend.call_count == initial_call_count
        # The call count should not have increased - circuit blocked the request

    async def test_open_circuit_provides_retry_after(
        self,
        consul_effect: MockConsulEffect,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test open circuit includes retry_after_seconds in error."""
        # Open the circuit
        mock_consul_backend.should_fail = True
        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service("failing-service")

        # Verify retry_after_seconds is present
        with pytest.raises(InfraUnavailableError) as exc_info:
            await consul_effect.register_service("blocked-service")

        retry_after = exc_info.value.model.context.get("retry_after_seconds")
        assert retry_after is not None
        assert isinstance(retry_after, int)
        # Should be close to 0 since timeout is 0.1s
        assert retry_after >= 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestEffectCircuitBreakerIsolation:
    """Test suite for per-backend circuit breaker isolation (G4 test 3)."""

    async def test_circuit_breaker_per_backend(
        self,
        consul_effect: MockConsulEffect,
        postgres_effect: MockPostgresEffect,
        mock_consul_backend: MockConsulBackend,
        mock_postgres_backend: MockPostgresBackend,
    ) -> None:
        """Test circuit breakers are independent per backend.

        G4 Acceptance Criteria Test 3:
        - Consul circuit breaker opens
        - PostgreSQL circuit breaker stays closed
        - Verify Consul operations blocked
        - Verify PostgreSQL operations allowed
        """
        # Phase 1: Verify both circuits start CLOSED
        assert consul_effect.get_circuit_state() == CircuitState.CLOSED
        assert postgres_effect.get_circuit_state() == CircuitState.CLOSED

        # Phase 2: Make Consul fail and open its circuit
        mock_consul_backend.should_fail = True
        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service("failing-consul-service")

        # Phase 3: Verify Consul circuit is OPEN
        assert consul_effect.get_circuit_state() == CircuitState.OPEN

        # Phase 4: Verify PostgreSQL circuit is still CLOSED
        assert postgres_effect.get_circuit_state() == CircuitState.CLOSED

        # Phase 5: Verify PostgreSQL operations still work
        result = await postgres_effect.execute_query("SELECT 1")
        assert result[0]["result"] == "success"
        assert postgres_effect.get_circuit_state() == CircuitState.CLOSED

        # Phase 6: Verify Consul operations are blocked
        with pytest.raises(InfraUnavailableError) as exc_info:
            await consul_effect.register_service("blocked-consul-service")
        assert "Circuit breaker is open" in exc_info.value.message

        # Phase 7: PostgreSQL can handle its own failures independently
        mock_postgres_backend.should_fail = True
        with pytest.raises(InfraConnectionError):
            await postgres_effect.execute_query("SELECT 1")

        # Postgres has 1 failure, Consul still blocked
        assert postgres_effect.get_failure_count() == 1
        assert consul_effect.get_circuit_state() == CircuitState.OPEN

    async def test_multiple_effects_recover_independently(
        self,
        consul_effect: MockConsulEffect,
        postgres_effect: MockPostgresEffect,
        mock_consul_backend: MockConsulBackend,
        mock_postgres_backend: MockPostgresBackend,
    ) -> None:
        """Test effects recover independently after failures."""
        # Open both circuits
        mock_consul_backend.should_fail = True
        mock_postgres_backend.should_fail = True

        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service("failing")
            with pytest.raises(InfraConnectionError):
                await postgres_effect.execute_query("SELECT 1")

        assert consul_effect.get_circuit_state() == CircuitState.OPEN
        assert postgres_effect.get_circuit_state() == CircuitState.OPEN

        # Wait for reset timeout
        await asyncio.sleep(0.15)

        # Only Consul recovers
        mock_consul_backend.should_fail = False
        mock_consul_backend.reset()

        result = await consul_effect.register_service("recovered")
        assert result["status"] == "registered"
        assert consul_effect.get_circuit_state() == CircuitState.CLOSED

        # PostgreSQL still failing
        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await postgres_effect.execute_query("SELECT 1")
        assert postgres_effect.get_circuit_state() == CircuitState.OPEN


@pytest.mark.unit
@pytest.mark.asyncio
class TestEffectCircuitBreakerCorrelationId:
    """Test suite for correlation ID propagation (G4 test 4)."""

    async def test_circuit_breaker_correlation_id_propagation(
        self,
        consul_effect: MockConsulEffect,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test correlation_id propagates through circuit breaker errors.

        G4 Acceptance Criteria Test 4:
        - Process intent with specific correlation_id
        - Trigger circuit breaker failure
        - Verify correlation_id in InfraUnavailableError context
        """
        # Phase 1: Create a specific correlation_id
        test_correlation_id = uuid4()

        # Phase 2: Open the circuit with failures (using the correlation_id)
        mock_consul_backend.should_fail = True
        for _ in range(3):
            with pytest.raises(InfraConnectionError) as exc_info:
                await consul_effect.register_service(
                    "failing-service",
                    correlation_id=test_correlation_id,
                )
            # Verify correlation_id in connection error
            assert exc_info.value.model.correlation_id == test_correlation_id

        # Phase 3: Verify circuit is OPEN
        assert consul_effect.get_circuit_state() == CircuitState.OPEN

        # Phase 4: Attempt operation with the same correlation_id
        with pytest.raises(InfraUnavailableError) as exc_info:
            await consul_effect.register_service(
                "blocked-service",
                correlation_id=test_correlation_id,
            )

        # Phase 5: Verify correlation_id in InfraUnavailableError
        error = exc_info.value
        assert error.model.correlation_id == test_correlation_id
        assert "Circuit breaker is open" in error.message

    async def test_different_correlation_ids_tracked_separately(
        self,
        consul_effect: MockConsulEffect,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test different correlation IDs are preserved in separate requests."""
        # Open the circuit
        mock_consul_backend.should_fail = True
        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service("failing")

        # Use different correlation IDs for blocked requests
        correlation_id_1 = uuid4()
        correlation_id_2 = uuid4()

        with pytest.raises(InfraUnavailableError) as exc_info_1:
            await consul_effect.register_service(
                "blocked-1", correlation_id=correlation_id_1
            )
        assert exc_info_1.value.model.correlation_id == correlation_id_1

        with pytest.raises(InfraUnavailableError) as exc_info_2:
            await consul_effect.register_service(
                "blocked-2", correlation_id=correlation_id_2
            )
        assert exc_info_2.value.model.correlation_id == correlation_id_2

        # Correlation IDs should be different
        assert correlation_id_1 != correlation_id_2

    async def test_correlation_id_generated_when_none_provided(
        self,
        consul_effect: MockConsulEffect,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test correlation_id is auto-generated when not provided."""
        # Open the circuit
        mock_consul_backend.should_fail = True
        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service("failing")

        # Attempt without correlation_id
        with pytest.raises(InfraUnavailableError) as exc_info:
            await consul_effect.register_service("blocked-service")
            # No correlation_id passed

        # Verify a correlation_id was generated
        error = exc_info.value
        assert error.model.correlation_id is not None
        assert isinstance(error.model.correlation_id, UUID)
        # UUID v4 format check
        assert error.model.correlation_id.version == 4


@pytest.mark.unit
@pytest.mark.asyncio
class TestEffectCircuitBreakerEdgeCases:
    """Test edge cases for effect-level circuit breaker behavior."""

    async def test_circuit_resets_on_shutdown_and_reinit(
        self,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test circuit state resets when effect is recreated."""
        # Create effect and open circuit
        effect1 = MockConsulEffect(
            backend=mock_consul_backend,
            failure_threshold=2,
            reset_timeout=60.0,
        )

        mock_consul_backend.should_fail = True
        for _ in range(2):
            with pytest.raises(InfraConnectionError):
                await effect1.register_service("failing")

        assert effect1.get_circuit_state() == CircuitState.OPEN

        # Create new effect instance - should have fresh state
        mock_consul_backend.should_fail = False
        mock_consul_backend.reset()

        effect2 = MockConsulEffect(
            backend=mock_consul_backend,
            failure_threshold=2,
            reset_timeout=60.0,
        )

        assert effect2.get_circuit_state() == CircuitState.CLOSED
        assert effect2.get_failure_count() == 0

        # New effect should work normally
        result = await effect2.register_service("new-service")
        assert result["status"] == "registered"

    async def test_threshold_of_one(
        self,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test circuit breaker with threshold=1 opens on first failure."""
        effect = MockConsulEffect(
            backend=mock_consul_backend,
            failure_threshold=1,
            reset_timeout=0.1,
        )

        mock_consul_backend.should_fail = True

        # First failure should open circuit
        with pytest.raises(InfraConnectionError):
            await effect.register_service("first-failure")

        assert effect.get_circuit_state() == CircuitState.OPEN

        # Immediately blocked
        with pytest.raises(InfraUnavailableError):
            await effect.register_service("blocked")

    async def test_zero_timeout_immediate_reset(
        self,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test circuit with zero timeout resets immediately."""
        effect = MockConsulEffect(
            backend=mock_consul_backend,
            failure_threshold=2,
            reset_timeout=0.0,  # Immediate reset
        )

        mock_consul_backend.should_fail = True

        # Open circuit
        for _ in range(2):
            with pytest.raises(InfraConnectionError):
                await effect.register_service("failing")

        assert effect.get_circuit_state() == CircuitState.OPEN

        # With 0 timeout, next check should auto-reset to HALF_OPEN
        mock_consul_backend.should_fail = False
        mock_consul_backend.reset()

        # Should succeed (auto-reset)
        result = await effect.register_service("recovered")
        assert result["status"] == "registered"
        assert effect.get_circuit_state() == CircuitState.CLOSED

    async def test_concurrent_operations_thread_safety(
        self,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test circuit breaker is thread-safe with concurrent operations."""
        effect = MockConsulEffect(
            backend=mock_consul_backend,
            failure_threshold=100,  # High threshold to not open during test
            reset_timeout=60.0,
        )

        # Run 50 concurrent operations
        tasks = [effect.register_service(f"service-{i}") for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 50
        assert all(r["status"] == "registered" for r in results)

        # Backend should have been called 50 times
        assert mock_consul_backend.call_count == 50

    async def test_success_resets_failure_count(
        self,
        consul_effect: MockConsulEffect,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test successful operation resets failure counter."""
        # Accumulate some failures (but not enough to open)
        mock_consul_backend.should_fail = True
        for _ in range(2):  # Threshold is 3
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service("failing")

        assert consul_effect.get_failure_count() == 2
        assert consul_effect.get_circuit_state() == CircuitState.CLOSED

        # Successful operation should reset counter
        mock_consul_backend.should_fail = False
        await consul_effect.register_service("success")

        assert consul_effect.get_failure_count() == 0
        assert consul_effect.get_circuit_state() == CircuitState.CLOSED


@pytest.mark.unit
@pytest.mark.asyncio
class TestEffectCircuitBreakerErrorContext:
    """Test error context structure for circuit breaker errors."""

    async def test_connection_error_includes_transport_type(
        self,
        consul_effect: MockConsulEffect,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test InfraConnectionError includes transport type in context."""
        mock_consul_backend.should_fail = True

        with pytest.raises(InfraConnectionError) as exc_info:
            await consul_effect.register_service("failing")

        error = exc_info.value
        assert (
            error.model.context.get("transport_type") == EnumInfraTransportType.CONSUL
        )
        assert error.model.context.get("operation") == "register_service"
        assert error.model.context.get("target_name") == "consul.test"

    async def test_unavailable_error_includes_circuit_state(
        self,
        consul_effect: MockConsulEffect,
        mock_consul_backend: MockConsulBackend,
    ) -> None:
        """Test InfraUnavailableError includes circuit state."""
        # Open circuit
        mock_consul_backend.should_fail = True
        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await consul_effect.register_service("failing")

        with pytest.raises(InfraUnavailableError) as exc_info:
            await consul_effect.register_service("blocked")

        error = exc_info.value
        assert error.model.context.get("circuit_state") == "open"
        assert "retry_after_seconds" in error.model.context

    async def test_postgres_uses_database_transport_type(
        self,
        postgres_effect: MockPostgresEffect,
        mock_postgres_backend: MockPostgresBackend,
    ) -> None:
        """Test PostgreSQL effect uses DATABASE transport type."""
        mock_postgres_backend.should_fail = True

        with pytest.raises(InfraConnectionError) as exc_info:
            await postgres_effect.execute_query("SELECT 1")

        error = exc_info.value
        assert (
            error.model.context.get("transport_type") == EnumInfraTransportType.DATABASE
        )
        assert error.model.context.get("target_name") == "postgres.test"
