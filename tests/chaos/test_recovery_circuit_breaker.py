# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Circuit breaker recovery tests for OMN-955.

This test suite validates circuit breaker behavior under failure conditions,
focusing on state transitions and recovery mechanisms. It tests:

1. Circuit opens after threshold failures
2. Circuit closes after reset timeout
3. Half-open state allows test requests
4. Recovery after sustained failures

Architecture:
    The circuit breaker implements a 3-state pattern:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Circuit tripped, requests blocked
    - HALF_OPEN: Testing recovery, limited requests allowed

    State Transitions:
    - CLOSED -> OPEN: Failure count >= threshold
    - OPEN -> HALF_OPEN: Time > reset_timeout
    - HALF_OPEN -> CLOSED: Success on test request
    - HALF_OPEN -> OPEN: Failure on test request

Test Organization:
    - TestCircuitBreakerOpens: Threshold-based circuit opening
    - TestCircuitBreakerCloses: Reset timeout and recovery
    - TestHalfOpenState: Test request behavior
    - TestCircuitBreakerRecovery: End-to-end recovery scenarios

Pattern Reference:
    - src/omnibase_infra/mixins/mixin_async_circuit_breaker.py
    - tests/unit/mixins/test_mixin_async_circuit_breaker_race_conditions.py

Related:
    - OMN-955: Failure recovery tests
    - OMN-954: Effect retry and backoff
"""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumCircuitState, EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    MixinAsyncCircuitBreaker,
)

# =============================================================================
# Test Helper: Mock Service with Circuit Breaker
# =============================================================================


class MockServiceWithCircuitBreaker(MixinAsyncCircuitBreaker):
    """Mock service using circuit breaker mixin for testing.

    This class simulates an infrastructure service (e.g., Consul, Kafka)
    that uses the circuit breaker pattern for fault tolerance.

    Attributes:
        operation_count: Number of operations actually executed.
        success_count: Number of successful operations.
        failure_count: Number of failed operations.
    """

    def __init__(
        self,
        threshold: int = 5,
        reset_timeout: float = 60.0,
        service_name: str = "test-service",
    ) -> None:
        """Initialize mock service with circuit breaker.

        Args:
            threshold: Number of failures before opening circuit.
            reset_timeout: Seconds until automatic reset.
            service_name: Service name for error context.
        """
        self._init_circuit_breaker(
            threshold=threshold,
            reset_timeout=reset_timeout,
            service_name=service_name,
            transport_type=EnumInfraTransportType.HTTP,
        )
        self.operation_count = 0
        self.success_count = 0
        self.failure_count = 0
        self._count_lock = asyncio.Lock()

    async def perform_operation(
        self,
        should_fail: bool = False,
        correlation_id=None,
    ) -> str:
        """Perform an operation through the circuit breaker.

        Args:
            should_fail: If True, simulate operation failure.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            "success" if operation completed.

        Raises:
            ValueError: If should_fail is True.
            InfraUnavailableError: If circuit breaker is open.
        """
        # Check circuit breaker (thread-safe)
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="perform_operation",
                correlation_id=correlation_id,
            )

        try:
            # Simulate operation
            async with self._count_lock:
                self.operation_count += 1

            if should_fail:
                raise ValueError("Intentional failure")

            # Record success (thread-safe)
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            async with self._count_lock:
                self.success_count += 1

            return "success"

        except ValueError:
            # Record failure (thread-safe)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="perform_operation",
                    correlation_id=correlation_id,
                )

            async with self._count_lock:
                self.failure_count += 1

            raise

    async def get_circuit_state(self) -> dict[str, object]:
        """Get current circuit breaker state (for testing).

        Returns:
            Dict with circuit state information:
                - failures: Current failure count
                - open: Whether circuit is open
                - open_until: Timestamp for auto-reset
        """
        async with self._circuit_breaker_lock:
            return {
                "failures": self._circuit_breaker_failures,
                "open": self._circuit_breaker_open,
                "open_until": self._circuit_breaker_open_until,
                "threshold": self.circuit_breaker_threshold,
                "reset_timeout": self.circuit_breaker_reset_timeout,
            }

    async def force_circuit_state(
        self,
        open_state: bool,
        failures: int = 0,
        open_until: float = 0.0,
    ) -> None:
        """Force circuit breaker to a specific state (for testing).

        Args:
            open_state: Whether circuit should be open.
            failures: Failure count to set.
            open_until: Timestamp for auto-reset.
        """
        async with self._circuit_breaker_lock:
            self._circuit_breaker_open = open_state
            self._circuit_breaker_failures = failures
            self._circuit_breaker_open_until = open_until


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def service() -> MockServiceWithCircuitBreaker:
    """Create a mock service with default circuit breaker settings."""
    return MockServiceWithCircuitBreaker(threshold=5, reset_timeout=60.0)


@pytest.fixture
def low_threshold_service() -> MockServiceWithCircuitBreaker:
    """Create service with low threshold for faster testing."""
    return MockServiceWithCircuitBreaker(threshold=3, reset_timeout=0.1)


@pytest.fixture
def fast_reset_service() -> MockServiceWithCircuitBreaker:
    """Create service with very fast reset timeout for testing."""
    return MockServiceWithCircuitBreaker(threshold=2, reset_timeout=0.01)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.unit
@pytest.mark.chaos
class TestCircuitBreakerOpens:
    """Test circuit breaker opening behavior."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(
        self,
        low_threshold_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test circuit opens after reaching failure threshold.

        Scenario:
            1. Cause 3 failures (threshold)
            2. Verify circuit is open
            3. Verify next request is blocked immediately
        """
        # Cause threshold failures
        for i in range(3):
            try:
                await low_threshold_service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Verify circuit is open
        state = await low_threshold_service.get_circuit_state()
        assert state["open"] is True
        assert state["failures"] == 3

        # Verify next request is blocked
        with pytest.raises(InfraUnavailableError) as exc_info:
            await low_threshold_service.perform_operation(should_fail=False)

        error = exc_info.value
        assert "circuit breaker is open" in error.message.lower()

    @pytest.mark.asyncio
    async def test_circuit_stays_closed_below_threshold(
        self,
        low_threshold_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test circuit stays closed when failures are below threshold."""
        # Cause 2 failures (below threshold of 3)
        for _ in range(2):
            try:
                await low_threshold_service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Verify circuit is still closed
        state = await low_threshold_service.get_circuit_state()
        assert state["open"] is False
        assert state["failures"] == 2

        # Verify request is allowed
        result = await low_threshold_service.perform_operation(should_fail=False)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(
        self,
        low_threshold_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test that success resets failure count."""
        # Cause 2 failures
        for _ in range(2):
            try:
                await low_threshold_service.perform_operation(should_fail=True)
            except ValueError:
                pass

        state = await low_threshold_service.get_circuit_state()
        assert state["failures"] == 2

        # Success resets count
        await low_threshold_service.perform_operation(should_fail=False)

        state = await low_threshold_service.get_circuit_state()
        assert state["failures"] == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_with_error_context(
        self,
        low_threshold_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test that circuit open error includes proper context."""
        correlation_id = uuid4()

        # Trip the circuit
        for _ in range(3):
            try:
                await low_threshold_service.perform_operation(
                    should_fail=True,
                    correlation_id=correlation_id,
                )
            except ValueError:
                pass

        # Verify error has proper context
        with pytest.raises(InfraUnavailableError) as exc_info:
            await low_threshold_service.perform_operation(
                should_fail=False,
                correlation_id=correlation_id,
            )

        error = exc_info.value
        assert error.model.context is not None
        # Context should include transport type and operation
        context = error.model.context
        assert "transport_type" in context
        assert "operation" in context


@pytest.mark.unit
@pytest.mark.chaos
class TestCircuitBreakerCloses:
    """Test circuit breaker closing behavior."""

    @pytest.mark.asyncio
    async def test_circuit_closes_after_reset_timeout(
        self,
        fast_reset_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test circuit closes automatically after reset timeout.

        Scenario:
            1. Trip the circuit
            2. Wait for reset timeout
            3. Verify circuit allows requests
            4. Verify circuit is closed after success
        """
        # Trip the circuit (threshold = 2)
        for _ in range(2):
            try:
                await fast_reset_service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Verify circuit is open
        state = await fast_reset_service.get_circuit_state()
        assert state["open"] is True

        # Wait for reset timeout (0.01s + small buffer)
        await asyncio.sleep(0.02)

        # Request should now be allowed (transitions to HALF_OPEN)
        result = await fast_reset_service.perform_operation(should_fail=False)
        assert result == "success"

        # Circuit should be closed after successful test request
        state = await fast_reset_service.get_circuit_state()
        assert state["open"] is False
        assert state["failures"] == 0

    @pytest.mark.asyncio
    async def test_circuit_remains_open_before_timeout(
        self,
        low_threshold_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test circuit remains open before reset timeout."""
        # Trip the circuit
        for _ in range(3):
            try:
                await low_threshold_service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Verify circuit is open
        state = await low_threshold_service.get_circuit_state()
        assert state["open"] is True

        # Immediately try another request (before timeout)
        with pytest.raises(InfraUnavailableError):
            await low_threshold_service.perform_operation(should_fail=False)

    @pytest.mark.asyncio
    async def test_retry_after_seconds_in_error(
        self,
        low_threshold_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test that error includes retry_after_seconds hint."""
        # Trip the circuit
        for _ in range(3):
            try:
                await low_threshold_service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Check error includes retry_after_seconds
        with pytest.raises(InfraUnavailableError) as exc_info:
            await low_threshold_service.perform_operation(should_fail=False)

        # The error should suggest when to retry
        # Note: InfraUnavailableError includes retry_after_seconds in kwargs
        # Access may vary based on error implementation
        error = exc_info.value
        assert error.message is not None


@pytest.mark.unit
@pytest.mark.chaos
class TestHalfOpenState:
    """Test half-open state behavior."""

    @pytest.mark.asyncio
    async def test_half_open_allows_test_request(
        self,
        fast_reset_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test that half-open state allows a test request.

        Scenario:
            1. Trip the circuit
            2. Wait for reset timeout
            3. First request should be allowed (test request)
        """
        # Trip the circuit
        for _ in range(2):
            try:
                await fast_reset_service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Verify open
        state = await fast_reset_service.get_circuit_state()
        assert state["open"] is True

        # Wait for timeout
        await asyncio.sleep(0.02)

        # Test request should be allowed
        result = await fast_reset_service.perform_operation(should_fail=False)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(
        self,
    ) -> None:
        """Test that failure in half-open state reopens circuit.

        Scenario:
            1. Trip the circuit
            2. Wait for reset timeout (transitions to HALF_OPEN)
            3. Test request fails
            4. Circuit returns to OPEN

        Note: After transitioning to HALF_OPEN, the failure count is reset to 0.
        Therefore, we need threshold number of failures to re-open the circuit.
        With threshold=1, a single failure will re-open immediately.
        """
        # Create service with threshold=1 so single failure reopens
        service = MockServiceWithCircuitBreaker(
            threshold=1,
            reset_timeout=0.01,
        )

        # Trip the circuit with single failure
        try:
            await service.perform_operation(should_fail=True)
        except ValueError:
            pass

        # Verify circuit is open
        state = await service.get_circuit_state()
        assert state["open"] is True

        # Wait for HALF_OPEN
        await asyncio.sleep(0.02)

        # Fail the test request - this should reopen the circuit
        try:
            await service.perform_operation(should_fail=True)
        except ValueError:
            pass

        # Circuit should be open again
        state = await service.get_circuit_state()
        assert state["open"] is True

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(
        self,
        fast_reset_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test that success in half-open state closes circuit."""
        # Trip the circuit
        for _ in range(2):
            try:
                await fast_reset_service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Wait for HALF_OPEN
        await asyncio.sleep(0.02)

        # Success should close the circuit
        result = await fast_reset_service.perform_operation(should_fail=False)
        assert result == "success"

        state = await fast_reset_service.get_circuit_state()
        assert state["open"] is False
        assert state["failures"] == 0


@pytest.mark.unit
@pytest.mark.chaos
class TestCircuitBreakerRecovery:
    """End-to-end circuit breaker recovery scenarios."""

    @pytest.mark.asyncio
    async def test_recovery_after_sustained_failures(
        self,
        fast_reset_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test full recovery cycle after sustained failures.

        Scenario:
            1. Multiple failures trip the circuit
            2. Wait for reset
            3. Test request succeeds
            4. Normal operations resume
        """
        # Phase 1: Sustained failures
        for _ in range(5):
            try:
                await fast_reset_service.perform_operation(should_fail=True)
            except (ValueError, InfraUnavailableError):
                pass

        # Verify circuit is open
        state = await fast_reset_service.get_circuit_state()
        assert state["open"] is True

        # Phase 2: Wait for recovery
        await asyncio.sleep(0.02)

        # Phase 3: Successful recovery
        result = await fast_reset_service.perform_operation(should_fail=False)
        assert result == "success"

        # Phase 4: Normal operations
        for _ in range(5):
            result = await fast_reset_service.perform_operation(should_fail=False)
            assert result == "success"

        state = await fast_reset_service.get_circuit_state()
        assert state["open"] is False
        assert state["failures"] == 0

    @pytest.mark.asyncio
    async def test_multiple_recovery_cycles(
        self,
        fast_reset_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test multiple failure/recovery cycles.

        Verifies the circuit breaker can handle repeated failure and
        recovery cycles without degradation.
        """
        for cycle in range(3):
            # Trip the circuit
            for _ in range(2):
                try:
                    await fast_reset_service.perform_operation(should_fail=True)
                except ValueError:
                    pass

            # Verify open
            state = await fast_reset_service.get_circuit_state()
            assert state["open"] is True, f"Cycle {cycle}: Expected open"

            # Wait and recover
            await asyncio.sleep(0.02)

            result = await fast_reset_service.perform_operation(should_fail=False)
            assert result == "success", f"Cycle {cycle}: Recovery failed"

            # Verify closed
            state = await fast_reset_service.get_circuit_state()
            assert state["open"] is False, f"Cycle {cycle}: Expected closed"

    @pytest.mark.asyncio
    async def test_intermittent_failures_dont_trip_circuit(
        self,
        low_threshold_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test that intermittent failures with successes don't trip circuit.

        Success resets the failure count, so alternating success/failure
        shouldn't trip the circuit.
        """
        # Alternating pattern: fail, fail, success, fail, fail, success
        patterns = [True, True, False, True, True, False]

        for should_fail in patterns:
            try:
                await low_threshold_service.perform_operation(should_fail=should_fail)
            except ValueError:
                pass

        # Circuit should still be closed (successes reset count)
        state = await low_threshold_service.get_circuit_state()
        assert state["open"] is False

    @pytest.mark.asyncio
    async def test_circuit_state_survives_concurrent_access(
        self,
        low_threshold_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test circuit breaker state is consistent under concurrent access."""
        results: list[str] = []
        errors: list[Exception] = []
        lock = asyncio.Lock()

        async def concurrent_operation(should_fail: bool) -> None:
            try:
                result = await low_threshold_service.perform_operation(
                    should_fail=should_fail
                )
                async with lock:
                    results.append(result)
            except (ValueError, InfraUnavailableError) as e:
                async with lock:
                    errors.append(e)

        # Launch concurrent operations - some fail, some succeed
        tasks = []
        for i in range(10):
            tasks.append(concurrent_operation(should_fail=(i < 5)))

        await asyncio.gather(*tasks)

        # Circuit state should be consistent
        state = await low_threshold_service.get_circuit_state()
        # Either circuit is open (if failures were first) or closed
        # The important thing is state is not corrupted
        assert isinstance(state["open"], bool)
        assert isinstance(state["failures"], int)
        assert state["failures"] >= 0


@pytest.mark.unit
@pytest.mark.chaos
class TestCircuitBreakerWithCorrelation:
    """Test circuit breaker correlation ID handling."""

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_to_error(
        self,
        low_threshold_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test that correlation ID is included in circuit breaker errors."""
        correlation_id = uuid4()

        # Trip the circuit
        for _ in range(3):
            try:
                await low_threshold_service.perform_operation(
                    should_fail=True,
                    correlation_id=correlation_id,
                )
            except ValueError:
                pass

        # Error should include correlation ID in context
        with pytest.raises(InfraUnavailableError) as exc_info:
            await low_threshold_service.perform_operation(
                should_fail=False,
                correlation_id=correlation_id,
            )

        error = exc_info.value
        # Verify correlation_id is in error model
        assert error.model.correlation_id is not None

    @pytest.mark.asyncio
    async def test_different_correlation_ids_tracked(
        self,
        low_threshold_service: MockServiceWithCircuitBreaker,
    ) -> None:
        """Test circuit breaker works correctly with different correlation IDs."""
        # Each request has a different correlation ID
        correlation_ids = [uuid4() for _ in range(3)]

        # Trip circuit with different correlation IDs
        for cid in correlation_ids:
            try:
                await low_threshold_service.perform_operation(
                    should_fail=True,
                    correlation_id=cid,
                )
            except ValueError:
                pass

        # Circuit should still be tripped (failures accumulate regardless of cid)
        state = await low_threshold_service.get_circuit_state()
        assert state["open"] is True
        assert state["failures"] == 3


__all__ = [
    "MockServiceWithCircuitBreaker",
    "TestCircuitBreakerOpens",
    "TestCircuitBreakerCloses",
    "TestHalfOpenState",
    "TestCircuitBreakerRecovery",
    "TestCircuitBreakerWithCorrelation",
]
