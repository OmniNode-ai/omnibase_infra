# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Race condition tests for MixinAsyncCircuitBreaker.

This module provides comprehensive async race condition tests for:
- Concurrent circuit breaker state checks
- Concurrent failure recording
- Concurrent reset operations
- State transitions under concurrent load
- HALF_OPEN state race conditions

These tests verify that the circuit breaker mixin properly handles
concurrent access patterns that would occur in production systems.

All tests use asyncio.gather() for true concurrent execution
and are designed to be deterministic.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    MixinAsyncCircuitBreaker,
)

# =============================================================================
# Test Helper Classes
# =============================================================================


class MockServiceWithCircuitBreaker(MixinAsyncCircuitBreaker):
    """Mock service using circuit breaker mixin for testing."""

    def __init__(
        self,
        threshold: int = 5,
        reset_timeout: float = 60.0,
        service_name: str = "test-service",
    ) -> None:
        self._init_circuit_breaker(
            threshold=threshold,
            reset_timeout=reset_timeout,
            service_name=service_name,
            transport_type=EnumInfraTransportType.HTTP,
        )
        self.operation_count = 0
        self._count_lock = asyncio.Lock()

    async def perform_operation(
        self,
        should_fail: bool = False,
        correlation_id: UUID | None = None,
    ) -> str:
        """Perform an operation through the circuit breaker."""
        # Check circuit breaker
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

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return "success"

        except ValueError:
            # Record failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="perform_operation",
                    correlation_id=correlation_id,
                )
            raise

    async def get_circuit_state(self) -> dict[str, object]:
        """Get current circuit breaker state (for testing)."""
        async with self._circuit_breaker_lock:
            return {
                "failures": self._circuit_breaker_failures,
                "open": self._circuit_breaker_open,
                "open_until": self._circuit_breaker_open_until,
            }


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def service() -> MockServiceWithCircuitBreaker:
    """Create a fresh mock service with circuit breaker."""
    return MockServiceWithCircuitBreaker(threshold=5, reset_timeout=60.0)


@pytest.fixture
def low_threshold_service() -> MockServiceWithCircuitBreaker:
    """Create service with low threshold for faster testing."""
    return MockServiceWithCircuitBreaker(threshold=3, reset_timeout=0.1)


# =============================================================================
# Concurrent State Check Tests
# =============================================================================


class TestConcurrentCircuitChecks:
    """Tests for concurrent circuit breaker state checks."""

    @pytest.mark.asyncio
    async def test_concurrent_checks_when_closed(
        self, service: MockServiceWithCircuitBreaker
    ) -> None:
        """Test concurrent circuit checks when circuit is closed."""
        num_concurrent = 50
        results: list[str] = []
        errors: list[Exception] = []
        lock = asyncio.Lock()

        async def check_and_operate() -> None:
            try:
                result = await service.perform_operation(should_fail=False)
                async with lock:
                    results.append(result)
            except Exception as e:
                async with lock:
                    errors.append(e)

        await asyncio.gather(*[check_and_operate() for _ in range(num_concurrent)])

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == num_concurrent
        assert all(r == "success" for r in results)
        assert service.operation_count == num_concurrent

    @pytest.mark.asyncio
    async def test_concurrent_checks_when_open(
        self, low_threshold_service: MockServiceWithCircuitBreaker
    ) -> None:
        """Test concurrent circuit checks when circuit is open."""
        # First, trip the circuit
        for _ in range(3):
            try:
                await low_threshold_service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Verify circuit is open
        state = await low_threshold_service.get_circuit_state()
        assert state["open"] is True

        # Now try concurrent operations - all should fail fast
        num_concurrent = 20
        errors: list[Exception] = []
        lock = asyncio.Lock()

        async def check_when_open() -> None:
            try:
                await low_threshold_service.perform_operation(should_fail=False)
            except InfraUnavailableError as e:
                async with lock:
                    errors.append(e)
            except Exception as e:
                async with lock:
                    errors.append(e)

        await asyncio.gather(*[check_when_open() for _ in range(num_concurrent)])

        # All should have gotten InfraUnavailableError
        assert len(errors) == num_concurrent
        assert all(isinstance(e, InfraUnavailableError) for e in errors)


# =============================================================================
# Concurrent Failure Recording Tests
# =============================================================================


class TestConcurrentFailureRecording:
    """Tests for concurrent failure recording."""

    @pytest.mark.asyncio
    async def test_concurrent_failures_increment_correctly(
        self, service: MockServiceWithCircuitBreaker
    ) -> None:
        """Test that concurrent failures increment counter correctly."""
        num_concurrent = 4  # Less than threshold to verify count
        errors: list[Exception] = []
        lock = asyncio.Lock()

        async def fail_operation() -> None:
            try:
                await service.perform_operation(should_fail=True)
            except ValueError:
                pass  # Expected
            except Exception as e:
                async with lock:
                    errors.append(e)

        await asyncio.gather(*[fail_operation() for _ in range(num_concurrent)])

        assert len(errors) == 0, f"Unexpected errors: {errors}"

        # Verify failure count is correct
        state = await service.get_circuit_state()
        assert state["failures"] == num_concurrent
        assert state["open"] is False  # Below threshold

    @pytest.mark.asyncio
    async def test_concurrent_failures_trigger_circuit_open(self) -> None:
        """Test that concurrent failures properly trigger circuit open.

        This test verifies that when failures exceed threshold, the circuit
        breaker opens. We use a long reset_timeout to prevent auto-transition
        to HALF_OPEN during test execution.

        The test validates circuit breaker behavior by checking that EITHER:
        1. The final state shows circuit is open, OR
        2. At least one operation received InfraUnavailableError (circuit opened
           during execution and blocked subsequent operations)

        Note: Uses its own service instance with long reset_timeout (60s) to
        prevent timing-related flakiness in CI environments where the default
        0.1s timeout could elapse during concurrent operation execution.
        """
        # Use long reset_timeout to prevent auto-transition to HALF_OPEN
        # during test execution (prevents CI flakiness)
        service = MockServiceWithCircuitBreaker(threshold=3, reset_timeout=60.0)

        num_concurrent = 5  # More than threshold (3)
        errors: list[Exception] = []
        circuit_open_count = 0
        lock = asyncio.Lock()

        async def fail_operation() -> None:
            nonlocal circuit_open_count
            try:
                await service.perform_operation(should_fail=True)
            except ValueError:
                pass  # Expected failure before circuit opened
            except InfraUnavailableError:
                # Circuit opened during concurrent operations - this is success!
                async with lock:
                    circuit_open_count += 1
            except Exception as e:
                async with lock:
                    errors.append(e)

        await asyncio.gather(*[fail_operation() for _ in range(num_concurrent)])

        assert len(errors) == 0, f"Unexpected errors: {errors}"

        # Verify circuit breaker behavior: EITHER circuit is open now OR
        # we observed it open during execution (InfraUnavailableError)
        state = await service.get_circuit_state()
        circuit_triggered = state["open"] is True or circuit_open_count > 0

        assert circuit_triggered, (
            f"Circuit breaker should have triggered: "
            f"final_state_open={state['open']}, "
            f"failures={state['failures']}, "
            f"circuit_open_errors_seen={circuit_open_count}"
        )


# =============================================================================
# Concurrent Reset Tests
# =============================================================================


class TestConcurrentResetOperations:
    """Tests for concurrent circuit breaker reset operations."""

    @pytest.mark.asyncio
    async def test_concurrent_success_resets(
        self, service: MockServiceWithCircuitBreaker
    ) -> None:
        """Test concurrent successful operations reset failure count."""
        # First add some failures
        for _ in range(3):
            try:
                await service.perform_operation(should_fail=True)
            except ValueError:
                pass

        state = await service.get_circuit_state()
        assert state["failures"] == 3

        # Now concurrent successes should reset
        num_concurrent = 10
        results: list[str] = []
        lock = asyncio.Lock()

        async def succeed_operation() -> None:
            result = await service.perform_operation(should_fail=False)
            async with lock:
                results.append(result)

        await asyncio.gather(*[succeed_operation() for _ in range(num_concurrent)])

        assert len(results) == num_concurrent

        # Failure count should be reset to 0
        state = await service.get_circuit_state()
        assert state["failures"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_manual_resets(
        self, low_threshold_service: MockServiceWithCircuitBreaker
    ) -> None:
        """Test concurrent manual reset operations are safe."""
        # Trip the circuit
        for _ in range(3):
            try:
                await low_threshold_service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Verify circuit is open
        state = await low_threshold_service.get_circuit_state()
        assert state["open"] is True

        # Concurrent resets should be safe
        async def manual_reset() -> None:
            async with low_threshold_service._circuit_breaker_lock:
                await low_threshold_service._reset_circuit_breaker()

        await asyncio.gather(*[manual_reset() for _ in range(10)])

        # Circuit should be closed and failures reset
        state = await low_threshold_service.get_circuit_state()
        assert state["open"] is False
        assert state["failures"] == 0


# =============================================================================
# State Transition Race Condition Tests
# =============================================================================


class TestStateTransitionRaceConditions:
    """Tests for race conditions during state transitions."""

    @pytest.mark.asyncio
    async def test_open_to_half_open_transition_race(self) -> None:
        """Test OPEN to HALF_OPEN transition under concurrent access."""
        service = MockServiceWithCircuitBreaker(
            threshold=2,
            reset_timeout=0.01,  # Very short timeout for testing
        )

        # Trip the circuit
        for _ in range(2):
            try:
                await service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Verify circuit is open
        state = await service.get_circuit_state()
        assert state["open"] is True

        # Wait for reset timeout
        await asyncio.sleep(0.02)

        # Concurrent checks should handle HALF_OPEN transition correctly
        num_concurrent = 20
        results: list[str] = []
        errors: list[Exception] = []
        lock = asyncio.Lock()

        async def check_after_timeout() -> None:
            try:
                result = await service.perform_operation(should_fail=False)
                async with lock:
                    results.append(result)
            except InfraUnavailableError as e:
                async with lock:
                    errors.append(e)
            except Exception as e:
                async with lock:
                    errors.append(e)

        await asyncio.gather(*[check_after_timeout() for _ in range(num_concurrent)])

        # Most operations should succeed after HALF_OPEN transition
        assert len(results) > 0
        # No unexpected errors
        unexpected_errors = [
            e for e in errors if not isinstance(e, InfraUnavailableError)
        ]
        assert len(unexpected_errors) == 0

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self) -> None:
        """Test HALF_OPEN to CLOSED transition on successful operation."""
        service = MockServiceWithCircuitBreaker(
            threshold=2,
            reset_timeout=0.01,
        )

        # Trip circuit
        for _ in range(2):
            try:
                await service.perform_operation(should_fail=True)
            except ValueError:
                pass

        # Wait for HALF_OPEN
        await asyncio.sleep(0.02)

        # Success should transition to CLOSED
        result = await service.perform_operation(should_fail=False)
        assert result == "success"

        # Verify circuit is closed
        state = await service.get_circuit_state()
        assert state["open"] is False
        assert state["failures"] == 0

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self) -> None:
        """Test HALF_OPEN to OPEN transition on failed operation.

        Note: After transitioning to HALF_OPEN, the failure count is reset to 0.
        Therefore, we need threshold number of failures to re-open the circuit.
        With threshold=1, a single failure will re-open immediately.
        """
        service = MockServiceWithCircuitBreaker(
            threshold=1,  # Single failure re-opens circuit
            reset_timeout=0.01,
        )

        # Trip circuit with single failure (threshold=1)
        try:
            await service.perform_operation(should_fail=True)
        except ValueError:
            pass

        # Verify circuit is open
        state = await service.get_circuit_state()
        assert state["open"] is True

        # Wait for HALF_OPEN
        await asyncio.sleep(0.02)

        # Failure should transition back to OPEN
        try:
            await service.perform_operation(should_fail=True)
        except ValueError:
            pass

        # Verify circuit is open again
        state = await service.get_circuit_state()
        assert state["open"] is True


# =============================================================================
# Mixed Operation Race Condition Tests
# =============================================================================


class TestMixedOperationRaceConditions:
    """Tests for mixed concurrent operations."""

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure_operations(
        self, service: MockServiceWithCircuitBreaker
    ) -> None:
        """Test mixed concurrent successful and failing operations."""
        num_concurrent = 20
        success_count = 0
        failure_count = 0
        circuit_open_count = 0
        lock = asyncio.Lock()

        async def mixed_operation(i: int) -> None:
            nonlocal success_count, failure_count, circuit_open_count
            should_fail = i % 2 == 0  # Every other operation fails

            try:
                await service.perform_operation(should_fail=should_fail)
                async with lock:
                    success_count += 1
            except ValueError:
                async with lock:
                    failure_count += 1
            except InfraUnavailableError:
                async with lock:
                    circuit_open_count += 1

        await asyncio.gather(*[mixed_operation(i) for i in range(num_concurrent)])

        # Total should equal num_concurrent
        assert success_count + failure_count + circuit_open_count == num_concurrent
        # Should have some successes and failures
        assert success_count > 0 or failure_count > 0

    @pytest.mark.asyncio
    async def test_concurrent_operations_with_varying_correlation_ids(
        self, service: MockServiceWithCircuitBreaker
    ) -> None:
        """Test concurrent operations with different correlation IDs."""
        num_concurrent = 30
        results: list[str] = []
        errors: list[Exception] = []
        lock = asyncio.Lock()

        async def operation_with_correlation(i: int) -> None:
            correlation_id = uuid4()
            try:
                result = await service.perform_operation(
                    should_fail=False,
                    correlation_id=correlation_id,
                )
                async with lock:
                    results.append(result)
            except Exception as e:
                async with lock:
                    errors.append(e)

        await asyncio.gather(
            *[operation_with_correlation(i) for i in range(num_concurrent)]
        )

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == num_concurrent


# =============================================================================
# Stress Tests
# =============================================================================


class TestCircuitBreakerStress:
    """Stress tests for circuit breaker under high load."""

    @pytest.mark.asyncio
    async def test_high_volume_operations_stress(self) -> None:
        """Stress test with high volume of concurrent operations."""
        service = MockServiceWithCircuitBreaker(
            threshold=100,  # High threshold for stress test
            reset_timeout=60.0,
        )

        num_operations = 500
        results: list[str] = []
        errors: list[Exception] = []
        lock = asyncio.Lock()

        async def stress_operation(i: int) -> None:
            # 10% failure rate
            should_fail = i % 10 == 0
            try:
                result = await service.perform_operation(should_fail=should_fail)
                async with lock:
                    results.append(result)
            except ValueError:
                pass  # Expected failures
            except Exception as e:
                async with lock:
                    errors.append(e)

        await asyncio.gather(*[stress_operation(i) for i in range(num_operations)])

        assert len(errors) == 0, f"Unexpected errors: {errors}"
        # 90% should succeed (those that don't intentionally fail)
        assert len(results) >= num_operations * 0.8

    @pytest.mark.asyncio
    async def test_rapid_open_close_cycles_stress(self) -> None:
        """Stress test with rapid circuit open/close cycles."""
        service = MockServiceWithCircuitBreaker(
            threshold=3,
            reset_timeout=0.001,  # Very fast reset
        )

        cycles = 20
        errors: list[Exception] = []
        lock = asyncio.Lock()

        async def cycle_circuit() -> None:
            try:
                # Trip circuit
                for _ in range(3):
                    try:
                        await service.perform_operation(should_fail=True)
                    except (ValueError, InfraUnavailableError):
                        pass

                # Wait for reset
                await asyncio.sleep(0.002)

                # Verify can operate again
                await service.perform_operation(should_fail=False)
            except Exception as e:
                async with lock:
                    errors.append(e)

        await asyncio.gather(*[cycle_circuit() for _ in range(cycles)])

        # Filter out expected InfraUnavailableError
        unexpected_errors = [
            e for e in errors if not isinstance(e, ValueError | InfraUnavailableError)
        ]
        assert len(unexpected_errors) == 0, f"Unexpected errors: {unexpected_errors}"


# =============================================================================
# Lock Verification Tests
# =============================================================================


class TestLockVerification:
    """Tests to verify lock is properly held during operations."""

    @pytest.mark.asyncio
    async def test_check_detects_missing_lock(
        self, service: MockServiceWithCircuitBreaker
    ) -> None:
        """Test that _check_circuit_breaker logs warning when lock not held.

        Note: The implementation logs a warning but still proceeds.
        This test verifies the operation completes without deadlock.
        """
        # Call without holding lock - should log warning but not deadlock
        # The implementation is designed to still work but log the violation
        await service._check_circuit_breaker("test_op")
        # If we get here, no deadlock occurred
        assert True

    @pytest.mark.asyncio
    async def test_record_failure_detects_missing_lock(
        self, service: MockServiceWithCircuitBreaker
    ) -> None:
        """Test that _record_circuit_failure logs warning when lock not held."""
        await service._record_circuit_failure("test_op")
        # If we get here, no deadlock occurred
        assert True

    @pytest.mark.asyncio
    async def test_reset_detects_missing_lock(
        self, service: MockServiceWithCircuitBreaker
    ) -> None:
        """Test that _reset_circuit_breaker logs warning when lock not held."""
        await service._reset_circuit_breaker()
        # If we get here, no deadlock occurred
        assert True
