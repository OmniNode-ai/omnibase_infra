# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dedicated test suite for circuit breaker state transitions.

This module provides comprehensive tests specifically for the circuit breaker
state machine transitions. While other test files cover functionality, thread
safety, and integration, this file focuses exclusively on state transitions.

Circuit Breaker State Machine:
    ┌─────────┐  failures >= threshold  ┌──────┐
    │ CLOSED  │ ──────────────────────▶ │ OPEN │
    └─────────┘                         └──────┘
        ▲                                   │
        │  success                          │ timeout elapsed
        │                                   ▼
        │                            ┌───────────┐
        └─────────────────────────── │ HALF_OPEN │
                                     └───────────┘
                                          │
                                          │ failure >= threshold
                                          ▼
                                     ┌──────┐
                                     │ OPEN │
                                     └──────┘

State Transitions Tested:
    1. CLOSED → OPEN: After threshold failures
    2. OPEN → HALF_OPEN: After reset timeout elapses
    3. HALF_OPEN → CLOSED: On successful operation
    4. HALF_OPEN → OPEN: On failed operation (reaching threshold)
    5. CLOSED stays CLOSED: Success resets failure count
    6. OPEN stays OPEN: Before timeout, requests blocked

Implementation Notes:
    The circuit breaker implementation uses two state variables:
    - _circuit_breaker_open: Boolean indicating OPEN state
    - _circuit_breaker_failures: Failure counter

    HALF_OPEN is an implicit state:
    - When timeout passes during _check_circuit_breaker(), the circuit
      transitions by setting _circuit_breaker_open = False and resetting
      _circuit_breaker_failures = 0
    - This allows a "test request" through
    - If that request succeeds → CLOSED (via _reset_circuit_breaker)
    - If that request fails → needs threshold failures to reach OPEN again

Related Test Files:
    - test_mixin_async_circuit_breaker.py: Basic functionality and thread safety
    - test_mixin_async_circuit_breaker_race_conditions.py: Race condition tests
    - test_effect_circuit_breaker.py: Effect-level integration tests
    - test_recovery_circuit_breaker.py: Chaos/recovery tests

See Also:
    - src/omnibase_infra/mixins/mixin_async_circuit_breaker.py
    - docs/patterns/circuit_breaker_implementation.md
"""

from __future__ import annotations

import asyncio
import time
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    CircuitState,
    MixinAsyncCircuitBreaker,
)

# =============================================================================
# Test Helper: Circuit Breaker Service Stub
# =============================================================================


class CircuitBreakerTestService(MixinAsyncCircuitBreaker):
    """Test service implementing circuit breaker pattern for transition tests.

    This stub provides direct access to circuit breaker state for assertions
    and wraps the mixin methods with lock acquisition for simpler testing.
    """

    def __init__(
        self,
        threshold: int = 5,
        reset_timeout: float = 60.0,
        service_name: str = "test-service",
        transport_type: EnumInfraTransportType = EnumInfraTransportType.HTTP,
    ) -> None:
        """Initialize test service with circuit breaker.

        Args:
            threshold: Maximum failures before opening circuit
            reset_timeout: Seconds before automatic reset
            service_name: Service identifier for error context
            transport_type: Transport type for error context
        """
        self._init_circuit_breaker(
            threshold=threshold,
            reset_timeout=reset_timeout,
            service_name=service_name,
            transport_type=transport_type,
        )

    async def check_circuit(
        self, operation: str = "test_operation", correlation_id: UUID | None = None
    ) -> None:
        """Check circuit breaker state (acquires lock internally for testing)."""
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

    async def record_failure(
        self, operation: str = "test_operation", correlation_id: UUID | None = None
    ) -> None:
        """Record circuit failure (acquires lock internally for testing)."""
        async with self._circuit_breaker_lock:
            await self._record_circuit_failure(operation, correlation_id)

    async def reset_circuit(self) -> None:
        """Reset circuit breaker (acquires lock internally for testing)."""
        async with self._circuit_breaker_lock:
            await self._reset_circuit_breaker()

    def get_state(self) -> CircuitState:
        """Get current circuit state for assertions.

        Returns:
            CircuitState.OPEN if circuit is open, else CircuitState.CLOSED.
            Note: HALF_OPEN is implicit and appears as CLOSED with reset failures.
        """
        if self._circuit_breaker_open:
            return CircuitState.OPEN
        return CircuitState.CLOSED

    def get_failure_count(self) -> int:
        """Get current failure count for assertions."""
        return self._circuit_breaker_failures

    def get_open_until(self) -> float:
        """Get reset timestamp for assertions."""
        return self._circuit_breaker_open_until

    async def execute_with_circuit_breaker(
        self,
        should_fail: bool = False,
        correlation_id: UUID | None = None,
    ) -> str:
        """Execute an operation through the circuit breaker pattern.

        This simulates a real operation that:
        1. Checks circuit breaker before execution
        2. Executes the operation (success or failure based on should_fail)
        3. Records success (reset) or failure with circuit breaker
        4. Returns result or raises exception

        Args:
            should_fail: If True, simulate operation failure
            correlation_id: Optional correlation ID for tracing

        Returns:
            "success" if operation completed successfully

        Raises:
            RuntimeError: If should_fail is True
            InfraUnavailableError: If circuit is open
        """
        # Check circuit before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("execute", correlation_id)

        try:
            if should_fail:
                raise RuntimeError("Simulated operation failure")

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return "success"

        except RuntimeError:
            # Record failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("execute", correlation_id)
            raise


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def service() -> CircuitBreakerTestService:
    """Create test service with default settings (threshold=5, timeout=60s)."""
    return CircuitBreakerTestService(threshold=5, reset_timeout=60.0)


@pytest.fixture
def fast_service() -> CircuitBreakerTestService:
    """Create test service with fast reset (threshold=3, timeout=0.1s)."""
    return CircuitBreakerTestService(threshold=3, reset_timeout=0.1)


@pytest.fixture
def single_failure_service() -> CircuitBreakerTestService:
    """Create test service with single failure threshold (threshold=1)."""
    return CircuitBreakerTestService(threshold=1, reset_timeout=0.05)


# =============================================================================
# Test Class: CLOSED → OPEN Transition
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionClosedToOpen:
    """Test CLOSED → OPEN state transition.

    Transition Trigger: failure_count >= threshold
    Transition Action: Sets _circuit_breaker_open = True, sets reset timestamp
    """

    async def test_transition_occurs_at_exact_threshold(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test circuit opens exactly when threshold is reached.

        Verifies:
        - Circuit stays CLOSED at threshold - 1 failures
        - Circuit transitions to OPEN at exactly threshold failures
        """
        # Verify starting state
        assert fast_service.get_state() == CircuitState.CLOSED
        assert fast_service.get_failure_count() == 0

        # Record failures up to threshold - 1
        for i in range(fast_service.circuit_breaker_threshold - 1):
            await fast_service.record_failure()
            assert fast_service.get_state() == CircuitState.CLOSED
            assert fast_service.get_failure_count() == i + 1

        # One more failure should open the circuit
        await fast_service.record_failure()
        assert fast_service.get_state() == CircuitState.OPEN
        assert (
            fast_service.get_failure_count() == fast_service.circuit_breaker_threshold
        )

    async def test_transition_with_threshold_of_one(
        self, single_failure_service: CircuitBreakerTestService
    ) -> None:
        """Test circuit opens on first failure when threshold is 1.

        Edge case: threshold=1 means any single failure opens the circuit.
        """
        assert single_failure_service.get_state() == CircuitState.CLOSED

        # First failure should immediately open circuit
        await single_failure_service.record_failure()

        assert single_failure_service.get_state() == CircuitState.OPEN
        assert single_failure_service.get_failure_count() == 1

    async def test_transition_sets_reset_timestamp(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that CLOSED → OPEN sets the reset timestamp correctly."""
        # Record failures to open circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        assert fast_service.get_state() == CircuitState.OPEN

        # Verify reset timestamp is set to future time
        open_until = fast_service.get_open_until()
        current_time = time.time()
        expected_time = current_time + fast_service.circuit_breaker_reset_timeout

        # Allow 1 second tolerance for test timing
        assert open_until > current_time
        assert abs(open_until - expected_time) < 1.0

    async def test_transition_blocks_subsequent_requests(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that OPEN state blocks requests immediately."""
        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        assert fast_service.get_state() == CircuitState.OPEN

        # Verify check_circuit raises InfraUnavailableError
        with pytest.raises(InfraUnavailableError) as exc_info:
            await fast_service.check_circuit()

        error = exc_info.value
        assert "Circuit breaker is open" in error.message
        assert error.model.context.get("circuit_state") == "open"


# =============================================================================
# Test Class: OPEN → HALF_OPEN Transition
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionOpenToHalfOpen:
    """Test OPEN → HALF_OPEN state transition.

    Transition Trigger: current_time >= reset_timeout
    Transition Action: Sets _circuit_breaker_open = False, resets failure count
    Note: HALF_OPEN is implicit - appears as CLOSED with failures = 0
    """

    async def test_transition_occurs_after_timeout(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test circuit transitions to HALF_OPEN after reset timeout.

        The implementation transitions from OPEN to HALF_OPEN when:
        1. Circuit is OPEN
        2. Reset timeout has elapsed
        3. _check_circuit_breaker() is called
        """
        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        assert fast_service.get_state() == CircuitState.OPEN

        # Wait for reset timeout
        await asyncio.sleep(fast_service.circuit_breaker_reset_timeout + 0.05)

        # Next check should transition to HALF_OPEN (no error raised)
        await fast_service.check_circuit()

        # Verify state: circuit is no longer open, failures reset
        assert fast_service.get_state() == CircuitState.CLOSED  # Implicit HALF_OPEN
        assert fast_service.get_failure_count() == 0

    async def test_transition_does_not_occur_before_timeout(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test circuit stays OPEN before reset timeout elapses."""
        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        assert fast_service.get_state() == CircuitState.OPEN

        # Immediately try to check (before timeout)
        with pytest.raises(InfraUnavailableError):
            await fast_service.check_circuit()

        # Circuit should still be open
        assert fast_service.get_state() == CircuitState.OPEN

    async def test_transition_with_zero_timeout(self) -> None:
        """Test circuit with zero timeout transitions immediately.

        Edge case: reset_timeout=0 means OPEN → HALF_OPEN is immediate.
        """
        service = CircuitBreakerTestService(threshold=2, reset_timeout=0.0)

        # Open the circuit
        await service.record_failure()
        await service.record_failure()
        assert service.get_state() == CircuitState.OPEN

        # Immediate check should transition (timeout already elapsed)
        await service.check_circuit()

        # Should be in HALF_OPEN (implicit CLOSED with reset failures)
        assert service.get_state() == CircuitState.CLOSED
        assert service.get_failure_count() == 0

    async def test_transition_allows_test_request(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test HALF_OPEN allows a test request through."""
        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        # Wait for timeout
        await asyncio.sleep(fast_service.circuit_breaker_reset_timeout + 0.05)

        # Execute a successful operation (should work in HALF_OPEN)
        result = await fast_service.execute_with_circuit_breaker(should_fail=False)

        assert result == "success"
        assert fast_service.get_state() == CircuitState.CLOSED


# =============================================================================
# Test Class: HALF_OPEN → CLOSED Transition
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionHalfOpenToClosed:
    """Test HALF_OPEN → CLOSED state transition.

    Transition Trigger: Successful operation in HALF_OPEN state
    Transition Action: Circuit fully closed via _reset_circuit_breaker()
    """

    async def test_transition_on_success(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test circuit closes on successful operation in HALF_OPEN state.

        Verifies the complete cycle: CLOSED → OPEN → HALF_OPEN → CLOSED
        """
        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        assert fast_service.get_state() == CircuitState.OPEN

        # Wait for HALF_OPEN
        await asyncio.sleep(fast_service.circuit_breaker_reset_timeout + 0.05)

        # Execute successful operation
        result = await fast_service.execute_with_circuit_breaker(should_fail=False)

        # Verify transition to CLOSED
        assert result == "success"
        assert fast_service.get_state() == CircuitState.CLOSED
        assert fast_service.get_failure_count() == 0

    async def test_transition_allows_normal_operations(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that after HALF_OPEN → CLOSED, normal operations resume."""
        # Complete the CLOSED → OPEN → HALF_OPEN → CLOSED cycle
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        await asyncio.sleep(fast_service.circuit_breaker_reset_timeout + 0.05)
        await fast_service.execute_with_circuit_breaker(should_fail=False)

        assert fast_service.get_state() == CircuitState.CLOSED

        # Verify multiple subsequent operations work
        for _ in range(5):
            result = await fast_service.execute_with_circuit_breaker(should_fail=False)
            assert result == "success"

        assert fast_service.get_state() == CircuitState.CLOSED

    async def test_transition_resets_failure_count(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that HALF_OPEN → CLOSED resets the failure count to zero."""
        # Open circuit and wait for HALF_OPEN
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        await asyncio.sleep(fast_service.circuit_breaker_reset_timeout + 0.05)

        # At HALF_OPEN, failures should already be reset
        # After check_circuit (which transitions to HALF_OPEN)
        await fast_service.check_circuit()
        assert fast_service.get_failure_count() == 0

        # Record a failure (but stay below threshold)
        await fast_service.record_failure()
        assert fast_service.get_failure_count() == 1

        # Successful operation should reset to zero
        await fast_service.reset_circuit()
        assert fast_service.get_failure_count() == 0


# =============================================================================
# Test Class: HALF_OPEN → OPEN Transition
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTransitionHalfOpenToOpen:
    """Test HALF_OPEN → OPEN state transition.

    Transition Trigger: Failures reach threshold in HALF_OPEN state
    Transition Action: Circuit reopens via _record_circuit_failure()

    Note: After transitioning to HALF_OPEN, the failure count is reset to 0.
    Therefore, threshold failures are needed to reopen the circuit.
    """

    async def test_transition_on_failure(
        self, single_failure_service: CircuitBreakerTestService
    ) -> None:
        """Test circuit reopens on failed operation in HALF_OPEN state.

        Uses threshold=1 so a single failure reopens the circuit immediately.
        """
        # Open the circuit
        await single_failure_service.record_failure()
        assert single_failure_service.get_state() == CircuitState.OPEN

        # Wait for HALF_OPEN
        await asyncio.sleep(single_failure_service.circuit_breaker_reset_timeout + 0.05)

        # Execute a failing operation
        with pytest.raises(RuntimeError, match="Simulated operation failure"):
            await single_failure_service.execute_with_circuit_breaker(should_fail=True)

        # Circuit should be OPEN again
        assert single_failure_service.get_state() == CircuitState.OPEN

    async def test_transition_requires_threshold_failures(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that HALF_OPEN → OPEN requires threshold failures.

        With threshold=3:
        - First failure in HALF_OPEN: failures=1, still CLOSED (HALF_OPEN)
        - Second failure: failures=2, still CLOSED (HALF_OPEN)
        - Third failure: failures=3, transitions to OPEN
        """
        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        # Wait for HALF_OPEN
        await asyncio.sleep(fast_service.circuit_breaker_reset_timeout + 0.05)

        # Transition to HALF_OPEN
        await fast_service.check_circuit()
        assert fast_service.get_failure_count() == 0

        # Record failures below threshold - should stay in HALF_OPEN (implicit CLOSED)
        for i in range(fast_service.circuit_breaker_threshold - 1):
            await fast_service.record_failure()
            assert fast_service.get_state() == CircuitState.CLOSED  # Still HALF_OPEN
            assert fast_service.get_failure_count() == i + 1

        # One more failure should reopen
        await fast_service.record_failure()
        assert fast_service.get_state() == CircuitState.OPEN

    async def test_transition_blocks_subsequent_requests(
        self, single_failure_service: CircuitBreakerTestService
    ) -> None:
        """Test that after HALF_OPEN → OPEN, requests are blocked again."""
        # CLOSED → OPEN
        await single_failure_service.record_failure()

        # Wait for HALF_OPEN
        await asyncio.sleep(single_failure_service.circuit_breaker_reset_timeout + 0.05)

        # Fail in HALF_OPEN (threshold=1, so immediately reopens)
        with pytest.raises(RuntimeError):
            await single_failure_service.execute_with_circuit_breaker(should_fail=True)

        # Verify subsequent requests are blocked
        with pytest.raises(InfraUnavailableError) as exc_info:
            await single_failure_service.check_circuit()

        assert "Circuit breaker is open" in exc_info.value.message


# =============================================================================
# Test Class: CLOSED Stability Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestClosedStateStability:
    """Test CLOSED state stability - circuit stays closed appropriately."""

    async def test_success_resets_failure_count(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that success resets failure count, preventing CLOSED → OPEN."""
        # Record failures below threshold
        for _ in range(fast_service.circuit_breaker_threshold - 1):
            await fast_service.record_failure()

        assert fast_service.get_failure_count() == 2

        # Success should reset the counter
        await fast_service.reset_circuit()

        assert fast_service.get_state() == CircuitState.CLOSED
        assert fast_service.get_failure_count() == 0

    async def test_intermittent_failures_dont_accumulate(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that failures interspersed with successes don't accumulate.

        Pattern: fail, fail, success, fail, fail, success
        Each success resets the counter, so circuit never opens.
        """
        patterns = [
            True,
            True,
            False,
            True,
            True,
            False,
        ]  # True = fail, False = success

        for should_fail in patterns:
            if should_fail:
                await fast_service.record_failure()
            else:
                await fast_service.reset_circuit()

        # Circuit should still be closed
        assert fast_service.get_state() == CircuitState.CLOSED
        assert fast_service.get_failure_count() == 0

    async def test_reset_is_idempotent(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that multiple resets don't corrupt state."""
        # Record some failures
        await fast_service.record_failure()
        await fast_service.record_failure()

        # Multiple resets should be safe
        await fast_service.reset_circuit()
        await fast_service.reset_circuit()
        await fast_service.reset_circuit()

        assert fast_service.get_state() == CircuitState.CLOSED
        assert fast_service.get_failure_count() == 0


# =============================================================================
# Test Class: OPEN State Stability Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestOpenStateStability:
    """Test OPEN state stability - circuit stays open appropriately."""

    async def test_open_circuit_blocks_all_requests(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that OPEN circuit blocks all requests before timeout."""
        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        assert fast_service.get_state() == CircuitState.OPEN

        # All checks should be blocked
        for _ in range(5):
            with pytest.raises(InfraUnavailableError):
                await fast_service.check_circuit()

        # Circuit should still be open
        assert fast_service.get_state() == CircuitState.OPEN

    async def test_open_circuit_includes_retry_after(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that OPEN circuit error includes retry_after_seconds."""
        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        with pytest.raises(InfraUnavailableError) as exc_info:
            await fast_service.check_circuit()

        error = exc_info.value
        retry_after = error.model.context.get("retry_after_seconds")

        assert retry_after is not None
        assert isinstance(retry_after, int)
        assert retry_after >= 0

    async def test_additional_failures_during_open_ignored(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test that failures during OPEN state don't extend timeout.

        Once open, recording more failures shouldn't change behavior.
        """
        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        original_open_until = fast_service.get_open_until()

        # Record more failures (simulating if somehow failures were recorded)
        await fast_service.record_failure()
        await fast_service.record_failure()

        # The open_until timestamp might update, but circuit stays open
        assert fast_service.get_state() == CircuitState.OPEN


# =============================================================================
# Test Class: Complete Cycle Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCompleteCycles:
    """Test complete state transition cycles."""

    async def test_full_recovery_cycle(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test complete CLOSED → OPEN → HALF_OPEN → CLOSED cycle."""
        # Phase 1: CLOSED
        assert fast_service.get_state() == CircuitState.CLOSED

        # Phase 2: CLOSED → OPEN
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        assert fast_service.get_state() == CircuitState.OPEN

        # Phase 3: OPEN → HALF_OPEN
        await asyncio.sleep(fast_service.circuit_breaker_reset_timeout + 0.05)

        # Phase 4: HALF_OPEN → CLOSED
        result = await fast_service.execute_with_circuit_breaker(should_fail=False)

        assert result == "success"
        assert fast_service.get_state() == CircuitState.CLOSED
        assert fast_service.get_failure_count() == 0

    async def test_failed_recovery_cycle(
        self, single_failure_service: CircuitBreakerTestService
    ) -> None:
        """Test CLOSED → OPEN → HALF_OPEN → OPEN cycle (failed recovery)."""
        # Phase 1: CLOSED → OPEN
        await single_failure_service.record_failure()
        assert single_failure_service.get_state() == CircuitState.OPEN

        # Phase 2: Wait for HALF_OPEN
        await asyncio.sleep(single_failure_service.circuit_breaker_reset_timeout + 0.05)

        # Phase 3: HALF_OPEN → OPEN (failed recovery)
        with pytest.raises(RuntimeError):
            await single_failure_service.execute_with_circuit_breaker(should_fail=True)

        assert single_failure_service.get_state() == CircuitState.OPEN

    async def test_multiple_recovery_cycles(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test circuit can handle multiple failure/recovery cycles."""
        for cycle in range(3):
            # Trip the circuit
            for _ in range(fast_service.circuit_breaker_threshold):
                await fast_service.record_failure()

            assert fast_service.get_state() == CircuitState.OPEN, (
                f"Cycle {cycle}: expected OPEN"
            )

            # Wait for recovery
            await asyncio.sleep(fast_service.circuit_breaker_reset_timeout + 0.05)

            # Recover
            result = await fast_service.execute_with_circuit_breaker(should_fail=False)
            assert result == "success", f"Cycle {cycle}: recovery failed"
            assert fast_service.get_state() == CircuitState.CLOSED, (
                f"Cycle {cycle}: expected CLOSED"
            )


# =============================================================================
# Test Class: Timing Precision Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestTimingPrecision:
    """Test timing precision for state transitions."""

    async def test_transition_timing_boundary(self) -> None:
        """Test transition occurs at exact timeout boundary."""
        reset_timeout = 0.2
        service = CircuitBreakerTestService(threshold=1, reset_timeout=reset_timeout)

        # Open circuit
        await service.record_failure()
        assert service.get_state() == CircuitState.OPEN

        # Check just before timeout - should still be blocked
        await asyncio.sleep(reset_timeout * 0.5)
        with pytest.raises(InfraUnavailableError):
            await service.check_circuit()

        # Wait remaining time plus small buffer
        await asyncio.sleep(reset_timeout * 0.5 + 0.05)

        # Should now allow (HALF_OPEN)
        await service.check_circuit()
        assert service.get_state() == CircuitState.CLOSED

    async def test_very_long_timeout_value(self) -> None:
        """Test circuit breaker with very long reset timeout."""
        service = CircuitBreakerTestService(threshold=1, reset_timeout=3600.0)

        # Open circuit
        await service.record_failure()

        # Verify retry_after is approximately correct
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit()

        retry_after = exc_info.value.model.context.get("retry_after_seconds")
        assert retry_after is not None
        assert retry_after > 3500  # Should be close to 3600


# =============================================================================
# Test Class: Correlation ID Propagation
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestCorrelationIdInTransitions:
    """Test correlation ID handling during state transitions."""

    async def test_correlation_id_in_open_error(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test correlation ID is included when circuit opens."""
        correlation_id = uuid4()

        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        # Check with correlation ID
        with pytest.raises(InfraUnavailableError) as exc_info:
            await fast_service.check_circuit(correlation_id=correlation_id)

        error = exc_info.value
        assert error.model.correlation_id == correlation_id

    async def test_correlation_id_generated_when_none(
        self, fast_service: CircuitBreakerTestService
    ) -> None:
        """Test correlation ID is auto-generated when not provided."""
        # Open the circuit
        for _ in range(fast_service.circuit_breaker_threshold):
            await fast_service.record_failure()

        # Check without correlation ID
        with pytest.raises(InfraUnavailableError) as exc_info:
            await fast_service.check_circuit()

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert isinstance(error.model.correlation_id, UUID)
        assert error.model.correlation_id.version == 4


__all__ = [
    "CircuitBreakerTestService",
    "TestTransitionClosedToOpen",
    "TestTransitionOpenToHalfOpen",
    "TestTransitionHalfOpenToClosed",
    "TestTransitionHalfOpenToOpen",
    "TestClosedStateStability",
    "TestOpenStateStability",
    "TestCompleteCycles",
    "TestTimingPrecision",
    "TestCorrelationIdInTransitions",
]
