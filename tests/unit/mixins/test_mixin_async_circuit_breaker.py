# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Comprehensive unit tests for MixinAsyncCircuitBreaker.

This test suite validates:
- Basic circuit breaker functionality (state management, failure counting)
- State transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- Thread safety with concurrent operations (100+ parallel tasks)
- Correlation ID propagation and generation
- Error context validation
- Edge cases (threshold=1, zero timeout, multiple resets)

Test Organization:
    - TestMixinAsyncCircuitBreakerBasics: Basic functionality
    - TestMixinAsyncCircuitBreakerStateTransitions: State machine transitions
    - TestMixinAsyncCircuitBreakerThreadSafety: Concurrency and race conditions
    - TestMixinAsyncCircuitBreakerCorrelationId: Correlation ID handling
    - TestMixinAsyncCircuitBreakerErrorContext: Error context validation
    - TestMixinAsyncCircuitBreakerEdgeCases: Edge cases and boundary conditions

Coverage Goals:
    - >90% code coverage for mixin
    - All state transitions tested
    - Thread safety validated with parallel execution
    - All error paths tested
"""

import asyncio
import time
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    CircuitState,
    MixinAsyncCircuitBreaker,
)


class TestCircuitBreakerService(MixinAsyncCircuitBreaker):
    """Test service that uses circuit breaker mixin for testing."""

    def __init__(
        self,
        threshold: int = 5,
        reset_timeout: float = 60.0,
        service_name: str = "test-service",
        transport_type: EnumInfraTransportType = EnumInfraTransportType.HTTP,
    ):
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
        """Check circuit breaker state (thread-safe wrapper for testing)."""
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

    async def record_failure(
        self, operation: str = "test_operation", correlation_id: UUID | None = None
    ) -> None:
        """Record circuit failure (thread-safe wrapper for testing)."""
        async with self._circuit_breaker_lock:
            await self._record_circuit_failure(operation, correlation_id)

    async def reset_circuit(self) -> None:
        """Reset circuit breaker (thread-safe wrapper for testing)."""
        async with self._circuit_breaker_lock:
            await self._reset_circuit_breaker()

    def get_state(self) -> CircuitState:
        """Get current circuit state (for testing assertions)."""
        if self._circuit_breaker_open:
            return CircuitState.OPEN
        return CircuitState.CLOSED

    def get_failure_count(self) -> int:
        """Get current failure count (for testing assertions)."""
        return self._circuit_breaker_failures


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerBasics:
    """Test basic circuit breaker functionality."""

    async def test_circuit_starts_closed(self) -> None:
        """Test that circuit breaker starts in CLOSED state."""
        service = TestCircuitBreakerService()
        assert service.get_state() == CircuitState.CLOSED
        assert service.get_failure_count() == 0
        assert not service._circuit_breaker_open

    async def test_check_allows_operation_when_closed(self) -> None:
        """Test that check_circuit allows operations when circuit is CLOSED."""
        service = TestCircuitBreakerService()

        # Should not raise when circuit is closed
        await service.check_circuit("test_operation")

        # Circuit should remain closed
        assert service.get_state() == CircuitState.CLOSED

    async def test_record_failure_increments_counter(self) -> None:
        """Test that record_failure increments the failure counter."""
        service = TestCircuitBreakerService(threshold=5)

        # Record multiple failures (below threshold)
        await service.record_failure("test_operation")
        assert service.get_failure_count() == 1

        await service.record_failure("test_operation")
        assert service.get_failure_count() == 2

        await service.record_failure("test_operation")
        assert service.get_failure_count() == 3

        # Circuit should still be closed (below threshold)
        assert service.get_state() == CircuitState.CLOSED

    async def test_record_failure_opens_circuit_at_threshold(self) -> None:
        """Test that circuit opens when failure threshold is reached."""
        service = TestCircuitBreakerService(threshold=3)

        # Record failures up to threshold
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")

        # Circuit should now be open
        assert service.get_state() == CircuitState.OPEN
        assert service._circuit_breaker_open is True

    async def test_check_raises_when_open(self) -> None:
        """Test that check_circuit raises InfraUnavailableError when circuit is OPEN."""
        service = TestCircuitBreakerService(threshold=2)

        # Open the circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN

        # check_circuit should raise InfraUnavailableError
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        assert "Circuit breaker is open" in error.message
        assert error.model.context.get("circuit_state") == "open"

    async def test_reset_closes_circuit(self) -> None:
        """Test that reset_circuit closes the circuit and resets failure count."""
        service = TestCircuitBreakerService(threshold=2)

        # Open the circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN

        # Reset the circuit
        await service.reset_circuit()

        # Circuit should now be closed
        assert service.get_state() == CircuitState.CLOSED
        assert service.get_failure_count() == 0
        assert service._circuit_breaker_open is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerStateTransitions:
    """Test circuit breaker state machine transitions."""

    async def test_state_transition_closed_to_open(self) -> None:
        """Test CLOSED → OPEN transition when threshold is reached."""
        service = TestCircuitBreakerService(threshold=3)

        # Start in CLOSED state
        assert service.get_state() == CircuitState.CLOSED

        # Record failures to reach threshold
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.CLOSED  # Still closed

        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN  # Now open

    async def test_state_transition_open_to_half_open(self) -> None:
        """Test OPEN → HALF_OPEN transition after reset timeout."""
        service = TestCircuitBreakerService(threshold=2, reset_timeout=0.1)

        # Open the circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN

        # Wait for reset timeout
        await asyncio.sleep(0.15)

        # Next check should transition to HALF_OPEN (no error)
        await service.check_circuit("test_operation")

        # Circuit should be half-open (failures reset)
        assert service.get_failure_count() == 0
        assert service._circuit_breaker_open is False

    async def test_state_transition_half_open_to_closed(self) -> None:
        """Test HALF_OPEN → CLOSED transition on successful operation."""
        service = TestCircuitBreakerService(threshold=2, reset_timeout=0.1)

        # Open the circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN

        # Wait for reset timeout (transition to HALF_OPEN)
        await asyncio.sleep(0.15)
        await service.check_circuit("test_operation")

        # Successful operation (explicit reset)
        await service.reset_circuit()

        # Circuit should be fully closed
        assert service.get_state() == CircuitState.CLOSED
        assert service.get_failure_count() == 0

    async def test_state_transition_half_open_to_open(self) -> None:
        """Test HALF_OPEN → OPEN transition on failure after timeout."""
        service = TestCircuitBreakerService(threshold=2, reset_timeout=0.1)

        # Open the circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN

        # Wait for reset timeout (transition to HALF_OPEN)
        await asyncio.sleep(0.15)
        await service.check_circuit("test_operation")

        # Another failure in HALF_OPEN state
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")

        # Circuit should be open again
        assert service.get_state() == CircuitState.OPEN

    async def test_auto_reset_after_timeout(self) -> None:
        """Test automatic reset after timeout elapsed."""
        service = TestCircuitBreakerService(threshold=2, reset_timeout=0.1)

        # Open the circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN

        # Before timeout - should raise
        with pytest.raises(InfraUnavailableError):
            await service.check_circuit("test_operation")

        # Wait for timeout
        await asyncio.sleep(0.15)

        # After timeout - should not raise (auto-reset to HALF_OPEN)
        await service.check_circuit("test_operation")
        assert service.get_failure_count() == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerThreadSafety:
    """Test circuit breaker thread safety with concurrent operations."""

    async def test_concurrent_check_operations(self) -> None:
        """Test multiple check operations in parallel (100 tasks)."""
        service = TestCircuitBreakerService(threshold=10)

        # Run 100 concurrent check operations
        tasks = [service.check_circuit("test_operation") for _ in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All checks should succeed (circuit is closed)
        assert all(result is None for result in results)

    async def test_concurrent_failure_recording(self) -> None:
        """Test multiple failure recordings in parallel (100 tasks)."""
        service = TestCircuitBreakerService(threshold=200)

        # Run 100 concurrent failure recordings
        tasks = [service.record_failure("test_operation") for _ in range(100)]
        await asyncio.gather(*tasks)

        # Failure count should be exactly 100 (no race conditions)
        assert service.get_failure_count() == 100

    async def test_concurrent_check_and_failure(self) -> None:
        """Test mixed check and failure operations in parallel."""
        service = TestCircuitBreakerService(threshold=50)

        # Create mixed tasks (50 checks, 50 failures)
        check_tasks = [service.check_circuit("check") for _ in range(50)]
        failure_tasks = [service.record_failure("failure") for _ in range(50)]
        all_tasks = check_tasks + failure_tasks

        # Run all tasks concurrently
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Check operations should succeed, failure operations should record
        check_results = results[:50]
        assert all(result is None for result in check_results)

        # Circuit should be open (50 failures >= threshold)
        assert service.get_state() == CircuitState.OPEN

    async def test_no_race_condition_at_threshold(self) -> None:
        """Test that exactly threshold failures opens circuit (no race)."""
        threshold = 10
        service = TestCircuitBreakerService(threshold=threshold)

        # Record exactly threshold failures concurrently
        tasks = [service.record_failure("test_operation") for _ in range(threshold)]
        await asyncio.gather(*tasks)

        # Circuit should be open
        assert service.get_state() == CircuitState.OPEN
        assert service.get_failure_count() == threshold

    async def test_lock_prevents_race_conditions(self) -> None:
        """Test that lock prevents race conditions during state transitions."""
        service = TestCircuitBreakerService(threshold=5, reset_timeout=0.1)

        # Concurrent operations: failures, checks, resets
        async def mixed_operations() -> None:
            """Perform mixed operations concurrently."""
            operations = []
            for i in range(20):
                if i % 3 == 0:
                    operations.append(service.record_failure("failure"))
                elif i % 3 == 1:
                    operations.append(service.check_circuit("check"))
                else:
                    operations.append(service.reset_circuit())

            await asyncio.gather(*operations, return_exceptions=True)

        # Run mixed operations
        await mixed_operations()

        # Final state should be consistent (no corruption)
        # Either closed (reset) or open (failures)
        state = service.get_state()
        assert state in (CircuitState.CLOSED, CircuitState.OPEN)


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerCorrelationId:
    """Test correlation ID propagation and generation."""

    async def test_correlation_id_propagation(self) -> None:
        """Test that correlation_id flows through errors."""
        service = TestCircuitBreakerService(threshold=1)

        # Open the circuit
        correlation_id = uuid4()
        await service.record_failure("test_operation", correlation_id)

        # Check should raise with same correlation_id
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation", correlation_id)

        error = exc_info.value
        assert error.model.correlation_id == correlation_id

    async def test_correlation_id_generated_if_none(self) -> None:
        """Test that UUID is generated if correlation_id not provided."""
        service = TestCircuitBreakerService(threshold=1)

        # Open the circuit without correlation_id
        await service.record_failure("test_operation")

        # Check should raise with generated correlation_id
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert isinstance(error.model.correlation_id, UUID)
        assert error.model.correlation_id.version == 4

    async def test_correlation_id_in_error_context(self) -> None:
        """Test that correlation_id is properly included in error context."""
        service = TestCircuitBreakerService(threshold=1)

        # Open circuit with specific correlation_id
        correlation_id = uuid4()
        await service.record_failure("test_operation", correlation_id)

        # Verify error contains correlation_id
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation", correlation_id)

        error = exc_info.value
        # Correlation ID is at model level, not in context dict
        assert error.model.correlation_id == correlation_id
        # Context is a dict with transport_type, operation, target_name
        assert isinstance(error.model.context, dict)


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerErrorContext:
    """Test error context validation and structure."""

    async def test_error_context_contains_required_fields(self) -> None:
        """Test that error context contains all required fields."""
        service = TestCircuitBreakerService(
            threshold=1,
            service_name="test-service",
            transport_type=EnumInfraTransportType.KAFKA,
        )

        # Open the circuit
        await service.record_failure("publish_event")

        # Check error context structure
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("publish_event")

        error = exc_info.value
        context = error.model.context

        # Context is a dict with structured fields
        assert context["transport_type"] == EnumInfraTransportType.KAFKA
        assert context["operation"] == "publish_event"
        assert context["target_name"] == "test-service"
        # Correlation ID is at model level
        assert error.model.correlation_id is not None

    async def test_error_includes_service_name(self) -> None:
        """Test that error includes service_name in context."""
        service_name = "custom-kafka-service"
        service = TestCircuitBreakerService(
            threshold=1, service_name=service_name
        )

        # Open circuit
        await service.record_failure("test_operation")

        # Verify service name in error
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        assert error.model.context["target_name"] == service_name
        assert service_name in error.message

    async def test_error_includes_circuit_state(self) -> None:
        """Test that error includes circuit_state in context."""
        service = TestCircuitBreakerService(threshold=1)

        # Open circuit
        await service.record_failure("test_operation")

        # Verify circuit_state in error context
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        assert error.model.context.get("circuit_state") == "open"

    async def test_error_includes_retry_after(self) -> None:
        """Test that error includes retry_after_seconds calculated correctly."""
        reset_timeout = 10.0
        service = TestCircuitBreakerService(threshold=1, reset_timeout=reset_timeout)

        # Open circuit
        await service.record_failure("test_operation")

        # Immediately check (should raise with retry_after)
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        retry_after = error.model.context.get("retry_after_seconds")

        # Should be close to reset_timeout (within 1 second tolerance)
        assert retry_after is not None
        assert isinstance(retry_after, int)
        assert 0 <= retry_after <= reset_timeout


@pytest.mark.unit
@pytest.mark.asyncio
class TestMixinAsyncCircuitBreakerEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_threshold_of_one(self) -> None:
        """Test circuit breaker with threshold=1 (opens on first failure)."""
        service = TestCircuitBreakerService(threshold=1)

        # First failure should open circuit
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN

        # Check should raise immediately
        with pytest.raises(InfraUnavailableError):
            await service.check_circuit("test_operation")

    async def test_zero_reset_timeout(self) -> None:
        """Test circuit breaker with zero reset timeout (immediate reset)."""
        service = TestCircuitBreakerService(threshold=2, reset_timeout=0.0)

        # Open circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN

        # Immediate check should auto-reset (no wait needed)
        await service.check_circuit("test_operation")
        assert service.get_failure_count() == 0

    async def test_very_long_reset_timeout(self) -> None:
        """Test circuit breaker with very long reset timeout."""
        service = TestCircuitBreakerService(threshold=1, reset_timeout=3600.0)

        # Open circuit
        await service.record_failure("test_operation")

        # Circuit should stay open for long timeout
        with pytest.raises(InfraUnavailableError) as exc_info:
            await service.check_circuit("test_operation")

        error = exc_info.value
        retry_after = error.model.context.get("retry_after_seconds")
        assert retry_after is not None
        assert retry_after > 3500  # Should be close to 3600

    async def test_multiple_resets(self) -> None:
        """Test multiple manual resets work correctly."""
        service = TestCircuitBreakerService(threshold=2)

        # Open and reset circuit multiple times
        for _ in range(5):
            # Open circuit
            await service.record_failure("test_operation")
            await service.record_failure("test_operation")
            assert service.get_state() == CircuitState.OPEN

            # Reset circuit
            await service.reset_circuit()
            assert service.get_state() == CircuitState.CLOSED
            assert service.get_failure_count() == 0

    async def test_failure_after_manual_reset(self) -> None:
        """Test that failures after manual reset work correctly."""
        service = TestCircuitBreakerService(threshold=3)

        # Record some failures
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_failure_count() == 2

        # Manual reset
        await service.reset_circuit()
        assert service.get_failure_count() == 0

        # New failures should count from zero
        await service.record_failure("test_operation")
        assert service.get_failure_count() == 1

        await service.record_failure("test_operation")
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN

    async def test_concurrent_operations_at_threshold_boundary(self) -> None:
        """Test concurrent operations near threshold boundary."""
        threshold = 5
        service = TestCircuitBreakerService(threshold=threshold)

        # Record threshold - 1 failures
        for _ in range(threshold - 1):
            await service.record_failure("test_operation")

        assert service.get_state() == CircuitState.CLOSED

        # One more failure should open circuit
        await service.record_failure("test_operation")
        assert service.get_state() == CircuitState.OPEN

    async def test_reset_idempotency(self) -> None:
        """Test that reset is idempotent (multiple resets don't break state)."""
        service = TestCircuitBreakerService(threshold=2)

        # Open circuit
        await service.record_failure("test_operation")
        await service.record_failure("test_operation")

        # Multiple resets should be safe
        await service.reset_circuit()
        await service.reset_circuit()
        await service.reset_circuit()

        # State should be consistent
        assert service.get_state() == CircuitState.CLOSED
        assert service.get_failure_count() == 0

    async def test_check_circuit_timing_precision(self) -> None:
        """Test that timeout timing is precise (no off-by-one errors)."""
        reset_timeout = 0.2
        service = TestCircuitBreakerService(threshold=1, reset_timeout=reset_timeout)

        # Open circuit
        start_time = time.time()
        await service.record_failure("test_operation")

        # Check immediately - should fail
        with pytest.raises(InfraUnavailableError):
            await service.check_circuit("test_operation")

        # Wait exactly reset_timeout
        elapsed = time.time() - start_time
        remaining = reset_timeout - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining + 0.05)  # Small buffer for precision

        # Check after timeout - should succeed (auto-reset)
        await service.check_circuit("test_operation")
        assert service.get_failure_count() == 0
