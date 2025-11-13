#!/usr/bin/env python3
"""
Unit tests for CircuitBreaker.

Tests circuit breaker pattern implementation including state transitions,
failure detection, recovery behavior, and concurrent operations.

ONEX v2.0 Compliance:
- Comprehensive test coverage for resilience patterns
- State machine validation
- Error handling verification
- Concurrent operation testing
"""

import asyncio

import pytest

from omninode_bridge.clients.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)


class TestCircuitBreakerInitialization:
    """Test circuit breaker initialization and configuration."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        cb = CircuitBreaker("test-service")

        assert cb.name == "test-service"
        assert cb.state == CircuitState.CLOSED
        assert cb.config.failure_threshold == 5
        assert cb.config.recovery_timeout == 60
        assert cb.config.half_open_max_calls == 3
        assert cb.config.success_threshold == 2

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=120,
            half_open_max_calls=5,
            success_threshold=3,
            timeout=60.0,
        )
        cb = CircuitBreaker("test-service", config)

        assert cb.config.failure_threshold == 10
        assert cb.config.recovery_timeout == 120
        assert cb.config.half_open_max_calls == 5
        assert cb.config.success_threshold == 3
        assert cb.config.timeout == 60.0

    def test_initial_metrics(self):
        """Test initial metrics state."""
        cb = CircuitBreaker("test-service")
        metrics = cb.get_metrics()

        assert metrics.name == "test-service"
        assert metrics.state == CircuitState.CLOSED
        assert metrics.failure_count == 0
        assert metrics.success_count == 0
        assert metrics.total_calls == 0
        assert metrics.consecutive_failures == 0
        assert metrics.consecutive_successes == 0
        assert metrics.error_rate == 0.0


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_closed_to_open_on_failure_threshold(self):
        """Test transition from CLOSED to OPEN when failure threshold is met."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            minimum_throughput=1,
            recovery_timeout=60,
        )
        cb = CircuitBreaker("test-service", config)

        # Simulate failures to trigger OPEN state
        for i in range(3):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        # Circuit should now be OPEN
        assert cb.state == CircuitState.OPEN
        assert cb.metrics.consecutive_failures == 3
        assert cb.metrics.failure_count == 3

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_recovery_timeout(self):
        """Test transition from OPEN to HALF_OPEN after recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=1,
            recovery_timeout=1,  # 1 second recovery timeout
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger OPEN state
        for i in range(2):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Next call should transition to HALF_OPEN
        try:
            async with cb.protect():
                pass  # Successful operation
        except CircuitBreakerError:
            pass

        # Should be in HALF_OPEN state after recovery timeout
        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self):
        """Test transition from HALF_OPEN to CLOSED after successful calls."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=1,
            recovery_timeout=1,
            success_threshold=2,  # Need 2 successes to close
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger OPEN state
        for i in range(2):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Transition to HALF_OPEN and then CLOSED with successful calls
        async with cb.protect():
            pass  # Success 1

        assert cb.state == CircuitState.HALF_OPEN

        async with cb.protect():
            pass  # Success 2 - should transition to CLOSED

        assert cb.state == CircuitState.CLOSED
        assert cb.metrics.consecutive_successes == 0  # Reset after closing
        assert cb.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self):
        """Test transition from HALF_OPEN back to OPEN on any failure."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=1,
            recovery_timeout=1,
            success_threshold=2,
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger OPEN state
        for i in range(2):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Transition to HALF_OPEN
        async with cb.protect():
            pass  # First success

        assert cb.state == CircuitState.HALF_OPEN

        # Fail in HALF_OPEN - should reopen circuit
        with pytest.raises(ValueError):
            async with cb.protect():
                raise ValueError("Test failure in half-open")

        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerFailureThreshold:
    """Test failure threshold triggering logic."""

    @pytest.mark.asyncio
    async def test_failure_threshold_requires_minimum_throughput(self):
        """Test that minimum throughput is required before opening circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            minimum_throughput=5,  # Require 5 calls before evaluating
        )
        cb = CircuitBreaker("test-service", config)

        # 3 failures but less than minimum throughput
        for i in range(3):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        # Should still be CLOSED due to minimum throughput not met
        assert cb.state == CircuitState.CLOSED
        assert cb.metrics.failure_count == 3
        assert cb.metrics.total_calls == 3

    @pytest.mark.asyncio
    async def test_consecutive_failures_trigger_open(self):
        """Test consecutive failures trigger OPEN state."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            minimum_throughput=3,
        )
        cb = CircuitBreaker("test-service", config)

        # 3 consecutive failures
        for i in range(3):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        assert cb.state == CircuitState.OPEN
        assert cb.metrics.consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_success_resets_consecutive_failures(self):
        """Test that success resets consecutive failure count."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            minimum_throughput=1,
        )
        cb = CircuitBreaker("test-service", config)

        # 2 failures
        for i in range(2):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        assert cb.metrics.consecutive_failures == 2

        # 1 success - should reset consecutive failures
        async with cb.protect():
            pass

        assert cb.metrics.consecutive_failures == 0
        assert cb.metrics.consecutive_successes == 1
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerRecoveryTimeout:
    """Test recovery timeout behavior."""

    @pytest.mark.asyncio
    async def test_open_blocks_requests_before_timeout(self):
        """Test that OPEN state blocks requests before recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=1,
            recovery_timeout=10,  # Long timeout
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger OPEN state
        for i in range(2):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        assert cb.state == CircuitState.OPEN

        # Should block requests immediately
        with pytest.raises(CircuitBreakerError, match="is OPEN, blocking requests"):
            async with cb.protect():
                pass

    @pytest.mark.asyncio
    async def test_recovery_timeout_allows_half_open_transition(self):
        """Test that recovery timeout allows transition to HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=1,
            recovery_timeout=1,  # Short timeout
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger OPEN state
        for i in range(2):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should allow transition to HALF_OPEN
        async with cb.protect():
            pass

        assert cb.state == CircuitState.HALF_OPEN


class TestCircuitBreakerHalfOpenState:
    """Test half-open state behavior."""

    @pytest.mark.asyncio
    async def test_half_open_max_calls_limit(self):
        """Test that HALF_OPEN state limits concurrent calls."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=1,
            recovery_timeout=1,
            half_open_max_calls=2,  # Allow only 2 concurrent calls
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger OPEN state
        for i in range(2):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Transition to HALF_OPEN
        async with cb.protect():
            pass

        assert cb.state == CircuitState.HALF_OPEN

        # Start two concurrent slow operations
        async def slow_operation():
            async with cb.protect():
                await asyncio.sleep(0.5)

        # Start max_calls operations
        tasks = [asyncio.create_task(slow_operation()) for _ in range(2)]

        # Wait a bit for tasks to start
        await asyncio.sleep(0.1)

        # Third call should be rejected due to max concurrent calls
        with pytest.raises(CircuitBreakerError, match="max concurrent calls reached"):
            async with cb.protect():
                pass

        # Wait for tasks to complete
        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_half_open_success_threshold_closes_circuit(self):
        """Test that reaching success threshold in HALF_OPEN closes circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=1,
            recovery_timeout=1,
            success_threshold=3,  # Need 3 successes
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger OPEN state
        for i in range(2):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Perform successful operations to close circuit
        for i in range(3):
            async with cb.protect():
                pass

        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_single_failure_reopens_circuit(self):
        """Test that any failure in HALF_OPEN reopens circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=1,
            recovery_timeout=1,
            success_threshold=3,
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger OPEN state
        for i in range(2):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # One success
        async with cb.protect():
            pass

        assert cb.state == CircuitState.HALF_OPEN

        # One failure - should reopen
        with pytest.raises(ValueError):
            async with cb.protect():
                raise ValueError("Test failure")

        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerConcurrentOperations:
    """Test concurrent operation handling."""

    @pytest.mark.asyncio
    async def test_concurrent_operations_in_closed_state(self):
        """Test that concurrent operations work correctly in CLOSED state."""
        cb = CircuitBreaker("test-service")

        async def operation(value: int) -> int:
            async with cb.protect():
                await asyncio.sleep(0.1)
                return value * 2

        # Run 10 concurrent operations
        results = await asyncio.gather(*[operation(i) for i in range(10)])

        assert results == [i * 2 for i in range(10)]
        assert cb.metrics.success_count == 10
        assert cb.metrics.total_calls == 10

    @pytest.mark.asyncio
    async def test_concurrent_failures_tracked_correctly(self):
        """Test that concurrent failures are tracked correctly."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            minimum_throughput=5,
        )
        cb = CircuitBreaker("test-service", config)

        async def failing_operation():
            async with cb.protect():
                raise ValueError("Test failure")

        # Run 5 concurrent failing operations
        results = await asyncio.gather(
            *[failing_operation() for _ in range(5)],
            return_exceptions=True,
        )

        assert all(isinstance(r, ValueError) for r in results)
        assert cb.metrics.failure_count == 5
        assert cb.metrics.total_calls == 5

    @pytest.mark.asyncio
    async def test_race_condition_on_state_transition(self):
        """Test race conditions during state transitions are handled safely."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            minimum_throughput=1,
            recovery_timeout=1,
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger OPEN state
        for i in range(3):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Multiple concurrent operations racing to transition to HALF_OPEN
        async def racing_operation():
            try:
                async with cb.protect():
                    await asyncio.sleep(0.1)
                return "success"
            except CircuitBreakerError:
                return "blocked"

        results = await asyncio.gather(*[racing_operation() for _ in range(5)])

        # Some operations should succeed, others might be blocked
        # The key is that the circuit breaker handles this safely
        assert cb.state in [CircuitState.HALF_OPEN, CircuitState.CLOSED]
        assert "success" in results  # At least one should succeed


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics tracking."""

    @pytest.mark.asyncio
    async def test_success_metrics_tracking(self):
        """Test successful operation metrics tracking."""
        cb = CircuitBreaker("test-service")

        for i in range(5):
            async with cb.protect():
                await asyncio.sleep(0.01)

        metrics = cb.get_metrics()
        assert metrics.success_count == 5
        assert metrics.failure_count == 0
        assert metrics.total_calls == 5
        assert metrics.consecutive_successes == 5
        assert metrics.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_failure_metrics_tracking(self):
        """Test failure metrics tracking."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker("test-service", config)

        for i in range(5):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError("Test failure")

        metrics = cb.get_metrics()
        assert metrics.success_count == 0
        assert metrics.failure_count == 5
        assert metrics.total_calls == 5
        assert metrics.consecutive_failures == 5
        assert metrics.error_rate == 1.0

    @pytest.mark.asyncio
    async def test_response_time_tracking(self):
        """Test response time tracking."""
        cb = CircuitBreaker("test-service")

        for i in range(3):
            async with cb.protect():
                await asyncio.sleep(0.1)

        metrics = cb.get_metrics()
        assert len(metrics.response_times) == 3
        assert all(rt >= 0.1 for rt in metrics.response_times)
        assert metrics.avg_response_time >= 0.1

    @pytest.mark.asyncio
    async def test_slow_call_tracking(self):
        """Test slow call tracking."""
        config = CircuitBreakerConfig(slow_call_threshold=0.05)
        cb = CircuitBreaker("test-service", config)

        # Fast call
        async with cb.protect():
            await asyncio.sleep(0.01)

        # Slow call
        async with cb.protect():
            await asyncio.sleep(0.1)

        metrics = cb.get_metrics()
        assert metrics.slow_calls == 1


class TestCircuitBreakerCallMethod:
    """Test circuit breaker call method."""

    @pytest.mark.asyncio
    async def test_call_method_with_successful_function(self):
        """Test call method with successful async function."""
        cb = CircuitBreaker("test-service")

        async def test_func(x: int, y: int) -> int:
            await asyncio.sleep(0.01)
            return x + y

        result = await cb.call(test_func, 10, 20)
        assert result == 30

    @pytest.mark.asyncio
    async def test_call_method_with_failing_function(self):
        """Test call method with failing async function."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker("test-service", config)

        async def failing_func() -> None:
            raise ValueError("Test failure")

        with pytest.raises(ValueError, match="Test failure"):
            await cb.call(failing_func)

    @pytest.mark.asyncio
    async def test_call_method_with_timeout(self):
        """Test call method with timeout."""
        config = CircuitBreakerConfig(timeout=0.1)
        cb = CircuitBreaker("test-service", config)

        async def slow_func() -> None:
            await asyncio.sleep(1.0)

        with pytest.raises(TimeoutError):
            await cb.call(slow_func)

    @pytest.mark.asyncio
    async def test_call_method_with_custom_timeout(self):
        """Test call method with custom timeout override."""
        config = CircuitBreakerConfig(timeout=10.0)
        cb = CircuitBreaker("test-service", config)

        async def slow_func() -> None:
            await asyncio.sleep(0.5)

        # Custom timeout lower than config
        with pytest.raises(TimeoutError):
            await cb.call(slow_func, timeout=0.1)


class TestCircuitBreakerManualControl:
    """Test manual circuit breaker control methods."""

    @pytest.mark.asyncio
    async def test_manual_reset(self):
        """Test manual reset to CLOSED state."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=1,
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger OPEN state
        for i in range(2):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Test failure {i}")

        assert cb.state == CircuitState.OPEN

        # Manual reset
        await cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.metrics.failure_count == 0
        assert cb.metrics.success_count == 0

    @pytest.mark.asyncio
    async def test_force_open(self):
        """Test forcing circuit to OPEN state."""
        cb = CircuitBreaker("test-service")

        assert cb.state == CircuitState.CLOSED

        # Force open
        await cb.force_open()

        assert cb.state == CircuitState.OPEN

        # Should block requests
        with pytest.raises(CircuitBreakerError):
            async with cb.protect():
                pass


class TestCircuitBreakerErrorHandling:
    """Test circuit breaker error handling."""

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test that timeout errors are handled as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=1,
            timeout=0.1,
        )
        cb = CircuitBreaker("test-service", config)

        # Trigger timeouts
        for i in range(2):
            with pytest.raises(TimeoutError):
                await cb.call(asyncio.sleep, 1.0)

        # Circuit should be OPEN due to timeouts
        assert cb.state == CircuitState.OPEN
        assert cb.metrics.failure_count == 2

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Test that exceptions are properly propagated."""
        cb = CircuitBreaker("test-service")

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError):
            async with cb.protect():
                raise CustomError("Custom error message")

    @pytest.mark.asyncio
    async def test_metrics_dict_serialization(self):
        """Test metrics dictionary serialization."""
        cb = CircuitBreaker("test-service")

        async with cb.protect():
            pass

        metrics_dict = cb.metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["name"] == "test-service"
        assert metrics_dict["state"] == "closed"
        assert metrics_dict["success_count"] == 1
        assert "error_rate" in metrics_dict
        assert "avg_response_time" in metrics_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
