"""Unit tests for circuit breaker implementation."""

import asyncio

import pytest

from omninode_bridge.clients.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)


@pytest.fixture
def circuit_breaker_config():
    """Provide circuit breaker configuration for testing."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=2,  # Short timeout for tests
        half_open_max_calls=2,
        success_threshold=2,
        timeout=5.0,
        minimum_throughput=1,  # Low threshold for tests
    )


@pytest.fixture
def circuit_breaker(circuit_breaker_config):
    """Provide circuit breaker instance for testing."""
    return CircuitBreaker("test-circuit", circuit_breaker_config)


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, circuit_breaker):
        """Test circuit breaker starts in CLOSED state."""
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_successful_calls_keep_circuit_closed(self, circuit_breaker):
        """Test successful calls maintain CLOSED state."""

        async def successful_operation():
            return "success"

        for _ in range(5):
            result = await circuit_breaker.call(successful_operation)
            assert result == "success"

        assert circuit_breaker.state == CircuitState.CLOSED
        metrics = circuit_breaker.get_metrics()
        assert metrics.success_count == 5
        assert metrics.failure_count == 0

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self, circuit_breaker):
        """Test circuit opens after failure threshold."""

        async def failing_operation():
            raise ValueError("Operation failed")

        # Fail threshold times
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_operation)

        # Circuit should be OPEN now
        assert circuit_breaker.state == CircuitState.OPEN

        # Next call should fail fast
        with pytest.raises(CircuitBreakerError, match="is OPEN"):
            await circuit_breaker.call(failing_operation)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open_after_timeout(
        self, circuit_breaker
    ):
        """Test circuit transitions to HALF_OPEN after recovery timeout."""

        async def failing_operation():
            raise ValueError("Operation failed")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)

        # Next call should transition to HALF_OPEN
        async def successful_operation():
            return "success"

        result = await circuit_breaker.call(successful_operation)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_after_successful_recovery(self, circuit_breaker):
        """Test circuit closes after successful recovery in HALF_OPEN state."""

        async def failing_operation():
            raise ValueError("Operation failed")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)

        # Successful calls to close circuit
        async def successful_operation():
            return "success"

        for _ in range(circuit_breaker.config.success_threshold):
            await circuit_breaker.call(successful_operation)

        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, circuit_breaker):
        """Test failure in HALF_OPEN state reopens circuit."""

        async def failing_operation():
            raise ValueError("Operation failed")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)

        # One successful call to transition to HALF_OPEN
        async def successful_operation():
            return "success"

        await circuit_breaker.call(successful_operation)
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Failure should reopen circuit
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == CircuitState.OPEN


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_track_success_count(self, circuit_breaker):
        """Test metrics correctly track successful calls."""

        async def successful_operation():
            return "success"

        for _ in range(10):
            await circuit_breaker.call(successful_operation)

        metrics = circuit_breaker.get_metrics()
        assert metrics.success_count == 10
        assert metrics.total_calls == 10

    @pytest.mark.asyncio
    async def test_metrics_track_failure_count(self, circuit_breaker):
        """Test metrics correctly track failed calls."""

        async def failing_operation():
            raise ValueError("Operation failed")

        # First 3 failures will be recorded, then circuit opens
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_operation)

        # After circuit opens, further calls are rejected
        for _ in range(2):
            with pytest.raises(CircuitBreakerError):
                await circuit_breaker.call(failing_operation)

        metrics = circuit_breaker.get_metrics()
        assert (
            metrics.failure_count == 3
        )  # Only the actual failures, not rejected calls
        assert metrics.total_calls == 3

    @pytest.mark.asyncio
    async def test_metrics_calculate_error_rate(self, circuit_breaker):
        """Test metrics calculate error rate correctly."""

        async def successful_operation():
            return "success"

        async def failing_operation():
            raise ValueError("Operation failed")

        # 7 successes, 3 failures = 30% error rate
        for _ in range(7):
            await circuit_breaker.call(successful_operation)

        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_operation)

        metrics = circuit_breaker.get_metrics()
        assert metrics.error_rate == pytest.approx(0.3, rel=0.01)

    @pytest.mark.asyncio
    async def test_metrics_track_consecutive_failures(self, circuit_breaker):
        """Test metrics track consecutive failures."""

        async def failing_operation():
            raise ValueError("Operation failed")

        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_operation)

        metrics = circuit_breaker.get_metrics()
        assert metrics.consecutive_failures == 3


class TestCircuitBreakerTimeout:
    """Test circuit breaker timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_counts_as_failure(self, circuit_breaker):
        """Test timeout is counted as failure."""

        async def slow_operation():
            await asyncio.sleep(10)  # Longer than timeout
            return "success"

        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_operation)

        metrics = circuit_breaker.get_metrics()
        assert metrics.failure_count == 1

    @pytest.mark.asyncio
    async def test_custom_timeout_override(self, circuit_breaker):
        """Test custom timeout can be specified per call."""

        async def slow_operation():
            await asyncio.sleep(1)
            return "success"

        # Should timeout with short timeout
        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_operation, timeout=0.1)

        # Should succeed with longer timeout
        result = await circuit_breaker.call(slow_operation, timeout=2.0)
        assert result == "success"


class TestCircuitBreakerManualControl:
    """Test manual circuit breaker control."""

    @pytest.mark.asyncio
    async def test_manual_reset(self, circuit_breaker):
        """Test manual reset to CLOSED state."""

        async def failing_operation():
            raise ValueError("Operation failed")

        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == CircuitState.OPEN

        # Manual reset
        await circuit_breaker.reset()

        assert circuit_breaker.state == CircuitState.CLOSED
        metrics = circuit_breaker.get_metrics()
        assert metrics.failure_count == 0
        assert metrics.success_count == 0

    @pytest.mark.asyncio
    async def test_manual_force_open(self, circuit_breaker):
        """Test manual force to OPEN state."""
        assert circuit_breaker.state == CircuitState.CLOSED

        # Force open
        await circuit_breaker.force_open()

        assert circuit_breaker.state == CircuitState.OPEN

        # Should reject calls
        async def operation():
            return "success"

        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.call(operation)


class TestCircuitBreakerProtectContext:
    """Test circuit breaker protect context manager."""

    @pytest.mark.asyncio
    async def test_protect_context_success(self, circuit_breaker):
        """Test protect context manager with successful operation."""
        result = None

        async with circuit_breaker.protect():
            result = "success"

        assert result == "success"
        metrics = circuit_breaker.get_metrics()
        assert metrics.success_count == 1

    @pytest.mark.asyncio
    async def test_protect_context_failure(self, circuit_breaker):
        """Test protect context manager with failing operation."""
        with pytest.raises(ValueError):
            async with circuit_breaker.protect():
                raise ValueError("Operation failed")

        metrics = circuit_breaker.get_metrics()
        assert metrics.failure_count == 1

    @pytest.mark.asyncio
    async def test_protect_context_blocks_when_open(self, circuit_breaker):
        """Test protect context blocks when circuit is OPEN."""
        # Force circuit open
        await circuit_breaker.force_open()

        # Should block immediately
        with pytest.raises(CircuitBreakerError):
            async with circuit_breaker.protect():
                pass


class TestCircuitBreakerEdgeCases:
    """Test circuit breaker edge cases."""

    @pytest.mark.asyncio
    async def test_minimum_throughput_requirement(self):
        """Test circuit doesn't open without minimum throughput."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            minimum_throughput=5,
        )
        cb = CircuitBreaker("test", config)

        async def failing_operation():
            raise ValueError("Operation failed")

        # Only 2 failures (below minimum throughput)
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(failing_operation)

        # Circuit should still be CLOSED
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_slow_call_tracking(self, circuit_breaker):
        """Test slow calls are tracked in metrics."""

        async def slow_operation():
            await asyncio.sleep(6)  # Longer than slow_call_threshold
            return "success"

        # Execute slow operation (with extended timeout)
        await circuit_breaker.call(slow_operation, timeout=10.0)

        metrics = circuit_breaker.get_metrics()
        assert metrics.slow_calls == 1

    @pytest.mark.asyncio
    async def test_response_time_tracking(self, circuit_breaker):
        """Test response times are tracked."""

        async def fast_operation():
            await asyncio.sleep(0.1)
            return "success"

        for _ in range(5):
            await circuit_breaker.call(fast_operation)

        metrics = circuit_breaker.get_metrics()
        assert len(metrics.response_times) == 5
        assert metrics.avg_response_time > 0
