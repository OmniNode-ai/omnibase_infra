"""Test suite for DatabaseCircuitBreaker."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from omninode_bridge.nodes.database_adapter_effect.v1_0_0.circuit_breaker import (
    CircuitBreakerOpenError,
    CircuitBreakerState,
    DatabaseCircuitBreaker,
)


class TestDatabaseCircuitBreaker:
    """Test suite for DatabaseCircuitBreaker."""

    def test_initialization_with_defaults(self):
        """Test circuit breaker initialization with default values."""
        cb = DatabaseCircuitBreaker()

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure_time is None
        assert cb.half_open_calls == 0

        # Default configuration
        assert cb.failure_threshold == 5
        assert cb.timeout_seconds == 60
        assert cb.half_open_max_calls == 3
        assert cb.half_open_success_threshold == 2

    def test_initialization_with_custom_values(self):
        """Test circuit breaker initialization with custom values."""
        cb = DatabaseCircuitBreaker(
            failure_threshold=10,
            timeout_seconds=120,
            half_open_max_calls=5,
            half_open_success_threshold=3,
        )

        assert cb.failure_threshold == 10
        assert cb.timeout_seconds == 120
        assert cb.half_open_max_calls == 5
        assert cb.half_open_success_threshold == 3

    def test_initialization_validation(self):
        """Test that invalid initialization values raise ValueError."""
        with pytest.raises(ValueError, match="failure_threshold must be > 0"):
            DatabaseCircuitBreaker(failure_threshold=0)

        with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
            DatabaseCircuitBreaker(timeout_seconds=0)

        with pytest.raises(ValueError, match="half_open_max_calls must be > 0"):
            DatabaseCircuitBreaker(half_open_max_calls=0)

        with pytest.raises(ValueError, match="half_open_success_threshold must be > 0"):
            DatabaseCircuitBreaker(half_open_success_threshold=0)

    def test_from_config_with_defaults(self):
        """Test creating circuit breaker from empty config."""
        cb = DatabaseCircuitBreaker.from_config({})

        # Should use default values
        assert cb.failure_threshold == 5
        assert cb.timeout_seconds == 60
        assert cb.half_open_max_calls == 3
        assert cb.half_open_success_threshold == 2

    def test_from_config_with_custom_values(self):
        """Test creating circuit breaker from custom config."""
        config = {
            "failure_threshold": 8,
            "recovery_timeout": 90,
            "half_open_max_calls": 4,
            "half_open_success_threshold": 1,
        }

        cb = DatabaseCircuitBreaker.from_config(config)

        assert cb.failure_threshold == 8
        assert cb.timeout_seconds == 90
        assert cb.half_open_max_calls == 4
        assert cb.half_open_success_threshold == 1

    @pytest.mark.asyncio
    async def test_execute_success_in_closed_state(self):
        """Test successful operation execution in CLOSED state."""
        cb = DatabaseCircuitBreaker()
        mock_operation = AsyncMock(return_value="success")

        result = await cb.execute(mock_operation, "arg1", kwarg1="value1")

        assert result == "success"
        mock_operation.assert_called_once_with("arg1", kwarg1="value1")

        # Check state remains CLOSED
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_execute_failure_in_closed_state(self):
        """Test failed operation execution in CLOSED state."""
        cb = DatabaseCircuitBreaker(failure_threshold=3)
        mock_operation = AsyncMock(side_effect=Exception("DB Error"))

        with pytest.raises(Exception, match="DB Error"):
            await cb.execute(mock_operation)

        mock_operation.assert_called_once()

        # Check failure count increments
        assert cb.failure_count == 1
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_execute_failure_threshold_opens_circuit(self):
        """Test that reaching failure threshold opens the circuit."""
        cb = DatabaseCircuitBreaker(failure_threshold=2)
        mock_operation = AsyncMock(side_effect=Exception("DB Error"))

        # First failure
        with pytest.raises(Exception):
            await cb.execute(mock_operation)
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 1

        # Second failure should open circuit
        with pytest.raises(Exception):
            await cb.execute(mock_operation)
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 2

    @pytest.mark.asyncio
    async def test_execute_rejects_in_open_state(self):
        """Test that operations are rejected in OPEN state."""
        cb = DatabaseCircuitBreaker()
        mock_operation = AsyncMock(return_value="success")

        # Manually set to OPEN state
        cb.state = CircuitBreakerState.OPEN
        cb.last_failure_time = datetime.now(UTC)

        with pytest.raises(CircuitBreakerOpenError, match="Circuit breaker is OPEN"):
            await cb.execute(mock_operation)

        # Operation should not be called
        mock_operation.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_half_open_state_success(self):
        """Test successful operations in HALF_OPEN state."""
        cb = DatabaseCircuitBreaker(
            failure_threshold=1, half_open_success_threshold=3  # Use 3 instead of 2
        )
        mock_operation = AsyncMock(return_value="success")

        # First, open the circuit
        cb.state = CircuitBreakerState.OPEN
        cb.last_failure_time = datetime.now(UTC) - timedelta(seconds=61)  # Past timeout

        # First execution should transition to HALF_OPEN and succeed
        result1 = await cb.execute(mock_operation)
        assert result1 == "success"
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.success_count == 1

        # Second execution should stay in HALF_OPEN and increment success_count
        result2 = await cb.execute(mock_operation)
        assert result2 == "success"
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.success_count == 2

        # Third execution should close the circuit
        result3 = await cb.execute(mock_operation)
        assert result3 == "success"
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_execute_half_open_state_failure_opens_circuit(self):
        """Test that failure in HALF_OPEN state opens circuit."""
        cb = DatabaseCircuitBreaker()
        mock_operation = AsyncMock(side_effect=Exception("DB Error"))

        # Set to HALF_OPEN state
        cb.state = CircuitBreakerState.HALF_OPEN

        with pytest.raises(Exception):
            await cb.execute(mock_operation)

        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_execute_sync_operation(self):
        """Test execution of synchronous operations."""
        cb = DatabaseCircuitBreaker()

        def sync_operation():
            return "sync_result"

        result = await cb.execute(sync_operation)
        assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_record_success_in_closed_state(self):
        """Test recording success in CLOSED state."""
        cb = DatabaseCircuitBreaker()
        cb.failure_count = 5

        await cb.record_success()

        assert cb.failure_count == 0
        assert cb._total_successes == 1

    @pytest.mark.asyncio
    async def test_record_success_in_half_open_state(self):
        """Test recording success in HALF_OPEN state."""
        cb = DatabaseCircuitBreaker(half_open_success_threshold=3)
        cb.state = CircuitBreakerState.HALF_OPEN

        # First success
        await cb.record_success()
        assert cb.success_count == 1
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Second success
        await cb.record_success()
        assert cb.success_count == 2
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Third success should close circuit (and reset success_count)
        await cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        # success_count is reset to 0 when transitioning to CLOSED

    @pytest.mark.asyncio
    async def test_record_failure_in_closed_state(self):
        """Test recording failure in CLOSED state."""
        cb = DatabaseCircuitBreaker(failure_threshold=5)
        cb.failure_count = 4

        await cb.record_failure()

        assert cb.failure_count == 5
        assert cb.state == CircuitBreakerState.OPEN
        assert cb._total_failures == 1

    @pytest.mark.asyncio
    async def test_record_failure_in_half_open_state(self):
        """Test recording failure in HALF_OPEN state."""
        cb = DatabaseCircuitBreaker()
        cb.state = CircuitBreakerState.HALF_OPEN

        await cb.record_failure()

        assert cb.state == CircuitBreakerState.OPEN

    def test_get_state(self):
        """Test getting current circuit breaker state."""
        cb = DatabaseCircuitBreaker()

        assert cb.get_state() == CircuitBreakerState.CLOSED

        cb.state = CircuitBreakerState.OPEN
        assert cb.get_state() == CircuitBreakerState.OPEN

    def test_get_metrics(self):
        """Test getting circuit breaker metrics."""
        cb = DatabaseCircuitBreaker(
            failure_threshold=3,
            timeout_seconds=45,
            half_open_max_calls=4,
        )

        # Set some state
        cb.state = CircuitBreakerState.OPEN
        cb.failure_count = 2
        cb._total_failures = 5
        cb._total_successes = 10
        cb._state_transitions = 3
        cb.last_failure_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        metrics = cb.get_metrics()

        assert metrics["state"] == "open"
        assert metrics["failure_count"] == 2
        assert metrics["total_failures"] == 5
        assert metrics["total_successes"] == 10
        assert metrics["state_transitions"] == 3
        assert metrics["last_failure_time"] == "2025-01-01T12:00:00+00:00"
        assert metrics["config"]["failure_threshold"] == 3
        assert metrics["config"]["timeout_seconds"] == 45
        assert metrics["config"]["half_open_max_calls"] == 4

    def test_should_attempt_reset_no_failure(self):
        """Test reset check when no failure has occurred."""
        cb = DatabaseCircuitBreaker()

        assert cb._should_attempt_reset() is True

    @pytest.mark.asyncio
    async def test_should_attempt_reset_timeout_elapsed(self):
        """Test reset check when timeout has elapsed."""
        cb = DatabaseCircuitBreaker(timeout_seconds=1)
        cb.last_failure_time = datetime.now(UTC)

        # Wait for timeout
        with patch(
            "omninode_bridge.nodes.database_adapter_effect.v1_0_0.circuit_breaker.datetime"
        ) as mock_dt:
            # Simulate time passing
            mock_dt.now.return_value = datetime.now(UTC) + timedelta(seconds=2)

            assert cb._should_attempt_reset() is True

    def test_should_attempt_reset_timeout_not_elapsed(self):
        """Test reset check when timeout has not elapsed."""
        cb = DatabaseCircuitBreaker(timeout_seconds=60)
        cb.last_failure_time = datetime.now(UTC)

        assert cb._should_attempt_reset() is False

    @pytest.mark.asyncio
    async def test_half_open_max_calls_limit(self):
        """Test that HALF_OPEN state respects max concurrent calls limit."""
        cb = DatabaseCircuitBreaker(half_open_max_calls=2)
        cb.state = CircuitBreakerState.HALF_OPEN
        cb.half_open_calls = 2

        mock_operation = AsyncMock(return_value="success")

        # Should reject when limit reached
        with pytest.raises(
            CircuitBreakerOpenError, match="max concurrent calls reached"
        ):
            await cb.execute(mock_operation)

    @pytest.mark.asyncio
    async def test_half_open_calls_decrement(self):
        """Test that half_open_calls decrement after execution."""
        cb = DatabaseCircuitBreaker(half_open_max_calls=3)
        cb.state = CircuitBreakerState.OPEN
        cb.last_failure_time = datetime.now(UTC) - timedelta(seconds=61)  # Past timeout

        mock_operation = AsyncMock(return_value="success")

        # First execution should increment to 1, then decrement to 0
        await cb.execute(mock_operation)
        assert cb.half_open_calls == 0

        # Second execution should increment to 1, then decrement to 0
        await cb.execute(mock_operation)
        assert cb.half_open_calls == 0

    @pytest.mark.asyncio
    async def test_state_transition_closed_to_open(self):
        """Test state transition from CLOSED to OPEN."""
        cb = DatabaseCircuitBreaker()

        await cb._transition_to_open()

        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.half_open_calls == 0
        assert cb._state_transitions == 1

    @pytest.mark.asyncio
    async def test_state_transition_open_to_half_open(self):
        """Test state transition from OPEN to HALF_OPEN."""
        cb = DatabaseCircuitBreaker()
        cb.state = CircuitBreakerState.OPEN

        await cb._transition_to_half_open()

        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.half_open_calls == 0
        assert cb._state_transitions == 1

    @pytest.mark.asyncio
    async def test_state_transition_half_open_to_closed(self):
        """Test state transition from HALF_OPEN to CLOSED."""
        cb = DatabaseCircuitBreaker()
        cb.state = CircuitBreakerState.HALF_OPEN

        await cb._transition_to_closed()

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.half_open_calls == 0
        assert cb._state_transitions == 1

    def test_circuit_breaker_state_enum(self):
        """Test CircuitBreakerState enum values."""
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"

    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError exception."""
        error = CircuitBreakerOpenError("Custom message")

        assert str(error) == "Custom message"
        assert error.message == "Custom message"

    def test_circuit_breaker_open_error_default_message(self):
        """Test CircuitBreakerOpenError default message."""
        error = CircuitBreakerOpenError()

        assert str(error) == "Circuit breaker is OPEN"
        assert error.message == "Circuit breaker is OPEN"
