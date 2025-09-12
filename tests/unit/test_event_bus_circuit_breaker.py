"""Unit tests for Event Bus Circuit Breaker.

Comprehensive unit tests covering:
- Circuit breaker state transitions
- Error handling and sanitization
- Queue management and dead letter queue
- Configuration validation
- Metrics accuracy
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from omnibase_core.model.core.model_onex_event import ModelOnexEvent
from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.core_error_codes import CoreErrorCode

from omnibase_infra.infrastructure.event_bus_circuit_breaker import (
    EventBusCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    EventBusMetrics
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration validation."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60
        assert config.success_threshold == 3
        assert config.timeout_seconds == 30
        assert config.max_queue_size == 1000
        assert config.dead_letter_enabled is True
        assert config.graceful_degradation is True
    
    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=120,
            success_threshold=5,
            timeout_seconds=60,
            max_queue_size=2000,
            dead_letter_enabled=False,
            graceful_degradation=False
        )
        
        assert config.failure_threshold == 10
        assert config.recovery_timeout == 120
        assert config.success_threshold == 5
        assert config.timeout_seconds == 60
        assert config.max_queue_size == 2000
        assert config.dead_letter_enabled is False
        assert config.graceful_degradation is False


class TestEventBusMetrics:
    """Test event bus metrics tracking."""
    
    def test_metrics_initialization(self):
        """Test metrics initialize with correct default values."""
        metrics = EventBusMetrics()
        
        assert metrics.total_events == 0
        assert metrics.successful_events == 0
        assert metrics.failed_events == 0
        assert metrics.queued_events == 0
        assert metrics.dropped_events == 0
        assert metrics.dead_letter_events == 0
        assert metrics.circuit_opens == 0
        assert metrics.circuit_closes == 0
        assert metrics.last_failure is None
        assert metrics.last_success is None


class TestEventBusCircuitBreaker:
    """Test event bus circuit breaker functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration with fast timeouts."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1,  # 1 second for fast testing
            success_threshold=2,
            timeout_seconds=5,
            max_queue_size=10,
            dead_letter_enabled=True,
            graceful_degradation=True
        )
    
    @pytest.fixture
    def circuit_breaker(self, config):
        """Create circuit breaker for testing."""
        return EventBusCircuitBreaker(config)
    
    @pytest.fixture
    def sample_event(self):
        """Create sample event for testing."""
        return ModelOnexEvent(
            event_type="test.event",
            correlation_id=uuid4(),
            payload={"test": "data"},
            timestamp=datetime.now().isoformat(),
            source="test",
            version="1.0"
        )
    
    def test_initial_state(self, circuit_breaker):
        """Test circuit breaker initializes in closed state."""
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED
        assert circuit_breaker.is_healthy() is True
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
    
    @pytest.mark.asyncio
    async def test_successful_publish_closed_state(self, circuit_breaker, sample_event):
        """Test successful event publishing in closed state."""
        # Mock successful publisher
        publisher_mock = AsyncMock()
        
        # Publish event
        result = await circuit_breaker.publish_event(sample_event, publisher_mock)
        
        # Verify success
        assert result is True
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        
        # Verify metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics.total_events == 1
        assert metrics.successful_events == 1
        assert metrics.failed_events == 0
        assert metrics.last_success is not None
        
        # Verify publisher was called
        publisher_mock.assert_called_once_with(sample_event)
    
    @pytest.mark.asyncio
    async def test_failure_handling_closed_state(self, circuit_breaker, sample_event):
        """Test failure handling in closed state."""
        # Mock failing publisher
        publisher_mock = AsyncMock(side_effect=Exception("Publisher error"))
        
        # Publish event (should handle failure gracefully)
        result = await circuit_breaker.publish_event(sample_event, publisher_mock)
        
        # Should queue event on failure (graceful degradation)
        assert result is False
        assert circuit_breaker.failure_count == 1
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED  # Still closed after 1 failure
        
        # Verify metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics.total_events == 1
        assert metrics.failed_events == 1
        assert metrics.queued_events == 1
        assert metrics.last_failure is not None
    
    @pytest.mark.asyncio
    async def test_circuit_opens_on_failure_threshold(self, circuit_breaker, sample_event):
        """Test circuit opens when failure threshold is reached."""
        # Mock failing publisher
        publisher_mock = AsyncMock(side_effect=Exception("Publisher error"))
        
        # Trigger failures to reach threshold (3 in test config)
        for i in range(3):
            await circuit_breaker.publish_event(sample_event, publisher_mock)
        
        # Circuit should now be open
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN
        assert circuit_breaker.is_healthy() is False
        
        # Verify metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics.failed_events == 3
        assert metrics.circuit_opens == 1
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, circuit_breaker, sample_event):
        """Test timeout handling during event publishing."""
        # Mock publisher that times out
        async def timeout_publisher(event):
            await asyncio.sleep(10)  # Longer than timeout_seconds (5)
        
        # Publish event
        result = await circuit_breaker.publish_event(sample_event, timeout_publisher)
        
        # Should handle timeout as failure
        assert result is False
        assert circuit_breaker.failure_count == 1
        
        # Verify metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics.failed_events == 1
        assert metrics.queued_events == 1
    
    @pytest.mark.asyncio
    async def test_open_circuit_behavior(self, circuit_breaker, sample_event):
        """Test behavior when circuit is open."""
        # Force circuit open
        await self._force_circuit_open(circuit_breaker, sample_event)
        
        # Mock publisher (should not be called)
        publisher_mock = AsyncMock()
        
        # Try to publish event
        result = await circuit_breaker.publish_event(sample_event, publisher_mock)
        
        # Should queue event without calling publisher
        assert result is False
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN
        
        # Publisher should not be called
        publisher_mock.assert_not_called()
        
        # Event should be queued
        metrics = circuit_breaker.get_metrics()
        assert metrics.queued_events > 0
    
    @pytest.mark.asyncio
    async def test_half_open_transition(self, circuit_breaker, sample_event):
        """Test transition from open to half-open state."""
        # Force circuit open
        await self._force_circuit_open(circuit_breaker, sample_event)
        
        # Wait for recovery timeout (1 second in test config)
        await asyncio.sleep(1.5)
        
        # Mock successful publisher
        publisher_mock = AsyncMock()
        
        # Next publish should transition to half-open
        result = await circuit_breaker.publish_event(sample_event, publisher_mock)
        
        # Should be in half-open state now
        assert circuit_breaker.get_state() == CircuitBreakerState.HALF_OPEN
        assert result is True  # Should succeed in half-open
    
    @pytest.mark.asyncio
    async def test_circuit_closure_from_half_open(self, circuit_breaker, sample_event):
        """Test circuit closure from half-open state after successful publishes."""
        # Force to half-open state
        await self._force_circuit_half_open(circuit_breaker, sample_event)
        
        # Mock successful publisher
        publisher_mock = AsyncMock()
        
        # Publish successful events to meet success threshold (2 in test config)
        for i in range(2):
            result = await circuit_breaker.publish_event(sample_event, publisher_mock)
            assert result is True
            await asyncio.sleep(0.1)
        
        # Circuit should now be closed
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED
        assert circuit_breaker.is_healthy() is True
        
        # Verify metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics.circuit_closes >= 1
    
    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, circuit_breaker, sample_event):
        """Test circuit reopens immediately on failure in half-open state."""
        # Force to half-open state
        await self._force_circuit_half_open(circuit_breaker, sample_event)
        
        # Mock failing publisher
        publisher_mock = AsyncMock(side_effect=Exception("Half-open failure"))
        
        # Publish failing event
        result = await circuit_breaker.publish_event(sample_event, publisher_mock)
        
        # Circuit should immediately open again
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN
        assert result is False
    
    @pytest.mark.asyncio
    async def test_queue_capacity_management(self, circuit_breaker, sample_event):
        """Test queue capacity management and overflow handling."""
        # Force circuit open
        await self._force_circuit_open(circuit_breaker, sample_event)
        
        # Mock publisher (won't be called in open state)
        publisher_mock = AsyncMock()
        
        # Fill queue beyond capacity (10 in test config)
        for i in range(15):
            await circuit_breaker.publish_event(sample_event, publisher_mock)
        
        # Check queue and dead letter metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics.queued_events <= 10  # Queue capacity limit
        assert metrics.dead_letter_events > 0  # Overflow to dead letter
        assert metrics.dropped_events >= 0  # Some may be dropped
    
    @pytest.mark.asyncio
    async def test_dead_letter_queue_structure(self, circuit_breaker, sample_event):
        """Test dead letter queue entry structure."""
        # Force circuit open and fill queue
        await self._force_circuit_open(circuit_breaker, sample_event)
        
        # Mock publisher
        publisher_mock = AsyncMock()
        
        # Exceed queue capacity to trigger dead letter
        for i in range(15):
            await circuit_breaker.publish_event(sample_event, publisher_mock)
        
        # Verify dead letter queue structure
        dlq = circuit_breaker.dead_letter_queue
        assert len(dlq) > 0
        
        dlq_entry = dlq[0]
        assert "event" in dlq_entry
        assert "timestamp" in dlq_entry
        assert "reason" in dlq_entry
        assert "circuit_state" in dlq_entry
        assert "retry_count" in dlq_entry
        
        # Verify event data is preserved
        assert dlq_entry["event"]["event_type"] == "test.event"
        assert dlq_entry["circuit_state"] == "open"
    
    @pytest.mark.asyncio
    async def test_fail_fast_mode(self, circuit_breaker, sample_event):
        """Test fail-fast mode when graceful degradation is disabled."""
        # Disable graceful degradation
        circuit_breaker.config.graceful_degradation = False
        
        # Force circuit open
        await self._force_circuit_open(circuit_breaker, sample_event)
        
        # Mock publisher
        publisher_mock = AsyncMock()
        
        # Should raise OnexError in fail-fast mode
        with pytest.raises(OnexError) as exc_info:
            await circuit_breaker.publish_event(sample_event, publisher_mock)
        
        # Verify error details
        assert exc_info.value.code == CoreErrorCode.INTEGRATION_SERVICE_UNAVAILABLE
        assert "circuit breaker open" in str(exc_info.value.message).lower()
    
    @pytest.mark.asyncio
    async def test_manual_circuit_reset(self, circuit_breaker, sample_event):
        """Test manual circuit breaker reset functionality."""
        # Force circuit open
        await self._force_circuit_open(circuit_breaker, sample_event)
        assert circuit_breaker.get_state() == CircuitBreakerState.OPEN
        
        # Manual reset
        await circuit_breaker.reset_circuit()
        
        # Should be closed now
        assert circuit_breaker.get_state() == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
        assert circuit_breaker.is_healthy() is True
    
    def test_health_status_comprehensive(self, circuit_breaker):
        """Test comprehensive health status reporting."""
        health_status = circuit_breaker.get_health_status()
        
        # Verify structure
        required_keys = [
            "circuit_state", "is_healthy", "failure_count",
            "queued_events", "dead_letter_events", "metrics"
        ]
        for key in required_keys:
            assert key in health_status
        
        # Verify metrics structure
        metrics = health_status["metrics"]
        assert "total_events" in metrics
        assert "successful_events" in metrics
        assert "failed_events" in metrics
        assert "success_rate" in metrics
        assert "circuit_opens" in metrics
        assert "circuit_closes" in metrics
        assert "last_failure" in metrics
        assert "last_success" in metrics
    
    @pytest.mark.asyncio
    async def test_concurrent_access_thread_safety(self, circuit_breaker, sample_event):
        """Test circuit breaker thread safety under concurrent access."""
        # Mock successful publisher
        publisher_mock = AsyncMock()
        
        # Create concurrent publish operations
        tasks = []
        for i in range(50):
            task = asyncio.create_task(
                circuit_breaker.publish_event(sample_event, publisher_mock)
            )
            tasks.append(task)
        
        # Wait for all operations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no exceptions from concurrent access
        for result in results:
            assert not isinstance(result, Exception)
        
        # Circuit should be in consistent state
        state = circuit_breaker.get_state()
        assert state in [CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN]
        
        # Metrics should be consistent
        metrics = circuit_breaker.get_metrics()
        assert metrics.total_events == 50
    
    @pytest.mark.asyncio
    async def test_metrics_accuracy(self, circuit_breaker, sample_event):
        """Test metrics accuracy across different operations."""
        initial_metrics = circuit_breaker.get_metrics()
        
        # Successful publishes
        successful_mock = AsyncMock()
        for i in range(5):
            await circuit_breaker.publish_event(sample_event, successful_mock)
        
        # Failed publishes
        failing_mock = AsyncMock(side_effect=Exception("Test error"))
        for i in range(3):
            await circuit_breaker.publish_event(sample_event, failing_mock)
        
        # Verify final metrics
        final_metrics = circuit_breaker.get_metrics()
        
        assert final_metrics.total_events == initial_metrics.total_events + 8
        assert final_metrics.successful_events == initial_metrics.successful_events + 5
        assert final_metrics.failed_events == initial_metrics.failed_events + 3
        
        # Success rate calculation
        expected_success_rate = (final_metrics.successful_events / final_metrics.total_events) * 100
        health_status = circuit_breaker.get_health_status()
        actual_success_rate = health_status["metrics"]["success_rate"]
        assert abs(actual_success_rate - expected_success_rate) < 0.01  # Floating point tolerance
    
    async def _force_circuit_open(self, circuit_breaker, sample_event):
        """Helper to force circuit breaker open."""
        failing_mock = AsyncMock(side_effect=Exception("Forced failure"))
        
        # Trigger enough failures to open circuit
        for i in range(circuit_breaker.config.failure_threshold):
            await circuit_breaker.publish_event(sample_event, failing_mock)
    
    async def _force_circuit_half_open(self, circuit_breaker, sample_event):
        """Helper to force circuit breaker to half-open state."""
        # First open the circuit
        await self._force_circuit_open(circuit_breaker, sample_event)
        
        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)
        
        # Force transition to half-open by attempting publish
        successful_mock = AsyncMock()
        await circuit_breaker.publish_event(sample_event, successful_mock)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])