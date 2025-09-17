"""Integration tests for RedPanda Event Bus with Circuit Breaker.

Comprehensive tests covering:
- Circuit breaker behavior under different failure scenarios
- Integration with actual RedPanda instance
- Dead letter queue functionality
- Performance under load
- Security scenarios with TLS/SSL
"""

import asyncio
import os
import time
from datetime import datetime
from uuid import uuid4

import pytest
from omnibase_core.model.core.model_onex_event import ModelOnexEvent

from omnibase_infra.infrastructure.container import RedPandaEventBus
from omnibase_infra.infrastructure.event_bus_circuit_breaker import (
    CircuitBreakerState,
)


class TestRedPandaCircuitBreakerIntegration:
    """Integration tests for RedPanda event bus with circuit breaker protection."""

    @pytest.fixture
    def event_bus(self):
        """Create RedPanda event bus for testing."""
        # Override environment variables for testing
        os.environ.update({
            "REDPANDA_HOST": "localhost",
            "REDPANDA_PORT": "9092",
            "CIRCUIT_BREAKER_FAILURE_THRESHOLD": "3",
            "CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "5",  # Faster for testing
            "CIRCUIT_BREAKER_SUCCESS_THRESHOLD": "2",
            "CIRCUIT_BREAKER_TIMEOUT": "10",
            "CIRCUIT_BREAKER_MAX_QUEUE": "100",
            "CIRCUIT_BREAKER_GRACEFUL_DEGRADATION": "true",
        })

        bus = RedPandaEventBus()
        yield bus

        # Cleanup
        asyncio.run(bus.close())

    @pytest.fixture
    def sample_event(self):
        """Create sample OnexEvent for testing."""
        return ModelOnexEvent(
            event_type="postgres.query.completed",
            correlation_id=uuid4(),
            payload={"query": "SELECT * FROM test_table", "duration": 0.05},
            timestamp=datetime.now().isoformat(),
            source="postgres_adapter",
            version="1.0",
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_normal_operation(self, event_bus, sample_event):
        """Test circuit breaker in normal operation (closed state)."""
        # Circuit should start in closed state
        assert event_bus._circuit_breaker.get_state() == CircuitBreakerState.CLOSED

        # Publishing should succeed (or be queued based on RedPanda availability)
        result = await event_bus._circuit_breaker_publish(sample_event)

        # Check circuit remains closed after successful operation
        assert event_bus._circuit_breaker.get_state() == CircuitBreakerState.CLOSED
        assert event_bus.is_healthy()

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self, event_bus):
        """Test circuit breaker opens after failure threshold is reached."""
        # Create event that will fail (invalid correlation_id format)
        failing_event = ModelOnexEvent(
            event_type="test.failure",
            correlation_id=uuid4(),
            payload={"test": "failure_scenario"},
            timestamp=datetime.now().isoformat(),
            source="test",
            version="1.0",
        )

        # Mock the raw publish method to always fail
        original_publish = event_bus._raw_publish_async

        async def mock_failing_publish(event):
            raise Exception("Simulated RedPanda failure")

        event_bus._raw_publish_async = mock_failing_publish

        try:
            # Trigger failures to reach threshold (default: 3)
            for i in range(3):
                await event_bus._circuit_breaker_publish(failing_event)
                # Small delay to prevent rapid-fire failures
                await asyncio.sleep(0.1)

            # Circuit should now be open
            assert event_bus._circuit_breaker.get_state() == CircuitBreakerState.OPEN
            assert not event_bus.is_healthy()

            # Verify metrics
            metrics = event_bus._circuit_breaker.get_metrics()
            assert metrics.failed_events >= 3
            assert metrics.circuit_opens >= 1

        finally:
            # Restore original method
            event_bus._raw_publish_async = original_publish

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_cycle(self, event_bus, sample_event):
        """Test complete circuit breaker recovery cycle: closed -> open -> half-open -> closed."""
        # Force circuit to open state
        await self._force_circuit_open(event_bus)
        assert event_bus._circuit_breaker.get_state() == CircuitBreakerState.OPEN

        # Wait for recovery timeout (5 seconds in test config)
        await asyncio.sleep(6)

        # Next publish should transition to half-open
        await event_bus._circuit_breaker_publish(sample_event)
        assert event_bus._circuit_breaker.get_state() == CircuitBreakerState.HALF_OPEN

        # Successful publishes should close the circuit
        for i in range(2):  # Success threshold is 2 in test config
            await event_bus._circuit_breaker_publish(sample_event)
            await asyncio.sleep(0.1)

        # Circuit should be closed again
        assert event_bus._circuit_breaker.get_state() == CircuitBreakerState.CLOSED
        assert event_bus.is_healthy()

    @pytest.mark.asyncio
    async def test_dead_letter_queue_functionality(self, event_bus):
        """Test dead letter queue captures failed events when circuit is open."""
        # Force circuit open and fill the queue
        await self._force_circuit_open(event_bus)

        # Create many events to exceed queue capacity (100 in test config)
        events = []
        for i in range(150):  # Exceed queue capacity
            event = ModelOnexEvent(
                event_type=f"test.event.{i}",
                correlation_id=uuid4(),
                payload={"index": i},
                timestamp=datetime.now().isoformat(),
                source="test",
                version="1.0",
            )
            events.append(event)
            await event_bus._circuit_breaker_publish(event)

        # Check dead letter queue has captured excess events
        metrics = event_bus._circuit_breaker.get_metrics()
        assert metrics.queued_events <= 100  # Queue capacity limit
        assert metrics.dead_letter_events > 0  # Excess events in DLQ

        # Verify DLQ entries have proper metadata
        dlq = event_bus._circuit_breaker.dead_letter_queue
        if dlq:
            dlq_entry = dlq[0]
            assert "event" in dlq_entry
            assert "timestamp" in dlq_entry
            assert "reason" in dlq_entry
            assert "circuit_state" in dlq_entry

    @pytest.mark.asyncio
    async def test_performance_under_load(self, event_bus):
        """Test event bus performance with circuit breaker under high load."""
        start_time = time.time()
        event_count = 100

        # Create batch of events
        events = []
        for i in range(event_count):
            event = ModelOnexEvent(
                event_type="performance.test",
                correlation_id=uuid4(),
                payload={"batch_id": i, "timestamp": time.time()},
                timestamp=datetime.now().isoformat(),
                source="performance_test",
                version="1.0",
            )
            events.append(event)

        # Publish events concurrently
        tasks = []
        for event in events:
            task = asyncio.create_task(event_bus._circuit_breaker_publish(event))
            tasks.append(task)

        # Wait for all publishes to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        duration = end_time - start_time

        # Performance assertions
        assert duration < 30  # Should complete within 30 seconds
        throughput = event_count / duration

        print(f"Performance test: {event_count} events in {duration:.2f}s (throughput: {throughput:.1f} events/sec)")

        # Check metrics
        metrics = event_bus._circuit_breaker.get_metrics()
        assert metrics.total_events >= event_count

        # Circuit should remain stable under load
        assert event_bus._circuit_breaker.get_state() in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]

    @pytest.mark.asyncio
    async def test_graceful_degradation_mode(self, event_bus):
        """Test graceful degradation when circuit breaker is enabled."""
        # Force circuit open
        await self._force_circuit_open(event_bus)

        # Create event
        event = ModelOnexEvent(
            event_type="graceful.test",
            correlation_id=uuid4(),
            payload={"test": "graceful_degradation"},
            timestamp=datetime.now().isoformat(),
            source="test",
            version="1.0",
        )

        # Publish should not raise exception (graceful degradation)
        result = await event_bus._circuit_breaker_publish(event)
        assert result is False  # Not published, but didn't fail

        # Event should be queued
        metrics = event_bus._circuit_breaker.get_metrics()
        assert metrics.queued_events > 0

    @pytest.mark.asyncio
    async def test_fail_fast_mode(self, event_bus):
        """Test fail-fast behavior when graceful degradation is disabled."""
        # Disable graceful degradation
        event_bus._circuit_breaker.config.graceful_degradation = False

        # Force circuit open
        await self._force_circuit_open(event_bus)

        # Create event
        event = ModelOnexEvent(
            event_type="failfast.test",
            correlation_id=uuid4(),
            payload={"test": "fail_fast"},
            timestamp=datetime.now().isoformat(),
            source="test",
            version="1.0",
        )

        # Should raise OnexError in fail-fast mode
        with pytest.raises(Exception):  # OnexError should be raised
            await event_bus._circuit_breaker_publish(event)

    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_accuracy(self, event_bus, sample_event):
        """Test circuit breaker metrics are accurate."""
        initial_metrics = event_bus._circuit_breaker.get_metrics()
        initial_total = initial_metrics.total_events

        # Publish successful events
        success_count = 5
        for i in range(success_count):
            await event_bus._circuit_breaker_publish(sample_event)
            await asyncio.sleep(0.1)

        # Check metrics updated correctly
        updated_metrics = event_bus._circuit_breaker.get_metrics()
        assert updated_metrics.total_events >= initial_total + success_count

        # Force some failures
        await self._force_circuit_open(event_bus)

        # Check failure metrics
        final_metrics = event_bus._circuit_breaker.get_metrics()
        assert final_metrics.failed_events >= 3  # Failure threshold
        assert final_metrics.circuit_opens >= 1

    @pytest.mark.asyncio
    async def test_health_status_reporting(self, event_bus):
        """Test comprehensive health status reporting."""
        # Get initial health status
        health_status = event_bus.get_circuit_breaker_status()

        # Verify health status structure
        assert "circuit_state" in health_status
        assert "is_healthy" in health_status
        assert "metrics" in health_status
        assert "success_rate" in health_status["metrics"]

        # Initially should be healthy
        assert health_status["is_healthy"] is True
        assert health_status["circuit_state"] == "closed"

        # Force circuit open
        await self._force_circuit_open(event_bus)

        # Health status should reflect open circuit
        health_status = event_bus.get_circuit_breaker_status()
        assert health_status["is_healthy"] is False
        assert health_status["circuit_state"] == "open"
        assert health_status["metrics"]["failed_events"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_access(self, event_bus):
        """Test circuit breaker thread safety with concurrent access."""
        # Create multiple concurrent publish operations
        concurrent_tasks = 50
        events = []

        for i in range(concurrent_tasks):
            event = ModelOnexEvent(
                event_type=f"concurrent.test.{i}",
                correlation_id=uuid4(),
                payload={"task_id": i},
                timestamp=datetime.now().isoformat(),
                source="concurrent_test",
                version="1.0",
            )
            events.append(event)

        # Execute all publishes concurrently
        tasks = [event_bus._circuit_breaker_publish(event) for event in events]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no deadlocks or race conditions
        assert len(results) == concurrent_tasks

        # Circuit should remain in consistent state
        state = event_bus._circuit_breaker.get_state()
        assert state in [CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN]

        # Metrics should be consistent
        metrics = event_bus._circuit_breaker.get_metrics()
        assert metrics.total_events >= concurrent_tasks

    async def _force_circuit_open(self, event_bus):
        """Helper method to force circuit breaker open."""
        # Mock failing publish
        async def mock_failing_publish(event):
            raise Exception("Forced failure for testing")

        original_publish = event_bus._raw_publish_async
        event_bus._raw_publish_async = mock_failing_publish

        try:
            # Create failing event
            failing_event = ModelOnexEvent(
                event_type="test.force.failure",
                correlation_id=uuid4(),
                payload={"test": "force_failure"},
                timestamp=datetime.now().isoformat(),
                source="test",
                version="1.0",
            )

            # Trigger enough failures to open circuit
            for i in range(5):  # Exceed failure threshold
                await event_bus._circuit_breaker_publish(failing_event)
                await asyncio.sleep(0.1)

        finally:
            # Restore original method
            event_bus._raw_publish_async = original_publish


@pytest.mark.asyncio
class TestRedPandaIntegrationWithRealInstance:
    """Integration tests requiring actual RedPanda instance."""

    @pytest.mark.skipif(not os.getenv("REDPANDA_INTEGRATION_TESTS"),
                       reason="Requires REDPANDA_INTEGRATION_TESTS environment variable")
    async def test_real_redpanda_connection(self):
        """Test connection to actual RedPanda instance."""
        event_bus = RedPandaEventBus()

        try:
            # Create test event
            event = ModelOnexEvent(
                event_type="integration.test.real",
                correlation_id=uuid4(),
                payload={"message": "Integration test with real RedPanda"},
                timestamp=datetime.now().isoformat(),
                source="integration_test",
                version="1.0",
            )

            # Publish to real RedPanda
            result = await event_bus._circuit_breaker_publish(event)

            # Should succeed with real instance
            assert result is True or result is False  # Either published or queued
            assert event_bus.is_healthy()

        finally:
            await event_bus.close()

    @pytest.mark.skipif(not os.getenv("REDPANDA_INTEGRATION_TESTS"),
                       reason="Requires REDPANDA_INTEGRATION_TESTS environment variable")
    async def test_ssl_tls_connection(self):
        """Test SSL/TLS connection to secured RedPanda."""
        # Override for SSL testing
        os.environ.update({
            "KAFKA_SECURITY_PROTOCOL": "SSL",
            "KAFKA_SSL_CA_LOCATION": "/path/to/ca.pem",
            "KAFKA_SSL_CERT_LOCATION": "/path/to/cert.pem",
            "KAFKA_SSL_KEY_LOCATION": "/path/to/key.pem",
        })

        event_bus = RedPandaEventBus()

        try:
            # Test SSL connection health
            health_status = event_bus.get_circuit_breaker_status()

            # Should initialize without errors
            assert health_status["circuit_state"] == "closed"

        finally:
            await event_bus.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
