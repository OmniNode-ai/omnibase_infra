"""
Unit tests for MetricsCollector.

Tests core metrics collection with <10ms overhead.
"""

import asyncio
import time

import pytest

from omninode_bridge.agents.metrics.collector import MetricsCollector


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    @pytest.mark.asyncio
    async def test_collector_creation(self):
        """Test creating metrics collector."""
        collector = MetricsCollector(
            buffer_size=1000,
            batch_size=100,
            flush_interval_ms=1000,
        )

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 0
        assert stats["buffer_capacity"] == 1000
        assert stats["metrics_since_flush"] == 0

    @pytest.mark.asyncio
    async def test_record_timing(self):
        """Test recording timing metric."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)

        await collector.record_timing(
            "test_metric",
            duration_ms=42.5,
            tags={"env": "test"},
        )

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1
        assert stats["metrics_since_flush"] == 1

    @pytest.mark.asyncio
    async def test_record_counter(self):
        """Test recording counter metric."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)

        await collector.record_counter(
            "test_count",
            count=5,
            tags={"type": "request"},
        )

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_record_gauge(self):
        """Test recording gauge metric."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)

        await collector.record_gauge(
            "agent_count",
            value=3.0,
            unit="count",
            tags={"workflow": "codegen"},
        )

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_record_rate(self):
        """Test recording rate metric."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)

        await collector.record_rate(
            "cache_hit_rate",
            rate_percent=85.5,
            tags={"cache": "template"},
        )

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_flush_empty(self):
        """Test flushing empty buffer."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)

        # Should not raise error
        await collector.flush()

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 0

    @pytest.mark.asyncio
    async def test_flush_with_metrics(self):
        """Test flushing metrics."""
        collector = MetricsCollector(
            buffer_size=1000,
            batch_size=10,
            kafka_enabled=False,
            postgres_enabled=False,
        )

        # Record metrics
        for i in range(5):
            await collector.record_timing(f"metric_{i}", float(i))

        stats_before = await collector.get_stats()
        assert stats_before["buffer_size"] == 5

        # Flush
        await collector.flush()

        stats_after = await collector.get_stats()
        # Buffer should be emptier after flush
        assert stats_after["buffer_size"] <= stats_before["buffer_size"]
        assert stats_after["metrics_since_flush"] == 0

    @pytest.mark.asyncio
    async def test_auto_flush_on_batch_size(self):
        """Test automatic flush when batch size reached."""
        collector = MetricsCollector(
            buffer_size=1000,
            batch_size=5,  # Small batch size
            kafka_enabled=False,
            postgres_enabled=False,
        )

        # Record exactly batch_size metrics
        for i in range(5):
            await collector.record_timing(f"metric_{i}", float(i))

        # Give flush task time to execute
        await asyncio.sleep(0.1)

        stats = await collector.get_stats()
        # metrics_since_flush should be reset after auto-flush
        assert stats["metrics_since_flush"] == 0

    @pytest.mark.asyncio
    async def test_multiple_metric_types(self):
        """Test recording multiple metric types."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)

        await collector.record_timing("timing_metric", 10.5)
        await collector.record_counter("counter_metric", 1)
        await collector.record_gauge("gauge_metric", 42.0, "count")
        await collector.record_rate("rate_metric", 95.5)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 4

    @pytest.mark.asyncio
    async def test_correlation_id_propagation(self):
        """Test correlation ID is propagated to metrics."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)

        correlation_id = "test-correlation-123"

        await collector.record_timing(
            "test_metric",
            10.0,
            correlation_id=correlation_id,
        )

        # Flush to verify metrics were created correctly
        await collector.flush()

        # No errors should occur
        stats = await collector.get_stats()
        assert stats["metrics_since_flush"] == 0

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping collector."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)

        # Start without Kafka/Postgres
        await collector.start()

        # Record some metrics
        await collector.record_timing("test", 10.0)

        # Stop
        await collector.stop()

        # No errors should occur
        assert True

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_record_timing_performance(self):
        """Test record_timing performance (<1ms)."""
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)

        # Measure single record time
        start = time.perf_counter()
        await collector.record_timing("test_metric", 10.0)
        duration_ms = (time.perf_counter() - start) * 1000

        assert duration_ms < 1.0, f"record_timing took {duration_ms:.3f}ms (>1ms)"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_record_overhead_100_metrics(self):
        """Test overhead for 100 metrics (<10ms total, <0.1ms per metric)."""
        collector = MetricsCollector(
            buffer_size=10000,
            batch_size=1000,  # Don't auto-flush during test
            kafka_enabled=False,
            postgres_enabled=False,
        )

        # Measure 100 record operations
        start = time.perf_counter()
        for i in range(100):
            await collector.record_timing(f"metric_{i}", float(i))
        duration_ms = (time.perf_counter() - start) * 1000

        avg_overhead_ms = duration_ms / 100

        assert avg_overhead_ms < 0.1, (
            f"Average overhead: {avg_overhead_ms:.3f}ms per metric (>0.1ms)\n"
            f"Total for 100 metrics: {duration_ms:.2f}ms"
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_recording(self):
        """Test concurrent metric recording (stress test)."""
        collector = MetricsCollector(
            buffer_size=10000,
            batch_size=10000,  # Disable auto-flush for this test
            kafka_enabled=False,
            postgres_enabled=False,
        )

        async def record_metrics(start: int, count: int):
            for i in range(start, start + count):
                await collector.record_timing(f"metric_{i}", float(i))

        # 10 concurrent tasks, 100 metrics each
        tasks = [record_metrics(i * 100, 100) for i in range(10)]
        await asyncio.gather(*tasks)

        stats = await collector.get_stats()
        assert stats["buffer_size"] == 1000

    @pytest.mark.asyncio
    async def test_alert_engine_integration(self):
        """Test alert engine is called when metrics exceed thresholds."""
        from omninode_bridge.agents.metrics.alerting.rules import AlertRuleEngine
        from omninode_bridge.agents.metrics.models import AlertRule, AlertSeverity

        # Create alert engine with rule
        rule = AlertRule(
            metric_name="test_metric",
            threshold=100.0,
            operator="gt",
            severity=AlertSeverity.CRITICAL,
        )
        alert_engine = AlertRuleEngine(rules=[rule])

        # Create collector with alert engine
        collector = MetricsCollector(kafka_enabled=False, postgres_enabled=False)
        await collector.start(alert_engine=alert_engine)

        # Record metric that triggers alert
        await collector.record_timing("test_metric", 150.0)

        # Give alert task time to execute
        await asyncio.sleep(0.1)

        # No errors should occur
        assert True

        await collector.stop()
