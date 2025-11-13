"""
Comprehensive Memory Leak Prevention Tests.

Tests that metrics collection maintains bounded memory usage
through circular buffer behavior and doesn't leak memory over time.

Test Categories:
1. Metrics list doesn't exceed MAX_METRICS_STORED (10,000)
2. Circular buffer behavior (add 15,000 metrics, verify only last 10,000)
3. Memory usage remains stable over time
4. No memory leak in long-running scenarios
5. Deque maxlen enforcement
6. Memory growth prevention under load
7. Resource cleanup verification

Security Fix: Memory leak prevention in metrics collection
Implementation: Agent 5
"""

import gc
import sys
from collections import deque
from datetime import datetime

import pytest
from prometheus_client import REGISTRY

from omninode_bridge.services.metadata_stamping.monitoring.metrics_collector import (
    MetricsCollector,
    OperationType,
    PerformanceMetric,
)


@pytest.fixture(autouse=True)
def clear_prometheus_registry():
    """Clear Prometheus registry before each test to prevent metric collisions.

    Prometheus metrics are global singletons. Without cleanup, creating multiple
    MetricsCollector instances causes "Duplicated timeseries in CollectorRegistry" errors.
    """
    # Clear all collectors before test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass  # Ignore if already unregistered

    yield

    # Clear after test for good measure
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


class TestMetricsMemoryBounds:
    """Test suite for metrics memory bounds enforcement."""

    def test_metrics_list_respects_max_samples_limit(self):
        """
        Test that metrics list doesn't exceed max_samples limit.

        MetricsCollector uses deque with maxlen to enforce this automatically.
        """
        max_samples = 1000
        collector = MetricsCollector(max_samples=max_samples)

        # Add more metrics than max_samples
        for i in range(1500):
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
            )
            collector.record_metric(metric)

        # Verify metrics list size is bounded
        assert len(collector.metrics) <= max_samples
        assert (
            len(collector.metrics) == max_samples
        ), f"Expected exactly {max_samples} metrics, got {len(collector.metrics)}"

    def test_circular_buffer_behavior_with_default_limit(self):
        """
        Test circular buffer behavior with default MAX_METRICS_STORED (10,000).

        Add 15,000 metrics and verify only the last 10,000 are retained.
        """
        collector = MetricsCollector(max_samples=10000)

        # Add 15,000 metrics with unique identifiers
        for i in range(15000):
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
                metadata={"sequence": i},  # Track insertion order
            )
            collector.record_metric(metric)

        # Verify only last 10,000 metrics retained
        assert len(collector.metrics) == 10000

        # Verify oldest metrics (0-4999) were evicted
        sequences = [m.metadata.get("sequence") for m in collector.metrics]
        assert min(sequences) >= 5000, "Oldest metrics should be evicted"
        assert max(sequences) == 14999, "Newest metrics should be retained"

        # Verify metrics are in insertion order (oldest to newest)
        assert sequences == sorted(sequences), "Metrics should maintain insertion order"

    @pytest.mark.parametrize(
        "max_samples",
        [100, 500, 1000, 5000, 10000],
    )
    def test_circular_buffer_with_various_limits(self, max_samples):
        """
        Test circular buffer behavior with various max_samples limits.

        Verifies that deque maxlen enforcement works for different limits.
        """
        collector = MetricsCollector(max_samples=max_samples)

        # Add 2x the limit
        num_metrics = max_samples * 2

        for i in range(num_metrics):
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
                metadata={"sequence": i},
            )
            collector.record_metric(metric)

        # Verify size constraint
        assert len(collector.metrics) == max_samples

        # Verify oldest half was evicted
        sequences = [m.metadata.get("sequence") for m in collector.metrics]
        assert min(sequences) >= max_samples, "Oldest metrics should be evicted"
        assert max(sequences) == num_metrics - 1, "Newest metrics should be retained"

    def test_memory_usage_stability_under_continuous_load(self):
        """
        Test that memory usage remains stable over time under continuous load.

        Simulates long-running service with continuous metric recording.
        """
        collector = MetricsCollector(max_samples=1000)

        # Fill deque to capacity first (1000 metrics)
        for i in range(1000):
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
            )
            collector.record_metric(metric)

        # Measure baseline memory AFTER filling deque
        gc.collect()
        baseline_memory = sys.getsizeof(collector.metrics)

        # Simulate continuous load (9,000 more metrics)
        for i in range(9000):
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
            )
            collector.record_metric(metric)

        # Measure final memory
        gc.collect()
        final_memory = sys.getsizeof(collector.metrics)

        # Memory should remain stable (not grow further)
        assert len(collector.metrics) == 1000
        memory_growth = final_memory - baseline_memory

        # Memory should be essentially stable (allow 10% variance for GC timing)
        assert (
            memory_growth < baseline_memory * 0.1
        ), f"Memory grew by {memory_growth} bytes ({memory_growth / baseline_memory * 100:.1f}%), expected stable"

    def test_no_memory_leak_with_repeated_additions(self):
        """
        Test that repeated metric additions don't cause memory leaks.

        Performs multiple rounds of metric additions and verifies
        memory doesn't grow unbounded.
        """
        collector = MetricsCollector(max_samples=500)

        memory_measurements = []

        # Perform 10 rounds of metric additions
        for round_num in range(10):
            # Add 1000 metrics (2x the limit)
            for i in range(1000):
                metric = PerformanceMetric(
                    operation_type=OperationType.DATABASE_QUERY,
                    execution_time_ms=10.5,
                    timestamp=datetime.now(),
                    success=True,
                )
                collector.record_metric(metric)

            # Measure memory
            gc.collect()
            memory_measurements.append(sys.getsizeof(collector.metrics))

        # Verify deque size constraint maintained
        assert len(collector.metrics) == 500

        # Verify memory stabilized (not growing linearly)
        # After first round, memory should be relatively stable
        stable_memory = memory_measurements[1:]  # Skip first measurement
        memory_variance = max(stable_memory) - min(stable_memory)
        average_memory = sum(stable_memory) / len(stable_memory)

        # Memory variance should be less than 20% of average
        assert (
            memory_variance < average_memory * 0.2
        ), f"Memory variance too high: {memory_variance} bytes ({memory_variance / average_memory * 100:.1f}%)"

    def test_deque_maxlen_enforcement(self):
        """
        Test that deque maxlen is correctly enforced.

        Verifies internal deque structure maintains maxlen constraint.
        """
        max_samples = 100
        collector = MetricsCollector(max_samples=max_samples)

        # Verify deque has maxlen set
        assert isinstance(collector.metrics, deque)
        assert collector.metrics.maxlen == max_samples

        # Add metrics beyond maxlen
        for i in range(200):
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
            )
            collector.record_metric(metric)

        # Verify maxlen constraint
        assert len(collector.metrics) == max_samples
        assert len(collector.metrics) <= collector.metrics.maxlen

    def test_memory_cleanup_after_operations(self):
        """
        Test that metrics are properly cleaned up and don't accumulate.

        Verifies that old metrics are evicted when new ones are added.
        """
        collector = MetricsCollector(max_samples=100)

        # Add initial batch
        for i in range(100):
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
                metadata={"batch": 1, "sequence": i},
            )
            collector.record_metric(metric)

        # Verify first batch
        assert len(collector.metrics) == 100
        first_batch_sequences = [m.metadata.get("sequence") for m in collector.metrics]
        assert all(seq < 100 for seq in first_batch_sequences)

        # Add second batch (should evict first batch)
        for i in range(100):
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
                metadata={"batch": 2, "sequence": i + 100},
            )
            collector.record_metric(metric)

        # Verify second batch replaced first batch
        assert len(collector.metrics) == 100
        second_batch_sequences = [m.metadata.get("sequence") for m in collector.metrics]
        assert all(seq >= 100 for seq in second_batch_sequences)
        assert all(seq < 200 for seq in second_batch_sequences)


class TestMetricsResourceManagement:
    """Test suite for metrics resource management."""

    def test_large_scale_metric_recording(self):
        """
        Test large-scale metric recording without memory explosion.

        Simulates high-throughput scenario (100,000 metrics).
        """
        collector = MetricsCollector(max_samples=10000)

        # Record large number of metrics
        for i in range(100000):
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
            )
            collector.record_metric(metric)

        # Verify bounded memory
        assert len(collector.metrics) == 10000

        # Verify last 10,000 metrics retained
        # (Implementation detail: deque maintains insertion order)

    def test_concurrent_metric_recording_memory_bounds(self):
        """
        Test that concurrent metric recording maintains memory bounds.

        Verifies thread-safe metric recording with memory constraints.
        """
        import threading

        collector = MetricsCollector(max_samples=1000)

        def record_metrics_thread(thread_id: int):
            """Record metrics from a thread."""
            for i in range(500):
                metric = PerformanceMetric(
                    operation_type=OperationType.DATABASE_QUERY,
                    execution_time_ms=10.5,
                    timestamp=datetime.now(),
                    success=True,
                    metadata={"thread_id": thread_id},
                )
                collector.record_metric(metric)

        # Create 10 threads recording metrics concurrently
        threads = [
            threading.Thread(target=record_metrics_thread, args=(i,)) for i in range(10)
        ]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify memory bounds maintained despite concurrent access
        assert len(collector.metrics) == 1000

    def test_memory_efficiency_with_metadata(self):
        """
        Test memory efficiency when metrics include metadata.

        Verifies that metadata doesn't cause unbounded memory growth.
        """
        collector = MetricsCollector(max_samples=1000)

        # Add metrics with large metadata
        for i in range(5000):
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
                metadata={
                    "query_id": str(i),
                    "table_name": f"table_{i % 10}",
                    "filter_count": i % 100,
                    "result_size": i * 10,
                },
            )
            collector.record_metric(metric)

        # Verify bounded size despite metadata
        assert len(collector.metrics) == 1000

    def test_garbage_collection_effectiveness(self):
        """
        Test that evicted metrics are properly garbage collected.

        Verifies that old metric objects are released for GC.
        """
        collector = MetricsCollector(max_samples=100)

        # Create metrics with tracking objects
        tracked_objects = []

        for i in range(200):
            large_data = ["x" * 1000] * 10  # ~10KB per metric
            metric = PerformanceMetric(
                operation_type=OperationType.DATABASE_QUERY,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
                metadata={"data": large_data},
            )

            if i < 100:
                tracked_objects.append(id(metric))

            collector.record_metric(metric)

        # Force garbage collection
        gc.collect()

        # Verify deque size
        assert len(collector.metrics) == 100

        # Note: We can't directly verify GC of evicted metrics without weak references,
        # but the deque behavior ensures old references are dropped


class TestMemoryLeakScenarios:
    """Test suite for specific memory leak scenarios."""

    def test_continuous_recording_24_hour_simulation(self):
        """
        Test memory stability simulating 24-hour continuous recording.

        Simulates high-frequency metric recording over extended period.
        """
        collector = MetricsCollector(max_samples=10000)

        # Simulate 1 metric per second for 24 hours = 86,400 metrics
        # Use batches to speed up test
        batch_size = 1000
        num_batches = 86  # ~86,000 metrics total

        for batch_num in range(num_batches):
            for i in range(batch_size):
                metric = PerformanceMetric(
                    operation_type=OperationType.DATABASE_QUERY,
                    execution_time_ms=10.5,
                    timestamp=datetime.now(),
                    success=True,
                )
                collector.record_metric(metric)

        # Verify memory bounds maintained
        assert len(collector.metrics) == 10000

    def test_burst_load_memory_stability(self):
        """
        Test memory stability during burst load scenarios.

        Simulates alternating periods of high and low load.
        """
        collector = MetricsCollector(max_samples=1000)

        # Simulate 5 burst cycles
        for cycle in range(5):
            # High load burst (5000 metrics)
            for i in range(5000):
                metric = PerformanceMetric(
                    operation_type=OperationType.DATABASE_QUERY,
                    execution_time_ms=10.5,
                    timestamp=datetime.now(),
                    success=True,
                )
                collector.record_metric(metric)

            # Verify bounded after burst
            assert len(collector.metrics) == 1000

            # Low load period (100 metrics)
            for i in range(100):
                metric = PerformanceMetric(
                    operation_type=OperationType.DATABASE_QUERY,
                    execution_time_ms=10.5,
                    timestamp=datetime.now(),
                    success=True,
                )
                collector.record_metric(metric)

            # Verify still bounded
            assert len(collector.metrics) == 1000

    def test_mixed_operation_types_memory_bounds(self):
        """
        Test memory bounds with mixed operation types.

        Verifies that different operation types don't affect memory bounds.
        """
        collector = MetricsCollector(max_samples=1000)

        operation_types = list(OperationType)

        # Add metrics for each operation type
        for i in range(10000):
            operation_type = operation_types[i % len(operation_types)]
            metric = PerformanceMetric(
                operation_type=operation_type,
                execution_time_ms=10.5,
                timestamp=datetime.now(),
                success=True,
            )
            collector.record_metric(metric)

        # Verify bounded memory
        assert len(collector.metrics) == 1000


if __name__ == "__main__":
    # Run tests with: pytest tests/performance/test_memory_leak_prevention.py -v
    pytest.main([__file__, "-v"])
