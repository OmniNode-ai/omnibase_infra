#!/usr/bin/env python3
"""
Dedicated Performance Benchmarks for Kafka Event Publishing.

This module provides comprehensive performance testing for Kafka/RedPanda event
publishing with focus on latency, throughput, and connection pool efficiency.

Performance Targets (from infrastructure code):
- Event publishing latency: <5ms per event
- Connection pool overhead: <5ms for producer acquisition/release
- Concurrent publishing: 100+ operations/second
- Batch publishing: >1000 events/second
- Connection pool utilization: Monitor at 90% threshold

Benchmark Categories:
1. Single Event Publishing Performance
   - Publish latency (p50, p95, p99)
   - Serialization overhead
   - Network round-trip time

2. Batch Publishing Performance
   - Batch throughput measurement
   - Batch size optimization
   - Memory usage per batch

3. Connection Pool Performance
   - Pool acquisition latency
   - Pool utilization efficiency
   - Concurrent operation handling

4. Event Serialization Performance
   - OnexEnvelopeV1 serialization
   - Large payload handling
   - Compression efficiency

Usage:
    # Run all Kafka benchmarks
    pytest tests/performance/test_kafka_performance.py -v

    # Run specific benchmark
    pytest tests/performance/test_kafka_performance.py::test_single_event_publish_latency -v

    # Generate benchmark report
    pytest tests/performance/test_kafka_performance.py --benchmark-only --benchmark-json=kafka.json

Note:
    These benchmarks use mocked Kafka infrastructure to avoid external dependencies.
    For real Kafka performance testing, use integration tests with actual infrastructure.
"""

import asyncio
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import psutil
import pytest

# Performance thresholds from infrastructure code
PERFORMANCE_THRESHOLDS = {
    "single_publish_ms": {"max": 5, "p95": 4, "p99": 4.5},
    "pool_acquisition_ms": {"max": 5, "p95": 3, "p99": 4},
    "batch_throughput_events_per_sec": {"min": 1000},
    "concurrent_throughput_ops_per_sec": {"min": 100},
    "serialization_overhead_ms": {"max": 2, "p95": 1.5},
}


def run_async_in_sync(coro):
    """
    Helper to run async code in a synchronous benchmark context.

    Creates a new event loop in a thread to avoid conflicts with pytest-asyncio.
    """

    def _run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread)
        return future.result()


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_kafka_producer():
    """Mock AIOKafkaProducer with realistic latency."""

    class MockProducer:
        def __init__(self, latency_ms: float = 1.0):
            self.latency = latency_ms / 1000.0
            self.published_count = 0
            self.published_bytes = 0

        async def send(self, topic: str, value: bytes, key: bytes = None):
            """Simulate event publishing with latency."""
            await asyncio.sleep(self.latency)
            self.published_count += 1
            self.published_bytes += len(value)
            return MagicMock()  # RecordMetadata mock

        async def start(self):
            """Simulate producer startup."""
            await asyncio.sleep(0.001)

        async def stop(self):
            """Simulate producer shutdown."""
            await asyncio.sleep(0.001)

    return MockProducer


@pytest.fixture
def mock_connection_pool():
    """Mock Kafka connection pool manager."""

    class MockConnectionPool:
        def __init__(self, pool_size: int = 10):
            self.pool_size = pool_size
            self.available = pool_size
            self.in_use = 0
            self.total_acquisitions = 0
            self.total_releases = 0
            self.wait_times = []

        async def acquire(self):
            """Simulate producer acquisition with wait time."""
            start = time.perf_counter()

            # Simulate wait if pool exhausted
            if self.available == 0:
                await asyncio.sleep(0.002)  # 2ms wait
            else:
                await asyncio.sleep(0.0001)  # 0.1ms for available producer

            wait_time_ms = (time.perf_counter() - start) * 1000
            self.wait_times.append(wait_time_ms)

            self.available -= 1
            self.in_use += 1
            self.total_acquisitions += 1

            return MagicMock()  # Producer mock

        async def release(self, producer):
            """Simulate producer release."""
            await asyncio.sleep(0)
            self.available += 1
            self.in_use -= 1
            self.total_releases += 1

        def get_metrics(self):
            """Get pool metrics."""
            return {
                "total_acquisitions": self.total_acquisitions,
                "total_releases": self.total_releases,
                "avg_wait_time_ms": (
                    sum(self.wait_times) / len(self.wait_times)
                    if self.wait_times
                    else 0
                ),
                "utilization": (
                    self.in_use / self.pool_size if self.pool_size > 0 else 0
                ),
            }

    return MockConnectionPool


@pytest.fixture
def sample_event_generator():
    """Generate sample events for benchmarking."""

    def _generate(count: int = 1, payload_size_kb: int = 1):
        """Generate N events with specified payload size."""
        events = []
        for i in range(count):
            # Create realistic event payload
            payload = {
                "event_id": str(uuid4()),
                "event_type": "metadata.stamp.created",
                "correlation_id": str(uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "payload": {
                    "file_hash": f"blake3_hash_{i:08d}",
                    "file_path": f"/data/file_{i:08d}.pdf",
                    "namespace": "omninode.services.metadata",
                    "metadata": {"size": payload_size_kb * 1024},
                    # Pad to desired size
                    "padding": "x" * (payload_size_kb * 1024 - 200),
                },
            }
            events.append(payload)
        return events

    return _generate


@pytest.fixture
def memory_tracker():
    """Track memory usage during benchmarks."""

    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process()
            self.baseline = None
            self.peak = 0
            self.samples = []

        def start(self):
            """Start memory tracking."""
            gc.collect()
            self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak = self.baseline
            self.samples = [self.baseline]

        def sample(self):
            """Take a memory sample."""
            current = self.process.memory_info().rss / 1024 / 1024  # MB
            self.samples.append(current)
            self.peak = max(self.peak, current)
            return current

        def stop(self):
            """Stop tracking and return statistics."""
            gc.collect()
            final = self.process.memory_info().rss / 1024 / 1024  # MB
            return {
                "baseline_mb": self.baseline,
                "peak_mb": self.peak,
                "final_mb": final,
                "delta_mb": final - self.baseline,
                "peak_delta_mb": self.peak - self.baseline,
            }

    return MemoryTracker()


# ============================================================================
# SINGLE EVENT PUBLISHING BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestSingleEventPublishing:
    """Benchmarks for single event publishing performance."""

    def test_single_event_publish_latency(
        self, benchmark, mock_kafka_producer, sample_event_generator
    ):
        """
        Benchmark: Single event publish latency.

        Target: <5ms (p95 < 4ms, p99 < 4.5ms)
        Measures: End-to-end publish latency including serialization
        """
        events = sample_event_generator(1, payload_size_kb=1)
        producer = mock_kafka_producer(latency_ms=1.0)

        async def _publish_event():
            """Publish single event."""
            event = events[0]
            payload_bytes = json.dumps(event).encode("utf-8")
            await producer.send("test-topic", payload_bytes)
            return len(payload_bytes)

        def _sync_publish():
            return run_async_in_sync(_publish_event())

        # Run benchmark
        result = benchmark.pedantic(_sync_publish, rounds=100, iterations=10)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000
        max_ms = stats.max * 1000

        print(
            f"\n[Performance] Single event publish - Mean: {mean_ms:.2f}ms, Max: {max_ms:.2f}ms"
        )
        print(f"[Event Size] {result} bytes")

        assert max_ms < PERFORMANCE_THRESHOLDS["single_publish_ms"]["max"]

    def test_large_event_publish_latency(
        self, benchmark, mock_kafka_producer, sample_event_generator
    ):
        """
        Benchmark: Large event (100KB) publish latency.

        Target: <20ms for large events
        Measures: Performance with large payloads
        """
        events = sample_event_generator(1, payload_size_kb=100)
        producer = mock_kafka_producer(latency_ms=2.0)

        async def _publish_large_event():
            """Publish large event."""
            event = events[0]
            payload_bytes = json.dumps(event).encode("utf-8")
            await producer.send("test-topic", payload_bytes)
            return len(payload_bytes)

        def _sync_publish_large():
            return run_async_in_sync(_publish_large_event())

        # Run benchmark
        result = benchmark.pedantic(_sync_publish_large, rounds=50, iterations=5)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] Large event (100KB) - Mean: {mean_ms:.2f}ms")
        print(f"[Event Size] {result / 1024:.2f} KB")

        assert mean_ms < 20.0


# ============================================================================
# BATCH PUBLISHING BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestBatchPublishing:
    """Benchmarks for batch publishing performance."""

    def test_batch_publish_throughput(
        self, benchmark, mock_kafka_producer, sample_event_generator
    ):
        """
        Benchmark: Batch publishing throughput.

        Target: >1000 events/second
        Measures: Maximum sustained throughput
        """
        events = sample_event_generator(1000, payload_size_kb=1)
        producer = mock_kafka_producer(latency_ms=0.5)

        async def _batch_publish():
            """Publish batch of events."""
            start = time.perf_counter()

            tasks = []
            for event in events:
                payload_bytes = json.dumps(event).encode("utf-8")
                tasks.append(producer.send("test-topic", payload_bytes))

            await asyncio.gather(*tasks)

            duration = time.perf_counter() - start
            throughput = len(events) / duration if duration > 0 else 0

            return {
                "count": len(events),
                "duration_s": duration,
                "throughput": throughput,
            }

        def _sync_batch_publish():
            return run_async_in_sync(_batch_publish())

        # Run benchmark
        result = benchmark.pedantic(_sync_batch_publish, rounds=5, iterations=2)

        throughput = result["throughput"]
        print(f"\n[Performance] Batch throughput: {throughput:.2f} events/second")

        assert (
            throughput
            >= PERFORMANCE_THRESHOLDS["batch_throughput_events_per_sec"]["min"]
        ), f"Throughput {throughput:.2f} below target"

    def test_batch_size_optimization(
        self, benchmark, mock_kafka_producer, sample_event_generator, memory_tracker
    ):
        """
        Benchmark: Optimal batch size determination.

        Tests: 10, 50, 100, 500, 1000 events per batch
        Measures: Throughput vs memory tradeoff
        """
        batch_sizes = [10, 50, 100, 500, 1000]
        results = []

        for batch_size in batch_sizes:
            events = sample_event_generator(batch_size, payload_size_kb=1)
            producer = mock_kafka_producer(latency_ms=0.5)

            memory_tracker.start()

            async def _batch_publish():
                tasks = []
                for event in events:
                    payload_bytes = json.dumps(event).encode("utf-8")
                    tasks.append(producer.send("test-topic", payload_bytes))

                start = time.perf_counter()
                await asyncio.gather(*tasks)
                duration = time.perf_counter() - start

                memory_tracker.sample()
                throughput = len(events) / duration if duration > 0 else 0

                return {"throughput": throughput, "duration_ms": duration * 1000}

            result = run_async_in_sync(_batch_publish())
            memory_stats = memory_tracker.stop()

            results.append(
                {
                    "batch_size": batch_size,
                    "throughput": result["throughput"],
                    "duration_ms": result["duration_ms"],
                    "memory_delta_mb": memory_stats["delta_mb"],
                }
            )

        # Print results table
        print("\n[Batch Size Optimization]")
        print("Size\tThroughput (ev/s)\tDuration (ms)\tMemory (MB)")
        for r in results:
            print(
                f"{r['batch_size']}\t{r['throughput']:.2f}\t\t{r['duration_ms']:.2f}\t\t{r['memory_delta_mb']:.2f}"
            )


# ============================================================================
# CONNECTION POOL BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestConnectionPoolPerformance:
    """Benchmarks for connection pool performance."""

    def test_pool_acquisition_latency(self, benchmark, mock_connection_pool):
        """
        Benchmark: Producer acquisition latency from pool.

        Target: <5ms (p95 < 3ms, p99 < 4ms)
        Measures: Pool overhead for producer acquisition/release
        """
        pool = mock_connection_pool(pool_size=10)

        async def _acquire_release():
            """Acquire and release producer from pool."""
            producer = await pool.acquire()
            await pool.release(producer)

        def _sync_acquire_release():
            return run_async_in_sync(_acquire_release())

        # Run benchmark
        benchmark.pedantic(_sync_acquire_release, rounds=100, iterations=10)

        metrics = pool.get_metrics()
        avg_wait_ms = metrics["avg_wait_time_ms"]

        print(f"\n[Performance] Pool acquisition - Avg: {avg_wait_ms:.2f}ms")
        print(f"[Pool Metrics] {metrics}")

        assert avg_wait_ms < PERFORMANCE_THRESHOLDS["pool_acquisition_ms"]["max"]

    def test_concurrent_pool_operations(self, benchmark, mock_connection_pool):
        """
        Benchmark: Concurrent pool operations throughput.

        Target: 100+ operations/second
        Measures: Pool efficiency under concurrent load
        """
        pool = mock_connection_pool(pool_size=10)

        async def _concurrent_operations():
            """Execute concurrent pool operations."""
            tasks = []
            for _ in range(100):

                async def _operation():
                    producer = await pool.acquire()
                    await asyncio.sleep(0.001)  # Simulate work
                    await pool.release(producer)

                tasks.append(_operation())

            start = time.perf_counter()
            await asyncio.gather(*tasks)
            duration = time.perf_counter() - start

            throughput = 100 / duration if duration > 0 else 0

            return {"throughput": throughput, "duration_s": duration}

        def _sync_concurrent():
            return run_async_in_sync(_concurrent_operations())

        # Run benchmark
        result = benchmark.pedantic(_sync_concurrent, rounds=5, iterations=2)

        throughput = result["throughput"]
        print(f"\n[Performance] Concurrent operations: {throughput:.2f} ops/second")

        assert (
            throughput
            >= PERFORMANCE_THRESHOLDS["concurrent_throughput_ops_per_sec"]["min"]
        )


# ============================================================================
# SERIALIZATION BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestEventSerializationPerformance:
    """Benchmarks for event serialization performance."""

    def test_onex_envelope_serialization(self, benchmark, sample_event_generator):
        """
        Benchmark: OnexEnvelopeV1 serialization overhead.

        Target: <2ms (p95 < 1.5ms)
        Measures: Envelope wrapping and JSON serialization time
        """
        events = sample_event_generator(1, payload_size_kb=1)

        def _serialize_envelope():
            """Serialize event with envelope."""
            event = events[0]

            # Simulate OnexEnvelopeV1 wrapping
            envelope = {
                "envelope_version": "v1",
                "event_id": str(uuid4()),
                "correlation_id": str(uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "payload": event,
            }

            # Serialize to JSON
            payload_bytes = json.dumps(envelope).encode("utf-8")
            return len(payload_bytes)

        # Run benchmark
        result = benchmark.pedantic(_serialize_envelope, rounds=100, iterations=10)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] Serialization - Mean: {mean_ms:.2f}ms")
        print(f"[Envelope Size] {result} bytes")

        assert mean_ms < PERFORMANCE_THRESHOLDS["serialization_overhead_ms"]["max"]


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================


def pytest_benchmark_update_json(config, benchmarks, output_json):
    """
    Customize benchmark JSON output with Kafka-specific metadata.

    Adds:
    - Performance thresholds
    - Test metadata
    - Kafka configuration
    """
    output_json["performance_thresholds"] = PERFORMANCE_THRESHOLDS
    output_json["component"] = "KafkaEventPublishing"
    output_json["test_metadata"] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "focus": "Event publishing latency, batch throughput, connection pool efficiency",
    }
