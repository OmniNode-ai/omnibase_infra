#!/usr/bin/env python3
"""
Dedicated Performance Benchmarks for NodeBridgeReducer.

This module provides comprehensive performance testing for the Bridge Reducer
with focus on aggregation throughput, streaming performance, and memory efficiency.

Performance Targets (from ROADMAP.md):
- Batch aggregation: >1000 items/second
- Aggregation latency: <1000ms for 1000 items (p95 < 800ms)
- Streaming throughput: >1000 items/second sustained
- Memory usage: <100MB delta for 10,000 items
- Namespace grouping: <1.2s for 1000 items across 10 namespaces

Benchmark Categories:
1. Batch Aggregation Performance
   - Small batch (100 items)
   - Medium batch (1000 items)
   - Large batch (10,000 items)
   - Throughput measurement

2. Streaming Aggregation Performance
   - Streaming window processing
   - Backpressure handling
   - Memory stability

3. Aggregation Type Performance
   - Namespace grouping
   - Time window aggregation
   - File type grouping
   - Custom aggregation

4. Memory Efficiency
   - Memory usage per item
   - Memory leak detection
   - Garbage collection efficiency

Usage:
    # Run all reducer benchmarks
    pytest tests/performance/test_reducer_performance.py -v

    # Run specific benchmark
    pytest tests/performance/test_reducer_performance.py::test_batch_aggregation_throughput -v

    # Generate benchmark report
    pytest tests/performance/test_reducer_performance.py --benchmark-only --benchmark-json=reducer.json
"""

import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import psutil
import pytest

# Performance thresholds from ROADMAP.md
PERFORMANCE_THRESHOLDS = {
    "batch_100_ms": {"max": 100, "p95": 80},
    "batch_1000_ms": {"max": 1000, "p95": 800},
    "batch_10000_ms": {"max": 10000, "p95": 8000},
    "throughput_items_per_sec": {"min": 1000},
    "namespace_grouping_ms": {"max": 1200, "p95": 1000},
    "memory_delta_mb": {"max_per_10k_items": 100},
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
def mock_container():
    """Mock ONEX container for dependency injection."""
    container = MagicMock()
    container.get_service = MagicMock(return_value=None)
    return container


@pytest.fixture
def reducer_node(mock_container):
    """Initialized NodeBridgeReducer instance."""
    try:
        from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

        return NodeBridgeReducer(container=mock_container)
    except (ImportError, Exception) as e:
        error_msg = str(e)
        if (
            "omnibase_core.utils.generation" in error_msg
            or "Contract model loading failed" in error_msg
        ):
            pytest.skip(
                "NodeBridgeReducer requires omnibase_core.utils.generation module"
            )
        else:
            raise


@pytest.fixture
def sample_metadata_generator():
    """Generate sample metadata for benchmarking."""
    from omninode_bridge.nodes.reducer.v1_0_0.models.model_stamp_metadata_input import (
        ModelStampMetadataInput,
    )

    def _generate(count: int = 1, namespace: str = "omninode.services.metadata"):
        """Generate N metadata records."""
        items = []
        for i in range(count):
            item = ModelStampMetadataInput(
                stamp_id=str(uuid4()),
                file_hash=f"blake3_{i:08d}",
                file_path=f"/data/file_{i:08d}.pdf",
                file_size=1024 * (i % 1000 + 1),
                namespace=namespace,
                content_type="application/pdf" if i % 2 == 0 else "image/png",
                workflow_id=uuid4(),
                workflow_state="completed" if i % 3 == 0 else "processing",
                processing_time_ms=float(i % 100),
            )
            items.append(item)
        return items

    return _generate


@pytest.fixture
def mock_reducer_contract():
    """Mock reducer contract for testing."""
    from omninode_bridge.nodes.reducer.v1_0_0.models.enum_aggregation_type import (
        EnumAggregationType,
    )

    class MockReducerContract:
        def __init__(self, items=None):
            self.input_state = {"items": items or []}
            self.input_stream = None
            self.aggregation = MagicMock()
            self.aggregation.aggregation_type = EnumAggregationType.NAMESPACE_GROUPING
            self.streaming = MagicMock()
            self.streaming.window_size = 5000
            self.streaming.batch_size = 100
            self.state_management = None

    return MockReducerContract


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
                "avg_mb": sum(self.samples) / len(self.samples),
                "memory_per_item_kb": (
                    (final - self.baseline) * 1024 / len(self.samples)
                    if len(self.samples) > 0
                    else 0
                ),
            }

    return MemoryTracker()


# ============================================================================
# BATCH AGGREGATION BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestBatchAggregationPerformance:
    """Benchmarks for batch aggregation performance."""

    def test_batch_aggregation_100_items(
        self,
        benchmark,
        reducer_node,
        sample_metadata_generator,
        mock_reducer_contract,
        memory_tracker,
    ):
        """
        Benchmark: Aggregate 100 items.

        Target: <100ms (p95 < 80ms)
        Measures: Small batch aggregation performance
        """
        items = sample_metadata_generator(100)
        contract = mock_reducer_contract(items)

        memory_tracker.start()

        async def _aggregate():
            result = await reducer_node.execute_reduction(contract)
            memory_tracker.sample()
            return result

        def _sync_aggregate():
            return run_async_in_sync(_aggregate())

        # Run benchmark
        result = benchmark.pedantic(_sync_aggregate, rounds=10, iterations=5)

        memory_stats = memory_tracker.stop()
        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000
        max_ms = stats.max * 1000

        print(f"\n[Performance] 100 items - Mean: {mean_ms:.2f}ms, Max: {max_ms:.2f}ms")
        print(f"[Memory] {memory_stats}")

        assert max_ms < PERFORMANCE_THRESHOLDS["batch_100_ms"]["max"]

    def test_batch_aggregation_1000_items(
        self,
        benchmark,
        reducer_node,
        sample_metadata_generator,
        mock_reducer_contract,
        memory_tracker,
    ):
        """
        Benchmark: Aggregate 1000 items.

        Target: <1000ms (p95 < 800ms)
        Measures: Medium batch aggregation performance
        """
        items = sample_metadata_generator(1000)
        contract = mock_reducer_contract(items)

        memory_tracker.start()

        async def _aggregate():
            result = await reducer_node.execute_reduction(contract)
            memory_tracker.sample()
            return result

        def _sync_aggregate():
            return run_async_in_sync(_aggregate())

        # Run benchmark
        result = benchmark.pedantic(_sync_aggregate, rounds=5, iterations=3)

        memory_stats = memory_tracker.stop()
        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000
        max_ms = stats.max * 1000

        print(
            f"\n[Performance] 1000 items - Mean: {mean_ms:.2f}ms, Max: {max_ms:.2f}ms"
        )
        print(f"[Memory] {memory_stats}")

        assert max_ms < PERFORMANCE_THRESHOLDS["batch_1000_ms"]["max"]

    def test_batch_aggregation_10000_items(
        self,
        benchmark,
        reducer_node,
        sample_metadata_generator,
        mock_reducer_contract,
        memory_tracker,
    ):
        """
        Benchmark: Aggregate 10,000 items.

        Target: <10s (p95 < 8s)
        Measures: Large batch aggregation performance and memory efficiency
        """
        items = sample_metadata_generator(10000)
        contract = mock_reducer_contract(items)

        memory_tracker.start()

        async def _aggregate():
            result = await reducer_node.execute_reduction(contract)
            # Sample memory every 1000 items processed
            memory_tracker.sample()
            return result

        def _sync_aggregate():
            return run_async_in_sync(_aggregate())

        # Run benchmark (fewer rounds for large dataset)
        result = benchmark.pedantic(_sync_aggregate, rounds=3, iterations=1)

        memory_stats = memory_tracker.stop()
        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] 10,000 items - Mean: {mean_ms:.2f}ms")
        print(f"[Memory] {memory_stats}")

        # Memory leak detection
        assert (
            memory_stats["delta_mb"]
            < PERFORMANCE_THRESHOLDS["memory_delta_mb"]["max_per_10k_items"]
        ), f"Memory delta {memory_stats['delta_mb']:.2f}MB exceeds threshold"

    def test_batch_aggregation_throughput(
        self,
        benchmark,
        reducer_node,
        sample_metadata_generator,
        mock_reducer_contract,
    ):
        """
        Benchmark: Aggregation throughput measurement.

        Target: >1000 items/second
        Measures: Maximum sustained aggregation rate
        """

        async def _measure_throughput():
            """Measure aggregation throughput."""
            items = sample_metadata_generator(5000)
            contract = mock_reducer_contract(items)

            start = time.perf_counter()
            await reducer_node.execute_reduction(contract)
            duration = time.perf_counter() - start

            throughput = len(items) / duration if duration > 0 else 0

            return {
                "items": len(items),
                "duration_s": duration,
                "throughput": throughput,
            }

        def _sync_throughput():
            return run_async_in_sync(_measure_throughput())

        # Run benchmark
        result = benchmark.pedantic(_sync_throughput, rounds=5, iterations=2)

        throughput = result["throughput"]
        print(f"\n[Performance] Throughput: {throughput:.2f} items/second")

        assert (
            throughput >= PERFORMANCE_THRESHOLDS["throughput_items_per_sec"]["min"]
        ), f"Throughput {throughput:.2f} below target {PERFORMANCE_THRESHOLDS['throughput_items_per_sec']['min']}"


# ============================================================================
# STREAMING AGGREGATION BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestStreamingAggregationPerformance:
    """Benchmarks for streaming aggregation performance."""

    def test_streaming_window_processing(
        self,
        benchmark,
        reducer_node,
        sample_metadata_generator,
        memory_tracker,
    ):
        """
        Benchmark: Streaming window processing.

        Target: Process 1000 items in windows of 100 in <1.5s
        Measures: Streaming performance and backpressure handling
        """
        items = sample_metadata_generator(1000)

        async def _stream_generator():
            """Async generator simulating streaming input."""
            for item in items:
                yield item
                await asyncio.sleep(0)  # Allow event loop

        memory_tracker.start()

        async def _process_stream():
            """Process streaming data."""
            from omninode_bridge.nodes.reducer.v1_0_0.models.enum_aggregation_type import (
                EnumAggregationType,
            )

            class StreamingContract:
                def __init__(self, stream):
                    self.input_stream = stream
                    self.input_state = None
                    self.streaming = MagicMock()
                    self.streaming.window_size = 5000
                    self.streaming.batch_size = 100
                    self.aggregation = MagicMock()
                    self.aggregation.aggregation_type = (
                        EnumAggregationType.NAMESPACE_GROUPING
                    )

            contract = StreamingContract(_stream_generator())
            result = await reducer_node.execute_reduction(contract)
            memory_tracker.sample()
            return result

        def _sync_process_stream():
            return run_async_in_sync(_process_stream())

        # Run benchmark
        result = benchmark.pedantic(_sync_process_stream, rounds=5, iterations=2)

        memory_stats = memory_tracker.stop()
        print(f"\n[Memory] Streaming 1000 items: {memory_stats}")


# ============================================================================
# AGGREGATION TYPE BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestAggregationTypePerformance:
    """Benchmarks for different aggregation types."""

    def test_namespace_grouping_performance(
        self,
        benchmark,
        reducer_node,
        sample_metadata_generator,
    ):
        """
        Benchmark: Multi-namespace aggregation.

        Target: 1000 items across 10 namespaces in <1.2s (p95 < 1.0s)
        Measures: Namespace grouping efficiency
        """
        from omninode_bridge.nodes.reducer.v1_0_0.models.enum_aggregation_type import (
            EnumAggregationType,
        )

        # Generate multi-namespace data
        items = []
        for i in range(10):  # 10 namespaces
            namespace = f"omninode.services.ns{i}"
            items.extend(sample_metadata_generator(100, namespace))

        class MockContract:
            def __init__(self):
                self.input_state = {"items": items}
                self.input_stream = None
                self.aggregation = MagicMock()
                self.aggregation.aggregation_type = (
                    EnumAggregationType.NAMESPACE_GROUPING
                )
                self.streaming = MagicMock()
                self.streaming.batch_size = 100

        async def _aggregate_namespaces():
            contract = MockContract()
            return await reducer_node.execute_reduction(contract)

        def _sync_aggregate_namespaces():
            return run_async_in_sync(_aggregate_namespaces())

        # Run benchmark
        result = benchmark.pedantic(_sync_aggregate_namespaces, rounds=5, iterations=3)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000
        max_ms = stats.max * 1000

        print(
            f"\n[Performance] Namespace grouping - Mean: {mean_ms:.2f}ms, Max: {max_ms:.2f}ms"
        )

        assert max_ms < PERFORMANCE_THRESHOLDS["namespace_grouping_ms"]["max"]


# ============================================================================
# MEMORY EFFICIENCY BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestMemoryEfficiency:
    """Benchmarks for memory efficiency."""

    async def test_memory_per_item_scaling(
        self,
        reducer_node,
        sample_metadata_generator,
        mock_reducer_contract,
        memory_tracker,
    ):
        """
        Test: Memory usage scaling per item.

        Target: Linear memory growth, <10KB per item
        Measures: Memory efficiency and leak detection
        """
        batch_sizes = [100, 500, 1000, 2000]
        memory_per_batch = []

        for size in batch_sizes:
            memory_tracker.start()

            items = sample_metadata_generator(size)
            contract = mock_reducer_contract(items)

            await reducer_node.execute_reduction(contract)

            stats = memory_tracker.stop()
            memory_per_item = stats["delta_mb"] * 1024 / size  # KB per item

            memory_per_batch.append(
                {
                    "size": size,
                    "delta_mb": stats["delta_mb"],
                    "memory_per_item_kb": memory_per_item,
                }
            )

            print(
                f"\n[Memory] {size} items: {stats['delta_mb']:.2f}MB ({memory_per_item:.2f} KB/item)"
            )

        # Verify linear scaling (memory per item should be relatively constant)
        avg_memory_per_item = sum(
            m["memory_per_item_kb"] for m in memory_per_batch
        ) / len(memory_per_batch)
        print(f"\n[Memory] Average: {avg_memory_per_item:.2f} KB/item")

        # Assert reasonable memory usage
        assert (
            avg_memory_per_item < 10
        ), f"Memory per item {avg_memory_per_item:.2f}KB exceeds 10KB threshold"


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================


def pytest_benchmark_update_json(config, benchmarks, output_json):
    """
    Customize benchmark JSON output with reducer-specific metadata.

    Adds:
    - Performance thresholds
    - Test metadata
    - Reducer configuration
    """
    output_json["performance_thresholds"] = PERFORMANCE_THRESHOLDS
    output_json["component"] = "NodeBridgeReducer"
    output_json["test_metadata"] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "focus": "Aggregation throughput, streaming performance, memory efficiency",
    }
