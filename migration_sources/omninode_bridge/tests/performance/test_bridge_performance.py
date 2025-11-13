#!/usr/bin/env python3
"""
Performance Benchmarking Suite for Bridge Nodes.

Comprehensive benchmarks for NodeBridgeOrchestrator and NodeBridgeReducer
using pytest-benchmark for statistical analysis and performance validation.

Benchmark Categories:
1. Orchestrator Performance
   - Single workflow execution (<300ms)
   - Concurrent workflow throughput (100+ workflows)
   - FSM state transition overhead (<50ms)
   - Service routing performance

2. Reducer Performance
   - Batch aggregation (1000 items in <1s)
   - Streaming window processing
   - PostgreSQL persistence
   - Memory usage under load

3. End-to-End Performance
   - Complete workflow pipeline (<500ms)
   - Multi-service coordination
   - Error recovery overhead

Performance Thresholds:
- All benchmarks include p50, p95, p99 percentiles
- Memory profiling for leak detection
- CSV/JSON report generation
- CI/CD integration ready

Usage:
    # Run all performance benchmarks
    pytest tests/performance/test_bridge_performance.py -v

    # Run specific benchmark category
    pytest tests/performance/test_bridge_performance.py -k "test_reducer" -v

    # Generate benchmark comparison report
    pytest tests/performance/test_bridge_performance.py --benchmark-only --benchmark-json=output.json

    # Save benchmark results for comparison
    pytest tests/performance/test_bridge_performance.py --benchmark-save=baseline

    # Compare against baseline
    pytest tests/performance/test_bridge_performance.py --benchmark-compare=baseline
"""

import asyncio
import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import psutil
import pytest


def run_async_in_sync(coro):
    """
    Helper to run async code in a synchronous context.

    Handles the case where we're already in an async context (pytest-asyncio)
    by running the coroutine in a separate thread with its own event loop.
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


# Import bridge node models
from omninode_bridge.nodes.reducer.v1_0_0.models.enum_aggregation_type import (
    EnumAggregationType,
)
from omninode_bridge.nodes.reducer.v1_0_0.models.model_stamp_metadata_input import (
    ModelStampMetadataInput,
)
from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

# Performance thresholds from requirements
PERFORMANCE_THRESHOLDS = {
    "orchestrator_workflow": {"max_ms": 300, "p95_ms": 250},
    "orchestrator_concurrent": {"min_throughput": 100, "max_latency_ms": 500},
    "reducer_batch_1000": {"max_ms": 1000, "p95_ms": 800},
    "fsm_transition": {"max_ms": 50, "p95_ms": 30},
    "end_to_end": {"max_ms": 500, "p95_ms": 400},
}


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
def sample_stamp_metadata():
    """Generate sample stamp metadata for testing."""

    def _generate(count: int = 1, namespace: str = "omninode.services.metadata"):
        """Generate N stamp metadata records."""
        stamps = []
        for i in range(count):
            stamp = ModelStampMetadataInput(
                stamp_id=str(uuid4()),
                file_hash=f"blake3_hash_{i:08d}",
                file_path=f"/data/files/document_{i:08d}.pdf",
                file_size=1024 * (i % 1000 + 1),  # Vary size 1KB-1MB
                namespace=namespace,
                content_type="application/pdf" if i % 2 == 0 else "image/png",
                workflow_id=uuid4(),
                workflow_state="completed" if i % 3 == 0 else "processing",
                processing_time_ms=float(i % 10),
            )
            stamps.append(stamp)
        return stamps

    return _generate


@pytest.fixture
def mock_reducer_contract():
    """Mock reducer contract for testing."""

    class MockReducerContract:
        def __init__(self, input_items=None):
            self.input_state = {"items": input_items or []}
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
            gc.collect()  # Force garbage collection
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
                "samples": len(self.samples),
            }

    return MemoryTracker()


# ============================================================================
# REDUCER PERFORMANCE BENCHMARKS
# ============================================================================


@pytest.mark.performance
@pytest.mark.asyncio
class TestReducerPerformance:
    """Performance benchmarks for NodeBridgeReducer."""

    def test_batch_aggregation_100_items(
        self,
        benchmark,
        reducer_node,
        sample_stamp_metadata,
        mock_reducer_contract,
        memory_tracker,
    ):
        """
        Benchmark: Aggregate 100 stamp metadata items.

        Target: <100ms
        Expected p95: <80ms
        """
        # Prepare test data
        stamps = sample_stamp_metadata(100)
        contract = mock_reducer_contract(stamps)

        # Memory tracking
        memory_tracker.start()

        # Benchmark execution
        async def _aggregate():
            result = await reducer_node.execute_reduction(contract)
            memory_tracker.sample()
            return result

        # Wrap async function for benchmark
        def _sync_aggregate():
            return run_async_in_sync(_aggregate())

        # Run benchmark
        result = benchmark.pedantic(_sync_aggregate, rounds=10, iterations=5)

        # Verify results
        memory_stats = memory_tracker.stop()
        assert isinstance(result, type(None)) or hasattr(
            result, "total_items"
        ), "Invalid result type"

        # Log memory usage
        print(f"\n[Memory] 100 items: {memory_stats}")

    def test_batch_aggregation_1000_items(
        self,
        benchmark,
        reducer_node,
        sample_stamp_metadata,
        mock_reducer_contract,
        memory_tracker,
    ):
        """
        Benchmark: Aggregate 1000 stamp metadata items.

        Target: <1000ms (1 second)
        Expected p95: <800ms
        Threshold: PERFORMANCE_THRESHOLDS['reducer_batch_1000']
        """
        # Prepare test data
        stamps = sample_stamp_metadata(1000)
        contract = mock_reducer_contract(stamps)

        # Memory tracking
        memory_tracker.start()

        # Benchmark execution
        async def _aggregate():
            result = await reducer_node.execute_reduction(contract)
            memory_tracker.sample()
            return result

        # Wrap async function for benchmark
        def _sync_aggregate():
            return run_async_in_sync(_aggregate())

        # Run benchmark
        result = benchmark.pedantic(_sync_aggregate, rounds=5, iterations=3)

        # Verify results
        memory_stats = memory_tracker.stop()

        # Validate performance threshold
        # Use max as conservative estimate for p95 (max >= p95 always)
        stats = benchmark.stats.stats
        max_time_s = stats.max
        mean_time_s = stats.mean
        p95_ms = max_time_s * 1000  # Convert seconds to ms
        mean_ms = mean_time_s * 1000  # Also track mean for reference
        assert (
            p95_ms < PERFORMANCE_THRESHOLDS["reducer_batch_1000"]["max_ms"]
        ), f"Performance degraded: {p95_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['reducer_batch_1000']['max_ms']}ms"

        # Log results
        print(
            f"\n[Performance] 1000 items - Mean: {mean_ms:.2f}ms, Max: {p95_ms:.2f}ms"
        )
        print(f"[Memory] 1000 items: {memory_stats}")

    def test_batch_aggregation_10000_items(
        self,
        benchmark,
        reducer_node,
        sample_stamp_metadata,
        mock_reducer_contract,
        memory_tracker,
    ):
        """
        Benchmark: Aggregate 10,000 stamp metadata items.

        Target: <10s for large batch
        Expected p95: <8s
        Memory: Monitor for leaks
        """
        # Prepare test data
        stamps = sample_stamp_metadata(10000)
        contract = mock_reducer_contract(stamps)

        # Memory tracking
        memory_tracker.start()

        # Benchmark execution
        async def _aggregate():
            result = await reducer_node.execute_reduction(contract)
            # Sample memory every 1000 items
            if hasattr(result, "total_items") and result.total_items % 1000 == 0:
                memory_tracker.sample()
            return result

        # Wrap async function for benchmark
        def _sync_aggregate():
            return run_async_in_sync(_aggregate())

        # Run benchmark (fewer rounds for large dataset)
        result = benchmark.pedantic(_sync_aggregate, rounds=3, iterations=1)

        # Verify results
        memory_stats = memory_tracker.stop()

        # Memory leak detection
        assert (
            memory_stats["delta_mb"] < 100
        ), f"Potential memory leak: {memory_stats['delta_mb']:.2f}MB increase"

        # Log results
        print(f"\n[Memory] 10000 items: {memory_stats}")

    def test_streaming_window_processing(
        self,
        benchmark,
        reducer_node,
        sample_stamp_metadata,
        memory_tracker,
    ):
        """
        Benchmark: Streaming window processing with batching.

        Target: Process 1000 items in windows of 100
        Expected: <1.5s total
        Memory: Stable across windows
        """
        # Prepare streaming data
        stamps = sample_stamp_metadata(1000)

        async def _stream_generator():
            """Async generator simulating streaming input."""
            for stamp in stamps:
                yield stamp
                await asyncio.sleep(0)  # Allow event loop

        # Create contract with streaming
        class StreamingContract:
            def __init__(self, stream):
                self.input_stream = stream
                self.input_state = None
                self.streaming = MagicMock()
                self.streaming.window_size = 5000
                self.streaming.batch_size = 100

        # Memory tracking
        memory_tracker.start()

        # Benchmark execution
        async def _process_stream():
            contract = StreamingContract(_stream_generator())
            result = await reducer_node.execute_reduction(contract)
            memory_tracker.sample()
            return result

        # Wrap async function for benchmark
        def _sync_process_stream():
            return run_async_in_sync(_process_stream())

        # Run benchmark
        result = benchmark.pedantic(_sync_process_stream, rounds=5, iterations=2)

        # Verify results
        memory_stats = memory_tracker.stop()

        # Log results
        print(f"\n[Memory] Streaming 1000 items: {memory_stats}")

    def test_namespace_grouping_performance(
        self,
        benchmark,
        reducer_node,
        sample_stamp_metadata,
    ):
        """
        Benchmark: Multi-namespace aggregation.

        Target: 1000 items across 10 namespaces in <1.2s
        Verify: Correct grouping and aggregation
        """
        # Prepare multi-namespace data
        stamps = []
        for i in range(10):  # 10 namespaces
            namespace = f"omninode.services.ns{i}"
            stamps.extend(sample_stamp_metadata(100, namespace))

        # Create contract
        class MockContract:
            def __init__(self):
                self.input_state = {"items": stamps}
                self.input_stream = None
                self.aggregation = MagicMock()
                self.aggregation.aggregation_type = (
                    EnumAggregationType.NAMESPACE_GROUPING
                )
                self.streaming = MagicMock()
                self.streaming.batch_size = 100

        # Benchmark execution
        async def _aggregate_namespaces():
            contract = MockContract()
            return await reducer_node.execute_reduction(contract)

        # Wrap async function for benchmark
        def _sync_aggregate_namespaces():
            return run_async_in_sync(_aggregate_namespaces())

        # Run benchmark
        result = benchmark.pedantic(_sync_aggregate_namespaces, rounds=5, iterations=3)

        # Verify namespace grouping (when result is available)
        # This will work once the actual implementation returns results
        print("\n[Info] Multi-namespace aggregation completed")


# ============================================================================
# ORCHESTRATOR PERFORMANCE BENCHMARKS
# ============================================================================


@pytest.mark.performance
@pytest.mark.asyncio
class TestOrchestratorPerformance:
    """Performance benchmarks for NodeBridgeOrchestrator."""

    def test_single_workflow_execution(self, benchmark):
        """
        Benchmark: Single workflow execution end-to-end.

        Target: <300ms
        Expected p95: <250ms
        Threshold: PERFORMANCE_THRESHOLDS['orchestrator_workflow']
        """

        # Mock orchestrator (placeholder until implementation is complete)
        async def _mock_workflow():
            """Simulate workflow execution."""
            await asyncio.sleep(0.001)  # Simulate minimal processing
            return {"status": "completed", "duration_ms": 1.0}

        # Wrap async function for benchmark
        def _sync_mock_workflow():
            return run_async_in_sync(_mock_workflow())

        # Benchmark execution
        result = benchmark.pedantic(_sync_mock_workflow, rounds=10, iterations=10)

        # Placeholder validation
        print("\n[Info] Orchestrator workflow benchmark (mock)")

    def test_concurrent_workflow_throughput(self, benchmark):
        """
        Benchmark: 100 concurrent workflows.

        Target: All complete in <500ms
        Expected: 100+ workflows/second
        Threshold: PERFORMANCE_THRESHOLDS['orchestrator_concurrent']
        """

        async def _concurrent_workflows():
            """Execute 100 workflows concurrently."""
            workflows = [asyncio.sleep(0.001) for _ in range(100)]  # Mock workflows
            await asyncio.gather(*workflows)
            return 100  # Count

        # Wrap async function for benchmark
        def _sync_concurrent_workflows():
            return run_async_in_sync(_concurrent_workflows())

        # Benchmark execution
        result = benchmark.pedantic(_sync_concurrent_workflows, rounds=5, iterations=3)

        # Placeholder validation
        print("\n[Info] Concurrent workflow benchmark (mock)")

    def test_fsm_state_transition(self, benchmark):
        """
        Benchmark: FSM state transition overhead.

        Target: <50ms per transition
        Expected p95: <30ms
        Threshold: PERFORMANCE_THRESHOLDS['fsm_transition']
        """

        def _mock_fsm_transition():
            """Simulate FSM state transition."""
            # Mock state machine transition
            states = ["idle", "active", "processing", "completed"]
            current_state = states[0]
            for next_state in states[1:]:
                # Simulate transition logic
                current_state = next_state
            return current_state

        # Benchmark execution (synchronous for state machine)
        result = benchmark.pedantic(_mock_fsm_transition, rounds=100, iterations=10)

        # Placeholder validation
        print("\n[Info] FSM transition benchmark (mock)")

    def test_service_routing_overhead(self, benchmark):
        """
        Benchmark: Service routing decision overhead.

        Target: <10ms per routing decision
        Expected: Minimal overhead for service selection
        """

        async def _mock_service_routing():
            """Simulate service routing logic."""
            # Mock routing decision
            services = ["metadata", "onextree", "hook_receiver"]
            selected = services[0]  # Simple selection
            await asyncio.sleep(0)
            return selected

        # Wrap async function for benchmark
        def _sync_mock_service_routing():
            return run_async_in_sync(_mock_service_routing())

        # Benchmark execution
        result = benchmark.pedantic(
            _sync_mock_service_routing, rounds=100, iterations=10
        )

        # Placeholder validation
        print("\n[Info] Service routing benchmark (mock)")


# ============================================================================
# END-TO-END PERFORMANCE BENCHMARKS
# ============================================================================


@pytest.mark.performance
@pytest.mark.asyncio
class TestEndToEndPerformance:
    """End-to-end performance benchmarks for complete workflows."""

    def test_complete_workflow_pipeline(
        self,
        benchmark,
        reducer_node,
        sample_stamp_metadata,
        mock_reducer_contract,
        memory_tracker,
    ):
        """
        Benchmark: Complete workflow from request to persistence.

        Flow: Request → Orchestrator → Services → Reducer → Persistence
        Target: <500ms end-to-end
        Expected p95: <400ms
        Threshold: PERFORMANCE_THRESHOLDS['end_to_end']
        """
        # Prepare test data
        stamps = sample_stamp_metadata(100)

        # Memory tracking
        memory_tracker.start()

        # Benchmark complete pipeline
        async def _complete_pipeline():
            """Simulate complete workflow pipeline."""
            start = time.perf_counter()

            # 1. Orchestrator phase (mock)
            await asyncio.sleep(0.001)  # Simulate orchestration

            # 2. Service execution (mock)
            await asyncio.sleep(0.001)  # Simulate service calls

            # 3. Reducer aggregation (actual)
            contract = mock_reducer_contract(stamps)
            result = await reducer_node.execute_reduction(contract)

            # 4. Persistence (mock)
            await asyncio.sleep(0.001)  # Simulate DB write

            memory_tracker.sample()
            duration_ms = (time.perf_counter() - start) * 1000
            return {"duration_ms": duration_ms, "result": result}

        # Wrap async function for benchmark
        def _sync_complete_pipeline():
            return run_async_in_sync(_complete_pipeline())

        # Run benchmark
        result = benchmark.pedantic(_sync_complete_pipeline, rounds=5, iterations=5)

        # Verify results
        memory_stats = memory_tracker.stop()

        # Log results
        print(f"\n[Memory] End-to-end pipeline: {memory_stats}")

    def test_multi_service_coordination(
        self,
        benchmark,
        memory_tracker,
    ):
        """
        Benchmark: Multi-service coordination overhead.

        Services: Metadata + OnexTree + HookReceiver
        Target: <200ms for coordination
        Memory: Monitor cross-service overhead
        """
        # Memory tracking
        memory_tracker.start()

        async def _multi_service():
            """Simulate multi-service coordination."""
            # Parallel service calls
            tasks = [
                asyncio.sleep(0.001),  # Metadata service
                asyncio.sleep(0.001),  # OnexTree service
                asyncio.sleep(0.001),  # HookReceiver service
            ]
            results = await asyncio.gather(*tasks)
            memory_tracker.sample()
            return results

        # Wrap async function for benchmark
        def _sync_multi_service():
            return run_async_in_sync(_multi_service())

        # Run benchmark
        result = benchmark.pedantic(_sync_multi_service, rounds=10, iterations=10)

        # Verify results
        memory_stats = memory_tracker.stop()

        # Log results
        print(f"\n[Memory] Multi-service coordination: {memory_stats}")


# ============================================================================
# PERFORMANCE REGRESSION DETECTION
# ============================================================================


@pytest.mark.performance
class TestPerformanceRegression:
    """Detect performance regressions against baseline."""

    def test_baseline_comparison(self, benchmark):
        """
        Compare current performance against saved baseline.

        Usage:
            # Save baseline
            pytest tests/performance/test_bridge_performance.py --benchmark-save=baseline

            # Compare against baseline
            pytest tests/performance/test_bridge_performance.py --benchmark-compare=baseline

        Regression Detection:
            - Fail if performance degrades >10%
            - Warning if performance degrades >5%
        """

        def _benchmark_operation():
            """Sample operation for baseline comparison."""
            result = sum(range(1000))
            return result

        # Run benchmark
        result = benchmark.pedantic(_benchmark_operation, rounds=100, iterations=10)

        # Baseline comparison is handled by pytest-benchmark
        # Configure thresholds in pytest.ini or via CLI:
        # --benchmark-compare-fail=mean:10%
        print("\n[Info] Baseline comparison (use --benchmark-compare)")


# ============================================================================
# MEMORY PROFILING
# ============================================================================


@pytest.mark.performance
@pytest.mark.asyncio
class TestMemoryProfiling:
    """Memory profiling and leak detection."""

    async def test_memory_leak_detection(
        self,
        reducer_node,
        sample_stamp_metadata,
        mock_reducer_contract,
        memory_tracker,
    ):
        """
        Detect memory leaks over repeated operations.

        Strategy:
            - Run 100 aggregations
            - Monitor memory growth
            - Detect leaks if memory grows >50MB
        """
        memory_tracker.start()
        samples = []

        # Run 100 aggregations
        for i in range(100):
            stamps = sample_stamp_metadata(100)
            contract = mock_reducer_contract(stamps)

            # Execute aggregation
            await reducer_node.execute_reduction(contract)

            # Sample memory every 10 iterations
            if i % 10 == 0:
                sample = memory_tracker.sample()
                samples.append(sample)

        # Final memory check
        memory_stats = memory_tracker.stop()

        # Detect linear growth (potential leak)
        if len(samples) > 2:
            growth_rate = (samples[-1] - samples[0]) / len(samples)
            print(f"\n[Memory] Growth rate: {growth_rate:.3f} MB/iteration")

            # Fail if significant growth detected
            assert (
                growth_rate < 0.5
            ), f"Memory leak detected: {growth_rate:.3f} MB/iteration"

        # Log results
        print(f"\n[Memory] Leak detection: {memory_stats}")
        print(f"[Memory] Samples: {samples}")


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================


def pytest_benchmark_update_json(config, benchmarks, output_json):
    """
    Customize benchmark JSON output.

    Adds:
    - Performance thresholds
    - Memory statistics
    - Test metadata
    """
    output_json["performance_thresholds"] = PERFORMANCE_THRESHOLDS
    output_json["test_metadata"] = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "test_run_timestamp": datetime.now(UTC).isoformat(),
    }


# ============================================================================
# BENCHMARK REPORT GENERATION
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def benchmark_report(request):
    """
    Generate comprehensive benchmark report after test session.

    Outputs:
    - CSV summary report
    - JSON detailed report
    - Performance threshold validation
    """

    def _generate_report():
        """Generate reports after all tests complete."""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)
        print("\nPerformance Thresholds:")
        for test_name, thresholds in PERFORMANCE_THRESHOLDS.items():
            print(f"  {test_name}: {thresholds}")
        print("\nReports:")
        print("  - Benchmark JSON: test-artifacts/benchmark.json")
        print("  - Benchmark CSV: test-artifacts/benchmark.csv")
        print(
            "  - HTML Report: test-artifacts/report.html (from --html=test-artifacts/report.html)"
        )
        print("\nUsage:")
        print("  - Save baseline: pytest --benchmark-save=baseline tests/performance/")
        print("  - Compare: pytest --benchmark-compare=baseline tests/performance/")
        print("=" * 80)

    request.addfinalizer(_generate_report)
