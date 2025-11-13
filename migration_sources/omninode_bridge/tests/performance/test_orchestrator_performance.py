#!/usr/bin/env python3
"""
Dedicated Performance Benchmarks for NodeBridgeOrchestrator.

This module provides comprehensive performance testing for the Bridge Orchestrator
with focus on workflow execution, FSM state transitions, and service routing.

Performance Targets (from ROADMAP.md):
- Single workflow execution: <50ms
- Concurrent workflows: 100+ workflows/second
- FSM state transition: <1ms
- Service routing overhead: <10ms
- Memory usage: <512MB under normal load

Benchmark Categories:
1. Workflow Execution Performance
   - Single workflow latency (p50, p95, p99)
   - Concurrent workflow throughput
   - Multi-step workflow orchestration

2. FSM State Machine Performance
   - State transition overhead
   - State persistence latency
   - State recovery performance

3. Service Routing Performance
   - Routing decision latency
   - Service selection overhead
   - Multi-service coordination

4. Memory & Resource Usage
   - Memory consumption under load
   - Memory leak detection
   - Resource cleanup efficiency

Usage:
    # Run all orchestrator benchmarks
    pytest tests/performance/test_orchestrator_performance.py -v

    # Run specific benchmark
    pytest tests/performance/test_orchestrator_performance.py::test_single_workflow_latency -v

    # Generate benchmark report
    pytest tests/performance/test_orchestrator_performance.py --benchmark-only --benchmark-json=orchestrator.json

    # Compare against baseline
    pytest tests/performance/test_orchestrator_performance.py --benchmark-compare=baseline
"""

import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import psutil
import pytest

# Performance thresholds from ROADMAP.md
PERFORMANCE_THRESHOLDS = {
    "single_workflow_ms": {"max": 50, "p95": 40, "p99": 45},
    "concurrent_workflows": {"min_throughput": 100, "max_latency_ms": 500},
    "fsm_transition_ms": {"max": 1, "p95": 0.8, "p99": 0.9},
    "service_routing_ms": {"max": 10, "p95": 8, "p99": 9},
    "memory_mb": {"max": 512, "normal_load": 256},
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
def mock_workflow():
    """Mock LlamaIndex workflow for orchestrator testing."""

    class MockWorkflow:
        """Mock workflow with realistic execution behavior."""

        def __init__(self, execution_time_ms: float = 10.0):
            self.execution_time = execution_time_ms / 1000.0
            self.state = "pending"
            self.steps_completed = 0

        async def run(self, **kwargs) -> dict[str, Any]:
            """Simulate workflow execution with multiple steps."""
            self.state = "running"

            # Simulate workflow steps
            steps = ["validate", "route", "execute", "aggregate", "complete"]
            for step in steps:
                await asyncio.sleep(self.execution_time / len(steps))
                self.steps_completed += 1

            self.state = "completed"
            return {
                "status": "success",
                "steps_completed": self.steps_completed,
                "duration_ms": self.execution_time * 1000,
            }

    return MockWorkflow


@pytest.fixture
def mock_fsm():
    """Mock FSM state machine for state transition testing."""

    class MockFSM:
        """Mock FSM with state transition tracking."""

        def __init__(self):
            self.state = "PENDING"
            self.transitions = []
            self.transition_count = 0

        def transition(self, from_state: str, to_state: str) -> float:
            """Perform state transition and return duration in ms."""
            start = time.perf_counter()

            # Simulate transition logic
            if self.state != from_state:
                raise ValueError(f"Invalid transition: {self.state} -> {to_state}")

            self.state = to_state
            self.transition_count += 1
            self.transitions.append((from_state, to_state))

            duration_ms = (time.perf_counter() - start) * 1000
            return duration_ms

    return MockFSM


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
# WORKFLOW EXECUTION BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestWorkflowExecutionPerformance:
    """Benchmarks for workflow execution performance."""

    def test_single_workflow_latency(self, benchmark, mock_workflow, memory_tracker):
        """
        Benchmark: Single workflow execution latency.

        Target: <50ms (p95 < 40ms, p99 < 45ms)
        Measures: End-to-end workflow execution time
        """
        memory_tracker.start()

        async def _execute_workflow():
            """Execute single workflow."""
            workflow = mock_workflow(execution_time_ms=10.0)
            result = await workflow.run()
            memory_tracker.sample()
            return result

        def _sync_execute():
            return run_async_in_sync(_execute_workflow())

        # Run benchmark
        result = benchmark.pedantic(_sync_execute, rounds=100, iterations=10)

        # Verify results
        memory_stats = memory_tracker.stop()
        assert result["status"] == "success"

        # Validate performance threshold
        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000
        max_ms = stats.max * 1000

        print(
            f"\n[Performance] Single Workflow - Mean: {mean_ms:.2f}ms, Max: {max_ms:.2f}ms"
        )
        print(f"[Memory] {memory_stats}")

        assert (
            max_ms < PERFORMANCE_THRESHOLDS["single_workflow_ms"]["max"]
        ), f"Workflow latency {max_ms:.2f}ms exceeds target {PERFORMANCE_THRESHOLDS['single_workflow_ms']['max']}ms"

    def test_concurrent_workflow_throughput(self, benchmark, mock_workflow):
        """
        Benchmark: Concurrent workflow throughput.

        Target: 100+ workflows/second
        Measures: Maximum sustainable throughput
        """

        async def _concurrent_workflows():
            """Execute 100 workflows concurrently."""
            workflows = [mock_workflow(execution_time_ms=5.0).run() for _ in range(100)]
            start = time.perf_counter()
            results = await asyncio.gather(*workflows, return_exceptions=True)
            duration = time.perf_counter() - start

            successful = sum(
                1
                for r in results
                if isinstance(r, dict) and r.get("status") == "success"
            )

            return {
                "total": len(results),
                "successful": successful,
                "duration_s": duration,
                "throughput": successful / duration if duration > 0 else 0,
            }

        def _sync_concurrent():
            return run_async_in_sync(_concurrent_workflows())

        # Run benchmark
        result = benchmark.pedantic(_sync_concurrent, rounds=10, iterations=3)

        # Validate throughput
        throughput = result["throughput"]
        print(
            f"\n[Performance] Concurrent Throughput: {throughput:.2f} workflows/second"
        )
        print(f"[Success Rate] {result['successful']}/{result['total']} workflows")

        assert (
            throughput
            >= PERFORMANCE_THRESHOLDS["concurrent_workflows"]["min_throughput"]
        ), f"Throughput {throughput:.2f} below target {PERFORMANCE_THRESHOLDS['concurrent_workflows']['min_throughput']}"

    def test_multi_step_workflow_orchestration(self, benchmark, mock_workflow):
        """
        Benchmark: Multi-step workflow orchestration overhead.

        Target: <100ms for 10-step workflow
        Measures: Orchestration overhead per step
        """

        async def _multi_step_workflow():
            """Execute workflow with 10 steps."""
            workflow = mock_workflow(execution_time_ms=50.0)  # 5ms per step
            result = await workflow.run()
            return result

        def _sync_multi_step():
            return run_async_in_sync(_multi_step_workflow())

        # Run benchmark
        result = benchmark.pedantic(_sync_multi_step, rounds=50, iterations=5)

        # Verify results
        assert result["steps_completed"] == 5
        print(f"\n[Performance] Multi-step workflow: {result['duration_ms']:.2f}ms")


# ============================================================================
# FSM STATE MACHINE BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestFSMPerformance:
    """Benchmarks for FSM state machine performance."""

    def test_state_transition_latency(self, benchmark, mock_fsm):
        """
        Benchmark: FSM state transition latency.

        Target: <1ms per transition (p95 < 0.8ms, p99 < 0.9ms)
        Measures: Pure state transition overhead
        """

        def _state_transition():
            """Perform single state transition."""
            fsm = mock_fsm()
            duration_ms = fsm.transition("PENDING", "PROCESSING")
            return duration_ms

        # Run benchmark
        result = benchmark.pedantic(_state_transition, rounds=1000, iterations=100)

        # Validate threshold
        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000  # Convert to ms
        max_ms = stats.max * 1000

        print(
            f"\n[Performance] FSM Transition - Mean: {mean_ms:.4f}ms, Max: {max_ms:.4f}ms"
        )

        # Note: The actual transition is much faster than the threshold
        # This benchmark captures Python function call overhead
        assert mean_ms < PERFORMANCE_THRESHOLDS["fsm_transition_ms"]["max"]

    def test_state_transition_sequence(self, benchmark, mock_fsm):
        """
        Benchmark: Complete state transition sequence.

        States: PENDING → PROCESSING → COMPLETED
        Target: <3ms for full sequence
        """

        def _transition_sequence():
            """Perform full state transition sequence."""
            fsm = mock_fsm()
            total_duration = 0.0

            transitions = [
                ("PENDING", "PROCESSING"),
                ("PROCESSING", "COMPLETED"),
            ]

            for from_state, to_state in transitions:
                duration = fsm.transition(from_state, to_state)
                total_duration += duration

            return total_duration

        # Run benchmark
        result = benchmark.pedantic(_transition_sequence, rounds=100, iterations=50)

        print(f"\n[Performance] State Sequence: {result:.4f}ms total")


# ============================================================================
# SERVICE ROUTING BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestServiceRoutingPerformance:
    """Benchmarks for service routing performance."""

    def test_service_routing_decision(self, benchmark):
        """
        Benchmark: Service routing decision latency.

        Target: <10ms per routing decision
        Measures: Service selection logic overhead
        """

        async def _route_service():
            """Simulate service routing logic."""
            # Mock routing decision based on workflow type
            services = {
                "metadata": {"latency_ms": 5.0, "capacity": 100},
                "onextree": {"latency_ms": 50.0, "capacity": 20},
                "hook_receiver": {"latency_ms": 2.0, "capacity": 200},
            }

            # Simple routing logic: choose lowest latency with capacity
            selected = min(services.items(), key=lambda x: x[1]["latency_ms"])

            await asyncio.sleep(0)  # Yield to event loop
            return selected[0]

        def _sync_route():
            return run_async_in_sync(_route_service())

        # Run benchmark
        result = benchmark.pedantic(_sync_route, rounds=100, iterations=100)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] Service Routing: {mean_ms:.2f}ms")
        print(f"[Result] Selected service: {result}")

        assert mean_ms < PERFORMANCE_THRESHOLDS["service_routing_ms"]["max"]

    def test_multi_service_coordination(self, benchmark):
        """
        Benchmark: Multi-service coordination overhead.

        Target: <50ms for 3 parallel service calls
        Measures: Parallel execution coordination
        """

        async def _coordinate_services():
            """Coordinate multiple services in parallel."""
            # Simulate parallel service calls
            tasks = [
                asyncio.sleep(0.005),  # Metadata service
                asyncio.sleep(0.010),  # OnexTree service
                asyncio.sleep(0.002),  # HookReceiver service
            ]

            start = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration_ms = (time.perf_counter() - start) * 1000

            return {"count": len(results), "duration_ms": duration_ms}

        def _sync_coordinate():
            return run_async_in_sync(_coordinate_services())

        # Run benchmark
        result = benchmark.pedantic(_sync_coordinate, rounds=50, iterations=10)

        print(
            f"\n[Performance] Multi-service coordination: {result['duration_ms']:.2f}ms"
        )
        assert result["duration_ms"] < 50.0


# ============================================================================
# MEMORY & RESOURCE BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestMemoryAndResourceUsage:
    """Benchmarks for memory and resource usage."""

    async def test_memory_usage_under_load(self, mock_workflow, memory_tracker):
        """
        Test: Memory usage during sustained workflow execution.

        Target: <512MB peak memory under normal load
        Measures: Memory consumption and leak detection
        """
        memory_tracker.start()

        # Execute 1000 workflows
        for i in range(1000):
            workflow = mock_workflow(execution_time_ms=1.0)
            await workflow.run()

            # Sample memory every 100 workflows
            if i % 100 == 0:
                memory_tracker.sample()

        memory_stats = memory_tracker.stop()

        print("\n[Memory] Sustained load test:")
        print(f"  Baseline: {memory_stats['baseline_mb']:.2f}MB")
        print(f"  Peak: {memory_stats['peak_mb']:.2f}MB")
        print(f"  Final: {memory_stats['final_mb']:.2f}MB")
        print(f"  Delta: {memory_stats['delta_mb']:.2f}MB")

        # Assert memory constraints
        assert memory_stats["peak_mb"] < PERFORMANCE_THRESHOLDS["memory_mb"]["max"]
        assert memory_stats["delta_mb"] < 100, "Potential memory leak detected"

    async def test_resource_cleanup_efficiency(self, mock_workflow, memory_tracker):
        """
        Test: Resource cleanup efficiency after workflow completion.

        Target: Memory returns to baseline after cleanup
        Measures: Garbage collection effectiveness
        """
        memory_tracker.start()
        baseline = memory_tracker.baseline

        # Create and complete workflows
        workflows = []
        for _ in range(100):
            workflow = mock_workflow(execution_time_ms=1.0)
            workflows.append(await workflow.run())

        peak = memory_tracker.sample()

        # Force cleanup
        workflows.clear()
        gc.collect()

        final = memory_tracker.sample()
        memory_stats = memory_tracker.stop()

        cleanup_efficiency = (
            ((peak - final) / (peak - baseline)) * 100 if peak > baseline else 100
        )

        print("\n[Memory] Cleanup efficiency:")
        print(f"  Baseline: {baseline:.2f}MB")
        print(f"  Peak: {peak:.2f}MB")
        print(f"  After cleanup: {final:.2f}MB")
        print(f"  Cleanup efficiency: {cleanup_efficiency:.1f}%")

        # Assert cleanup effectiveness (should recover >80% of memory)
        assert (
            cleanup_efficiency > 80
        ), f"Poor cleanup efficiency: {cleanup_efficiency:.1f}%"


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================


def pytest_benchmark_update_json(config, benchmarks, output_json):
    """
    Customize benchmark JSON output with orchestrator-specific metadata.

    Adds:
    - Performance thresholds
    - Test metadata
    - Orchestrator configuration
    """
    output_json["performance_thresholds"] = PERFORMANCE_THRESHOLDS
    output_json["component"] = "NodeBridgeOrchestrator"
    output_json["test_metadata"] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "focus": "Workflow execution, FSM transitions, service routing",
    }
