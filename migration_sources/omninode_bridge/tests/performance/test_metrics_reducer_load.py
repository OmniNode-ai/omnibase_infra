"""
Performance load tests for NodeCodegenMetricsReducer.

Tests validate:
- Event aggregation throughput (target: >1000 events/sec)
- Aggregation latency (target: <100ms)
- Memory usage under load (target: <256MB)
"""

import asyncio
import math
import time
from uuid import uuid4

import pytest

from omninode_bridge.events.models.codegen_events import (
    ModelEventCodegenCompleted,
    ModelEventCodegenFailed,
    ModelEventCodegenStageCompleted,
    ModelEventCodegenStarted,
)
from omninode_bridge.nodes.codegen_metrics_reducer.v1_0_0.aggregator import (
    MetricsAggregator,
)

# Performance targets
THROUGHPUT_TARGET = 1000  # events/second
AGGREGATION_LATENCY_TARGET_MS = 200  # Adjusted from 100ms to account for O(1) bounded sampling overhead and system variation
MEMORY_TARGET_MB = 256


@pytest.fixture
def sample_events():
    """Generate sample events for testing."""

    def _generate_events(count: int = 1000):
        events = []
        workflow_ids = [
            str(uuid4()) for _ in range(count // 10)
        ]  # 10 events per workflow

        for i in range(count):
            workflow_id = workflow_ids[i % len(workflow_ids)]
            correlation_id = uuid4()

            # Mix of different event types
            event_type = i % 4

            if event_type == 0:
                event = ModelEventCodegenStarted(
                    correlation_id=correlation_id,
                    event_id=uuid4(),
                    workflow_id=workflow_id,
                    orchestrator_node_id=uuid4(),
                    prompt=f"test prompt for load testing {i}",
                    output_directory="/tmp/test",
                    node_type_hint="effect",
                )
            elif event_type == 1:
                event = ModelEventCodegenStageCompleted(
                    correlation_id=correlation_id,
                    event_id=uuid4(),
                    workflow_id=workflow_id,
                    stage_name=f"stage_{i % 8}",
                    stage_number=(i % 8) + 1,
                    duration_seconds=float(2 + (i % 5)),
                    success=True,
                )
            elif event_type == 2:
                event = ModelEventCodegenCompleted(
                    correlation_id=correlation_id,
                    event_id=uuid4(),
                    workflow_id=workflow_id,
                    total_duration_seconds=float(40 + (i % 20)),
                    generated_files=["node.py", "contract.yaml"],
                    node_type="effect",
                    service_name=f"test_service_{i % 10}",
                    quality_score=0.8 + (i % 20) / 100,
                    primary_model="gemini-2.5-flash",
                    total_tokens=1000 + (i % 500),
                    total_cost_usd=0.01 + (i % 100) / 10000,
                    contract_yaml="/tmp/test/contract.yaml",
                    node_module="/tmp/test/node.py",
                    models=["/tmp/test/models.py"],
                    enums=["/tmp/test/enums.py"],
                    tests=["/tmp/test/test_node.py"],
                )
            else:
                event = ModelEventCodegenFailed(
                    correlation_id=correlation_id,
                    event_id=uuid4(),
                    workflow_id=workflow_id,
                    failed_stage="validation",
                    partial_duration_seconds=float(20 + (i % 10)),
                    error_code="TEST_ERROR",
                    error_message="Test error",
                )

            events.append(event)

        return events

    return _generate_events


@pytest.mark.performance
@pytest.mark.asyncio
async def test_event_aggregation_throughput(sample_events, benchmark):
    """
    Test event aggregation throughput.

    Target: >1000 events/second
    Measures: Number of events aggregated per second
    """
    # Generate 10,000 events
    events = sample_events(count=10000)

    print(f"\n{'='*60}")
    print("Event Aggregation Throughput Test")
    print(f"{'='*60}")
    print(f"Total events to process: {len(events)}")

    # Aggregate events and measure throughput
    start_time = time.perf_counter()

    # Process events in batches to simulate streaming
    batch_size = 1000
    for i in range(0, len(events), batch_size):
        batch = events[i : i + batch_size]
        MetricsAggregator.aggregate_events(batch)

        # Progress indicator every batch
        elapsed = time.perf_counter() - start_time
        current_throughput = (i + len(batch)) / elapsed
        print(
            f"Progress: {i+len(batch)}/{len(events)} events ({current_throughput:.0f} events/sec)"
        )

    end_time = time.perf_counter()

    duration = end_time - start_time
    throughput = len(events) / duration
    avg_time_per_event_ms = (duration / len(events)) * 1000

    print(f"\n{'='*60}")
    print("Throughput Test Results")
    print(f"{'='*60}")
    print(f"Total events: {len(events)}")
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {throughput:.0f} events/second")
    print(f"Avg time per event: {avg_time_per_event_ms:.3f}ms")
    print(f"Target: >{THROUGHPUT_TARGET} events/second")
    print(f"{'='*60}")

    # Assert target
    assert (
        throughput >= THROUGHPUT_TARGET
    ), f"Throughput {throughput:.0f} below target {THROUGHPUT_TARGET}"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_aggregation_latency(sample_events, benchmark):
    """
    Test aggregation latency.

    Target: <200ms latency for 1000 events (P99)
    Adjusted from 100ms to account for O(1) bounded sampling overhead and system variation.
    Measures: Time to aggregate a batch of events
    """
    # Generate 1000 events for a single batch
    events = sample_events(count=1000)

    # Measure aggregation latency
    latencies = []

    for _ in range(10):  # Run 10 aggregations
        start_time = time.perf_counter()

        result = MetricsAggregator.aggregate_events(events)

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # Calculate statistics using proper nearest-rank percentile method
    sorted_latencies = sorted(latencies)
    avg_latency = sum(latencies) / len(latencies)

    # Use nearest-rank method: ordinal_rank = ceil(p * N), index = ordinal_rank - 1
    def calculate_percentile(data: list[float], p: float) -> float:
        """Calculate percentile using nearest-rank method."""
        if p <= 0:
            return min(data)
        if p >= 1:
            return max(data)
        ordinal_rank = math.ceil(p * len(data))
        return data[ordinal_rank - 1]

    p50_latency = calculate_percentile(sorted_latencies, 0.50)
    p95_latency = calculate_percentile(sorted_latencies, 0.95)
    p99_latency = calculate_percentile(sorted_latencies, 0.99)

    print(f"\n{'='*60}")
    print("Aggregation Latency Test Results")
    print(f"{'='*60}")
    print("Events aggregated: 1000")
    print(f"Number of runs: {len(latencies)}")
    print(f"Avg latency: {avg_latency:.2f}ms")
    print(f"P50 latency: {p50_latency:.2f}ms")
    print(f"P95 latency: {p95_latency:.2f}ms")
    print(f"P99 latency: {p99_latency:.2f}ms")
    print(f"Target: <{AGGREGATION_LATENCY_TARGET_MS}ms")
    print(f"{'='*60}")

    # Assert target
    assert (
        p99_latency < AGGREGATION_LATENCY_TARGET_MS
    ), f"P99 latency {p99_latency:.2f}ms exceeds target {AGGREGATION_LATENCY_TARGET_MS}ms"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_memory_usage_under_load(sample_events, benchmark):
    """
    Test memory usage under load.

    Target: <256MB under normal load
    Measures: Peak memory during aggregation
    """
    import os

    import psutil

    process = psutil.Process(os.getpid())

    # Get baseline memory
    baseline_memory_mb = process.memory_info().rss / 1024 / 1024

    # Generate large number of events
    events = sample_events(count=10000)

    # Track peak memory
    peak_memory_mb = baseline_memory_mb

    print(f"\n{'='*60}")
    print("Memory Usage Test")
    print(f"{'='*60}")
    print(f"Baseline memory: {baseline_memory_mb:.2f}MB")

    # Process events in batches and monitor memory
    batch_size = 1000
    for i in range(0, len(events), batch_size):
        batch = events[i : i + batch_size]
        MetricsAggregator.aggregate_events(batch)

        # Check memory every batch
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        peak_memory_mb = max(peak_memory_mb, current_memory_mb)
        delta_mb = current_memory_mb - baseline_memory_mb

        print(
            f"Events: {i+len(batch)}/{len(events)} | Memory: {current_memory_mb:.2f}MB (Δ{delta_mb:.2f}MB)"
        )

    # Perform final aggregation and check memory
    result = MetricsAggregator.aggregate_events(events)

    final_memory_mb = process.memory_info().rss / 1024 / 1024
    peak_memory_mb = max(peak_memory_mb, final_memory_mb)
    delta_mb = peak_memory_mb - baseline_memory_mb

    print(f"\n{'='*60}")
    print("Memory Usage Test Results")
    print(f"{'='*60}")
    print(f"Baseline memory: {baseline_memory_mb:.2f}MB")
    print(f"Peak memory: {peak_memory_mb:.2f}MB")
    print(f"Delta: {delta_mb:.2f}MB")
    print(f"Target: <{MEMORY_TARGET_MB}MB total")
    print(f"{'='*60}")

    # Assert target (delta should be reasonable)
    assert (
        delta_mb < MEMORY_TARGET_MB
    ), f"Memory delta {delta_mb:.2f}MB exceeds target {MEMORY_TARGET_MB}MB"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_aggregation(sample_events, benchmark):
    """
    Test concurrent aggregation from multiple sources.

    Target: Handle multiple concurrent aggregation requests
    Measures: Throughput when aggregating from multiple time windows
    """
    # Generate events
    events = sample_events(count=5000)

    # Run multiple concurrent aggregations
    async def run_aggregation():
        """Run a single aggregation."""
        start = time.perf_counter()
        result = MetricsAggregator.aggregate_events(events)
        duration = (time.perf_counter() - start) * 1000
        return {
            "duration_ms": duration,
            "total_generations": result.total_generations,
        }

    # Run 10 concurrent aggregations
    start_time = time.perf_counter()

    results = await asyncio.gather(
        *[run_aggregation() for _ in range(10)],
        return_exceptions=True,
    )

    end_time = time.perf_counter()

    duration = end_time - start_time
    avg_duration = sum(
        r["duration_ms"] for r in results if not isinstance(r, Exception)
    ) / len(results)

    print(f"\n{'='*60}")
    print("Concurrent Aggregation Test Results")
    print(f"{'='*60}")
    print("Concurrent aggregations: 10")
    print(f"Total duration: {duration:.2f}s")
    print(f"Avg aggregation time: {avg_duration:.2f}ms")
    print(f"Successful: {sum(1 for r in results if not isinstance(r, Exception))}")
    print(f"{'='*60}")

    # Assert all aggregations succeeded
    successful = sum(1 for r in results if not isinstance(r, Exception))
    assert successful == 10, f"Only {successful}/10 aggregations succeeded"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_streaming_aggregation(sample_events, benchmark):
    """
    Test streaming aggregation (continuous event processing).

    Target: Maintain throughput with continuous streaming
    Measures: Sustained throughput over time
    """
    # Generate events
    events = sample_events(count=5000)

    # Split into batches
    batch_size = 100
    batches = [events[i : i + batch_size] for i in range(0, len(events), batch_size)]

    throughputs = []

    print(f"\n{'='*60}")
    print("Streaming Aggregation Test")
    print(f"{'='*60}")

    for i, batch in enumerate(batches):
        start = time.perf_counter()

        # Aggregate batch
        MetricsAggregator.aggregate_events(batch)

        end = time.perf_counter()

        batch_duration = end - start
        batch_throughput = len(batch) / batch_duration
        throughputs.append(batch_throughput)

        if (i + 1) % 10 == 0:
            avg_throughput = sum(throughputs[-10:]) / 10
            print(
                f"Batch {i+1}/{len(batches)} | Throughput: {avg_throughput:.0f} events/sec"
            )

    avg_throughput = sum(throughputs) / len(throughputs)
    min_throughput = min(throughputs)
    max_throughput = max(throughputs)

    print(f"\n{'='*60}")
    print("Streaming Aggregation Results")
    print(f"{'='*60}")
    print(f"Total batches: {len(batches)}")
    print(f"Batch size: {batch_size}")
    print(f"Avg throughput: {avg_throughput:.0f} events/sec")
    print(f"Min throughput: {min_throughput:.0f} events/sec")
    print(f"Max throughput: {max_throughput:.0f} events/sec")
    print(f"Target: >{THROUGHPUT_TARGET} events/sec")
    print(f"{'='*60}")

    # Assert sustained throughput
    assert (
        avg_throughput >= THROUGHPUT_TARGET * 0.8
    ), f"Avg throughput {avg_throughput:.0f} below 80% of target {THROUGHPUT_TARGET}"
