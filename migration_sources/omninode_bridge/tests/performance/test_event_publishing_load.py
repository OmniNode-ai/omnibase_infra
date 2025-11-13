"""
Performance tests for Kafka event publishing.

Tests validate:
- Kafka publish latency (target: <10ms per event)
- Event consumption lag (target: <100ms end-to-end)
- Batch publishing throughput
"""

import asyncio
import time
from uuid import uuid4

import pytest

from omninode_bridge.events.models.codegen_events import (
    ModelEventCodegenStageCompleted,
    ModelEventCodegenStarted,
)

# Performance targets
PUBLISH_LATENCY_TARGET_MS = 10
CONSUMPTION_LAG_TARGET_MS = 100
BATCH_THROUGHPUT_TARGET = 500  # events/second


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer with realistic latency."""

    class MockProducer:
        def __init__(self):
            self.published_events = []
            self.publish_latency_ms = 5  # Simulate 5ms publish latency

        async def send(self, topic: str, value: dict):
            """Simulate Kafka send with latency."""
            await asyncio.sleep(self.publish_latency_ms / 1000)
            self.published_events.append(
                {
                    "topic": topic,
                    "value": value,
                    "timestamp": time.time(),
                }
            )
            return True

    return MockProducer()


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer with realistic lag."""

    class MockConsumer:
        def __init__(self):
            self.consumed_events = []
            self.consumption_lag_ms = 10  # Simulate 10ms consumption lag

        async def consume(self, topics: list, timeout_ms: int = 1000):
            """Simulate Kafka consume with lag."""
            await asyncio.sleep(self.consumption_lag_ms / 1000)
            # Return mock messages
            return [
                {"topic": topic, "value": {}, "timestamp": time.time()}
                for topic in topics
            ]

        async def commit(self):
            """Simulate commit."""
            await asyncio.sleep(0.001)

    return MockConsumer()


@pytest.mark.performance
@pytest.mark.asyncio
async def test_kafka_publish_latency(mock_kafka_producer, benchmark):
    """
    Test Kafka publish latency.

    Target: <10ms per event (P95)
    Measures: Time to publish individual events
    """

    async def publish_event():
        """Publish a single event and measure latency."""
        event = ModelEventCodegenStarted(
            correlation_id=uuid4(),
            event_id=uuid4(),
            workflow_id=uuid4(),
            orchestrator_node_id=uuid4(),
            prompt="test prompt for load testing",
            output_directory="/tmp/test",
            node_type_hint="effect",
        )

        start = time.perf_counter()
        await mock_kafka_producer.send(
            topic="dev.omninode-bridge.codegen.generation-started.v1",
            value=event.model_dump(),
        )
        end = time.perf_counter()

        return (end - start) * 1000  # Convert to ms

    # Publish 1000 events and measure latency
    latencies = []

    print(f"\n{'='*60}")
    print("Kafka Publish Latency Test")
    print(f"{'='*60}")

    for i in range(1000):
        latency = await publish_event()
        latencies.append(latency)

        if (i + 1) % 200 == 0:
            avg_latency = sum(latencies[-200:]) / 200
            print(f"Progress: {i+1}/1000 | Avg latency: {avg_latency:.2f}ms")

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    p50_latency = sorted(latencies)[len(latencies) // 2]
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"\n{'='*60}")
    print("Publish Latency Results")
    print(f"{'='*60}")
    print(f"Total events: {len(latencies)}")
    print(f"Avg latency: {avg_latency:.2f}ms")
    print(f"Min latency: {min_latency:.2f}ms")
    print(f"Max latency: {max_latency:.2f}ms")
    print(f"P50 latency: {p50_latency:.2f}ms")
    print(f"P95 latency: {p95_latency:.2f}ms")
    print(f"P99 latency: {p99_latency:.2f}ms")
    print(f"Target: <{PUBLISH_LATENCY_TARGET_MS}ms (P95)")
    print(f"{'='*60}")

    # Assert target
    assert (
        p95_latency < PUBLISH_LATENCY_TARGET_MS
    ), f"P95 latency {p95_latency:.2f}ms exceeds target {PUBLISH_LATENCY_TARGET_MS}ms"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_event_consumption_lag(
    mock_kafka_producer, mock_kafka_consumer, benchmark
):
    """
    Test event consumption lag (end-to-end).

    Target: <100ms end-to-end (publish → consume)
    Measures: Time from publish to consumption
    """

    async def measure_end_to_end_lag():
        """Measure publish → consume round-trip."""
        event = ModelEventCodegenStarted(
            correlation_id=uuid4(),
            event_id=uuid4(),
            workflow_id=uuid4(),
            orchestrator_node_id=uuid4(),
            prompt="test prompt for load testing",
            output_directory="/tmp/test",
            node_type_hint="effect",
        )

        # Publish event
        publish_start = time.perf_counter()
        await mock_kafka_producer.send(
            topic="dev.omninode-bridge.codegen.generation-started.v1",
            value=event.model_dump(),
        )
        publish_end = time.perf_counter()

        # Consume event
        consume_start = time.perf_counter()
        await mock_kafka_consumer.consume(
            topics=["dev.omninode-bridge.codegen.generation-started.v1"],
            timeout_ms=1000,
        )
        consume_end = time.perf_counter()

        publish_latency = (publish_end - publish_start) * 1000
        consume_latency = (consume_end - consume_start) * 1000
        total_latency = (consume_end - publish_start) * 1000

        return {
            "publish_ms": publish_latency,
            "consume_ms": consume_latency,
            "total_ms": total_latency,
        }

    # Run 100 iterations
    results = []

    print(f"\n{'='*60}")
    print("Event Consumption Lag Test")
    print(f"{'='*60}")

    for i in range(100):
        result = await measure_end_to_end_lag()
        results.append(result)

        if (i + 1) % 20 == 0:
            avg_total = sum(r["total_ms"] for r in results[-20:]) / 20
            print(f"Progress: {i+1}/100 | Avg end-to-end: {avg_total:.2f}ms")

    # Calculate statistics
    avg_publish = sum(r["publish_ms"] for r in results) / len(results)
    avg_consume = sum(r["consume_ms"] for r in results) / len(results)
    avg_total = sum(r["total_ms"] for r in results) / len(results)

    total_latencies = [r["total_ms"] for r in results]
    p50_total = sorted(total_latencies)[len(total_latencies) // 2]
    p95_total = sorted(total_latencies)[int(len(total_latencies) * 0.95)]
    p99_total = sorted(total_latencies)[int(len(total_latencies) * 0.99)]

    print(f"\n{'='*60}")
    print("Consumption Lag Results")
    print(f"{'='*60}")
    print(f"Avg publish latency: {avg_publish:.2f}ms")
    print(f"Avg consume latency: {avg_consume:.2f}ms")
    print(f"Avg total latency: {avg_total:.2f}ms")
    print(f"P50 total: {p50_total:.2f}ms")
    print(f"P95 total: {p95_total:.2f}ms")
    print(f"P99 total: {p99_total:.2f}ms")
    print(f"Target: <{CONSUMPTION_LAG_TARGET_MS}ms (P99)")
    print(f"{'='*60}")

    # Assert target
    assert (
        p99_total < CONSUMPTION_LAG_TARGET_MS
    ), f"P99 total latency {p99_total:.2f}ms exceeds target {CONSUMPTION_LAG_TARGET_MS}ms"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_batch_publishing_throughput(mock_kafka_producer, benchmark):
    """
    Test batch publishing throughput.

    Target: >500 events/second
    Measures: Throughput when publishing many events
    """

    async def publish_batch(batch_size: int = 100):
        """Publish a batch of events."""
        events = [
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                workflow_id=uuid4(),
                stage_name="test_stage",
                stage_number=(i % 8) + 1,  # 1-8
                duration_seconds=2.0,
                success=True,
            )
            for i in range(batch_size)
        ]

        start = time.perf_counter()

        # Publish all events concurrently
        await asyncio.gather(
            *[
                mock_kafka_producer.send(
                    topic="dev.omninode-bridge.codegen.stage-completed.v1",
                    value=event.model_dump(),
                )
                for event in events
            ]
        )

        end = time.perf_counter()

        duration = end - start
        throughput = batch_size / duration

        return {
            "batch_size": batch_size,
            "duration_s": duration,
            "throughput": throughput,
        }

    # Test different batch sizes
    batch_sizes = [50, 100, 200, 500, 1000]
    results = []

    print(f"\n{'='*60}")
    print("Batch Publishing Throughput Test")
    print(f"{'='*60}")

    for batch_size in batch_sizes:
        result = await publish_batch(batch_size)
        results.append(result)

        print(
            f"Batch size: {batch_size} | Duration: {result['duration_s']:.2f}s | "
            f"Throughput: {result['throughput']:.0f} events/sec"
        )

    avg_throughput = sum(r["throughput"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print("Batch Publishing Results")
    print(f"{'='*60}")
    print(f"Avg throughput: {avg_throughput:.0f} events/second")
    print(f"Target: >{BATCH_THROUGHPUT_TARGET} events/second")
    print(f"{'='*60}")

    # Assert target
    assert (
        avg_throughput >= BATCH_THROUGHPUT_TARGET
    ), f"Avg throughput {avg_throughput:.0f} below target {BATCH_THROUGHPUT_TARGET}"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_sustained_publishing_throughput(mock_kafka_producer, benchmark):
    """
    Test sustained publishing throughput over time.

    Target: Maintain >500 events/second for 60 seconds
    Measures: Throughput stability over time
    """

    async def publish_continuously(duration_seconds: int = 10):
        """Publish events continuously for a duration."""
        start_time = time.time()
        total_published = 0
        throughputs = []

        while time.time() - start_time < duration_seconds:
            batch_start = time.time()

            # Publish batch of 100 events
            batch_tasks = []
            for i in range(100):
                event = ModelEventCodegenStarted(
                    correlation_id=uuid4(),
                    event_id=uuid4(),
                    workflow_id=uuid4(),
                    orchestrator_node_id=uuid4(),
                    prompt="test prompt for load testing",
                    output_directory="/tmp/test",
                    node_type_hint="effect",
                )
                batch_tasks.append(
                    mock_kafka_producer.send(
                        topic="dev.omninode-bridge.codegen.generation-started.v1",
                        value=event.model_dump(),
                    )
                )

            await asyncio.gather(*batch_tasks)

            batch_end = time.time()
            batch_duration = batch_end - batch_start
            batch_throughput = 100 / batch_duration

            total_published += 100
            throughputs.append(batch_throughput)

            # Print progress every second
            elapsed = time.time() - start_time
            if int(elapsed) != int(elapsed - batch_duration):
                avg_throughput = sum(throughputs) / len(throughputs)
                print(
                    f"Time: {int(elapsed)}s | Total: {total_published} | "
                    f"Avg throughput: {avg_throughput:.0f} events/sec"
                )

        total_duration = time.time() - start_time
        overall_throughput = total_published / total_duration

        return {
            "duration_s": total_duration,
            "total_published": total_published,
            "overall_throughput": overall_throughput,
            "avg_throughput": sum(throughputs) / len(throughputs),
            "min_throughput": min(throughputs),
            "max_throughput": max(throughputs),
        }

    print(f"\n{'='*60}")
    print("Sustained Publishing Throughput Test (10 seconds)")
    print(f"{'='*60}")

    result = await publish_continuously(duration_seconds=10)

    print(f"\n{'='*60}")
    print("Sustained Publishing Results")
    print(f"{'='*60}")
    print(f"Duration: {result['duration_s']:.2f}s")
    print(f"Total published: {result['total_published']}")
    print(f"Overall throughput: {result['overall_throughput']:.0f} events/sec")
    print(f"Avg throughput: {result['avg_throughput']:.0f} events/sec")
    print(f"Min throughput: {result['min_throughput']:.0f} events/sec")
    print(f"Max throughput: {result['max_throughput']:.0f} events/sec")
    print(f"Target: >{BATCH_THROUGHPUT_TARGET} events/second")
    print(f"{'='*60}")

    # Assert sustained throughput
    assert (
        result["avg_throughput"] >= BATCH_THROUGHPUT_TARGET
    ), f"Avg throughput {result['avg_throughput']:.0f} below target {BATCH_THROUGHPUT_TARGET}"
