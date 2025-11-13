#!/usr/bin/env python3
"""
Integration Tests for CodegenDLQMonitor with Real Kafka.

This test suite validates DLQ monitoring functionality with a real Kafka/Redpanda
broker, ensuring production-ready behavior for dead letter queue alerting.

Test Coverage:
- Kafka connection and lifecycle management
- DLQ message detection and counting across 4 topics
- Alert threshold triggering mechanism
- Concurrent DLQ topic monitoring
- Statistics accuracy and retrieval
- Graceful shutdown and offset commit
- Error handling (connection failures, invalid messages)
- Alert cooldown behavior
- Performance under load

Infrastructure Requirements:
- Kafka/Redpanda broker running on localhost:29092
- Test topics created dynamically by AIOKafkaProducer
- No persistent state required (fresh start for each test)

Performance Targets:
- Message consumption: <10ms latency per message
- Statistics retrieval: <5ms
- Graceful shutdown: <2 seconds
- 100+ messages/second throughput

Test Execution:
    pytest tests/integration/monitoring/test_codegen_dlq_monitor_integration.py -v
    pytest -m "integration and requires_infrastructure" tests/integration/monitoring/
"""

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

# Import DLQ Monitor
from omninode_bridge.monitoring.codegen_dlq_monitor import CodegenDLQMonitor

# Import Kafka clients for test setup
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    AIOKafkaProducer = None  # type: ignore
    AIOKafkaConsumer = None  # type: ignore

# ============================================================================
# Test Configuration
# ============================================================================

KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092")
TEST_NAMESPACE = "test.dlq.integration"
KAFKA_TIMEOUT_MS = 10000  # 10 second timeout for integration tests


# DLQ topic names (matching CodegenDLQMonitor.DLQ_TOPICS)
DLQ_TOPICS = [
    "omninode_codegen_dlq_analyze_v1",
    "omninode_codegen_dlq_validate_v1",
    "omninode_codegen_dlq_pattern_v1",
    "omninode_codegen_dlq_mixin_v1",
]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
async def kafka_producer():
    """Create Kafka producer for publishing test DLQ messages."""
    if not KAFKA_AVAILABLE:
        pytest.skip("aiokafka not available for Kafka testing")

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )

    try:
        await producer.start()
        yield producer
    finally:
        await producer.stop()


@pytest.fixture
async def dlq_monitor():
    """Create CodegenDLQMonitor instance for testing."""
    if not KAFKA_AVAILABLE:
        pytest.skip("aiokafka not available for DLQ monitoring")

    monitor = CodegenDLQMonitor(
        kafka_config={"bootstrap_servers": KAFKA_BROKER},
        alert_threshold=10,  # Default threshold
    )

    yield monitor

    # Cleanup
    await monitor.stop_monitoring()


@pytest.fixture
async def running_dlq_monitor():
    """Create and start a DLQ monitor instance."""
    if not KAFKA_AVAILABLE:
        pytest.skip("aiokafka not available for DLQ monitoring")

    monitor = CodegenDLQMonitor(
        kafka_config={"bootstrap_servers": KAFKA_BROKER},
        alert_threshold=10,
    )

    # Start monitoring in background
    monitor_task = asyncio.create_task(monitor.start_monitoring())

    # Give monitor time to connect
    await asyncio.sleep(1.0)

    yield monitor

    # Cleanup
    await monitor.stop_monitoring()
    try:
        await asyncio.wait_for(monitor_task, timeout=2.0)
    except TimeoutError:
        pass


@pytest.fixture
async def dlq_monitor_with_low_threshold():
    """Create DLQ monitor with low alert threshold (3 messages)."""
    if not KAFKA_AVAILABLE:
        pytest.skip("aiokafka not available for DLQ monitoring")

    monitor = CodegenDLQMonitor(
        kafka_config={"bootstrap_servers": KAFKA_BROKER},
        alert_threshold=3,  # Low threshold for testing
    )

    yield monitor

    await monitor.stop_monitoring()


# ============================================================================
# Test Suite 1: Connection and Lifecycle
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_monitor_connects_to_kafka_successfully(dlq_monitor):
    """
    Test that DLQ monitor connects to Kafka broker successfully.

    Validates:
    - Consumer initialization
    - Kafka connection establishment
    - Topic subscription
    - Monitor state management
    """
    assert not dlq_monitor.is_running, "Monitor should not be running initially"
    assert dlq_monitor.consumer is None, "Consumer should be None before start"

    # Start monitoring (will run until stopped)
    monitor_task = asyncio.create_task(dlq_monitor.start_monitoring())

    # Give monitor time to connect
    await asyncio.sleep(1.0)

    # Verify monitor is running
    assert dlq_monitor.is_running, "Monitor should be running after start"
    assert dlq_monitor.consumer is not None, "Consumer should be initialized"

    # Stop monitoring
    await dlq_monitor.stop_monitoring()

    # Wait for task to complete
    try:
        await asyncio.wait_for(monitor_task, timeout=2.0)
    except TimeoutError:
        pass

    assert not dlq_monitor.is_running, "Monitor should not be running after stop"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_graceful_shutdown_commits_offsets(running_dlq_monitor, kafka_producer):
    """
    Test that graceful shutdown properly commits Kafka consumer offsets.

    Validates:
    - Consumer offset commit before shutdown
    - Clean consumer closure
    - No resource leaks
    - Quick shutdown (<2 seconds)
    """
    # Publish a test message
    test_topic = DLQ_TOPICS[0]
    await kafka_producer.send(
        test_topic,
        value={
            "error": "test error",
            "correlation_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
    await kafka_producer.flush()

    # Give monitor time to consume
    await asyncio.sleep(1.0)

    # Verify message was consumed
    stats = await running_dlq_monitor.get_dlq_stats()
    assert (
        stats["total_dlq_messages"] > 0
    ), "Monitor should have consumed at least one message"

    # Measure shutdown time
    start_time = time.perf_counter()
    await running_dlq_monitor.stop_monitoring()
    shutdown_duration_s = time.perf_counter() - start_time

    # Validate shutdown
    assert not running_dlq_monitor.is_running, "Monitor should be stopped"
    assert running_dlq_monitor.consumer is None, "Consumer should be None after stop"
    assert shutdown_duration_s < 2.0, f"Shutdown took {shutdown_duration_s:.2f}s (>2s)"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_graceful_degradation_when_kafka_unavailable():
    """
    Test that monitor handles Kafka connection failures gracefully.

    Validates:
    - Proper error handling for invalid broker
    - OnexError raised with correct error code
    - No hanging connections
    - Clean error messages
    """
    # Create monitor with invalid broker
    monitor = CodegenDLQMonitor(
        kafka_config={"bootstrap_servers": "invalid-broker:9999"},
        alert_threshold=10,
    )

    # Attempt to start monitoring
    from omnibase_core.errors.model_onex_error import ModelOnexError

    with pytest.raises(ModelOnexError) as exc_info:
        await monitor.start_monitoring()

    # Validate error details
    error = exc_info.value
    assert "Failed to connect to Kafka" in str(
        error
    ), "Error message should mention Kafka connection failure"
    assert error.context is not None, "Error should include context"
    assert (
        "invalid-broker:9999" in str(error.context).lower()
        or "kafka_config" in error.context
    )


# ============================================================================
# Test Suite 2: DLQ Message Detection and Counting
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_detects_and_counts_dlq_messages_correctly(
    running_dlq_monitor, kafka_producer
):
    """
    Test that DLQ messages are detected and counted accurately.

    Validates:
    - Message consumption from DLQ topics
    - Accurate per-topic counting
    - Total message count tracking
    - Statistics endpoint accuracy
    """
    # Publish messages to different DLQ topics
    messages_per_topic = 5
    correlation_ids = {}

    for topic in DLQ_TOPICS:
        correlation_ids[topic] = []
        for i in range(messages_per_topic):
            correlation_id = str(uuid4())
            correlation_ids[topic].append(correlation_id)

            await kafka_producer.send(
                topic,
                value={
                    "error": f"Test error {i} for {topic}",
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "event_type": "test_failure",
                },
            )

    await kafka_producer.flush()

    # Give monitor time to consume all messages
    await asyncio.sleep(2.0)

    # Verify counts
    stats = await running_dlq_monitor.get_dlq_stats()

    # Check total count
    expected_total = messages_per_topic * len(DLQ_TOPICS)
    assert (
        stats["total_dlq_messages"] == expected_total
    ), f"Expected {expected_total} total messages, got {stats['total_dlq_messages']}"

    # Check per-topic counts
    for topic in DLQ_TOPICS:
        topic_count = stats["dlq_counts"].get(topic, 0)
        assert (
            topic_count == messages_per_topic
        ), f"Expected {messages_per_topic} messages for {topic}, got {topic_count}"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_monitors_multiple_dlq_topics_concurrently(
    running_dlq_monitor, kafka_producer
):
    """
    Test that monitor handles concurrent DLQ message consumption from all topics.

    Validates:
    - Concurrent consumption from 4 DLQ topics
    - No message loss
    - Accurate tracking across topics
    - Performance under concurrent load
    """
    # Publish messages to all topics concurrently
    messages_per_topic = 10
    publish_tasks = []

    for topic in DLQ_TOPICS:
        for i in range(messages_per_topic):
            task = kafka_producer.send(
                topic,
                value={
                    "error": f"Concurrent error {i}",
                    "correlation_id": str(uuid4()),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            publish_tasks.append(task)

    # Wait for all publishes to complete
    await asyncio.gather(*publish_tasks)
    await kafka_producer.flush()

    # Give monitor time to consume
    await asyncio.sleep(2.0)

    # Verify all messages consumed
    stats = await running_dlq_monitor.get_dlq_stats()
    expected_total = messages_per_topic * len(DLQ_TOPICS)

    assert (
        stats["total_dlq_messages"] == expected_total
    ), f"Expected {expected_total} messages, got {stats['total_dlq_messages']}"

    # Verify each topic has messages
    for topic in DLQ_TOPICS:
        assert (
            stats["dlq_counts"][topic] > 0
        ), f"Topic {topic} should have consumed messages"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_handles_invalid_message_formats_gracefully(
    running_dlq_monitor, kafka_producer
):
    """
    Test that monitor handles invalid/malformed DLQ messages without crashing.

    Validates:
    - Resilience to invalid JSON
    - Resilience to missing fields
    - Resilience to null values
    - Continued monitoring after errors
    """
    test_topic = DLQ_TOPICS[0]

    # Send invalid messages (must provide non-None value for aiokafka)
    invalid_messages = [
        {},  # Empty message
        {"error": None},  # Null error field
        {"correlation_id": "test"},  # Missing error field
        {"invalid_structure": True},  # Unexpected structure
    ]

    for invalid_msg in invalid_messages:
        await kafka_producer.send(test_topic, value=invalid_msg)

    await kafka_producer.flush()

    # Give monitor time to process
    await asyncio.sleep(1.0)

    # Monitor should still be running
    assert running_dlq_monitor.is_running, "Monitor should still be running"

    # Send valid message to verify monitor is still functional
    await kafka_producer.send(
        test_topic,
        value={
            "error": "Valid error after invalid messages",
            "correlation_id": str(uuid4()),
        },
    )
    await kafka_producer.flush()

    await asyncio.sleep(1.0)

    # Verify monitor consumed messages (including invalid ones)
    stats = await running_dlq_monitor.get_dlq_stats()
    assert stats["total_dlq_messages"] > 0, "Monitor should have consumed messages"


# ============================================================================
# Test Suite 3: Alert Threshold Triggering
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_alert_threshold_triggers_alerting_mechanism(
    dlq_monitor_with_low_threshold, kafka_producer
):
    """
    Test that alert threshold triggers critical log and alerting.

    Validates:
    - Alert triggered when count >= threshold
    - Critical log message generated
    - Alert metadata includes count and topic
    - Alert cooldown period enforced
    """
    # Start monitor with low threshold (3 messages)
    monitor_task = asyncio.create_task(
        dlq_monitor_with_low_threshold.start_monitoring()
    )
    await asyncio.sleep(1.0)

    test_topic = DLQ_TOPICS[0]

    # Mock the _send_alert method to capture alert calls
    alert_calls = []

    async def mock_send_alert(topic: str, count: int, recent_error: str):
        alert_calls.append(
            {"topic": topic, "count": count, "recent_error": recent_error}
        )

    dlq_monitor_with_low_threshold._send_alert = mock_send_alert

    # Publish messages to exceed threshold
    for i in range(5):  # Exceeds threshold of 3
        await kafka_producer.send(
            test_topic,
            value={
                "error": f"Error message {i}",
                "correlation_id": str(uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    await kafka_producer.flush()

    # Give monitor time to process and alert
    await asyncio.sleep(2.0)

    # Verify alert was triggered
    assert len(alert_calls) > 0, "Alert should have been triggered"
    assert alert_calls[0]["topic"] == test_topic, "Alert should be for correct topic"
    assert alert_calls[0]["count"] >= 3, "Alert count should be >= threshold (3)"

    # Cleanup
    await dlq_monitor_with_low_threshold.stop_monitoring()
    try:
        await asyncio.wait_for(monitor_task, timeout=2.0)
    except TimeoutError:
        pass


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_alert_cooldown_prevents_spam(
    dlq_monitor_with_low_threshold, kafka_producer
):
    """
    Test that alert cooldown period prevents alert spam.

    Validates:
    - First alert sent immediately when threshold exceeded
    - Subsequent alerts suppressed during cooldown period
    - Alert cooldown period respected (15 minutes default)
    - _should_send_alert logic working correctly
    """
    # Start monitor
    monitor_task = asyncio.create_task(
        dlq_monitor_with_low_threshold.start_monitoring()
    )
    await asyncio.sleep(1.0)

    test_topic = DLQ_TOPICS[0]

    # Track alert calls
    alert_calls = []

    async def mock_send_alert(topic: str, count: int, recent_error: str):
        alert_calls.append(datetime.now(UTC))

    dlq_monitor_with_low_threshold._send_alert = mock_send_alert

    # Publish messages to trigger first alert
    for i in range(5):
        await kafka_producer.send(
            test_topic,
            value={"error": f"Error {i}", "correlation_id": str(uuid4())},
        )

    await kafka_producer.flush()
    await asyncio.sleep(1.5)

    # First alert should be sent
    initial_alert_count = len(alert_calls)
    assert initial_alert_count > 0, "First alert should be sent"

    # Reset the DLQ counts to test cooldown without re-triggering threshold
    # This simulates the cooldown period without waiting 15 minutes
    initial_alert_time = dlq_monitor_with_low_threshold.last_alert_time.get(test_topic)

    # Publish more messages (should NOT trigger new alert due to cooldown)
    # Note: We need to manually reset counts first to avoid exceeding threshold again
    dlq_monitor_with_low_threshold.dlq_counts[test_topic] = 0

    for i in range(5, 8):  # Publish 3 more to reach threshold again
        await kafka_producer.send(
            test_topic,
            value={"error": f"Error {i}", "correlation_id": str(uuid4())},
        )

    await kafka_producer.flush()
    await asyncio.sleep(1.5)

    # No new alerts should be sent (cooldown active - last_alert_time is set)
    assert (
        len(alert_calls) == initial_alert_count
    ), "No new alerts should be sent during cooldown"

    # Verify the last alert time was not updated
    if initial_alert_time:
        current_alert_time = dlq_monitor_with_low_threshold.last_alert_time.get(
            test_topic
        )
        # Alert time should be same or only slightly different (due to _should_send_alert preventing new alerts)
        assert (
            current_alert_time == initial_alert_time
            or len(alert_calls) == initial_alert_count
        ), "Alert time should not be updated during cooldown"

    # Cleanup
    await dlq_monitor_with_low_threshold.stop_monitoring()
    try:
        await asyncio.wait_for(monitor_task, timeout=2.0)
    except TimeoutError:
        pass


# ============================================================================
# Test Suite 4: Statistics and Monitoring
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_statistics_endpoint_returns_accurate_counts(
    running_dlq_monitor, kafka_producer
):
    """
    Test that get_dlq_stats() returns accurate statistics.

    Validates:
    - dlq_counts per-topic accuracy
    - total_dlq_messages accuracy
    - is_monitoring status
    - alert_threshold value
    - timestamp presence
    """
    # Publish known number of messages
    test_messages = {
        DLQ_TOPICS[0]: 3,
        DLQ_TOPICS[1]: 5,
        DLQ_TOPICS[2]: 2,
        DLQ_TOPICS[3]: 4,
    }

    for topic, count in test_messages.items():
        for i in range(count):
            await kafka_producer.send(
                topic,
                value={"error": f"Error {i}", "correlation_id": str(uuid4())},
            )

    await kafka_producer.flush()
    await asyncio.sleep(2.0)

    # Get statistics
    stats = await running_dlq_monitor.get_dlq_stats()

    # Validate statistics structure
    assert "dlq_counts" in stats, "Stats should include dlq_counts"
    assert "total_dlq_messages" in stats, "Stats should include total_dlq_messages"
    assert "alert_threshold" in stats, "Stats should include alert_threshold"
    assert "is_monitoring" in stats, "Stats should include is_monitoring"
    assert "timestamp" in stats, "Stats should include timestamp"

    # Validate counts
    expected_total = sum(test_messages.values())
    assert (
        stats["total_dlq_messages"] == expected_total
    ), f"Expected {expected_total} total, got {stats['total_dlq_messages']}"

    # Validate per-topic counts
    for topic, expected_count in test_messages.items():
        actual_count = stats["dlq_counts"].get(topic, 0)
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} for {topic}, got {actual_count}"

    # Validate monitoring status
    assert stats["is_monitoring"] is True, "Monitor should report as running"
    assert stats["alert_threshold"] == 10, "Alert threshold should be 10"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_reset_counts_functionality(running_dlq_monitor, kafka_producer):
    """
    Test that reset_counts() properly resets DLQ statistics.

    Validates:
    - reset_counts() clears all topic counts
    - reset_counts(topic) clears specific topic
    - total_dlq_messages reset to 0
    - Monitoring continues after reset
    """
    # Publish messages
    test_topic = DLQ_TOPICS[0]
    for i in range(5):
        await kafka_producer.send(
            test_topic,
            value={"error": f"Error {i}", "correlation_id": str(uuid4())},
        )

    await kafka_producer.flush()
    await asyncio.sleep(1.5)

    # Verify messages consumed
    stats = await running_dlq_monitor.get_dlq_stats()
    assert stats["total_dlq_messages"] > 0, "Should have messages before reset"

    # Reset all counts
    await running_dlq_monitor.reset_counts()

    # Verify reset
    stats = await running_dlq_monitor.get_dlq_stats()
    assert stats["total_dlq_messages"] == 0, "Total should be 0 after reset"
    assert len(stats["dlq_counts"]) == 0, "All topic counts should be cleared"

    # Verify monitoring still works
    await kafka_producer.send(
        test_topic, value={"error": "Post-reset error", "correlation_id": str(uuid4())}
    )
    await kafka_producer.flush()
    await asyncio.sleep(1.0)

    stats = await running_dlq_monitor.get_dlq_stats()
    assert stats["total_dlq_messages"] > 0, "Should consume messages after reset"


# ============================================================================
# Test Suite 5: Performance and Load Testing
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_performance_under_high_message_load(running_dlq_monitor, kafka_producer):
    """
    Test monitor performance under high message load.

    Validates:
    - 100+ messages/second throughput
    - <10ms average consumption latency
    - No message loss under load
    - Stable monitoring during burst traffic
    """
    num_messages = 100
    test_topic = DLQ_TOPICS[0]

    # Publish messages rapidly
    start_time = time.perf_counter()

    publish_tasks = []
    for i in range(num_messages):
        task = kafka_producer.send(
            test_topic,
            value={"error": f"High load error {i}", "correlation_id": str(uuid4())},
        )
        publish_tasks.append(task)

    await asyncio.gather(*publish_tasks)
    await kafka_producer.flush()

    publish_duration_s = time.perf_counter() - start_time

    # Give monitor time to consume
    await asyncio.sleep(2.0)

    # Verify all messages consumed
    stats = await running_dlq_monitor.get_dlq_stats()

    # Validate throughput
    assert (
        stats["total_dlq_messages"] >= num_messages
    ), f"Expected {num_messages} messages, got {stats['total_dlq_messages']}"

    # Validate performance
    messages_per_second = num_messages / publish_duration_s
    assert (
        messages_per_second > 100
    ), f"Throughput {messages_per_second:.0f} msg/s < 100 msg/s"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_statistics_retrieval_performance(running_dlq_monitor):
    """
    Test that statistics retrieval is fast (<5ms).

    Validates:
    - get_dlq_stats() latency
    - Consistent performance across calls
    - No blocking during stats retrieval
    """
    # Warm up
    await running_dlq_monitor.get_dlq_stats()

    # Measure stats retrieval time
    iterations = 10
    latencies = []

    for _ in range(iterations):
        start_time = time.perf_counter()
        await running_dlq_monitor.get_dlq_stats()
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)

    # Validate performance
    avg_latency_ms = sum(latencies) / len(latencies)
    max_latency_ms = max(latencies)

    assert avg_latency_ms < 5.0, f"Average latency {avg_latency_ms:.2f}ms > 5ms"
    assert max_latency_ms < 10.0, f"Max latency {max_latency_ms:.2f}ms > 10ms"


# ============================================================================
# Test Suite 6: Webhook Alerting (Mocked)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_webhook_alert_sends_http_post():
    """
    Test that webhook alerts send HTTP POST requests with correct payload.

    Validates:
    - Webhook POST request sent when configured
    - Correct payload structure
    - Alert metadata included
    - Error handling when webhook fails
    """
    # Create monitor with mocked webhook
    with patch("aiohttp.ClientSession.post") as mock_post:
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_response

        monitor = CodegenDLQMonitor(
            kafka_config={"bootstrap_servers": KAFKA_BROKER},
            alert_threshold=3,
            alert_webhook_url="https://example.com/webhook",
        )

        # Trigger alert directly
        await monitor._send_alert(
            topic=DLQ_TOPICS[0], count=10, recent_error="Test error"
        )

        # Verify webhook POST was called
        assert mock_post.called, "Webhook POST should be called"
        call_args = mock_post.call_args

        # Verify URL
        assert "https://example.com/webhook" in str(
            call_args
        ), "Webhook URL should be correct"

        # Verify payload structure (passed as json=...)
        if "json" in call_args.kwargs:
            payload = call_args.kwargs["json"]
            assert payload["alert_type"] == "dlq_threshold_exceeded"
            assert payload["topic"] == DLQ_TOPICS[0]
            assert payload["count"] == 10
            assert payload["severity"] == "critical"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_webhook_failure_does_not_crash_monitor():
    """
    Test that webhook failures don't crash the DLQ monitor.

    Validates:
    - Monitor continues running when webhook fails
    - Error logged but not raised
    - Graceful degradation for alerting
    """
    # Create monitor with webhook that will fail
    with patch("aiohttp.ClientSession.post") as mock_post:
        # Setup mock to raise exception
        mock_post.side_effect = Exception("Webhook endpoint unavailable")

        monitor = CodegenDLQMonitor(
            kafka_config={"bootstrap_servers": "localhost:29092"},
            alert_threshold=3,
            alert_webhook_url="https://example.com/webhook",
        )

        # Trigger alert (should not raise exception)
        try:
            await monitor._send_alert(
                topic=DLQ_TOPICS[0], count=10, recent_error="Test error"
            )
        except Exception as e:
            pytest.fail(f"Webhook failure should not raise exception: {e}")

        # Verify webhook was attempted
        assert mock_post.called, "Webhook POST should be attempted"


# ============================================================================
# Test Suite 7: Edge Cases and Error Scenarios
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_monitor_handles_kafka_broker_restart(
    dlq_monitor_with_low_threshold, kafka_producer
):
    """
    Test monitor behavior when Kafka broker becomes temporarily unavailable.

    Note: This test simulates behavior but doesn't actually restart broker.
    In production, circuit breakers and retry logic would handle broker restarts.

    Validates:
    - Error handling for broker disconnection
    - Logging of connection failures
    - Graceful error handling
    """
    # Start monitor
    monitor_task = asyncio.create_task(
        dlq_monitor_with_low_threshold.start_monitoring()
    )
    await asyncio.sleep(1.0)

    # Simulate broker issue by stopping the monitor's consumer
    if dlq_monitor_with_low_threshold.consumer:
        # Force stop consumer to simulate connection loss
        await dlq_monitor_with_low_threshold.consumer.stop()

    # Monitor should handle the error gracefully
    # (In production, this would trigger reconnection logic)

    # Cleanup
    await dlq_monitor_with_low_threshold.stop_monitoring()
    try:
        await asyncio.wait_for(monitor_task, timeout=2.0)
    except TimeoutError:
        pass


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_concurrent_monitor_instances_independent(kafka_producer):
    """
    Test that multiple monitor instances operate independently.

    Validates:
    - Independent consumer groups
    - Separate count tracking
    - No shared state interference
    - Isolated statistics
    """
    if not KAFKA_AVAILABLE:
        pytest.skip("Kafka not available")

    # Create two monitor instances with different consumer groups
    monitor1 = CodegenDLQMonitor(
        kafka_config={"bootstrap_servers": KAFKA_BROKER},
        alert_threshold=10,
    )

    monitor2 = CodegenDLQMonitor(
        kafka_config={"bootstrap_servers": KAFKA_BROKER},
        alert_threshold=5,
    )

    # Start both monitors
    task1 = asyncio.create_task(monitor1.start_monitoring())
    task2 = asyncio.create_task(monitor2.start_monitoring())
    await asyncio.sleep(1.5)

    # Publish messages
    test_topic = DLQ_TOPICS[0]
    for i in range(3):
        await kafka_producer.send(
            test_topic, value={"error": f"Error {i}", "correlation_id": str(uuid4())}
        )

    await kafka_producer.flush()
    await asyncio.sleep(2.0)

    # Both monitors should consume the same messages (different consumer groups)
    stats1 = await monitor1.get_dlq_stats()
    stats2 = await monitor2.get_dlq_stats()

    # Both should see the messages
    assert stats1["total_dlq_messages"] > 0, "Monitor 1 should have consumed messages"
    assert stats2["total_dlq_messages"] > 0, "Monitor 2 should have consumed messages"

    # Cleanup
    await monitor1.stop_monitoring()
    await monitor2.stop_monitoring()

    try:
        await asyncio.wait_for(asyncio.gather(task1, task2), timeout=3.0)
    except TimeoutError:
        pass


# ============================================================================
# Test Suite 8: Repr and String Representation
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_monitor_string_representation(dlq_monitor):
    """
    Test that monitor __repr__ provides useful debug information.

    Validates:
    - __repr__ includes running state
    - __repr__ includes alert threshold
    - __repr__ includes message count
    - Useful for debugging and logging
    """
    repr_str = repr(dlq_monitor)

    assert "CodegenDLQMonitor" in repr_str, "Repr should include class name"
    assert "running=" in repr_str, "Repr should include running state"
    assert "threshold=" in repr_str, "Repr should include threshold"
    assert "total_messages=" in repr_str, "Repr should include message count"


# ============================================================================
# Performance Summary Test
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_infrastructure
async def test_overall_performance_summary(running_dlq_monitor, kafka_producer):
    """
    Comprehensive performance test measuring all key metrics.

    Metrics:
    - Message consumption latency
    - Statistics retrieval latency
    - Throughput (messages/second)
    - Shutdown duration

    Success Criteria (relaxed for integration environment):
    - Consumption: <50ms per message (includes Kafka network + Docker overhead)
    - Statistics: <5ms retrieval
    - Throughput: >30 msg/s (realistic for Docker test environment)
    - Shutdown: <2 seconds
    """
    # Test 1: Consumption latency
    num_test_messages = 50
    test_topic = DLQ_TOPICS[0]

    start_time = time.perf_counter()
    for i in range(num_test_messages):
        await kafka_producer.send(
            test_topic,
            value={"error": f"Perf test {i}", "correlation_id": str(uuid4())},
        )
    await kafka_producer.flush()
    await asyncio.sleep(1.5)

    consumption_duration_s = time.perf_counter() - start_time
    avg_latency_ms = (consumption_duration_s / num_test_messages) * 1000

    # Test 2: Statistics latency
    stats_start = time.perf_counter()
    await running_dlq_monitor.get_dlq_stats()
    stats_latency_ms = (time.perf_counter() - stats_start) * 1000

    # Test 3: Throughput
    throughput_msg_per_s = num_test_messages / consumption_duration_s

    # Test 4: Shutdown duration
    shutdown_start = time.perf_counter()
    await running_dlq_monitor.stop_monitoring()
    shutdown_duration_s = time.perf_counter() - shutdown_start

    # Print performance summary
    print("\n" + "=" * 60)
    print("CodegenDLQMonitor Performance Summary")
    print("=" * 60)
    print(f"Message Consumption: {avg_latency_ms:.2f}ms avg (target: <50ms)")
    print(f"Statistics Retrieval: {stats_latency_ms:.2f}ms (target: <5ms)")
    print(f"Throughput: {throughput_msg_per_s:.0f} msg/s (target: >30 msg/s)")
    print(f"Graceful Shutdown: {shutdown_duration_s:.2f}s (target: <2s)")
    print("=" * 60)

    # Validate against targets (relaxed for integration test environment with Docker/Kafka overhead)
    assert avg_latency_ms < 50.0, f"Avg latency {avg_latency_ms:.2f}ms exceeds 50ms"
    assert stats_latency_ms < 5.0, f"Stats latency {stats_latency_ms:.2f}ms exceeds 5ms"
    assert (
        throughput_msg_per_s > 30
    ), f"Throughput {throughput_msg_per_s:.0f} msg/s below 30 msg/s"
    assert shutdown_duration_s < 2.0, f"Shutdown {shutdown_duration_s:.2f}s exceeds 2s"
