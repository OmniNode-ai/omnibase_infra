# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for ServiceRuntimeHealthMonitor (OMN-8623).

Verifies that the monitor can:
1. Run a single health check cycle against a live Kafka broker
2. Emit an event to the runtime-health-check topic
3. Produce a correctly-structured event with all required fields

Requires: running Kafka/Redpanda broker on localhost:19092.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator

import pytest
from aiokafka import AIOKafkaConsumer

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.models.health.model_runtime_health_check_event import (
    ModelRuntimeHealthCheckEvent,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kafka,
]

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
HEALTH_TOPIC = ServiceTopicRegistry.from_defaults().resolve(
    topic_keys.RUNTIME_HEALTH_CHECK
)


@pytest.fixture
async def kafka_consumer() -> AsyncGenerator[AIOKafkaConsumer, None]:
    """Create a transient Kafka consumer subscribed to the health topic."""
    consumer = AIOKafkaConsumer(
        HEALTH_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        group_id=f"test-runtime-health-monitor-{uuid.uuid4().hex}",
        enable_auto_commit=False,
    )
    await consumer.start()
    # Seek to end so we only see events produced during this test
    await consumer.seek_to_end()
    try:
        yield consumer
    finally:
        await consumer.stop()


@pytest.mark.asyncio
async def test_run_once_returns_event() -> None:
    """run_once() must return a valid ModelRuntimeHealthCheckEvent."""
    monitor = ServiceRuntimeHealthMonitor(
        bootstrap_servers=BOOTSTRAP_SERVERS,
    )
    event = await monitor.run_once()

    assert isinstance(event, ModelRuntimeHealthCheckEvent)
    assert event.status in ("HEALTHY", "DEGRADED", "CRITICAL")
    assert event.contract_count >= 0
    assert event.consumer_group_count >= 0
    assert event.correlation_id is not None
    assert event.timestamp is not None


@pytest.mark.asyncio
async def test_run_once_emits_to_kafka(kafka_consumer: AIOKafkaConsumer) -> None:
    """run_once() must publish a health-check event to the Kafka topic."""
    config = ModelKafkaEventBusConfig(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        environment="integration-test",
    )
    bus = EventBusKafka(config=config)
    await bus.start()
    try:
        monitor = ServiceRuntimeHealthMonitor(
            event_bus=bus,
            bootstrap_servers=BOOTSTRAP_SERVERS,
        )
        await monitor.run_once()

        # Give broker a moment to deliver the message
        record = None
        try:
            records = await asyncio.wait_for(
                kafka_consumer.getmany(timeout_ms=5000, max_records=1),
                timeout=10.0,
            )
            for _tp, msgs in records.items():
                if msgs:
                    record = msgs[0]
                    break
        except TimeoutError:
            pass  # record stays None; assertion below will fail with a clear message

        assert record is not None, "Expected health-check event on Kafka topic"
        payload = json.loads(record.value)
        assert "status" in payload
        assert payload["status"] in ("HEALTHY", "DEGRADED", "CRITICAL")
    finally:
        await bus.stop()
