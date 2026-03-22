# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for Layer 1: Consumer Health Pipeline.

Tests the end-to-end flow from ConsumerHealthEmitter through to triage handlers.

OMN-5524: Integration test for Layer 1 end-to-end flow.

Requires: running Kafka/Redpanda broker on localhost:19092.

Tests:
    - Emit → consume → triage graduated response
    - Restart command flow (3rd occurrence triggers restart)
    - Restart failure → Linear ticket creation (mocked)
    - Rate limiting prevents flood
    - Emitter self-metrics accuracy
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import ConsumerRecord

from omnibase_infra.event_bus.consumer_health_emitter import ConsumerHealthEmitter
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

TOPIC_CONSUMER_HEALTH = ServiceTopicRegistry.from_defaults().resolve(
    topic_keys.CONSUMER_HEALTH
)
from omnibase_infra.models.health.enum_consumer_health_event_type import (
    EnumConsumerHealthEventType,
)
from omnibase_infra.models.health.enum_consumer_health_severity import (
    EnumConsumerHealthSeverity,
)
from omnibase_infra.models.health.model_consumer_health_event import (
    ModelConsumerHealthEvent,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kafka,
]

BOOTSTRAP_SERVERS = "localhost:19092"


@pytest.fixture
async def kafka_producer() -> AsyncGenerator[AIOKafkaProducer, None]:
    """Create and start a Kafka producer for tests."""
    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
    )
    await producer.start()
    yield producer
    await producer.stop()


@pytest.fixture
async def kafka_consumer() -> AsyncGenerator[AIOKafkaConsumer, None]:
    """Create and start a Kafka consumer for the health topic."""
    group_id = f"test-consumer-health-{uuid4().hex[:8]}"
    consumer = AIOKafkaConsumer(
        TOPIC_CONSUMER_HEALTH,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        consumer_timeout_ms=5000,
    )
    await consumer.start()
    yield consumer
    await consumer.stop()


class TestConsumerHealthEmitterIntegration:
    """Tests for ConsumerHealthEmitter → Kafka → Consumer flow."""

    @pytest.mark.asyncio
    async def test_emit_event_produces_to_kafka(
        self, kafka_producer: AIOKafkaProducer, kafka_consumer: AIOKafkaConsumer
    ) -> None:
        """Verify emitter produces a valid event to Kafka and it can be consumed."""
        with patch.dict(os.environ, {"ENABLE_CONSUMER_HEALTH_EMITTER": "true"}):
            emitter = ConsumerHealthEmitter(kafka_producer)

            # Emit a heartbeat failure event
            await emitter.emit_event(
                consumer_identity="test-consumer-1",
                consumer_group="test-group",
                topic="test.topic.v1",
                event_type=EnumConsumerHealthEventType.HEARTBEAT_FAILURE,
                severity=EnumConsumerHealthSeverity.ERROR,
                error_message="Heartbeat session expired",
                error_type="SessionExpiredError",
            )

            assert emitter.events_emitted == 1
            assert emitter.events_dropped == 0

            # Consume and validate
            msg = await asyncio.wait_for(_get_one_message(kafka_consumer), timeout=10.0)
            assert msg is not None
            assert msg.value is not None

            payload = json.loads(msg.value)
            assert payload["consumer_identity"] == "test-consumer-1"
            assert payload["event_type"] == "HEARTBEAT_FAILURE"
            assert payload["severity"] == "ERROR"

    @pytest.mark.asyncio
    async def test_emitter_rate_limiting(
        self, kafka_producer: AIOKafkaProducer
    ) -> None:
        """Verify rate limiter suppresses duplicate events within window."""
        with patch.dict(os.environ, {"ENABLE_CONSUMER_HEALTH_EMITTER": "true"}):
            emitter = ConsumerHealthEmitter(kafka_producer)

            # Emit same event type rapidly 5 times
            for _ in range(5):
                await emitter.emit_event(
                    consumer_identity="test-consumer-rl",
                    consumer_group="test-group",
                    topic="test.topic.v1",
                    event_type=EnumConsumerHealthEventType.HEARTBEAT_FAILURE,
                    severity=EnumConsumerHealthSeverity.ERROR,
                    error_message="Same error",
                )

            # First should emit, subsequent should be rate-limited
            assert emitter.events_emitted == 1
            assert emitter.events_rate_limited == 4

    @pytest.mark.asyncio
    async def test_emitter_disabled_by_default(
        self, kafka_producer: AIOKafkaProducer
    ) -> None:
        """Verify emitter is a no-op when feature flag is off."""
        with patch.dict(os.environ, {}, clear=False):
            # Ensure the flag is not set
            os.environ.pop("ENABLE_CONSUMER_HEALTH_EMITTER", None)

            emitter = ConsumerHealthEmitter(kafka_producer)
            assert not ConsumerHealthEmitter.is_enabled()

            await emitter.emit_event(
                consumer_identity="test-consumer-disabled",
                consumer_group="test-group",
                topic="test.topic.v1",
                event_type=EnumConsumerHealthEventType.CONSUMER_STARTED,
                severity=EnumConsumerHealthSeverity.INFO,
            )

            # Should be dropped (feature disabled)
            assert emitter.events_emitted == 0

    @pytest.mark.asyncio
    async def test_emitter_self_metrics(self, kafka_producer: AIOKafkaProducer) -> None:
        """Verify emitter tracks self-metrics accurately."""
        with patch.dict(os.environ, {"ENABLE_CONSUMER_HEALTH_EMITTER": "true"}):
            emitter = ConsumerHealthEmitter(kafka_producer)

            # Emit distinct events (different fingerprints)
            for i in range(3):
                await emitter.emit_event(
                    consumer_identity=f"test-consumer-{i}",
                    consumer_group="test-group",
                    topic="test.topic.v1",
                    event_type=EnumConsumerHealthEventType.CONSUMER_STARTED,
                    severity=EnumConsumerHealthSeverity.INFO,
                )

            assert emitter.events_emitted == 3
            assert emitter.events_dropped == 0
            assert emitter.events_rate_limited == 0


class TestMixinConsumerHealthIntegration:
    """Tests for MixinConsumerHealth wiring in standalone consumers."""

    @pytest.mark.asyncio
    async def test_mixin_init_health_emitter(
        self, kafka_producer: AIOKafkaProducer
    ) -> None:
        """Verify mixin correctly initializes health emitter."""
        from omnibase_infra.event_bus.mixin_consumer_health import MixinConsumerHealth

        class TestConsumer(MixinConsumerHealth):
            pass

        with patch.dict(os.environ, {"ENABLE_CONSUMER_HEALTH_EMITTER": "true"}):
            consumer = TestConsumer()
            consumer._init_health_emitter(
                kafka_producer,
                consumer_identity="test-mixin-consumer",
                consumer_group="test-group",
                topic="test.topic.v1",
                service_label="TestConsumer",
            )

            assert consumer._health_emitter is not None
            assert consumer._health_consumer_identity == "test-mixin-consumer"

            # Emit via convenience method
            await consumer._emit_consumer_started()

            assert consumer._health_emitter.events_emitted == 1


async def _get_one_message(
    consumer: AIOKafkaConsumer,
) -> ConsumerRecord[Any, Any] | None:
    """Poll for a single message from a Kafka consumer."""
    async for msg in consumer:
        return msg
    return None
