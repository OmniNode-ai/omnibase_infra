# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for Layer 2: Runtime Log Event Bridge Pipeline.

Tests the end-to-end flow from Python logging → RuntimeLogEventBridge → Kafka.

OMN-5526: Integration test for Layer 2.

Requires: running Kafka/Redpanda broker on localhost:19092.

Tests:
    - Log error → Kafka event flow
    - Circular logging prevention
    - Rate limiting
    - Bridge metrics accuracy
    - Allowlist-only attachment
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

# ONEX_FLAG_EXEMPT: test fixture — env var toggled in tests below
_FLAG = "ENABLE_RUNTIME_LOG_BRIDGE"
from aiokafka.structs import ConsumerRecord

from omnibase_infra.event_bus.topic_constants import TOPIC_RUNTIME_ERROR
from omnibase_infra.observability.runtime_log_event_bridge import (
    RuntimeLogEventBridge,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kafka,
]

BOOTSTRAP_SERVERS = "localhost:19092"


@pytest.fixture
async def kafka_producer() -> AsyncGenerator[AIOKafkaProducer, None]:
    """Create and start a Kafka producer for the bridge."""
    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
    )
    await producer.start()
    yield producer
    await producer.stop()


@pytest.fixture
async def kafka_consumer() -> AsyncGenerator[AIOKafkaConsumer, None]:
    """Create and start a Kafka consumer for the runtime error topic."""
    group_id = f"test-runtime-error-{uuid4().hex[:8]}"
    consumer = AIOKafkaConsumer(
        TOPIC_RUNTIME_ERROR,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        consumer_timeout_ms=5000,
    )
    await consumer.start()
    yield consumer
    await consumer.stop()


class _EnableBridgeCtx:
    """Context manager to enable the runtime log bridge feature flag."""

    def __enter__(self) -> _EnableBridgeCtx:
        os.environ[_FLAG] = "true"
        return self

    def __exit__(self, *args: object) -> None:
        os.environ.pop(_FLAG, None)


class TestRuntimeLogBridgeIntegration:
    """Tests for RuntimeLogEventBridge → Kafka flow."""

    @pytest.mark.asyncio
    async def test_log_error_produces_kafka_event(
        self,
        kafka_producer: AIOKafkaProducer,
        kafka_consumer: AIOKafkaConsumer,
    ) -> None:
        """Verify an ERROR log record flows through the bridge to Kafka."""
        os.environ[_FLAG] = "true"
        try:
            bridge = RuntimeLogEventBridge(
                producer=kafka_producer,
                hostname="test-host",
                service_label="test-service",
            )
            await bridge.start()

            # Create a test logger and attach bridge
            test_logger_name = f"test.logger.{uuid4().hex[:8]}"
            bridge.attach_to_loggers([test_logger_name])

            test_logger = logging.getLogger(test_logger_name)
            test_logger.setLevel(logging.WARNING)

            # Emit an error log
            test_logger.error("Database connection failed to host %s", "db-primary")

            # Give the drain loop time to process
            await asyncio.sleep(0.5)

            assert bridge.events_emitted == 1
            assert bridge.events_dropped == 0

            # Consume and validate
            msg = await asyncio.wait_for(_get_one_message(kafka_consumer), timeout=10.0)
            assert msg is not None
            assert msg.value is not None

            payload = json.loads(msg.value)
            assert payload["log_level"] == "ERROR"
            assert payload["hostname"] == "test-host"
            assert payload["service_label"] == "test-service"
            assert "Database connection failed" in payload["raw_message"]

            # Cleanup
            bridge.detach_from_loggers([test_logger_name])
            await bridge.stop()
        finally:
            os.environ.pop(_FLAG, None)

    @pytest.mark.asyncio
    async def test_circular_logging_prevention(
        self,
        kafka_producer: AIOKafkaProducer,
    ) -> None:
        """Verify the bridge does not capture its own log records."""
        os.environ[_FLAG] = "true"
        try:
            bridge = RuntimeLogEventBridge(
                producer=kafka_producer,
                hostname="test-host",
                service_label="test-service",
            )
            await bridge.start()

            # The bridge's own logger name starts with the module name
            bridge_logger = logging.getLogger(
                "omnibase_infra.observability.runtime_log_event_bridge._bridge"
            )
            bridge_logger.setLevel(logging.WARNING)

            # Attach bridge to its own module logger (should be filtered)
            bridge.attach_to_loggers(
                ["omnibase_infra.observability.runtime_log_event_bridge"]
            )

            # Emit via bridge's internal logger
            bridge_logger.error("This should NOT be captured")
            await asyncio.sleep(0.3)

            # Bridge should not have emitted anything (circular prevention)
            assert bridge.events_emitted == 0

            bridge.detach_from_loggers(
                ["omnibase_infra.observability.runtime_log_event_bridge"]
            )
            await bridge.stop()
        finally:
            os.environ.pop(_FLAG, None)

    @pytest.mark.asyncio
    async def test_rate_limiting(
        self,
        kafka_producer: AIOKafkaProducer,
    ) -> None:
        """Verify rate limiter suppresses duplicate log events."""
        os.environ[_FLAG] = "true"
        try:
            bridge = RuntimeLogEventBridge(
                producer=kafka_producer,
                hostname="test-host",
                service_label="test-service",
            )
            await bridge.start()

            test_logger_name = f"test.ratelimit.{uuid4().hex[:8]}"
            bridge.attach_to_loggers([test_logger_name])
            test_logger = logging.getLogger(test_logger_name)
            test_logger.setLevel(logging.WARNING)

            # Emit same error message 10 times rapidly
            for _ in range(10):
                test_logger.error("Repeated error message")

            await asyncio.sleep(0.5)

            # Max 5 per fingerprint per 5-minute window
            assert bridge.events_emitted <= 5
            assert bridge.events_rate_limited >= 5

            bridge.detach_from_loggers([test_logger_name])
            await bridge.stop()
        finally:
            os.environ.pop(_FLAG, None)

    @pytest.mark.asyncio
    async def test_bridge_disabled_by_default(
        self,
        kafka_producer: AIOKafkaProducer,
    ) -> None:
        """Verify bridge is a no-op when feature flag is off."""
        os.environ.pop(_FLAG, None)

        bridge = RuntimeLogEventBridge(
            producer=kafka_producer,
        )
        assert not RuntimeLogEventBridge.is_enabled()

        test_logger_name = f"test.disabled.{uuid4().hex[:8]}"
        bridge.attach_to_loggers([test_logger_name])
        test_logger = logging.getLogger(test_logger_name)
        test_logger.setLevel(logging.WARNING)

        test_logger.error("This should be silently dropped")
        await asyncio.sleep(0.2)

        # No events should be emitted or queued
        assert bridge.events_emitted == 0

        bridge.detach_from_loggers([test_logger_name])

    @pytest.mark.asyncio
    async def test_bridge_metrics_accuracy(
        self,
        kafka_producer: AIOKafkaProducer,
    ) -> None:
        """Verify bridge self-metrics track correctly."""
        os.environ[_FLAG] = "true"
        try:
            bridge = RuntimeLogEventBridge(
                producer=kafka_producer,
                hostname="test-host",
                service_label="test-service",
            )
            await bridge.start()

            test_logger_name = f"test.metrics.{uuid4().hex[:8]}"
            bridge.attach_to_loggers([test_logger_name])
            test_logger = logging.getLogger(test_logger_name)
            test_logger.setLevel(logging.WARNING)

            # Emit 3 distinct errors
            test_logger.error("Error A unique %s", uuid4())
            test_logger.error("Error B unique %s", uuid4())
            test_logger.warning("Warning C unique %s", uuid4())

            await asyncio.sleep(0.5)

            assert bridge.events_emitted == 3
            assert bridge.events_dropped == 0
            assert bridge.events_rate_limited == 0

            bridge.detach_from_loggers([test_logger_name])
            await bridge.stop()
        finally:
            os.environ.pop(_FLAG, None)


async def _get_one_message(
    consumer: AIOKafkaConsumer,
) -> ConsumerRecord[Any, Any] | None:
    """Poll for a single message from a Kafka consumer."""
    async for msg in consumer:
        return msg
    return None
