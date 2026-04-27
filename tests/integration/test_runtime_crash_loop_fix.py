# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for the runtime crash-loop fix (OMN-9551).

Exercises the two coupled fixes that together prevent the boot-time
restart loop:

1. **OMN-9552**: ServiceRuntimeHealthMonitor suppresses ``_emit()`` while
   inside the boot grace window so a not-yet-provisioned health-check
   topic cannot trip the circuit breaker on first boot.
2. **OMN-9553**: EventBusKafka._publish_with_retry catches
   ``UnknownTopicOrPartitionError`` *before* ``KafkaError`` and skips the
   circuit-breaker failure counter — a missing topic is a configuration
   error, not a broker connectivity failure.

These tests cover the contract between the two: even if the monitor
emitted past the grace window, a topic-not-found error must not cascade
into circuit-breaker open and block subsequent publishes to healthy
topics.

Related Tickets:
    - OMN-9551: parent — runtime crash loop fix
    - OMN-9552: boot grace window
    - OMN-9553: exclude UnknownTopicOrPartitionError from CB
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiokafka.errors import UnknownTopicOrPartitionError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventHeaders
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
)

pytestmark = [pytest.mark.integration]


@pytest.fixture
async def kafka_bus_with_mock_producer() -> AsyncGenerator[EventBusKafka, None]:
    """EventBusKafka with a mocked AIOKafkaProducer — no real broker needed."""
    mock_producer = AsyncMock()
    mock_producer.start = AsyncMock()
    mock_producer.stop = AsyncMock()
    mock_producer._closed = False
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=mock_producer,
    ):
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="integration-test",
        )
        bus = EventBusKafka(config=config)
        yield bus
        try:
            await bus.close()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass


class TestBootGraceSuppressesEmit:
    """Monitor must not call event_bus.publish_envelope inside the grace window."""

    @pytest.mark.asyncio
    async def test_emit_suppressed_when_inside_grace_window(self) -> None:
        bus = MagicMock()
        bus.publish_envelope = AsyncMock()

        monitor = ServiceRuntimeHealthMonitor(
            event_bus=bus,
            bootstrap_servers="",
            boot_grace_seconds=9999.0,  # effectively never expires
        )
        event = await monitor.run_once()
        await monitor._emit(event)

        bus.publish_envelope.assert_not_called()

    @pytest.mark.asyncio
    async def test_emit_proceeds_after_grace_window_expires(self) -> None:
        bus = MagicMock()
        bus.publish_envelope = AsyncMock()

        monitor = ServiceRuntimeHealthMonitor(
            event_bus=bus,
            bootstrap_servers="",
            boot_grace_seconds=0.0,  # already expired at construction
        )
        # Backdate so elapsed clearly exceeds 0.0 (avoids monotonic-clock races).
        monitor._started_at = time.monotonic() - 1.0

        await monitor.run_once()  # run_once() invokes _emit() internally

        assert bus.publish_envelope.call_count >= 1, (
            "Expected publish_envelope to be called once grace window has expired"
        )


class TestTopicNotFoundDoesNotCascadeIntoCircuitOpen:
    """Even if the monitor emits to an unprovisioned topic, the CB stays closed."""

    @pytest.mark.asyncio
    async def test_unknown_topic_does_not_open_circuit_for_subsequent_healthy_publish(
        self, kafka_bus_with_mock_producer: EventBusKafka
    ) -> None:
        bus = kafka_bus_with_mock_producer

        # Drive the failure counter close to threshold via a real KafkaError, then
        # confirm a subsequent UnknownTopicOrPartitionError does not push it over.
        bus._circuit_breaker_failures = bus.circuit_breaker_threshold - 1
        before = bus._circuit_breaker_failures

        bus._producer = AsyncMock()
        bus._producer.send = AsyncMock(side_effect=UnknownTopicOrPartitionError())

        headers = ModelEventHeaders(
            correlation_id=uuid4(),
            source="ServiceRuntimeHealthMonitor",
            event_type="runtime-health-check",
            timestamp=datetime.now(UTC),
        )

        with pytest.raises(ProtocolConfigurationError):
            await bus._publish_with_retry(
                topic="onex.evt.omnibase-infra.runtime-health-check.v1",
                key=None,
                value=b"{}",
                kafka_headers=[],
                headers=headers,
            )

        assert bus._circuit_breaker_failures == before, (
            "Topic-not-found must not increment the circuit-breaker failure counter"
        )
        assert not bus._circuit_breaker_open, (
            "Circuit breaker must remain closed after topic-not-found, so "
            "subsequent publishes to healthy topics are not blocked"
        )
