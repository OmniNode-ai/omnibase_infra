# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that UnknownTopicOrPartitionError does NOT trip the circuit breaker.

Mirror pattern from test_kafka_event_bus.py: EventBusKafka fixture with mocked
AIOKafkaProducer so no real broker is needed.

Related Tickets:
    - OMN-9553: Exclude UnknownTopicOrPartitionError from circuit breaker failure count
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from aiokafka.errors import UnknownTopicOrPartitionError

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventHeaders
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

_TEST_SERVERS = "localhost:9092"
_TEST_ENV = "test"


@pytest.fixture
async def bus_with_mock_producer() -> EventBusKafka:
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
            bootstrap_servers=_TEST_SERVERS,
            environment=_TEST_ENV,
        )
        bus = EventBusKafka(config=config)
        yield bus
        try:
            await bus.close()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass


class TestCircuitBreakerTopicError:
    """UnknownTopicOrPartitionError is a config error, not a broker failure."""

    @pytest.mark.asyncio
    async def test_unknown_topic_does_not_trip_circuit_breaker(
        self, bus_with_mock_producer: EventBusKafka
    ) -> None:
        """Publishing to a nonexistent topic must NOT increment the CB failure counter."""
        bus = bus_with_mock_producer
        before = bus._circuit_breaker_failures

        bus._producer = AsyncMock()
        bus._producer.send = AsyncMock(side_effect=UnknownTopicOrPartitionError())

        headers = ModelEventHeaders(
            correlation_id=uuid4(),
            source="test",
            event_type="test.event",
            timestamp=datetime.now(UTC),
        )

        try:
            await bus._publish_with_retry(
                topic="onex.evt.nonexistent.v1",
                key=None,
                value=b"test",
                kafka_headers=[],
                headers=headers,
            )
        except Exception:  # noqa: BLE001 — expected to raise; we only care about CB state
            pass

        assert bus._circuit_breaker_failures == before, (
            "UnknownTopicOrPartitionError must not increment circuit breaker failure counter"
        )
