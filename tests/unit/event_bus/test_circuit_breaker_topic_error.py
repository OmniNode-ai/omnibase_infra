# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that UnknownTopicOrPartitionError does NOT trip the circuit breaker.

Mirror pattern from test_kafka_event_bus.py: EventBusKafka fixture with mocked
AIOKafkaProducer so no real broker is needed.

Related Tickets:
    - OMN-9553: Exclude UnknownTopicOrPartitionError from circuit breaker failure count
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from aiokafka.errors import KafkaError, UnknownTopicOrPartitionError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventHeaders
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

TEST_SERVERS: str = "localhost:9092"
TEST_ENV: str = "test"


@pytest.fixture
async def bus_with_mock_producer() -> AsyncGenerator[EventBusKafka, None]:
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
            bootstrap_servers=TEST_SERVERS,
            environment=TEST_ENV,
        )
        bus = EventBusKafka(config=config)
        yield bus
        try:
            await bus.close()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass


class TestCircuitBreakerTopicError:
    """UnknownTopicOrPartitionError is a config error, not a broker failure."""

    def test_unknown_topic_is_subclass_of_kafka_error(self) -> None:
        """Document the IS-A relationship that makes except-ordering critical."""
        assert issubclass(UnknownTopicOrPartitionError, KafkaError), (
            "UnknownTopicOrPartitionError must remain a KafkaError subclass — "
            "the except chain in _publish_with_retry depends on this ordering."
        )

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

        with pytest.raises(ProtocolConfigurationError):
            await bus._publish_with_retry(
                topic="onex.evt.nonexistent.v1",
                key=None,
                value=b"test",
                kafka_headers=[],
                headers=headers,
            )

        assert bus._circuit_breaker_failures == before, (
            "UnknownTopicOrPartitionError must not increment circuit breaker failure counter"
        )

    @pytest.mark.asyncio
    async def test_unknown_topic_does_not_trip_circuit_breaker_near_threshold(
        self, bus_with_mock_producer: EventBusKafka
    ) -> None:
        """CB must not open even when counter is already at threshold - 1."""
        bus = bus_with_mock_producer
        bus._circuit_breaker_failures = bus.circuit_breaker_threshold - 1
        before = bus._circuit_breaker_failures

        bus._producer = AsyncMock()
        bus._producer.send = AsyncMock(side_effect=UnknownTopicOrPartitionError())

        headers = ModelEventHeaders(
            correlation_id=uuid4(),
            source="test",
            event_type="test.event",
            timestamp=datetime.now(UTC),
        )

        with pytest.raises(ProtocolConfigurationError):
            await bus._publish_with_retry(
                topic="onex.evt.nonexistent.v1",
                key=None,
                value=b"test",
                kafka_headers=[],
                headers=headers,
            )

        assert bus._circuit_breaker_failures == before, (
            "UnknownTopicOrPartitionError must not push failures over the CB threshold"
        )
        assert not bus._circuit_breaker_open, (
            "Circuit breaker must remain closed after topic-not-found error"
        )
