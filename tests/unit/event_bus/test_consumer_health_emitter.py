# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ConsumerHealthEmitter.

Ticket: OMN-5516
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.event_bus.consumer_health_emitter import ConsumerHealthEmitter
from omnibase_infra.models.health.enum_consumer_health_event_type import (
    EnumConsumerHealthEventType,
)
from omnibase_infra.models.health.enum_consumer_health_severity import (
    EnumConsumerHealthSeverity,
)
from omnibase_infra.models.health.model_consumer_health_event import (
    ModelConsumerHealthEvent,
)


@pytest.fixture
def mock_producer() -> AsyncMock:
    """Create a mock AIOKafkaProducer."""
    producer = AsyncMock()
    producer.send = AsyncMock()
    return producer


@pytest.fixture
def emitter(mock_producer: AsyncMock) -> ConsumerHealthEmitter:
    """Create a ConsumerHealthEmitter with mock producer."""
    return ConsumerHealthEmitter(mock_producer)


def _make_event(
    consumer_identity: str = "consumer-1",
    event_type: EnumConsumerHealthEventType = EnumConsumerHealthEventType.HEARTBEAT_FAILURE,
) -> ModelConsumerHealthEvent:
    return ModelConsumerHealthEvent.create(
        consumer_identity=consumer_identity,
        consumer_group="group-1",
        topic="test-topic",
        event_type=event_type,
        severity=EnumConsumerHealthSeverity.ERROR,
    )


@pytest.mark.unit
class TestConsumerHealthEmitter:
    """Tests for ConsumerHealthEmitter."""

    @patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_EMITTER": "true"})
    async def test_emit_sends_to_producer(
        self, emitter: ConsumerHealthEmitter, mock_producer: AsyncMock
    ) -> None:
        event = _make_event()
        await emitter.emit(event)
        mock_producer.send.assert_called_once()
        assert emitter.events_emitted == 1

    @patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_EMITTER": ""})
    async def test_emit_skipped_when_disabled(
        self, emitter: ConsumerHealthEmitter, mock_producer: AsyncMock
    ) -> None:
        event = _make_event()
        await emitter.emit(event)
        mock_producer.send.assert_not_called()
        assert emitter.events_emitted == 0

    @patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_EMITTER": "true"})
    async def test_rate_limiting(
        self, emitter: ConsumerHealthEmitter, mock_producer: AsyncMock
    ) -> None:
        event = _make_event()
        await emitter.emit(event)
        await emitter.emit(event)  # Same fingerprint, should be rate-limited
        assert mock_producer.send.call_count == 1
        assert emitter.events_emitted == 1
        assert emitter.events_rate_limited == 1

    @patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_EMITTER": "true"})
    async def test_different_fingerprints_not_rate_limited(
        self, emitter: ConsumerHealthEmitter, mock_producer: AsyncMock
    ) -> None:
        event1 = _make_event(consumer_identity="c1")
        event2 = _make_event(consumer_identity="c2")
        await emitter.emit(event1)
        await emitter.emit(event2)
        assert mock_producer.send.call_count == 2
        assert emitter.events_emitted == 2

    @patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_EMITTER": "true"})
    async def test_emit_failure_is_tolerated(
        self, emitter: ConsumerHealthEmitter, mock_producer: AsyncMock
    ) -> None:
        mock_producer.send.side_effect = Exception("broker down")
        event = _make_event()
        await emitter.emit(event)  # Should not raise
        assert emitter.events_dropped == 1
        assert emitter.events_emitted == 0

    def test_is_enabled_returns_false_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert not ConsumerHealthEmitter.is_enabled()

    def test_is_enabled_returns_true_when_set(self) -> None:
        with patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_EMITTER": "true"}):
            assert ConsumerHealthEmitter.is_enabled()

    @patch.dict("os.environ", {"ENABLE_CONSUMER_HEALTH_EMITTER": "true"})
    async def test_emit_event_convenience(
        self, emitter: ConsumerHealthEmitter, mock_producer: AsyncMock
    ) -> None:
        await emitter.emit_event(
            consumer_identity="c1",
            consumer_group="g1",
            topic="t1",
            event_type=EnumConsumerHealthEventType.SESSION_TIMEOUT,
            severity=EnumConsumerHealthSeverity.CRITICAL,
            error_message="session expired",
        )
        mock_producer.send.assert_called_once()
        assert emitter.events_emitted == 1
