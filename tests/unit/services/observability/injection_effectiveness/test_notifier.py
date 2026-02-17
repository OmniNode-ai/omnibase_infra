# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for EffectivenessInvalidationNotifier.

Tests the Kafka notification publisher for effectiveness data changes.

Related Tickets:
    - OMN-2303: Activate effectiveness consumer and populate measurement tables
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.services.observability.injection_effectiveness.notifier import (
    TOPIC_EFFECTIVENESS_INVALIDATION,
    EffectivenessInvalidationNotifier,
)


@pytest.fixture
def mock_producer() -> AsyncMock:
    """Create a mock AIOKafkaProducer."""
    producer = AsyncMock()
    producer.send_and_wait = AsyncMock()
    return producer


class TestEffectivenessInvalidationNotifier:
    """Tests for EffectivenessInvalidationNotifier."""

    @pytest.mark.asyncio
    async def test_notify_publishes_event(self, mock_producer: AsyncMock) -> None:
        """Successful notification publishes JSON to Kafka."""
        notifier = EffectivenessInvalidationNotifier(mock_producer)
        correlation_id = uuid4()

        await notifier.notify(
            tables_affected=("injection_effectiveness", "latency_breakdowns"),
            rows_written=42,
            source="kafka_consumer",
            correlation_id=correlation_id,
        )

        mock_producer.send_and_wait.assert_awaited_once()
        call_args = mock_producer.send_and_wait.call_args

        # send_and_wait is called with positional topic and keyword args
        topic = call_args.args[0] if call_args.args else call_args.kwargs.get("topic")
        assert topic == TOPIC_EFFECTIVENESS_INVALIDATION

        # Verify value is bytes (JSON-encoded)
        value = call_args.kwargs.get("value")
        assert isinstance(value, bytes)

        # Verify key is the correlation_id
        key = call_args.kwargs.get("key")
        assert str(correlation_id).encode("utf-8") == key

    @pytest.mark.asyncio
    async def test_notify_skips_zero_rows(self, mock_producer: AsyncMock) -> None:
        """Zero rows_written does not publish."""
        notifier = EffectivenessInvalidationNotifier(mock_producer)

        await notifier.notify(
            tables_affected=("injection_effectiveness",),
            rows_written=0,
            source="batch_compute",
        )

        mock_producer.send_and_wait.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_notify_skips_negative_rows(self, mock_producer: AsyncMock) -> None:
        """Negative rows_written does not publish."""
        notifier = EffectivenessInvalidationNotifier(mock_producer)

        await notifier.notify(
            tables_affected=("injection_effectiveness",),
            rows_written=-1,
            source="batch_compute",
        )

        mock_producer.send_and_wait.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_notify_generates_correlation_id_when_none(
        self, mock_producer: AsyncMock
    ) -> None:
        """When no correlation_id is provided, one is generated."""
        notifier = EffectivenessInvalidationNotifier(mock_producer)

        await notifier.notify(
            tables_affected=("pattern_hit_rates",),
            rows_written=5,
            source="batch_compute",
        )

        mock_producer.send_and_wait.assert_awaited_once()
        # Key should be a UUID string encoded as bytes
        key = mock_producer.send_and_wait.call_args.kwargs["key"]
        assert isinstance(key, bytes)
        assert len(key) == 36  # UUID string length

    @pytest.mark.asyncio
    async def test_notify_custom_topic(self, mock_producer: AsyncMock) -> None:
        """Custom topic is used when specified."""
        custom_topic = "custom.invalidation.topic"
        notifier = EffectivenessInvalidationNotifier(mock_producer, topic=custom_topic)

        await notifier.notify(
            tables_affected=("injection_effectiveness",),
            rows_written=1,
            source="kafka_consumer",
        )

        call_args = mock_producer.send_and_wait.call_args
        topic = call_args.args[0] if call_args.args else call_args.kwargs.get("topic")
        assert topic == custom_topic

    @pytest.mark.asyncio
    async def test_notify_suppresses_producer_errors(
        self, mock_producer: AsyncMock
    ) -> None:
        """Producer errors are logged but not raised (fire-and-forget)."""
        mock_producer.send_and_wait = AsyncMock(side_effect=RuntimeError("Kafka down"))
        notifier = EffectivenessInvalidationNotifier(mock_producer)

        # Should not raise
        await notifier.notify(
            tables_affected=("injection_effectiveness",),
            rows_written=10,
            source="kafka_consumer",
        )


class TestModelEffectivenessInvalidationEvent:
    """Tests for the invalidation event model."""

    def test_model_construction(self) -> None:
        from omnibase_infra.services.observability.injection_effectiveness.models.model_invalidation_event import (
            ModelEffectivenessInvalidationEvent,
        )

        event = ModelEffectivenessInvalidationEvent(
            tables_affected=("injection_effectiveness", "latency_breakdowns"),
            rows_written=42,
            source="kafka_consumer",
        )

        assert event.event_type == "effectiveness_data_changed"
        assert event.tables_affected == (
            "injection_effectiveness",
            "latency_breakdowns",
        )
        assert event.rows_written == 42
        assert event.source == "kafka_consumer"
        assert event.correlation_id is not None
        assert event.emitted_at is not None

    def test_model_frozen(self) -> None:
        from omnibase_infra.services.observability.injection_effectiveness.models.model_invalidation_event import (
            ModelEffectivenessInvalidationEvent,
        )

        event = ModelEffectivenessInvalidationEvent(
            tables_affected=("injection_effectiveness",),
            rows_written=1,
            source="batch_compute",
        )

        with pytest.raises(Exception):
            event.rows_written = 99  # type: ignore[misc]

    def test_model_serialization(self) -> None:
        from omnibase_infra.services.observability.injection_effectiveness.models.model_invalidation_event import (
            ModelEffectivenessInvalidationEvent,
        )

        event = ModelEffectivenessInvalidationEvent(
            tables_affected=("pattern_hit_rates",),
            rows_written=5,
            source="batch_compute",
        )

        data = event.model_dump(mode="json")
        assert data["event_type"] == "effectiveness_data_changed"
        assert data["rows_written"] == 5
        assert isinstance(data["correlation_id"], str)
        assert isinstance(data["emitted_at"], str)
