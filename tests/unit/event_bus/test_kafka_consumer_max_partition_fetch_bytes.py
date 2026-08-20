# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that AIOKafkaConsumer receives max_partition_fetch_bytes aligned to
the producer's max_request_size (OMN-16267 CodeRabbit follow-up).

aiokafka's own AIOKafkaConsumer default for max_partition_fetch_bytes
(1_048_576) is smaller than this repo's producer default max_request_size
(1_048_588). A record batch in that gap passes the producer's client-side
size check, is accepted by the broker, and then trips RecordTooLargeError
(or a silent skip-and-advance on the async-iterator consumption path) on
the consumer side. Both AIOKafkaConsumer construction sites -- the initial
create in _start_consumer_for_topic_unlocked and the recreate-on-retry
inside its metadata-propagation retry loop -- must pass
max_partition_fetch_bytes=self._config.max_request_size.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka.errors import UnknownTopicOrPartitionError

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

TEST_BOOTSTRAP_SERVERS: str = "localhost:9092"


@pytest.mark.unit
class TestConsumerMaxPartitionFetchBytes:
    """Verify AIOKafkaConsumer receives max_partition_fetch_bytes from config."""

    @pytest.fixture
    def mock_producer(self) -> AsyncMock:
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer._closed = False
        return producer

    @pytest.mark.asyncio
    async def test_initial_consumer_receives_max_partition_fetch_bytes(
        self, mock_producer: AsyncMock
    ) -> None:
        """The first AIOKafkaConsumer() construction gets max_partition_fetch_bytes
        equal to config.max_request_size."""
        mock_consumer = AsyncMock()
        mock_consumer.start = AsyncMock()
        mock_consumer.stop = AsyncMock()
        consumer_cls = MagicMock(return_value=mock_consumer)

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
        ):
            config = ModelKafkaEventBusConfig(bootstrap_servers=TEST_BOOTSTRAP_SERVERS)
            event_bus = EventBusKafka(config=config)

            await event_bus._start_consumer_for_topic("events", "my-group")

            consumer_cls.assert_called_once()
            call_kwargs = consumer_cls.call_args.kwargs
            assert call_kwargs["max_partition_fetch_bytes"] == config.max_request_size
            assert config.max_request_size == 1_048_588

    @pytest.mark.asyncio
    async def test_custom_max_request_size_propagates_to_consumer(
        self, mock_producer: AsyncMock
    ) -> None:
        """A custom max_request_size on the config reaches the consumer too,
        not just the producer -- the two must stay aligned."""
        mock_consumer = AsyncMock()
        mock_consumer.start = AsyncMock()
        mock_consumer.stop = AsyncMock()
        consumer_cls = MagicMock(return_value=mock_consumer)

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
        ):
            config = ModelKafkaEventBusConfig(
                bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
                max_request_size=5_000_000,
            )
            event_bus = EventBusKafka(config=config)

            await event_bus._start_consumer_for_topic("events", "my-group")

            call_kwargs = consumer_cls.call_args.kwargs
            assert call_kwargs["max_partition_fetch_bytes"] == 5_000_000

    @pytest.mark.asyncio
    async def test_recreated_consumer_after_metadata_retry_receives_max_partition_fetch_bytes(
        self, mock_producer: AsyncMock
    ) -> None:
        """The consumer recreated inside the metadata-propagation retry loop
        (after UnknownTopicOrPartitionError on the first start()) also gets
        max_partition_fetch_bytes -- the second AIOKafkaConsumer() construction
        site, not just the first."""
        failing_consumer = AsyncMock()
        failing_consumer.start = AsyncMock(
            side_effect=UnknownTopicOrPartitionError("topic metadata not ready")
        )
        failing_consumer.stop = AsyncMock()

        succeeding_consumer = AsyncMock()
        succeeding_consumer.start = AsyncMock()
        succeeding_consumer.stop = AsyncMock()

        consumer_cls = MagicMock(side_effect=[failing_consumer, succeeding_consumer])

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.asyncio.sleep",
                AsyncMock(),
            ),
        ):
            config = ModelKafkaEventBusConfig(
                bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
                timeout_seconds=30,
            )
            event_bus = EventBusKafka(config=config)

            await event_bus._start_consumer_for_topic("events", "my-group")

            assert consumer_cls.call_count == 2, (
                "expected exactly one retry recreation after the metadata error"
            )
            recreated_call_kwargs = consumer_cls.call_args_list[1].kwargs
            assert (
                recreated_call_kwargs["max_partition_fetch_bytes"]
                == config.max_request_size
            )
