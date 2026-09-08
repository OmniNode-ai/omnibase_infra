# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that AIOKafkaConsumer receives the per-partition fetch bound from its
own typed config field (OMN-15837), not from the producer's max_request_size.

OMN-16267 originally pinned ``max_partition_fetch_bytes`` to
``config.max_request_size`` on the theory that a smaller bound would trip
RecordTooLargeError, or a silent skip-and-advance, on a record at the
producer's ceiling. OMN-15837 decouples them: the bound is a per-consumer
buffer ceiling, and an auto-wired runtime holds one consumer per wired topic,
so a 1 MiB bound multiplied by 382 wired consumers reserved ~382 MiB against a
1.5 GiB container limit and OOM-looped the runtime. The coupling is also not
required -- a KIP-74 broker returns the first record of a partition in full
regardless of the fetch bound, so at least one record always decodes and the
skip-and-advance branch in aiokafka's fetcher is unreachable. Both
AIOKafkaConsumer construction sites -- the initial create in
``_start_consumer_for_topic_unlocked`` and the recreate-on-retry inside its
metadata-propagation retry loop -- must pass
``max_partition_fetch_bytes=self._config.max_partition_fetch_bytes``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka.errors import UnknownTopicOrPartitionError

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

TEST_BOOTSTRAP_SERVERS: str = "localhost:9092"

# The measured maximum payload on the live bus (OMN-15837): 240,027 bytes, on
# onex.snapshot.projection.live-events.v1. The default bound must exceed it so
# the ordinary case still batches instead of degrading to one record per fetch.
MEASURED_MAX_LIVE_PAYLOAD_BYTES: int = 240_027


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
        """The first AIOKafkaConsumer() construction gets the config's own
        max_partition_fetch_bytes, which defaults to 256 KiB."""
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
            assert (
                call_kwargs["max_partition_fetch_bytes"]
                == config.max_partition_fetch_bytes
            )
            assert config.max_partition_fetch_bytes == 262_144

    @pytest.mark.asyncio
    async def test_consumer_bound_is_decoupled_from_producer_max_request_size(
        self, mock_producer: AsyncMock
    ) -> None:
        """Raising the producer ceiling must NOT drag the consumer buffer up
        with it -- that coupling is what multiplied by the wired consumer count
        and OOM-looped the runtime (OMN-15837)."""
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
            assert call_kwargs["max_partition_fetch_bytes"] == 262_144
            assert call_kwargs["max_partition_fetch_bytes"] != config.max_request_size

    @pytest.mark.asyncio
    async def test_custom_max_partition_fetch_bytes_propagates_to_consumer(
        self, mock_producer: AsyncMock
    ) -> None:
        """An explicitly configured bound reaches the consumer."""
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
                max_partition_fetch_bytes=524_288,
            )
            event_bus = EventBusKafka(config=config)

            await event_bus._start_consumer_for_topic("events", "my-group")

            call_kwargs = consumer_cls.call_args.kwargs
            assert call_kwargs["max_partition_fetch_bytes"] == 524_288

    @pytest.mark.asyncio
    async def test_recreated_consumer_after_metadata_retry_receives_max_partition_fetch_bytes(
        self, mock_producer: AsyncMock
    ) -> None:
        """The consumer recreated inside the metadata-propagation retry loop
        (after UnknownTopicOrPartitionError on the first start()) also gets the
        bound -- the second AIOKafkaConsumer() construction site, not just the
        first. Without this the retry path silently reverts to aiokafka's own
        1 MiB default, which is the very number this ticket removes."""
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
                == config.max_partition_fetch_bytes
            )


@pytest.mark.unit
class TestMaxPartitionFetchBytesConfigField:
    """The typed config field itself: default, bounds, env override."""

    def test_default_is_256_kib(self) -> None:
        config = ModelKafkaEventBusConfig(bootstrap_servers=TEST_BOOTSTRAP_SERVERS)
        assert config.max_partition_fetch_bytes == 262_144

    def test_default_exceeds_the_largest_measured_live_payload(self) -> None:
        """The bound is only worth 256 KiB if ordinary traffic still fits under
        it. 240,027 bytes is the largest value measured across 30,859 sampled
        records on the lab bus (OMN-15837)."""
        config = ModelKafkaEventBusConfig(bootstrap_servers=TEST_BOOTSTRAP_SERVERS)
        assert config.max_partition_fetch_bytes > MEASURED_MAX_LIVE_PAYLOAD_BYTES

    def test_default_frees_more_than_the_measured_oom_overshoot(self) -> None:
        """382 wired consumers x the reclaimed per-consumer bytes must exceed
        the measured overshoot (anon-rss 1,557,120 kB at kill against a
        MemLimit of 1,572,864 kB is the tightest of the sampled kills; the
        widest overshoot measured was ~50 MB)."""
        config = ModelKafkaEventBusConfig(bootstrap_servers=TEST_BOOTSTRAP_SERVERS)
        wired_consumers = 382
        reclaimed = wired_consumers * (
            config.max_request_size - config.max_partition_fetch_bytes
        )
        measured_overshoot_bytes = 50 * 1024 * 1024
        assert reclaimed > measured_overshoot_bytes
        assert reclaimed > 280 * 1024 * 1024

    def test_rejects_value_below_floor(self) -> None:
        with pytest.raises(ValueError):
            ModelKafkaEventBusConfig(
                bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
                max_partition_fetch_bytes=1024,
            )

    def test_rejects_value_above_ceiling(self) -> None:
        with pytest.raises(ValueError):
            ModelKafkaEventBusConfig(
                bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
                max_partition_fetch_bytes=52_428_801,
            )

    def test_accepts_floor_and_ceiling_exactly(self) -> None:
        low = ModelKafkaEventBusConfig(
            bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
            max_partition_fetch_bytes=16_384,
        )
        high = ModelKafkaEventBusConfig(
            bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
            max_partition_fetch_bytes=52_428_800,
        )
        assert low.max_partition_fetch_bytes == 16_384
        assert high.max_partition_fetch_bytes == 52_428_800

    def test_env_override_applies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", TEST_BOOTSTRAP_SERVERS)
        monkeypatch.setenv("KAFKA_MAX_PARTITION_FETCH_BYTES", "524288")
        config = ModelKafkaEventBusConfig.default()
        assert config.max_partition_fetch_bytes == 524_288

    def test_env_override_absent_keeps_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control for the test above: without the variable the
        default stands, so a passing override test is not vacuous."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", TEST_BOOTSTRAP_SERVERS)
        monkeypatch.delenv("KAFKA_MAX_PARTITION_FETCH_BYTES", raising=False)
        config = ModelKafkaEventBusConfig.default()
        assert config.max_partition_fetch_bytes == 262_144
