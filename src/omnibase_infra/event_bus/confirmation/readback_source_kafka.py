# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Kafka readback surface: assign + seek to the produced coordinate (OMN-15861).

The ``gateway_canary_probe`` readback (OMN-15741) joins a consumer group at
``auto_offset_reset="latest"`` and polls until it recognises a marker payload.
That shape is right for a healthcheck -- it does not know where its record
landed -- but wrong here, because a ``ModelPublishReceipt`` *does* carry the
exact coordinate. So this source keeps the probe's produce-then-poll-with-a-
deadline structure and replaces the scan with the precise primitive:

* ``assign`` one ``TopicPartition`` -- no consumer group, no rebalance, no
  committed-offset side effects. A confirmation must never mutate group state;
  joining a group to read one record would move committed offsets for whoever
  else uses that group id.
* ``end_offsets`` -- the high-water mark. ``hwm > receipt.offset`` is the
  broker's own statement that the record at that offset is committed and
  readable. This is the actual durability fact.
* ``seek`` + one fetch -- proves the record is not merely counted but readable,
  and that its coordinate is the one the receipt claims.

An unreachable broker raises out of ``observe``; ``BrokerReadbackStrategy``
turns that into ``UNKNOWN`` (fails closed). It is deliberately NOT caught here:
this layer reports facts, the strategy decides policy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition

from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt


@runtime_checkable
class ProtocolSeekableConsumer(Protocol):
    """The consumer surface a coordinate readback needs.

    Narrowed to five calls so the readback loop is unit-testable against a fake
    without a broker, and so this module never grows an accidental dependency on
    the rest of the aiokafka consumer API.
    """

    async def start(self) -> None:
        """Connect to the cluster."""
        ...

    async def stop(self) -> None:
        """Disconnect. MUST be safe to call after a failed ``start``."""
        ...

    def assign(self, partitions: list[TopicPartition]) -> None:
        """Bind to explicit partitions without joining a consumer group."""
        ...

    def seek(self, partition: TopicPartition, offset: int) -> None:
        """Position the fetch cursor at ``offset``."""
        ...

    async def end_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        """Return each partition's high-water mark (next offset to be written)."""
        ...

    async def getone(self) -> object:
        """Fetch the next record at the current cursor.

        Typed as ``object`` rather than aiokafka's ``ConsumerRecord`` so this
        protocol stays transport-shaped: the readback only ever reads
        ``topic``/``partition``/``offset`` off the result, and pinning the
        concrete aiokafka class here would make the seam untestable without
        constructing one.
        """
        ...


class KafkaReadbackSource:
    """Confirms a Kafka coordinate by assigning, seeking, and fetching it.

    Args:
        config: Bus config supplying bootstrap servers and auth. The configured
            ``bootstrap_servers`` is compared against ``receipt.cluster`` so a
            receipt from a different cluster is never confirmed here.
        consumer_factory: Builds the consumer for one readback. Injected so the
            loop can be exercised without a broker; defaults to a real,
            group-less ``AIOKafkaConsumer``.
    """

    def __init__(
        self,
        config: ModelKafkaEventBusConfig,
        *,
        consumer_factory: Callable[[], ProtocolSeekableConsumer] | None = None,
    ) -> None:
        self._config = config
        self._consumer_factory = consumer_factory or self._build_default_consumer

    @property
    def transport(self) -> EnumInfraTransportType:
        """This source can only answer for Kafka receipts."""
        return EnumInfraTransportType.KAFKA

    def _build_default_consumer(self) -> ProtocolSeekableConsumer:
        """A group-less consumer: assignment only, no committed-offset effects."""
        consumer = AIOKafkaConsumer(
            bootstrap_servers=self._config.bootstrap_servers,
            enable_auto_commit=False,
            group_id=None,
            retry_backoff_ms=self._config.reconnect_backoff_ms,
            **build_aiokafka_auth_kwargs(self._config),
        )
        return consumer  # type: ignore[return-value]

    async def observe(
        self,
        receipt: ModelPublishReceipt,
        *,
        deadline_seconds: float,
    ) -> bool:
        """Return whether the broker holds a readable record at ``receipt``.

        Raises:
            Exception: Any broker/transport failure. The caller
                (``BrokerReadbackStrategy``) maps this to ``UNKNOWN``; it is not
                translated to ``False`` here because "could not ask" is not
                "answered no".
        """
        if receipt.cluster != self._config.bootstrap_servers:
            return False

        deadline_at = time.monotonic() + deadline_seconds
        consumer = self._consumer_factory()
        await consumer.start()
        try:
            partition = TopicPartition(receipt.topic, receipt.partition)
            consumer.assign([partition])

            # Poll the high-water mark until it passes the receipt's offset.
            # hwm is the offset that will be assigned NEXT, so the record at
            # `receipt.offset` is committed exactly when hwm > receipt.offset.
            while True:
                end_offsets = await consumer.end_offsets([partition])
                high_water_mark = end_offsets.get(partition)
                if high_water_mark is not None and high_water_mark > receipt.offset:
                    break
                if time.monotonic() >= deadline_at:
                    return False
                await asyncio.sleep(min(0.05, max(deadline_at - time.monotonic(), 0.0)))

            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                return False

            consumer.seek(partition, receipt.offset)
            record = await asyncio.wait_for(consumer.getone(), timeout=remaining)
            return bool(
                getattr(record, "topic", None) == receipt.topic
                and int(getattr(record, "partition", -1)) == receipt.partition
                and int(getattr(record, "offset", -1)) == receipt.offset
            )
        except TimeoutError:
            # The record's offset was committed but the fetch did not complete in
            # budget. Not observed -> UNCONFIRMED, and the outbox retries.
            return False
        finally:
            await consumer.stop()


__all__: list[str] = [
    "KafkaReadbackSource",
    "ProtocolSeekableConsumer",
]
