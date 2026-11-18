# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka implementation of the boot subscription demand split (OMN-18087).

One group-less consumer measures the boot set's subscribe topics and then
watches the idle ones. It joins no consumer group and commits no offset, so it
costs no group join (each join costs the broker's group_initial_rebalance_delay,
3 s on the dev lane) and moves no offset a real consumer will later read from.

A topic is in use when any of its partitions holds a retained record (its high
watermark is above its low watermark), or when it cannot be read at all, so that
a topic not yet provisioned keeps its contract on the boot path. The measure is
the broker's own state at boot; nothing in it is a list of topics or nodes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection

from aiokafka.admin import AIOKafkaAdminClient
from aiokafka.structs import TopicPartition

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.kafka_transport import create_offset_consumer
from omnibase_infra.handlers.kafka_idle_topic_watches import (
    IdleTopicWatch,
    nothing_to_stop,
    watch_for_first_records,
)
from omnibase_infra.topics.topic_namespace import apply_topic_namespace

logger = logging.getLogger(__name__)

# Partitions per offset request; one request across a whole broker timed out on
# the lab (measured by the topic-activity sampler, 2026-09-26).
_OFFSET_CHUNK = 100


def _log_demand_measure(
    topics: Collection[str], retained: Collection[str], readable: Collection[str]
) -> None:
    """Log the split's counts in one line (OMN-24223).

    A topic is retained (holds a record), unreadable (missing or not visible) or
    idle and watched; ``retained + unreadable`` is the number in use. The line
    starts with the words the boot log is searched for.
    """
    logger.info(
        "Boot subscription demand measure: topics=%d retained=%d unreadable=%d "
        "idle_watched=%d (OMN-24223)",
        len(topics),
        len(retained),
        len(set(topics) - set(readable)),
        len(set(readable) - set(retained)),
    )


class HandlerKafkaTopicDemandSource:
    """Measure and watch topics with one group-less, commit-free consumer."""

    def __init__(self, container: ModelONEXContainer) -> None:
        # The registry uses the interface as a key, never as a constructor.
        key: type[object] = ProtocolEventBusSubscriber
        bus = container.get_service(key)
        if not isinstance(bus, EventBusKafka):
            raise TypeError("the demand split reads the Kafka bus's own broker")
        self._config = bus.config

    @property
    def handler_type(self) -> EnumHandlerType:
        """INFRA_HANDLER: reads the broker's offsets and records."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """EFFECT: talks to the Kafka broker."""
        return EnumHandlerTypeCategory.EFFECT

    async def watch_idle_topics(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
    ) -> tuple[frozenset[str], Callable[[], Awaitable[None]]]:
        try:
            return await self._measure_and_watch(topics, on_first_record)
        except Exception:  # boundary: an unmeasurable broker keeps the boot path
            logger.exception(
                "Boot subscription demand measure failed; every contract joins "
                "at boot (OMN-18087)"
            )
            return frozenset(topics), nothing_to_stop

    async def _measure_and_watch(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
    ) -> tuple[frozenset[str], Callable[[], Awaitable[None]]]:
        auth = build_aiokafka_auth_kwargs(self._config)
        # The bus addresses a lane-namespaced physical topic for each contract
        # topic; read the same physical topics and report contract names back.
        logical_of = {apply_topic_namespace(topic): topic for topic in topics}
        # A group-less consumer only knows partitions for topics it has been
        # assigned, so the partitions come from the admin client.
        admin = AIOKafkaAdminClient(
            bootstrap_servers=self._config.bootstrap_servers, **auth
        )
        await admin.start()
        try:
            existing = set(await admin.list_topics())
            wanted = sorted(set(logical_of) & existing)
            described = await admin.describe_topics(wanted) if wanted else []
        finally:
            await admin.close()
        partitions = [
            TopicPartition(str(entry["topic"]), int(partition["partition"]))
            for entry in described
            if not entry.get("error_code")
            for partition in entry["partitions"]
        ]
        if not partitions:
            # Nothing to measure or watch: the boot set is empty (a runtime
            # profile that owns no auto-wired contract) or none of its topics
            # exists yet, so every topic is unreadable and in use. Starting a
            # reader here would stop it before aiokafka's group-less
            # coordinator task had run once, and that stop raises
            # CancelledError out of this call, past the boundary above, and ends
            # the kernel (tenant-projection-writer restart loop on the laptop
            # profile, run 40881579955).
            _log_demand_measure(logical_of, (), ())
            return frozenset(logical_of.values()), nothing_to_stop
        consumer = create_offset_consumer(self._config)
        await consumer.start()
        low: dict[TopicPartition, int] = {}
        high: dict[TopicPartition, int] = {}
        try:
            # Assigning first loads the metadata the offset requests wait on;
            # bounded chunks keep one request from timing out on a large broker.
            consumer.assign(partitions)
            for start in range(0, len(partitions), _OFFSET_CHUNK):
                chunk = partitions[start : start + _OFFSET_CHUNK]
                low.update(await consumer.beginning_offsets(chunk))
                high.update(await consumer.end_offsets(chunk))
        except BaseException:
            await consumer.stop()
            raise
        # A topic that cannot be read here (not yet provisioned, or not visible
        # to this principal) counts as in use: it cannot be watched, so its
        # contract joins at boot exactly as it did before.
        # A partition the broker returned no offset for is unreadable too.
        measured = [tp for tp in partitions if tp in low and tp in high]
        with_records = {tp.topic for tp in measured if high[tp] > low[tp]}
        readable = {tp.topic for tp in measured}
        in_use = frozenset(
            logical
            for physical, logical in logical_of.items()
            if physical in with_records or physical not in readable
        )
        watched = [tp for tp in measured if tp.topic not in with_records]
        _log_demand_measure(logical_of, with_records, readable)
        if not watched:
            await consumer.stop()
            return in_use, nothing_to_stop

        consumer.assign(watched)
        for tp in watched:
            consumer.seek(tp, high[tp])

        idle = IdleTopicWatch(logical_of[tp.topic] for tp in watched)

        async def on_first_physical_record(physical: str) -> None:
            # _watch reports PHYSICAL (lane-namespaced) names on both its arrival
            # path and its fallback path; map them back to the contract names.
            await idle.start(logical_of[physical], on_first_record)

        task = asyncio.create_task(
            watch_for_first_records(
                consumer,
                watched,
                on_first_physical_record,
                self._config.consumer_poll_timeout_ms,
            ),
            name="boot-subscription-demand-watch",
        )

        def _log_watch_death(done: asyncio.Task[None]) -> None:
            idle.release()
            if not done.cancelled() and done.exception() is not None:
                logger.error(
                    "Boot subscription demand watch died; deferred contracts "
                    "will not start on demand (OMN-18087)",
                    exc_info=done.exception(),
                )

        task.add_done_callback(_log_watch_death)

        async def stop() -> None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                await consumer.stop()

        return in_use, stop


__all__: list[str] = ["HandlerKafkaTopicDemandSource"]
