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

from aiokafka import AIOKafkaConsumer
from aiokafka.admin import AIOKafkaAdminClient
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.kafka_transport import create_groupless_consumer
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.topics.topic_namespace import apply_topic_namespace

logger = logging.getLogger(__name__)

_POLL_TIMEOUT_MS = 1000
# Partitions per offset request; one request across a whole broker timed out on
# the lab (measured by the topic-activity sampler, 2026-09-26).
_OFFSET_CHUNK = 100


class HandlerKafkaTopicDemandSource:
    """Measure and watch topics with one group-less, commit-free consumer."""

    def __init__(self, config: ModelKafkaEventBusConfig) -> None:
        self._config = config

    async def watch_idle_topics(
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
        consumer = create_groupless_consumer(self._config)
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
        with_records = {tp.topic for tp in partitions if high[tp] > low[tp]}
        readable = {tp.topic for tp in partitions}
        in_use = frozenset(
            logical
            for physical, logical in logical_of.items()
            if physical in with_records or physical not in readable
        )
        watched = [tp for tp in partitions if tp.topic not in with_records]
        if not watched:
            await consumer.stop()

            async def _nothing_to_stop() -> None:
                return None

            return in_use, _nothing_to_stop

        consumer.assign(watched)
        for tp in watched:
            consumer.seek(tp, high[tp])

        async def on_first_physical_record(physical: str) -> None:
            await on_first_record(logical_of[physical])

        task = asyncio.create_task(
            self._watch(consumer, watched, on_first_physical_record),
            name="boot-subscription-demand-watch",
        )

        async def stop() -> None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                await consumer.stop()

        return in_use, stop

    async def _watch(
        self,
        consumer: AIOKafkaConsumer,
        watched: list[TopicPartition],
        on_first_record: Callable[[str], Awaitable[None]],
    ) -> None:
        remaining = list(watched)
        while remaining:
            batches = await consumer.getmany(timeout_ms=_POLL_TIMEOUT_MS)
            waiting = {tp.topic for tp in remaining}
            arrived = sorted(
                {tp.topic for tp, records in batches.items() if records} & waiting
            )
            if not arrived:
                continue
            # Pause rather than reassign: reassigning resets every remaining
            # partition's position, and a record arriving in that gap is lost.
            started = [tp for tp in remaining if tp.topic in arrived]
            remaining = [tp for tp in remaining if tp.topic not in arrived]
            consumer.pause(*started)
            for topic in arrived:
                try:
                    await on_first_record(topic)
                except Exception:  # boundary: one topic's start never ends the watch
                    logger.exception(
                        "Demand start failed for topic %s (OMN-18087)", topic
                    )


__all__: list[str] = ["HandlerKafkaTopicDemandSource"]
