# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka implementation of the boot subscription demand split (OMN-18087).

One group-less consumer measures the boot set's subscribe topics and then
watches the idle ones. It joins no consumer group and commits no offset, so it
costs no group join (each join costs the broker's group_initial_rebalance_delay,
3 s on the dev lane) and moves no offset a real consumer will later read from.

A topic is in use when any of its partitions holds a retained record (its high
watermark is above its low watermark). The measure is the broker's own state at
boot; nothing in it is a list of topics or nodes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection

from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

logger = logging.getLogger(__name__)

_POLL_TIMEOUT_MS = 1000


class HandlerKafkaTopicDemandSource:
    """Measure and watch topics with one group-less, commit-free consumer."""

    def __init__(self, config: ModelKafkaEventBusConfig) -> None:
        self._config = config

    async def watch_idle_topics(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
    ) -> tuple[frozenset[str], Callable[[], Awaitable[None]]]:
        consumer = AIOKafkaConsumer(
            bootstrap_servers=self._config.bootstrap_servers,
            group_id=None,
            enable_auto_commit=False,
            **build_aiokafka_auth_kwargs(self._config),
        )
        await consumer.start()
        try:
            # topics() refreshes cluster metadata for every topic the principal
            # can describe; a topic it cannot see is left out and stays idle.
            visible = await consumer.topics()
            partitions = [
                TopicPartition(topic, partition)
                for topic in sorted(set(topics) & visible)
                for partition in sorted(consumer.partitions_for_topic(topic) or ())
            ]
            low = await consumer.beginning_offsets(partitions) if partitions else {}
            high = await consumer.end_offsets(partitions) if partitions else {}
        except BaseException:
            await consumer.stop()
            raise
        in_use = frozenset(tp.topic for tp in partitions if high[tp] > low[tp])
        watched = [tp for tp in partitions if tp.topic not in in_use]
        if not watched:
            await consumer.stop()

            async def _nothing_to_stop() -> None:
                return None

            return in_use, _nothing_to_stop

        consumer.assign(watched)
        for tp in watched:
            consumer.seek(tp, high[tp])
        task = asyncio.create_task(
            self._watch(consumer, watched, on_first_record),
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
