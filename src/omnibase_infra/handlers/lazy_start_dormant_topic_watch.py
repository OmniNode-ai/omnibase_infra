# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Watch the topics of dormant contracts for a first message (OMN-26316).

A contract the lane left dormant has no subscription, so a client that writes a
command straight to the broker (the house routing overlay's publisher does) reaches
no consumer, and the front door's two ways in (the ingress topic and the runtime's
own publish gate) never see it. One group-less, commit-free consumer reads the
topics those contracts subscribe to from their current end. It joins no group and
moves no offset a real consumer will later read from, the same reader the boot
subscription demand split uses (OMN-18087). The first record to reach a topic calls
``on_first_record`` once for that topic; the caller wakes the dormant consumers
there, whose own subscription then takes the topic over.

A topic the broker does not hold cannot be read, so a message to it (the broker
creates it on the write, or refuses the write) is not watched; the count is logged.
A topic whose wake failed is not watched again in this process: the wake is logged
with the contract's name, and the next request through the ingress or the publish
gate wakes it again.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Mapping

from aiokafka.admin import AIOKafkaAdminClient
from aiokafka.structs import TopicPartition

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.kafka_transport import create_offset_consumer
from omnibase_infra.handlers.kafka_idle_topic_watches import (
    nothing_to_stop,
    watch_for_first_records,
)
from omnibase_infra.topics.topic_namespace import apply_topic_namespace

logger = logging.getLogger(__name__)


class LazyStartDormantTopicWatch:
    """Watch dormant contracts' topics with one group-less, commit-free consumer."""

    def __init__(self, container: ModelONEXContainer) -> None:
        # The registry uses the interface as a key, never as a constructor.
        key: type[object] = ProtocolEventBusSubscriber
        bus = container.get_service(key)
        if not isinstance(bus, EventBusKafka):
            raise TypeError("the dormant-topic watch reads the Kafka bus's own broker")
        self._config = bus.config

    async def watch(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
        request_chunk: int,
    ) -> Callable[[], Awaitable[None]]:
        """Call ``on_first_record(topic)`` once, on the first record after now.

        A lane's dormant set is in the thousands and one request across a whole
        broker timed out on the lab (measured by the topic-activity sampler,
        2026-09-26), so topics are described and partitions are measured
        ``request_chunk`` at a time (the policy's ``topic_request_chunk``).

        Returns the stop of the watch. An unreadable broker raises: the caller
        logs it and the front door's other two ways in stay.
        """
        # The bus addresses a lane-namespaced physical topic for each contract
        # topic; read the same physical topics and report contract names back.
        logical_of = {apply_topic_namespace(topic): topic for topic in topics}
        auth = build_aiokafka_auth_kwargs(self._config)
        admin = AIOKafkaAdminClient(
            bootstrap_servers=self._config.bootstrap_servers, **auth
        )
        await admin.start()
        try:
            existing = set(await admin.list_topics())
            wanted = sorted(set(logical_of) & existing)
            described: list[Mapping[str, object]] = []
            for start in range(0, len(wanted), request_chunk):
                described.extend(
                    await admin.describe_topics(wanted[start : start + request_chunk])
                )
        finally:
            await admin.close()
        partitions = [
            TopicPartition(str(entry["topic"]), int(partition["partition"]))
            for entry in described
            if not entry.get("error_code") and isinstance(entry["partitions"], list)
            for partition in entry["partitions"]
        ]
        if not partitions:
            logger.info(
                "Lazy start: no topic of the %d dormant contract topic(s) exists on "
                "the broker; nothing to watch (OMN-26316)",
                len(logical_of),
            )
            return nothing_to_stop
        consumer = create_offset_consumer(self._config)
        await consumer.start()
        end: dict[TopicPartition, int] = {}
        try:
            consumer.assign(partitions)
            for start in range(0, len(partitions), request_chunk):
                chunk = partitions[start : start + request_chunk]
                end.update(await consumer.end_offsets(chunk))
        except BaseException:
            await consumer.stop()
            raise
        watched = [tp for tp in partitions if tp in end]
        consumer.assign(watched)
        for tp in watched:
            consumer.seek(tp, end[tp])
        logger.info(
            "Lazy start: watching %d of %d dormant contract topic(s) for a first "
            "message; the rest are not on the broker (OMN-26316)",
            len({tp.topic for tp in watched}),
            len(logical_of),
        )

        async def on_first_physical_record(physical: str) -> None:
            await on_first_record(logical_of[physical])

        task = asyncio.create_task(
            watch_for_first_records(
                consumer,
                watched,
                on_first_physical_record,
                self._config.consumer_poll_timeout_ms,
            ),
            name="lazy-start-dormant-topic-watch",
        )

        def _log_watch_death(done: asyncio.Task[None]) -> None:
            if not done.cancelled() and done.exception() is not None:
                logger.error(
                    "Lazy start: the dormant-topic watch died; a message to a "
                    "dormant consumer's topic no longer wakes it (OMN-26316)",
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

        return stop


__all__: list[str] = ["LazyStartDormantTopicWatch"]
