# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A group-less single-topic Kafka reader (OMN-17427)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Final

from aiokafka.errors import KafkaError

if TYPE_CHECKING:
    from aiokafka import AIOKafkaConsumer

    from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage

__all__ = ["UNGROUPED_AUTO_OFFSET_RESET", "ModelKafkaUngroupedReader"]

logger = logging.getLogger(__name__)

#: Where a group-less read begins: the log end at start. A reply read must not
#: see replies to earlier commands.
UNGROUPED_AUTO_OFFSET_RESET: Final[str] = "latest"


class ModelKafkaUngroupedReader:
    """One group-less consumer of one topic, feeding one callback (OMN-17427).

    A one-shot request/reply caller reads one reply and leaves. Joining a
    consumer group for that read leaves the group behind: the consumer
    unsubscribes, the group goes Empty, and its auto-committed offsets keep it on
    the broker until offset retention expires it. Measured on the dev broker on
    2026-10-29: 8835 groups, 8191 of them Empty, about 6978 of those minted one
    per call by the SCM and work-ledger callers. On a one-shard broker with
    several hundred live consumers that starved group coordination and detached
    the GitHub write consumer.

    This reader takes the topic's partitions directly (no group, so no join, no
    rebalance, no coordinator round trip) and commits nothing, so nothing is left
    on the broker when it stops. It begins at the log end and ``start`` does not
    return until every assigned partition has a resolved position, so a record
    published as soon as ``start`` returns is read.
    """

    def __init__(
        self,
        *,
        consumer: AIOKafkaConsumer,
        topic: str,
        on_message: Callable[[ModelEventMessage], Awaitable[None]],
        to_message: Callable[[object, str], ModelEventMessage],
    ) -> None:
        self._consumer = consumer
        self._topic = topic
        self._on_message = on_message
        self._to_message = to_message
        self._task: asyncio.Task[None] | None = None
        self._stopped = False

    async def start(self, *, timeout_seconds: float) -> None:
        """Assign the partitions, resolve their positions, then begin reading.

        Raises ``TimeoutError`` when the assignment or a position does not
        resolve in ``timeout_seconds``; the consumer is stopped first.
        """
        try:
            async with asyncio.timeout(timeout_seconds):
                await self._consumer.start()
                for partition in self._consumer.assignment():
                    await self._consumer.position(partition)
        except BaseException:
            await self._stop_consumer()
            raise
        self._task = asyncio.create_task(self._read())

    async def stop(self) -> None:
        """Stop reading and release the consumer. Safe to call twice."""
        if self._stopped:
            return
        self._stopped = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._stop_consumer()

    async def _stop_consumer(self) -> None:
        try:
            await self._consumer.stop()
        except (KafkaError, OSError, RuntimeError, TimeoutError):
            logger.warning(
                "Error stopping group-less consumer for topic %s",
                self._topic,
                exc_info=True,
            )

    async def _read(self) -> None:
        while True:
            record = await self._consumer.getone()
            try:
                await self._on_message(self._to_message(record, self._topic))
            except Exception:
                logger.exception(
                    "Group-less read of topic %s: callback failed", self._topic
                )
