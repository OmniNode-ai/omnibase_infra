# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pins a serial batch's fetch positions while a record is unfinished (OMN-19355)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiokafka import AIOKafkaConsumer
    from aiokafka.structs import TopicPartition

__all__ = ["ModelSerialBatchWithhold"]

logger = logging.getLogger(__name__)


class ModelSerialBatchWithhold:
    """Pins a serial batch's fetch positions at its unfinished records.

    After ``getmany`` the consumer's fetch position is past the whole batch.
    Commits are completion-based (OMN-21229), so that position is never
    committed, but it is where the loop would carry on from. ``engage`` seeks
    the in-flight record's partition back to that record and every partition
    of the batch not yet started back to its first record, so the loop
    refetches from the unfinished records instead of continuing a batch it
    only half ran (OMN-19355, OMN-15232).
    """

    __slots__ = ("_consumer", "_positions", "engaged")

    def __init__(
        self,
        consumer: AIOKafkaConsumer,
        positions: list[tuple[TopicPartition, int]],
    ) -> None:
        self._consumer = consumer
        self._positions = positions
        self.engaged = False

    def engage(self) -> None:
        if self.engaged:
            return
        self.engaged = True
        for partition, offset in self._positions:
            try:
                self._consumer.seek(partition, offset)
            except Exception:
                logger.exception(
                    "slow_dispatch_withhold_failed topic=%s partition=%s offset=%s "
                    "-- could not pin the fetch position at an unfinished record; "
                    "the batch may carry on past it (OMN-19355)",
                    partition.topic,
                    partition.partition,
                    offset,
                )
