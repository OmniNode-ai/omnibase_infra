# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for the consumer surface the rejoin supervisor uses (OMN-18640).

Declared here, rather than typing the supervisor against ``AIOKafkaConsumer``
directly, for one reason that is the point of the ticket: the wedge is only
reproducible against a consumer that can be told to behave as the 2026-09-18
one did -- assignment held, leaders answering, fetch position frozen. A
structural protocol lets that fake be a first-class typed implementation rather
than a mock, so the replay test checks the same contract the runtime uses.

Every method here is one ``AIOKafkaConsumer`` already exposes, with the same
signature, so ``AIOKafkaConsumer`` satisfies it without an adapter.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol

from aiokafka.structs import ConsumerRecord, TopicPartition

#: One fetched Kafka record. The runtime's consumers deserialize in
#: ``_kafka_msg_to_model`` rather than in the client, so both type parameters
#: are the wire form.
type FetchedRecord = ConsumerRecord[bytes, bytes]


class ProtocolRejoinableConsumer(Protocol):
    """Minimal consumer surface needed to detect a wedge and recover from it."""

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = ...,
        max_records: int | None = ...,
    ) -> Mapping[TopicPartition, Sequence[FetchedRecord]]:
        """Fetch buffered records, returning after at most ``timeout_ms``."""
        ...

    def assignment(self) -> set[TopicPartition]:
        """Partitions currently assigned to this consumer."""
        ...

    async def position(self, partition: TopicPartition) -> int:
        """The consumer's own next-fetch offset for a partition."""
        ...

    async def end_offsets(
        self, partitions: Collection[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        """Log end offsets from the partition LEADERS.

        Deliberately not ``highwater``, which returns the value cached by the
        last successful fetch and therefore reads as caught-up on exactly the
        consumer this supervisor exists to catch.
        """
        ...

    def seek(self, partition: TopicPartition, offset: int) -> None:
        """Move the fetch position."""
        ...

    async def stop(self) -> None:
        """Close the consumer and leave its group."""
        ...


__all__ = ["FetchedRecord", "ProtocolRejoinableConsumer"]
