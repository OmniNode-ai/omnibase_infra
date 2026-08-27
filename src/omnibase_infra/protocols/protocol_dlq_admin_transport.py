# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Read-only Kafka admin surface the DLQ probe needs (OMN-16769).

Declared structurally, and deliberately narrow: these five methods are the
ENTIRE broker surface this node touches, and every one of them is a READ.
There is no produce, no commit, no topic create/delete/alter, and no
consumer-group mutation reachable through this protocol — the scheduled
workflow is dry-run-safe by construction rather than by discipline.

Modelled on the existing ``ProtocolKafkaLagConsumer`` /
``AdapterKafkaAdminLag`` pair (OMN-12632), which exists for the same
reason: keep ``aiokafka`` out of the import path at parse time and let
tests supply a fake that mirrors the real client's shapes — including its
sharp edges (``offsets_for_times`` returning ``None``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

# (topic, partition). A plain tuple rather than aiokafka's TopicPartition so
# the protocol — and every test fake implementing it — stays import-free.
type TopicPartition = tuple[str, int]


class ProtocolDlqAdminTransport(Protocol):
    """Read-only offset/metadata surface over a Kafka-compatible broker."""

    async def list_topics(self) -> Sequence[str]:
        """Every topic name the broker knows about."""
        ...

    async def partitions_for_topic(self, topic: str) -> Sequence[int]:
        """Partition ids for one topic."""
        ...

    async def beginning_offsets(
        self, partitions: Sequence[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        """Log-start offset per partition. MOVES as retention trims the log."""
        ...

    async def end_offsets(
        self, partitions: Sequence[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        """High-water mark per partition (lifetime records ever written)."""
        ...

    async def offsets_for_times(
        self, partition_timestamps: Mapping[TopicPartition, int]
    ) -> Mapping[TopicPartition, int | None]:
        """First offset at/after each partition's timestamp (epoch ms).

        Returns ``None`` for a partition with no record at or after the
        requested timestamp — i.e. nothing arrived in the window. Callers
        MUST normalize that to the partition's high-water mark; treating a
        ``None`` as offset 0 would report the topic's entire lifetime
        volume as one window's arrivals.
        """
        ...


__all__ = ["ProtocolDlqAdminTransport", "TopicPartition"]
