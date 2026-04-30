# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka replay consumer protocol."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from aiokafka import TopicPartition

from omnibase_infra.nodes.node_kafka_replay_compute.protocols.protocol_kafka_message import (
    ProtocolKafkaMessage,
)
from omnibase_infra.nodes.node_kafka_replay_compute.protocols.protocol_offset_and_timestamp import (
    ProtocolOffsetAndTimestamp,
)


class ProtocolKafkaReplayConsumer(Protocol):
    """Subset of AIOKafkaConsumer used by replay compute."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def partitions_for_topic(self, topic: str) -> set[int] | None: ...

    async def beginning_offsets(
        self, partitions: Iterable[TopicPartition]
    ) -> Mapping[TopicPartition, int]: ...

    async def end_offsets(
        self, partitions: Iterable[TopicPartition]
    ) -> Mapping[TopicPartition, int]: ...

    async def offsets_for_times(
        self, timestamps: Mapping[TopicPartition, int]
    ) -> Mapping[TopicPartition, ProtocolOffsetAndTimestamp | None]: ...

    def assign(self, partitions: Iterable[TopicPartition]) -> None: ...

    def seek(self, partition: TopicPartition, offset: int) -> None: ...

    async def getmany(
        self, *partitions: TopicPartition, timeout_ms: int
    ) -> Mapping[TopicPartition, Sequence[ProtocolKafkaMessage]]: ...


__all__ = ["ProtocolKafkaReplayConsumer"]
