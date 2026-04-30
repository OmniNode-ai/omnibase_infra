# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka replay consumer protocol."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from omnibase_infra.nodes.node_kafka_replay_compute.protocols.protocol_kafka_message import (
    ProtocolKafkaMessage,
)
from omnibase_infra.nodes.node_kafka_replay_compute.protocols.protocol_offset_and_timestamp import (
    ProtocolOffsetAndTimestamp,
)
from omnibase_infra.nodes.node_kafka_replay_compute.protocols.protocol_topic_partition import (
    ProtocolTopicPartition,
)


class ProtocolKafkaReplayConsumer(Protocol):
    """Subset of AIOKafkaConsumer used by replay compute."""

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def partitions_for_topic(self, topic: str) -> set[int] | None:
        raise NotImplementedError

    async def beginning_offsets(
        self, partitions: Iterable[ProtocolTopicPartition]
    ) -> Mapping[ProtocolTopicPartition, int]:
        raise NotImplementedError

    async def end_offsets(
        self, partitions: Iterable[ProtocolTopicPartition]
    ) -> Mapping[ProtocolTopicPartition, int]:
        raise NotImplementedError

    async def offsets_for_times(
        self, timestamps: Mapping[ProtocolTopicPartition, int]
    ) -> Mapping[ProtocolTopicPartition, ProtocolOffsetAndTimestamp | None]:
        raise NotImplementedError

    def assign(self, partitions: Iterable[ProtocolTopicPartition]) -> None:
        raise NotImplementedError

    def seek(self, partition: ProtocolTopicPartition, offset: int) -> None:
        raise NotImplementedError

    async def getmany(
        self, *partitions: ProtocolTopicPartition, timeout_ms: int
    ) -> Mapping[ProtocolTopicPartition, Sequence[ProtocolKafkaMessage]]:
        raise NotImplementedError


__all__ = ["ProtocolKafkaReplayConsumer"]
