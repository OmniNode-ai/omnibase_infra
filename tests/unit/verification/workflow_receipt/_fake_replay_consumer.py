# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-memory ``ProtocolKafkaReplayConsumer`` fixture for OMN-15095's
verifier tests. Same shape as
``tests/unit/nodes/node_kafka_replay_compute/_defb_corpus.py``'s
``FakeReplayConsumer`` -- this drives the REAL ``HandlerKafkaReplay`` through
its documented test seam (``consumer_factory``), not a hand-rolled mock of
the handler itself. Not a test module (leading underscore).
"""

from __future__ import annotations

from dataclasses import dataclass

from aiokafka import TopicPartition


@dataclass(frozen=True)
class _FixtureRecord:
    topic: str
    partition: int
    offset: int
    value: bytes | None


class FakeReplayConsumer:
    """In-memory replay consumer over one topic's recorded byte records."""

    def __init__(self, records_by_topic: dict[str, list[bytes | None]]) -> None:
        self._records_by_topic = records_by_topic
        self._positions: dict[TopicPartition, int] = {}
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def partitions_for_topic(self, topic: str) -> set[int] | None:
        if topic not in self._records_by_topic:
            return None
        return {0}

    async def beginning_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return dict.fromkeys(partitions, 0)

    async def end_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        return {
            partition: len(self._records_by_topic[partition.topic])
            for partition in partitions
        }

    async def offsets_for_times(
        self, timestamps: dict[TopicPartition, int]
    ) -> dict[TopicPartition, None]:
        return dict.fromkeys(timestamps)

    def assign(self, partitions: list[TopicPartition]) -> None:
        for partition in partitions:
            self._positions.setdefault(partition, 0)

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self._positions[partition] = offset

    async def getmany(
        self, *partitions: TopicPartition, timeout_ms: int
    ) -> dict[TopicPartition, list[_FixtureRecord]]:
        del timeout_ms
        batch: dict[TopicPartition, list[_FixtureRecord]] = {}
        for partition in partitions:
            position = self._positions.get(partition, 0)
            values = self._records_by_topic[partition.topic]
            if position >= len(values):
                continue
            batch[partition] = [
                _FixtureRecord(
                    topic=partition.topic,
                    partition=partition.partition,
                    offset=position,
                    value=values[position],
                )
            ]
            self._positions[partition] = position + 1
        return batch


__all__ = ["FakeReplayConsumer"]
