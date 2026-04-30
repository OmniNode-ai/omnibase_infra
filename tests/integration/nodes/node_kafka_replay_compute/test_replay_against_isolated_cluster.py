# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for Kafka replay compute using an isolated fixture."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest
from aiokafka import TopicPartition

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.projection import ModelSequenceInfo
from omnibase_infra.nodes.node_kafka_replay_compute.handlers.handler_replay import (
    HandlerKafkaReplay,
)
from omnibase_infra.nodes.node_kafka_replay_compute.models import ModelKafkaReplayInput


@dataclass(frozen=True)
class _FixtureRecord:
    topic: str
    partition: int
    offset: int
    value: bytes


class _IsolatedKafkaConsumer:
    def __init__(self, records_by_topic: dict[str, list[bytes]]) -> None:
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
            value = values[position]
            batch[partition] = [
                _FixtureRecord(
                    topic=partition.topic,
                    partition=partition.partition,
                    offset=position,
                    value=value,
                )
            ]
            self._positions[partition] = position + 1
        return batch


def _envelope_bytes(event_type: str, ordinal: int) -> bytes:
    envelope = ModelEventEnvelope[dict[str, object]](
        payload={"ordinal": ordinal},
        correlation_id=uuid4(),
        event_type=event_type,
    )
    return envelope.model_dump_json().encode("utf-8")


@pytest.fixture
def isolated_kafka_records() -> dict[str, list[bytes]]:
    return {
        "dispatch_worker-completed.v1": [
            _envelope_bytes("dispatch_worker-completed.v1", i) for i in range(10)
        ],
        "dispatch-outcome-evaluated.v1": [
            _envelope_bytes("dispatch-outcome-evaluated.v1", i) for i in range(5)
        ],
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replay_against_isolated_cluster(
    isolated_kafka_records: dict[str, list[bytes]],
) -> None:
    handler = HandlerKafkaReplay(
        consumer_factory=lambda _: _IsolatedKafkaConsumer(isolated_kafka_records)
    )

    result = await handler.handle(
        ModelKafkaReplayInput(
            topics=[
                "dispatch_worker-completed.v1",
                "dispatch-outcome-evaluated.v1",
            ],
            target_cluster_bootstrap="isolated-redpanda:9092",
            target_consumer_group="omn-10392-isolated-replay",
            expected_event_count=15,
            progress_interval_events=5,
        )
    )

    assert result.events_replayed == 15
    assert result.last_offset_per_topic == {
        "dispatch_worker-completed.v1": 9,
        "dispatch-outcome-evaluated.v1": 4,
    }
    assert result.sequence_info_per_topic[
        "dispatch_worker-completed.v1:0"
    ] == ModelSequenceInfo.from_kafka(partition=0, offset=9)
    assert len(result.correlation_id_chain) == 15
    assert result.failed_event_offsets == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replay_resumes_from_sequence_info(
    isolated_kafka_records: dict[str, list[bytes]],
) -> None:
    handler = HandlerKafkaReplay(
        consumer_factory=lambda _: _IsolatedKafkaConsumer(isolated_kafka_records)
    )

    result = await handler.handle(
        ModelKafkaReplayInput(
            topics=["dispatch_worker-completed.v1"],
            target_cluster_bootstrap="isolated-redpanda:9092",
            target_consumer_group="omn-10392-isolated-replay-resume",
            expected_event_count=6,
            resume_from={
                "dispatch_worker-completed.v1:0": ModelSequenceInfo.from_kafka(
                    partition=0,
                    offset=3,
                )
            },
        )
    )

    assert result.events_replayed == 6
    assert result.last_offset_per_topic == {"dispatch_worker-completed.v1": 9}
