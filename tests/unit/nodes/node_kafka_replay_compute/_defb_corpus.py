# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared def-B parity corpus for node_kafka_replay_compute (OMN-14819).

Single source of truth for the replay input corpus + isolated-consumer fixture used
by BOTH the def-B parity/guard test and the adequacy/hand-flip receipt recorder, so
the selected ``input_hashes`` in the committed receipts are reproducible from the
same inputs the parity test drives. Not a test module (leading underscore -> not
collected by pytest).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from aiokafka import TopicPartition

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.projection import ModelSequenceInfo
from omnibase_infra.nodes.node_kafka_replay_compute.handlers.handler_replay import (
    HandlerKafkaReplay,
)
from omnibase_infra.nodes.node_kafka_replay_compute.models import (
    ModelKafkaReplayInput,
    ModelKafkaReplayOutput,
)

# Deterministic, explicitly-injected isolated target (never .201).
_BOOTSTRAP = "isolated-redpanda:9092"


@dataclass(frozen=True)
class _FixtureRecord:
    topic: str
    partition: int
    offset: int
    value: bytes | None


class FakeReplayConsumer:
    """In-memory replay consumer supporting good, empty (None), and corrupt records."""

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


def _good(ordinal: int) -> bytes:
    envelope = ModelEventEnvelope[dict[str, object]](
        payload={"ordinal": ordinal},
        correlation_id=uuid4(),
        event_type="dispatch_worker-completed.v1",
    )
    return envelope.model_dump_json().encode("utf-8")


def _good_run(n: int) -> list[bytes | None]:
    return [_good(i) for i in range(n)]


@dataclass(frozen=True)
class ReplayCase:
    """One replay parity case: an input, its isolated records, and its expectation."""

    name: str
    command: ModelKafkaReplayInput
    records: dict[str, list[bytes | None]]
    expected_events: int | None = None
    expected_last_offset: dict[str, int] = field(default_factory=dict)
    expected_correlation_len: int | None = None
    expected_failed_offsets: list[int] = field(default_factory=list)
    raises: bool = False


_T1 = "dispatch_worker-completed.v1"
_T2 = "dispatch-outcome-evaluated.v1"


def replay_cases() -> list[ReplayCase]:
    """The full def-B parity corpus (branch-exercising, deterministic outcomes)."""
    corrupt: list[bytes | None] = [_good(0), None, b"not-json", _good(3), _good(4)]
    return [
        ReplayCase(
            name="two_topic_full_replay",
            command=ModelKafkaReplayInput(
                topics=[_T1, _T2],
                target_cluster_bootstrap=_BOOTSTRAP,
                target_consumer_group="omn-14819-full",
                expected_event_count=15,
                progress_interval_events=5,
            ),
            records={_T1: _good_run(10), _T2: _good_run(5)},
            expected_events=15,
            expected_last_offset={_T1: 9, _T2: 4},
            expected_correlation_len=15,
        ),
        ReplayCase(
            name="resume_from_sequence",
            command=ModelKafkaReplayInput(
                topics=[_T1],
                target_cluster_bootstrap=_BOOTSTRAP,
                target_consumer_group="omn-14819-resume",
                expected_event_count=6,
                resume_from={
                    f"{_T1}:0": ModelSequenceInfo.from_kafka(partition=0, offset=3)
                },
            ),
            records={_T1: _good_run(10)},
            expected_events=6,
            expected_last_offset={_T1: 9},
            expected_correlation_len=6,
        ),
        ReplayCase(
            name="from_offset_numeric",
            command=ModelKafkaReplayInput(
                topics=[_T1],
                target_cluster_bootstrap=_BOOTSTRAP,
                target_consumer_group="omn-14819-fromoffset",
                from_offset="2",
            ),
            records={_T1: _good_run(10)},
            expected_events=8,
            expected_last_offset={_T1: 9},
            expected_correlation_len=8,
        ),
        ReplayCase(
            name="to_offset_numeric_bound",
            command=ModelKafkaReplayInput(
                topics=[_T1],
                target_cluster_bootstrap=_BOOTSTRAP,
                target_consumer_group="omn-14819-tooffset",
                to_offset="5",
            ),
            records={_T1: _good_run(10)},
            expected_events=5,
            expected_last_offset={_T1: 4},
            expected_correlation_len=5,
        ),
        ReplayCase(
            name="to_offset_timestamp",
            command=ModelKafkaReplayInput(
                topics=[_T1],
                target_cluster_bootstrap=_BOOTSTRAP,
                target_consumer_group="omn-14819-timestamp",
                to_offset="2020-01-01T00:00:00",
            ),
            records={_T1: _good_run(4)},
            expected_events=4,
            expected_last_offset={_T1: 3},
            expected_correlation_len=4,
        ),
        ReplayCase(
            name="corrupt_records_tracked",
            command=ModelKafkaReplayInput(
                topics=[_T1],
                target_cluster_bootstrap=_BOOTSTRAP,
                target_consumer_group="omn-14819-corrupt",
            ),
            records={_T1: corrupt},
            expected_events=5,
            expected_last_offset={_T1: 4},
            expected_correlation_len=3,
            expected_failed_offsets=[1, 2],
        ),
        ReplayCase(
            name="expected_count_mismatch_raises",
            command=ModelKafkaReplayInput(
                topics=[_T1],
                target_cluster_bootstrap=_BOOTSTRAP,
                target_consumer_group="omn-14819-mismatch",
                expected_event_count=999,
                max_empty_polls=2,
            ),
            records={_T1: _good_run(10)},
            raises=True,
        ),
    ]


def run_case(case: ReplayCase) -> ModelKafkaReplayOutput:
    """Drive one case through the live def-B ``handle`` with the isolated consumer."""
    handler = HandlerKafkaReplay(
        consumer_factory=lambda _command: FakeReplayConsumer(case.records)
    )
    return asyncio.run(handler.handle(case.command))


def corpus_inputs() -> list[ModelKafkaReplayInput]:
    """The corpus input models, in stable order (recorder candidate pool)."""
    return [case.command for case in replay_cases()]
