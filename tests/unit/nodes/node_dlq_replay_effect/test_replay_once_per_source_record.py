# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20589: one replay per source record, however many subscribers dead-lettered it.

WHAT WAS MEASURED (dev-202 lane, 2026-10-05, read-only). Two runtime ticks
published at 08:41:06Z and 08:41:13Z became 63 records each on
``onex.intent.platform.runtime-tick.v1``. Bucketed by ``x-replay-count``, the
copies were 1, 2, 4, 8, 16 and 32. The intents DLQ held 4, 8, 16, 32, 64 and 128
dead letters at retry_count 0 to 5. Two auto-wired subscriptions,
``node_consumer_flow_prune_effect`` and ``node_dead_letter_prune_effect``,
each raised KeyError on every delivery of a tick, and each wrote its OWN dead
letter for the same source record.

WHY IT DOUBLES. ``HandlerDlqReplay`` replays every dead letter independently.
A replay publishes the source record onto its original topic, where EVERY
subscriber receives it again, so a record that N subscribers reject comes back
N times per round: N**g copies at round g, until ``max_replay_count`` stops it.
With N=2 and the default cap of 5, the total is 2**0 + ... + 2**5 = 63.
OMN-17896 (the ``should_replay`` docstring) measured the same signature and
closed only the boundary-terminal record class.

THE FIX UNDER TEST. Dead letters that name the same source coordinate
(original topic, partition and offset) are failures of ONE delivered record.
One replay re-delivers it to every subscriber, so they need one replay between
them, not one each. The rest complete as SKIPPED so their offsets commit.

The tests drive the real ``HandlerDlqReplay`` over a closed loop: the replay
producer appends to the source log, and every append is dead-lettered once per
failing subscriber, the way the auto-wired boundary does it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import cast
from uuid import UUID, uuid4

import pytest

from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_replay_run_result import (
    ModelDlqReplayRunResult,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_unparseable_dlq_record import (
    DlqDrainRecord,
)

pytestmark = pytest.mark.unit

_INTENTS_DLQ = "onex.dlq.omnibase-infra.intents.v1"  # onex-topic-allow: the DLQ the measured dead letters landed on
_TICK_TOPIC = "onex.intent.platform.runtime-tick.v1"  # onex-topic-allow: the source topic of the measured storm
_OUTER_TIMEOUT_SECONDS = 10.0
_MAX_RUNS = 200


def _config(**overrides: object) -> ModelDlqReplayEngineConfig:
    fields: dict[str, object] = {
        "bootstrap_servers": "test-broker:9092",
        "dlq_topic": _INTENTS_DLQ,
        "max_replay_count": 5,
        "max_run_duration_seconds": 2.0,
        "idle_probe_seconds": 0.05,
    }
    fields.update(overrides)
    return ModelDlqReplayEngineConfig.model_validate(fields)


def _dead_letter(
    *,
    dlq_offset: int,
    retry_count: int,
    correlation_id: UUID,
    source_offset: int | None,
    source_partition: int | None = 0,
    subscriber: str = "node_consumer_flow_prune_effect",
) -> ModelDlqMessage:
    """The shape the auto-wired boundary writes for one rejecting subscriber."""
    return ModelDlqMessage(
        original_topic=_TICK_TOPIC,
        original_key=str(correlation_id),
        original_value=json.dumps(
            {"payload": {"tick_id": str(correlation_id)}, "event_type": "tick"}
        ),
        original_offset=None if source_offset is None else str(source_offset),
        original_partition=source_partition,
        failure_reason=(
            "HandlerDispatchFailureError: dispatcher "
            f"'dispatcher.auto.{subscriber}' failed: KeyError"
        ),
        failure_timestamp="2026-10-05T08:41:06Z",
        correlation_id=correlation_id,
        retry_count=retry_count,
        error_type="HandlerDispatchFailureError",
        dlq_offset=dlq_offset,
        dlq_partition=0,
        raw_payload={"original_topic": _TICK_TOPIC},
    )


class _Bus:
    """A source topic whose every record ``subscribers`` subscribers reject.

    ``source`` holds the replay count of each record published on the source
    topic. Each append is dead-lettered once per failing subscriber, carrying
    the replay count forward as ``retry_count`` exactly as the boundary does
    (``x-replay-count`` read back into the dead letter, OMN-14551).
    """

    def __init__(self, subscribers: int) -> None:
        self.subscribers = subscribers
        self.correlation_id = uuid4()
        self.source: list[int] = []
        self.dlq: list[ModelDlqMessage] = []

    def publish_source(self, retry_count: int) -> None:
        source_offset = len(self.source)
        self.source.append(retry_count)
        for index in range(self.subscribers):
            self.dlq.append(
                _dead_letter(
                    dlq_offset=len(self.dlq),
                    retry_count=retry_count,
                    correlation_id=self.correlation_id,
                    source_offset=source_offset,
                    subscriber=f"subscriber_{index}",
                )
            )


class _LoopConsumer:
    """One DLQ partition read from its committed offset; sees appends live."""

    def __init__(
        self, config: ModelDlqReplayEngineConfig, dlq: list[ModelDlqMessage]
    ) -> None:
        self.config = config
        self._dlq = dlq
        self.committed = 0

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def commit_offsets(self, offsets: Mapping[tuple[str, int], int]) -> None:
        for next_offset in offsets.values():
            self.committed = next_offset

    async def consume_messages(self) -> AsyncIterator[DlqDrainRecord]:
        index = self.committed
        while index < len(self._dlq):
            yield self._dlq[index]
            index += 1


class _LoopReplayProducer:
    """Publishing a replay appends the record to the source topic again."""

    def __init__(self, bus: _Bus) -> None:
        self._bus = bus
        self.replayed: list[ModelDlqMessage] = []

    async def replay_message(
        self, message: ModelDlqMessage, replay_correlation_id: UUID
    ) -> None:
        self.replayed.append(message)
        self._bus.publish_source(message.retry_count + 1)


class _RecordingReplayProducer:
    def __init__(self) -> None:
        self.replayed: list[ModelDlqMessage] = []

    async def replay_message(
        self, message: ModelDlqMessage, replay_correlation_id: UUID
    ) -> None:
        self.replayed.append(message)


class _ConfirmingQuarantine:
    def __init__(self) -> None:
        self.quarantined: list[ModelDlqMessage] = []

    async def quarantine_message(
        self, message: ModelDlqMessage, reason: str, correlation_id: UUID
    ) -> object:
        self.quarantined.append(message)
        return object()


def _handler(
    consumer: _LoopConsumer, producer: object, quarantine: _ConfirmingQuarantine
) -> HandlerDlqReplay:
    return HandlerDlqReplay(
        consumers={_INTENTS_DLQ: cast("DLQConsumer", consumer)},
        producer=cast("DLQProducer", producer),
        quarantine_producer=cast("DLQQuarantineProducer", quarantine),
    )


async def _settle(
    bus: _Bus, config: ModelDlqReplayEngineConfig
) -> tuple[_LoopConsumer, _ConfirmingQuarantine, list[ModelDlqReplayRunResult]]:
    """Publish one source record and run the replay until the DLQ is drained."""
    consumer = _LoopConsumer(config, bus.dlq)
    quarantine = _ConfirmingQuarantine()
    handler = _handler(consumer, _LoopReplayProducer(bus), quarantine)
    bus.publish_source(0)
    results: list[ModelDlqReplayRunResult] = []
    for _ in range(_MAX_RUNS):
        if consumer.committed >= len(bus.dlq):
            break
        results.append(await asyncio.wait_for(handler.run(), _OUTER_TIMEOUT_SECONDS))
    assert consumer.committed == len(bus.dlq), "the replay loop did not settle"
    return consumer, quarantine, results


@pytest.mark.unit
class TestOneReplayPerSourceRecord:
    @pytest.mark.asyncio
    async def test_one_replay_per_source_two_failing_subscribers_do_not_double(
        self,
    ) -> None:
        """RED at 8095e6696: 63 source copies, 1/2/4/8/16/32 by replay count."""
        bus = _Bus(subscribers=2)
        config = _config()

        await _settle(bus, config)

        by_round = [bus.source.count(r) for r in range(config.max_replay_count + 1)]
        assert len(bus.source) == config.max_replay_count + 1, (
            f"one source record became {len(bus.source)} copies; by replay "
            f"count {by_round}"
        )
        assert by_round == [1] * (config.max_replay_count + 1)

    @pytest.mark.asyncio
    async def test_one_replay_per_source_holds_when_duplicates_span_runs(
        self,
    ) -> None:
        """Each run handles one dead letter, so siblings land in different runs."""
        bus = _Bus(subscribers=3)
        config = _config(max_records_per_run=1)

        await _settle(bus, config)

        assert len(bus.source) == config.max_replay_count + 1

    @pytest.mark.asyncio
    async def test_one_replay_per_source_duplicates_complete_and_commit(
        self,
    ) -> None:
        bus = _Bus(subscribers=2)
        config = _config()

        consumer, quarantine, results = await _settle(bus, config)

        statuses = [r.status for run in results for r in run.results]
        rounds = config.max_replay_count + 1
        # Every round leaves two dead letters for one source record: the
        # first is replayed (or, at the cap, quarantined) and its sibling is
        # completed as a duplicate.
        assert statuses.count(EnumReplayStatus.COMPLETED) == config.max_replay_count
        assert statuses.count(EnumReplayStatus.SKIPPED) == rounds - 1
        assert len(quarantine.quarantined) == 2, (
            "both capped dead letters reach quarantine: refusal is decided per "
            "dead letter and is never skipped as a duplicate"
        )
        assert EnumReplayStatus.FAILED not in statuses
        assert consumer.committed == len(bus.dlq) == 2 * rounds

    @pytest.mark.asyncio
    async def test_one_replay_per_source_single_subscriber_is_unchanged(
        self,
    ) -> None:
        bus = _Bus(subscribers=1)
        config = _config()

        await _settle(bus, config)

        assert len(bus.source) == config.max_replay_count + 1


@pytest.mark.unit
class TestPositiveControl:
    @pytest.mark.asyncio
    async def test_positive_control_same_correlation_different_source_offsets_both_replay(
        self,
    ) -> None:
        correlation_id = uuid4()
        dlq = [
            _dead_letter(
                dlq_offset=0,
                retry_count=0,
                correlation_id=correlation_id,
                source_offset=10,
            ),
            _dead_letter(
                dlq_offset=1,
                retry_count=0,
                correlation_id=correlation_id,
                source_offset=11,
            ),
        ]
        producer = _RecordingReplayProducer()
        handler = _handler(
            _LoopConsumer(_config(), dlq), producer, _ConfirmingQuarantine()
        )

        await asyncio.wait_for(handler.run(), _OUTER_TIMEOUT_SECONDS)

        assert [m.dlq_offset for m in producer.replayed] == [0, 1]

    @pytest.mark.asyncio
    async def test_positive_control_dead_letters_without_a_source_coordinate_each_replay(
        self,
    ) -> None:
        correlation_id = uuid4()
        dlq = [
            _dead_letter(
                dlq_offset=index,
                retry_count=0,
                correlation_id=correlation_id,
                source_offset=None,
                source_partition=None,
            )
            for index in range(2)
        ]
        producer = _RecordingReplayProducer()
        handler = _handler(
            _LoopConsumer(_config(), dlq), producer, _ConfirmingQuarantine()
        )

        await asyncio.wait_for(handler.run(), _OUTER_TIMEOUT_SECONDS)

        assert [m.dlq_offset for m in producer.replayed] == [0, 1]
