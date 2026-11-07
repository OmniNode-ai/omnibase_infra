# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22822: a replayed dead letter returns only to the subscriber that failed it.

WHAT WAS MEASURED. ``node_dlq_replay_effect`` replays a dead letter by
publishing the original record back onto its source topic, where EVERY consumer
group subscribed to that topic receives it again -- not only the group whose
handler failed it. A subscriber that had already handled the record (and a
subscriber that does not care about it) was handed it a second time, and
re-processed it.

THE FIX UNDER TEST, in three seams:

1. The dead-letter envelope names the failing consumer group
   (``consumer_group``), written by ``MixinKafkaDlq`` on both of its paths.
2. ``ModelDlqMessage`` parses it and ``DLQProducer.replay_message`` stamps it on
   the replayed record as ``x-replay-target-group``.
3. ``EventBusKafka._process_consumed_record`` skips a record whose target group
   is present and is not the subscriber's own group; it commits normally (no
   rewind). A record with no such header behaves exactly as before.

The fixture is the recorded shape: ONE source record (topic, partition, offset)
that two consumer groups both failed, so it carries two dead letters, each
written by the real publisher for its own group.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.topic_constants import build_dlq_topic
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
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_unparseable_dlq_record import (
    DlqDrainRecord,
)
from omnibase_infra.topics import SUFFIX_PLATFORM_DLQ_MESSAGE

pytestmark = pytest.mark.unit

# The wire name, spelled out once here on purpose: the production constant is
# asserted EQUAL to it (test_header_name_is_one_shared_constant), so renaming
# the constant cannot silently change what is on the wire.
_TARGET_HEADER = "x-replay-target-group"

# The source topic the recorded record was delivered on, the DLQ topic this node
# drains and the durable quarantine sink, all from the repo's topic constants.
_SOURCE_TOPIC = SUFFIX_PLATFORM_DLQ_MESSAGE
_DLQ_TOPIC = build_dlq_topic("events")
_QUARANTINE_TOPIC = build_dlq_topic("quarantine")

_GROUP_A = "dev.omnibase-infra.node_subscriber_a.consume.v1"
_GROUP_B = "dev.omnibase-infra.node_subscriber_b.consume.v1"
_GROUP_C = "dev.omnibase-infra.node_subscriber_c.consume.v1"

_SOURCE_PARTITION = 0
_SOURCE_OFFSET = 7
_RECORD_BODY = json.dumps(
    {"event_type": "fixture", "payload": {"tick_id": "omn22822"}}
).encode("utf-8")


# --------------------------------------------------------------------------
# The recorded fixture: two dead letters for one source record, written by the
# real publisher, one per failing consumer group.
# --------------------------------------------------------------------------


class _CapturingProducer:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_and_wait(
        self,
        topic: str,
        *,
        value: bytes,
        key: bytes | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> object:
        self.sent.append(
            {"topic": topic, "value": value, "key": key, "headers": headers}
        )
        return object()


class _DlqHost(MixinKafkaDlq):
    def __init__(self) -> None:
        self._config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            dead_letter_topic=_DLQ_TOPIC,
        )
        self._environment = "test"
        self._group = "test-group"
        self._producer = cast("Any", _CapturingProducer())
        self._producer_lock = asyncio.Lock()
        self._timeout_seconds = 5
        self._init_dlq()

    def _model_headers_to_kafka(self, headers: object) -> list[tuple[str, bytes]]:
        return []


class _RawRecord:
    def __init__(self, headers: object = ()) -> None:
        self.key = b"omn22822-key"
        self.value = _RECORD_BODY
        self.offset = _SOURCE_OFFSET
        self.partition = _SOURCE_PARTITION
        self.headers = headers


async def _raw_dead_letter(group: str) -> dict[str, Any]:
    host = _DlqHost()
    persisted = await host._publish_raw_to_dlq(
        original_topic=_SOURCE_TOPIC,
        raw_msg=_RawRecord(),
        error=RuntimeError("subscriber rejected the record"),
        correlation_id=UUID(int=22822),
        failure_type="handler_exception",
        consumer_group=group,
    )
    assert persisted is True
    sent = cast("_CapturingProducer", host._producer).sent
    return cast("dict[str, Any]", json.loads(sent[0]["value"]))


def _typed_message() -> ModelEventMessage:
    from omnibase_infra.event_bus.models import ModelEventHeaders

    return ModelEventMessage(
        topic=_SOURCE_TOPIC,
        key=b"omn22822-key",
        value=_RECORD_BODY,
        headers=ModelEventHeaders(
            source="test", event_type="fixture", timestamp=_now()
        ),
        offset=str(_SOURCE_OFFSET),
        partition=_SOURCE_PARTITION,
    )


def _now() -> Any:
    from datetime import UTC, datetime

    return datetime.now(UTC)


async def _typed_dead_letter(group: str) -> dict[str, Any]:
    host = _DlqHost()
    persisted = await host._publish_to_dlq(
        original_topic=_SOURCE_TOPIC,
        failed_message=_typed_message(),
        error=RuntimeError("subscriber rejected the record"),
        correlation_id=UUID(int=22822),
        consumer_group=group,
    )
    assert persisted is True
    sent = cast("_CapturingProducer", host._producer).sent
    return cast("dict[str, Any]", json.loads(sent[0]["value"]))


def _parse(payload: dict[str, Any], dlq_offset: int) -> ModelDlqMessage:
    return ModelDlqMessage.from_kafka_message(
        payload, dlq_offset=dlq_offset, dlq_partition=0
    )


# --------------------------------------------------------------------------
# Seam 1: the envelope names the failing group.
# --------------------------------------------------------------------------


class TestEnvelopeCarriesFailingGroup:
    @pytest.mark.asyncio
    async def test_raw_path_envelope_names_the_failing_group(self) -> None:
        assert (await _raw_dead_letter(_GROUP_A))["consumer_group"] == _GROUP_A
        assert (await _raw_dead_letter(_GROUP_B))["consumer_group"] == _GROUP_B

    @pytest.mark.asyncio
    async def test_typed_path_envelope_names_the_failing_group(self) -> None:
        assert (await _typed_dead_letter(_GROUP_A))["consumer_group"] == _GROUP_A
        assert (await _typed_dead_letter(_GROUP_B))["consumer_group"] == _GROUP_B

    @pytest.mark.asyncio
    @pytest.mark.parametrize("placeholder", ["unknown", "auto-wiring"])
    async def test_a_placeholder_group_is_not_written_as_a_replay_target(
        self, placeholder: str
    ) -> None:
        """A label that is no subscriber's group would make every consumer skip it."""
        assert (await _raw_dead_letter(placeholder))["consumer_group"] is None
        assert (await _typed_dead_letter(placeholder))["consumer_group"] is None


# --------------------------------------------------------------------------
# Seam 2: the model parses it; the replay producer stamps the header.
# --------------------------------------------------------------------------


class TestModelParsesConsumerGroup:
    @pytest.mark.asyncio
    async def test_model_reads_the_group_off_the_recorded_envelope(self) -> None:
        a = _parse(await _raw_dead_letter(_GROUP_A), 0)
        b = _parse(await _raw_dead_letter(_GROUP_B), 1)
        assert a.consumer_group == _GROUP_A
        assert b.consumer_group == _GROUP_B

    def test_legacy_envelope_without_the_key_parses_to_none(self) -> None:
        legacy = {
            "original_topic": _SOURCE_TOPIC,
            "original_message": {"key": None, "value": "{}", "offset": 1},
            "correlation_id": str(uuid4()),
        }
        assert _parse(legacy, 0).consumer_group is None

    @pytest.mark.parametrize("junk", [None, "", "   ", 7, ["g"]])
    def test_a_non_string_or_blank_group_parses_to_none(self, junk: object) -> None:
        payload = {
            "original_topic": _SOURCE_TOPIC,
            "original_message": {"key": None, "value": "{}", "offset": 1},
            "correlation_id": str(uuid4()),
            "consumer_group": junk,
        }
        assert _parse(payload, 0).consumer_group is None


class _FakeKafkaProducer:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_and_wait(self, topic: str, **kwargs: Any) -> None:
        self.sent.append({"topic": topic, **kwargs})


def _replay_producer(
    *, add_replay_headers: bool = True
) -> tuple[DLQProducer, _FakeKafkaProducer]:
    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_DLQ_TOPIC,
        quarantine_topic=_QUARANTINE_TOPIC,
        add_replay_headers=add_replay_headers,
    )
    producer = DLQProducer(config)
    fake = _FakeKafkaProducer()
    producer._producer = cast("Any", fake)
    producer._started = True
    return producer, fake


class TestReplayStampsTheTargetGroup:
    @pytest.mark.asyncio
    async def test_replay_publishes_the_header_for_the_failing_group(self) -> None:
        a = _parse(await _raw_dead_letter(_GROUP_A), 0)
        b = _parse(await _raw_dead_letter(_GROUP_B), 1)
        producer, fake = _replay_producer()

        await producer.replay_message(a, uuid4())
        await producer.replay_message(b, uuid4())

        assert dict(fake.sent[0]["headers"])[_TARGET_HEADER] == _GROUP_A.encode()
        assert dict(fake.sent[1]["headers"])[_TARGET_HEADER] == _GROUP_B.encode()
        # Same source record, same body: only the target differs.
        assert fake.sent[0]["value"] == fake.sent[1]["value"] == _RECORD_BODY

    @pytest.mark.asyncio
    async def test_replay_of_a_headerless_legacy_dead_letter_has_no_target(
        self,
    ) -> None:
        legacy = {
            "original_topic": _SOURCE_TOPIC,
            "original_message": {
                "key": None,
                "value": _RECORD_BODY.decode(),
                "offset": 1,
            },
            "correlation_id": str(uuid4()),
        }
        producer, fake = _replay_producer()

        await producer.replay_message(_parse(legacy, 0), uuid4())

        assert _TARGET_HEADER not in dict(fake.sent[0]["headers"])

    @pytest.mark.asyncio
    async def test_no_headers_at_all_when_replay_headers_are_off(self) -> None:
        a = _parse(await _raw_dead_letter(_GROUP_A), 0)
        producer, fake = _replay_producer(add_replay_headers=False)

        await producer.replay_message(a, uuid4())

        assert fake.sent[0]["headers"] == []

    @pytest.mark.asyncio
    async def test_golden_replay_header_set_for_a_targeted_dead_letter(self) -> None:
        """Golden: the exact header NAMES a targeted replay puts on the wire."""
        a = _parse(await _raw_dead_letter(_GROUP_A), 0)
        producer, fake = _replay_producer()

        await producer.replay_message(a, uuid4())

        assert sorted(dict(fake.sent[0]["headers"])) == [
            "correlation_id",
            "x-original-dlq-offset",
            "x-replay-correlation-id",
            "x-replay-count",
            "x-replay-target-group",
            "x-replayed-at",
            "x-replayed-by",
        ]

    def test_header_name_is_one_shared_constant(self) -> None:
        from omnibase_infra.event_bus import event_bus_kafka
        from omnibase_infra.models.event_bus import model_dlq_replay_headers
        from omnibase_infra.nodes.node_dlq_replay_effect import engine_dlq_replay

        assert model_dlq_replay_headers.REPLAY_TARGET_GROUP_HEADER == _TARGET_HEADER
        # The producer and the consumer import this very object.
        assert event_bus_kafka.REPLAY_TARGET_GROUP_HEADER is (
            model_dlq_replay_headers.REPLAY_TARGET_GROUP_HEADER
        )
        assert engine_dlq_replay.REPLAY_TARGET_GROUP_HEADER is (
            model_dlq_replay_headers.REPLAY_TARGET_GROUP_HEADER
        )


# --------------------------------------------------------------------------
# Seam 3: the consumer skips a record targeted at another group.
# --------------------------------------------------------------------------

_BUS_TOPIC = _SOURCE_TOPIC


def _kafka_record(headers: list[tuple[str, bytes]] | None) -> MagicMock:
    msg = MagicMock()
    msg.topic = _BUS_TOPIC
    msg.partition = _SOURCE_PARTITION
    msg.offset = _SOURCE_OFFSET
    msg.timestamp = 0
    msg.key = None
    msg.headers = headers if headers is not None else []
    msg.value = _RECORD_BODY
    return msg


async def _bus_with_two_groups() -> tuple[
    EventBusKafka, dict[str, AsyncMock], AsyncMock
]:
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    config = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092", environment="dev"
    )
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
        return_value=producer,
    ):
        bus = EventBusKafka(config=config)
        await bus.start()
    callbacks = {_GROUP_A: AsyncMock(), _GROUP_B: AsyncMock()}
    bus._subscribers[_BUS_TOPIC] = [
        (group, f"sub-{index}", callbacks[group])
        for index, group in enumerate((_GROUP_A, _GROUP_B))
    ]
    return bus, callbacks, producer


class TestConsumerHonoursTheTarget:
    @pytest.mark.asyncio
    async def test_group_a_skips_a_record_targeted_at_group_b_and_b_processes_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus, callbacks, _ = await _bus_with_two_groups()
        record = _kafka_record([(_TARGET_HEADER, _GROUP_B.encode())])
        consumer = MagicMock()

        with caplog.at_level(logging.INFO):
            rewound_a = await bus._process_consumed_record(
                record, _BUS_TOPIC, _GROUP_A, uuid4(), consumer
            )
            rewound_b = await bus._process_consumed_record(
                record, _BUS_TOPIC, _GROUP_B, uuid4(), consumer
            )

        callbacks[_GROUP_A].assert_not_called()
        callbacks[_GROUP_B].assert_awaited_once()
        # Skip is a normal commit: no rewind for either group.
        assert rewound_a is False
        assert rewound_b is False
        skip_logs = [r for r in caplog.records if _GROUP_B in r.getMessage()]
        assert skip_logs
        assert all(r.levelno == logging.INFO for r in skip_logs)
        await bus.close()

    @pytest.mark.asyncio
    async def test_group_a_processes_a_record_targeted_at_group_a(self) -> None:
        bus, callbacks, _ = await _bus_with_two_groups()
        record = _kafka_record([(_TARGET_HEADER, _GROUP_A.encode())])

        await bus._process_consumed_record(
            record, _BUS_TOPIC, _GROUP_A, uuid4(), MagicMock()
        )

        callbacks[_GROUP_A].assert_awaited_once()
        await bus.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("headers", [None, [("x-replay-count", b"1")]])
    async def test_a_record_with_no_target_header_is_processed_by_every_group(
        self, headers: list[tuple[str, bytes]] | None
    ) -> None:
        bus, callbacks, _ = await _bus_with_two_groups()
        record = _kafka_record(headers)

        for group in (_GROUP_A, _GROUP_B):
            await bus._process_consumed_record(
                record, _BUS_TOPIC, group, uuid4(), MagicMock()
            )

        callbacks[_GROUP_A].assert_awaited_once()
        callbacks[_GROUP_B].assert_awaited_once()
        await bus.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", [None, b"", b"   "])
    async def test_an_empty_target_header_means_no_target_and_is_processed(
        self, value: bytes | None
    ) -> None:
        bus, callbacks, _ = await _bus_with_two_groups()
        record = _kafka_record(
            cast("list[tuple[str, bytes]]", [(_TARGET_HEADER, value)])
        )

        await bus._process_consumed_record(
            record, _BUS_TOPIC, _GROUP_A, uuid4(), MagicMock()
        )

        callbacks[_GROUP_A].assert_awaited_once()
        await bus.close()

    @pytest.mark.asyncio
    async def test_a_skipped_record_is_not_dead_lettered_by_the_skipping_group(
        self,
    ) -> None:
        """A malformed targeted record must not be dead-lettered by a bystander."""
        bus, _callbacks, producer = await _bus_with_two_groups()
        # A header the converter cannot parse: any group that converts this
        # record dead-letters it as a deserialization error.
        record = _kafka_record(
            [(_TARGET_HEADER, _GROUP_B.encode()), ("timestamp", b"not-a-timestamp")]
        )

        rewound = await bus._process_consumed_record(
            record, _BUS_TOPIC, _GROUP_A, uuid4(), MagicMock()
        )

        assert rewound is False
        producer.send_and_wait.assert_not_called()
        await bus.close()


# --------------------------------------------------------------------------
# Error chain: the closed loop. Source records fail in groups A and B, are
# dead-lettered, replayed through the real handler and producer, and delivered
# by a bus that applies the consumer rule. Group C never fails.
# --------------------------------------------------------------------------


class _LoopBus:
    """A source topic with groups A and B failing every delivery, C succeeding."""

    FAILING = (_GROUP_A, _GROUP_B)
    SUBSCRIBERS = (_GROUP_A, _GROUP_B, _GROUP_C)

    def __init__(self) -> None:
        # (replay count, target group header or None)
        self.source: list[tuple[int, str | None]] = []
        self.dlq: list[ModelDlqMessage] = []
        self.deliveries: dict[str, int] = dict.fromkeys(self.SUBSCRIBERS, 0)

    def publish_source(self, retry_count: int, target: str | None) -> None:
        offset = len(self.source)
        self.source.append((retry_count, target))
        for group in self.SUBSCRIBERS:
            if target is not None and target != group:
                continue  # the consumer rule under test: skip, commit
            self.deliveries[group] += 1
            if group in self.FAILING:
                self.dlq.append(
                    ModelDlqMessage(
                        original_topic=_SOURCE_TOPIC,
                        original_key=None,
                        original_value=_RECORD_BODY.decode(),
                        original_offset=str(offset),
                        original_partition=_SOURCE_PARTITION,
                        failure_reason="HandlerDispatchFailureError",
                        correlation_id=UUID(int=22822),
                        retry_count=retry_count,
                        error_type="HandlerDispatchFailureError",
                        consumer_group=group,
                        dlq_offset=len(self.dlq),
                        dlq_partition=0,
                    )
                )


class _LoopConsumer:
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
    def __init__(self, bus: _LoopBus) -> None:
        self._bus = bus
        self.targets: list[str | None] = []

    async def replay_message(
        self, message: ModelDlqMessage, replay_correlation_id: UUID
    ) -> None:
        self.targets.append(message.consumer_group)
        self._bus.publish_source(message.retry_count + 1, message.consumer_group)


class _Quarantine:
    def __init__(self) -> None:
        self.quarantined: list[ModelDlqMessage] = []

    async def quarantine_message(
        self, message: ModelDlqMessage, reason: str, correlation_id: UUID
    ) -> object:
        self.quarantined.append(message)
        return object()


class TestErrorChainStaysWithTheFailingGroups:
    @pytest.mark.asyncio
    async def test_replay_chain_reaches_only_failing_groups_and_does_not_double(
        self,
    ) -> None:
        max_replay = 3
        config = ModelDlqReplayEngineConfig.model_validate(
            {
                "bootstrap_servers": "test-broker:9092",
                "dlq_topic": _DLQ_TOPIC,
                "max_replay_count": max_replay,
                "max_run_duration_seconds": 2.0,
                "idle_probe_seconds": 0.05,
            }
        )
        bus = _LoopBus()
        consumer = _LoopConsumer(config, bus.dlq)
        quarantine = _Quarantine()
        replay = _LoopReplayProducer(bus)
        handler = HandlerDlqReplay(
            consumers={_DLQ_TOPIC: cast("DLQConsumer", consumer)},
            producer=cast("DLQProducer", replay),
            quarantine_producer=cast("DLQQuarantineProducer", quarantine),
            tracking=None,
        )
        bus.publish_source(0, None)
        statuses: list[EnumReplayStatus] = []
        for _ in range(100):
            if consumer.committed >= len(bus.dlq):
                break
            run = await asyncio.wait_for(handler.run(), 10.0)
            statuses.extend(r.status for r in run.results)
        assert consumer.committed == len(bus.dlq), "the replay chain did not settle"

        # Each failing group's dead letter is replayed to that group alone, so
        # the chain is linear: one record per failing group per round. It is
        # not the old 2**g fan-out.
        assert len(bus.source) == 1 + 2 * max_replay
        # The group that never failed saw the record exactly once, not once
        # per replay round.
        assert bus.deliveries[_GROUP_C] == 1
        # Every replay was addressed; none went to everyone.
        assert None not in replay.targets
        assert EnumReplayStatus.FAILED not in statuses
        # At the cap each failing group's dead letter is quarantined, by group.
        assert sorted(m.consumer_group or "" for m in quarantine.quarantined) == [
            _GROUP_A,
            _GROUP_B,
        ]


# --------------------------------------------------------------------------
# OMN-20589 dedupe, per (source record, group).
# --------------------------------------------------------------------------


class _RecordingReplayProducer:
    def __init__(self) -> None:
        self.replayed: list[ModelDlqMessage] = []

    async def replay_message(
        self, message: ModelDlqMessage, replay_correlation_id: UUID
    ) -> None:
        self.replayed.append(message)


def _sibling(dlq_offset: int, group: str | None) -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=_SOURCE_TOPIC,
        original_value=_RECORD_BODY.decode(),
        original_offset=str(_SOURCE_OFFSET),
        original_partition=_SOURCE_PARTITION,
        correlation_id=UUID(int=22822),
        consumer_group=group,
        dlq_offset=dlq_offset,
        dlq_partition=0,
    )


async def _replayed_by_handler(groups: list[str | None]) -> list[ModelDlqMessage]:
    config = ModelDlqReplayEngineConfig.model_validate(
        {
            "bootstrap_servers": "test-broker:9092",
            "dlq_topic": _DLQ_TOPIC,
            "max_run_duration_seconds": 2.0,
            "idle_probe_seconds": 0.05,
        }
    )
    dlq = [_sibling(index, group) for index, group in enumerate(groups)]
    producer = _RecordingReplayProducer()
    handler = HandlerDlqReplay(
        consumers={_DLQ_TOPIC: cast("DLQConsumer", _LoopConsumer(config, dlq))},
        producer=cast("DLQProducer", producer),
        quarantine_producer=cast("DLQQuarantineProducer", _Quarantine()),
        tracking=None,
    )
    await asyncio.wait_for(handler.run(), 10.0)
    return producer.replayed


class TestDedupeIsPerSourceRecordAndGroup:
    @pytest.mark.asyncio
    async def test_siblings_from_different_groups_are_each_replayed(self) -> None:
        replayed = await _replayed_by_handler([_GROUP_A, _GROUP_B])
        assert [m.consumer_group for m in replayed] == [_GROUP_A, _GROUP_B]

    @pytest.mark.asyncio
    async def test_siblings_from_the_same_group_are_replayed_once(self) -> None:
        replayed = await _replayed_by_handler([_GROUP_A, _GROUP_A])
        assert [m.dlq_offset for m in replayed] == [0]

    @pytest.mark.asyncio
    async def test_unaddressed_siblings_are_replayed_once_as_before(self) -> None:
        replayed = await _replayed_by_handler([None, None])
        assert [m.dlq_offset for m in replayed] == [0]

    @pytest.mark.asyncio
    async def test_an_unaddressed_replay_answers_an_addressed_sibling(self) -> None:
        """It reached every group, the sibling's included."""
        replayed = await _replayed_by_handler([None, _GROUP_B])
        assert [m.dlq_offset for m in replayed] == [0]
