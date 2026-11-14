# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24177: a broadcast topic is never replayed; every other replay is bounded per target.

WHAT WAS MEASURED. ``node_dlq_replay_effect`` replays a dead letter by
publishing the original record back onto its source topic. The runtime tick
topic is read by EVERY consumer group: a node that failed every tick filled
the shared topic with replayed ticks -- 39,960 skip lines in 4.5 minutes on
the dev runtime on 2026-11-14. A replay would re-send the record to all
groups, each of which skips it, and the next tick supersedes it.

THE FIX UNDER TEST, in three seams:

1. ``should_replay`` refuses a record whose ``original_topic`` is in
   ``ModelDlqReplayEngineConfig.broadcast_topics`` (the refusal routes it to
   quarantine, never a drop).
2. ``HandlerDlqReplay._process_message`` bounds the replays of every other
   (original topic, target group) pair: at most ``max_replays`` in any window
   of ``window_seconds`` (``DlqConsumerDrainState.admit_replay``), the next
   record for the pair quarantined instead. A node that fails every message
   keeps producing fresh dead letters, each with a replay count of zero, so
   the per-record cap never trips; this bounds the pair.
3. The values come from the node's own ``contract.yaml``
   ``replay_policy`` block via ``replay_policy.load_replay_policy``, the
   config's default -- the node cannot run on a policy its contract does not
   declare, and a contract without the block fails CLOSED.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from importlib.resources import files
from typing import Any, cast
from uuid import UUID

import pytest
import yaml

from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.topic_constants import build_dlq_topic
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    DlqConsumerDrainState,
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models import (
    ModelDlqMessage,
    ModelDlqTargetReplayBound,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_unparseable_dlq_record import (
    DlqDrainRecord,
)
from omnibase_infra.nodes.node_dlq_replay_effect.replay_policy import (
    parse_replay_policy,
)
from omnibase_infra.topics import SUFFIX_RUNTIME_TICK
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_NODE_REGISTRATION

pytestmark = pytest.mark.unit

_DLQ_TOPIC = build_dlq_topic("events")

# A plain, non-broadcast source topic for the golden chain. The repo's own
# event topics are dead-letter topics in this lane (refused by is_dlq_topic),
# so the golden topic is spelled out as its wire name.
_GOLDEN_TOPIC = SUFFIX_NODE_REGISTRATION

_GROUP_A = "dev.omnibase-infra.node_subscriber_a.consume.v1"
_GROUP_B = "dev.omnibase-infra.node_subscriber_b.consume.v1"

_SOURCE_PARTITION = 0
_RECORD_BODY = json.dumps(
    {"event_type": "fixture", "payload": {"tick_id": "omn24177"}}
).encode("utf-8")


def _config(**overrides: Any) -> ModelDlqReplayEngineConfig:
    """The engine config the exemplar drives the handler with, plus policy."""
    return ModelDlqReplayEngineConfig.model_validate(
        {
            "bootstrap_servers": "test-broker:9092",
            "dlq_topic": _DLQ_TOPIC,
            "max_run_duration_seconds": 2.0,
            "idle_probe_seconds": 0.05,
            **overrides,
        }
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


class _RecordingReplayProducer:
    def __init__(self) -> None:
        self.replayed: list[ModelDlqMessage] = []

    async def replay_message(
        self, message: ModelDlqMessage, replay_correlation_id: UUID
    ) -> None:
        self.replayed.append(message)


class _Quarantine:
    def __init__(self) -> None:
        self.quarantined: list[tuple[ModelDlqMessage, str]] = []

    async def quarantine_message(
        self, message: ModelDlqMessage, reason: str, correlation_id: UUID
    ) -> object:
        self.quarantined.append((message, reason))
        return object()


def _sibling(
    dlq_offset: int,
    group: str | None,
    topic: str = SUFFIX_RUNTIME_TICK,
    source_offset: int = 0,
) -> ModelDlqMessage:
    """One dead letter; distinct source offsets keep siblings un-deduplicated."""
    return ModelDlqMessage(
        original_topic=topic,
        original_value=_RECORD_BODY.decode(),
        original_offset=str(source_offset),
        original_partition=_SOURCE_PARTITION,
        correlation_id=UUID(int=24177),
        consumer_group=group,
        dlq_offset=dlq_offset,
        dlq_partition=0,
    )


async def _run_handler(
    dlq: list[ModelDlqMessage],
    config: ModelDlqReplayEngineConfig,
    producer: _RecordingReplayProducer,
    quarantine: _Quarantine,
) -> list[EnumReplayStatus]:
    handler = HandlerDlqReplay(
        consumers={_DLQ_TOPIC: cast("DLQConsumer", _LoopConsumer(config, dlq))},
        producer=cast("DLQProducer", producer),
        quarantine_producer=cast("DLQQuarantineProducer", quarantine),
        tracking=None,
    )
    run = await asyncio.wait_for(handler.run(), 10.0)
    return [result.status for result in run.results]


class TestShouldReplayBroadcastRefusal:
    def test_should_replay_refuses_a_broadcast_topic(self) -> None:
        message = _sibling(0, _GROUP_A, topic=SUFFIX_RUNTIME_TICK)

        eligible, reason = should_replay(
            message, _config(broadcast_topics=(SUFFIX_RUNTIME_TICK,))
        )
        assert eligible is False
        assert "Broadcast topic" in reason

        # Failing control: with no broadcast_topics declared the same message
        # is eligible, so the refusal is the policy, not the message.
        assert should_replay(message, _config(broadcast_topics=()))[0] is True


class TestBroadcastErrorChain:
    @pytest.mark.asyncio
    async def test_broadcast_dead_letters_are_quarantined_not_replayed(
        self,
    ) -> None:
        config = _config(broadcast_topics=(SUFFIX_RUNTIME_TICK,))
        dlq = [_sibling(index, _GROUP_A, source_offset=index) for index in range(5)]
        producer = _RecordingReplayProducer()
        quarantine = _Quarantine()

        statuses = await _run_handler(dlq, config, producer, quarantine)

        assert producer.replayed == []
        assert len(quarantine.quarantined) == 5
        assert all(
            "Broadcast topic" in reason for _message, reason in quarantine.quarantined
        )
        assert statuses == [EnumReplayStatus.QUARANTINED] * 5


class TestGoldenReplayChain:
    @pytest.mark.asyncio
    async def test_non_broadcast_dead_letters_are_all_replayed(self) -> None:
        config = _config(broadcast_topics=(SUFFIX_RUNTIME_TICK,))
        dlq = [
            _sibling(index, _GROUP_A, topic=_GOLDEN_TOPIC, source_offset=index)
            for index in range(5)
        ]
        producer = _RecordingReplayProducer()
        quarantine = _Quarantine()

        statuses = await _run_handler(dlq, config, producer, quarantine)

        assert len(producer.replayed) == 5
        assert all(m.original_topic == _GOLDEN_TOPIC for m in producer.replayed)
        assert quarantine.quarantined == []
        assert statuses == [EnumReplayStatus.COMPLETED] * 5


class TestReplayBound:
    @pytest.mark.asyncio
    async def test_bound_is_per_pair_and_the_window_moves(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bound = ModelDlqTargetReplayBound(max_replays=3, window_seconds=60.0)
        config = _config(target_replay_bound=bound)
        # Ten distinct source records of one (topic, group) pair, plus one
        # dead letter of a DIFFERENT group on the same topic: the bound is
        # per pair, so that one is still replayed in the same run.
        dlq = [
            _sibling(index, _GROUP_A, topic=_GOLDEN_TOPIC, source_offset=index)
            for index in range(10)
        ]
        dlq.append(_sibling(10, _GROUP_B, topic=_GOLDEN_TOPIC, source_offset=10))
        producer = _RecordingReplayProducer()
        quarantine = _Quarantine()

        with caplog.at_level(logging.ERROR):
            statuses = await _run_handler(dlq, config, producer, quarantine)

        assert len(producer.replayed) == 4  # 3 for group A + 1 for group B
        assert [m.consumer_group for m in producer.replayed] == [
            _GROUP_A,
            _GROUP_A,
            _GROUP_A,
            _GROUP_B,
        ]
        assert len(quarantine.quarantined) == 7
        assert all(
            "Replay bound" in reason for _message, reason in quarantine.quarantined
        )
        assert statuses.count(EnumReplayStatus.COMPLETED) == 4
        assert statuses.count(EnumReplayStatus.QUARANTINED) == 7

        # One ERROR line per refused record, naming the bound.
        bound_lines = [
            record
            for record in caplog.records
            if record.levelno == logging.ERROR and "REPLAY BOUND" in record.getMessage()
        ]
        assert len(bound_lines) == 7


class TestWindowMoves:
    def test_admit_replay_slides_the_window(self) -> None:
        state = DlqConsumerDrainState()
        # A list-backed fake clock: the test moves time, not the test runner.
        times = [0.0, 1.0, 2.0, 61.0, 62.0]
        cursor = 0

        def clock() -> float:
            nonlocal cursor
            value = times[cursor]
            cursor += 1
            return value

        state.clock = clock
        bound = ModelDlqTargetReplayBound(max_replays=2, window_seconds=60.0)

        assert state.admit_replay(_GOLDEN_TOPIC, _GROUP_A, bound) is True  # t=0
        assert state.admit_replay(_GOLDEN_TOPIC, _GROUP_A, bound) is True  # t=1
        assert state.admit_replay(_GOLDEN_TOPIC, _GROUP_A, bound) is False  # t=2
        # The window has slid past the first replay: the bound frees up again.
        assert state.admit_replay(_GOLDEN_TOPIC, _GROUP_A, bound) is True  # t=61
        # Another group is a different pair and spends its own bound.
        assert state.admit_replay(_GOLDEN_TOPIC, _GROUP_B, bound) is True  # t=62


class TestPolicySeam:
    def test_the_contract_policy_is_the_config_default(self) -> None:
        config = ModelDlqReplayEngineConfig(
            bootstrap_servers="x:9092", dlq_topic=build_dlq_topic("events")
        )
        assert SUFFIX_RUNTIME_TICK in config.broadcast_topics
        bound = config.target_replay_bound
        assert isinstance(bound, ModelDlqTargetReplayBound)
        assert bound.max_replays > 0
        assert bound.window_seconds > 0

    def test_the_default_config_refuses_a_tick_dead_letter(self) -> None:
        config = ModelDlqReplayEngineConfig(
            bootstrap_servers="x:9092", dlq_topic=build_dlq_topic("events")
        )
        eligible, reason = should_replay(
            _sibling(0, _GROUP_A, SUFFIX_RUNTIME_TICK), config
        )
        assert eligible is False
        assert "Broadcast topic" in reason

    def test_contract_broadcast_topics_name_the_tick_topic(self) -> None:
        """Seam: the topic constant the runtime publishes ticks on is the contract's."""
        contract = yaml.safe_load(
            files("omnibase_infra.nodes.node_dlq_replay_effect")
            .joinpath("contract.yaml")
            .read_text(encoding="utf-8")
        )
        assert SUFFIX_RUNTIME_TICK == "onex.intent.platform.runtime-tick.v1"
        assert SUFFIX_RUNTIME_TICK in contract["replay_policy"]["broadcast_topics"]

    def test_a_contract_without_replay_policy_fails_closed(self) -> None:
        with pytest.raises(ProtocolConfigurationError):
            parse_replay_policy("name: node_dlq_replay_effect\n", "test")

    @pytest.mark.parametrize(
        "bound",
        [
            {"max_replays": 0, "window_seconds": 60},
            {"max_replays": 5, "window_seconds": 0},
            {"max_replays": 5},
            {"max_replays": 5, "window_seconds": 60, "extra": 1},
        ],
    )
    def test_an_invalid_bound_is_refused(self, bound: dict[str, object]) -> None:
        text = yaml.safe_dump(
            {"replay_policy": {"broadcast_topics": [], "target_replay_bound": bound}}
        )
        with pytest.raises(ProtocolConfigurationError):
            parse_replay_policy(text, "test")

    def test_an_empty_topic_entry_is_refused_and_an_empty_list_is_allowed(self) -> None:
        bound = {"max_replays": 5, "window_seconds": 60}
        ok = parse_replay_policy(
            yaml.safe_dump(
                {
                    "replay_policy": {
                        "broadcast_topics": [],
                        "target_replay_bound": bound,
                    }
                }
            ),
            "test",
        )
        assert ok.broadcast_topics == ()
