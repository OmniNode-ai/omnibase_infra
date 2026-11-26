# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316 -- a dispatch-deadline dead letter is quarantined, never replayed.

Measured on the dev lane 2026-11-25: ``node_dlq_replay_effect`` republished one
dead-lettered ``dod-verify-start`` intent about every 10 minutes. Each replay
outlived the 600 s dispatch deadline and left an abandoned handler running, and
the fourth orphan took /health to 503.

This drives the real ``HandlerDlqReplay`` drain loop, the real engine config and
the contract-declared replay policy over an in-memory scratch bus (an in-memory
consumer, replay producer and quarantine producer). It needs no broker and never
skips. The assertion is on what reached each side of the bus: the replay
producer received nothing and the quarantine producer received every deadline
dead letter, each reason naming ``dispatch_deadline_exceeded``. A
``consumer_error`` dead letter in the same drain is the positive control: it is
replayed, so a handler that quarantined everything would fail here.

Ticket: OMN-26316
Evidence-Ticket: OMN-26316
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumDlqFailureClass
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.nodes.node_dlq_replay_effect.replay_policy import (
    load_replay_policy,
)
from tests.unit.nodes.node_dlq_replay_effect.test_handler_dlq_replay import _handler

pytestmark = pytest.mark.integration

# onex-topic-allow: quotes the live topic whose dead letters were measured
_COMMAND_TOPIC = "onex.cmd.omnimarket.dod-verify-start.v1"
_DLQ_TOPIC = "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: live DLQ
_DEADLINE = EnumDlqFailureClass.DISPATCH_DEADLINE_EXCEEDED.value
_CORRELATION = UUID("2590d878-2d28-4784-ab87-c31c21fc3977")


def _config() -> ModelDlqReplayEngineConfig:
    return ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_DLQ_TOPIC,
        max_replay_count=5,
    )


def _dead_letter(
    *, failure_class: str, offset: int, correlation: UUID
) -> ModelDlqMessage:
    return ModelDlqMessage.from_kafka_message(
        {
            "original_topic": _COMMAND_TOPIC,
            "original_message": {"value": '{"ticket": "OMN-17427"}', "offset": offset},
            "failure_reason": "handler exceeded the 600 s dispatch deadline",
            "failure_timestamp": "2026-11-25T19:33:00Z",
            "correlation_id": str(correlation),
            "retry_count": 0,
            "error_type": "DispatchDeadlineExceededError",
            "failure_class": failure_class,
        },
        dlq_offset=offset,
        dlq_partition=0,
    )


async def test_deadline_dead_letters_are_quarantined_and_not_replayed() -> None:
    assert load_replay_policy().deadline_exceeded_max_replays == 0
    deadline = [
        _dead_letter(failure_class=_DEADLINE, offset=o, correlation=_CORRELATION)
        for o in (120620, 120621, 120622, 120624)
    ]
    transient = _dead_letter(
        failure_class=EnumDlqFailureClass.CONSUMER_ERROR.value,
        offset=120625,
        correlation=uuid4(),
    )
    handler, _consumer, producer, quarantine = _handler(
        [*deadline, transient], _config()
    )

    result = await handler.run()

    assert result.quarantined == len(deadline)
    assert result.completed == 1
    # Replay side of the bus: only the transient failure was republished.
    assert producer.replayed == [(_COMMAND_TOPIC, transient.correlation_id)]
    # Quarantine side of the bus: every deadline dead letter, named by class.
    assert len(quarantine.quarantined) == len(deadline)
    assert all(_DEADLINE in reason for _m, reason, _p in quarantine.quarantined)
