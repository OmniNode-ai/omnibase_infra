# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316 -- a dispatch deadline expiry is not replayed without bound.

Measured on the dev lane 2026-11-25 (19:23Z to 19:53Z): four dod-verify-start
dispatches at offsets 120620 to 120624 each outlived the 600 s dispatch
deadline and all carried the same dead-lettered intent (one correlation id).
``node_dlq_replay_effect`` republished it about every 10 minutes. A deadline
expiry leaves the handler running, abandoned, so every replay adds one more
orphan dispatch, and a fourth orphan passes the runtime's limit of 3 and
/health answers 503.

Nothing in ``should_replay`` treated the class as special, so a deadline
dead letter was replayed up to ``max_replay_count`` (5), and the count was
read off each copy's own ``retry_count`` alone.

The replay policy now declares ``deadline_exceeded_max_replays``. A deadline
dead letter is replayed at most that many times ACROSS its dead letters
(lineage by correlation id, not per copy) and then quarantined by name.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumDlqFailureClass
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.nodes.node_dlq_replay_effect.replay_policy import (
    load_replay_policy,
)
from tests.unit.nodes.node_dlq_replay_effect.test_handler_dlq_replay import _handler

pytestmark = pytest.mark.unit

# onex-topic-allow: quotes the live topic whose dead letters were measured
_COMMAND_TOPIC = "onex.cmd.omnimarket.dod-verify-start.v1"
_DLQ_TOPIC = "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: live DLQ
_DEADLINE = EnumDlqFailureClass.DISPATCH_DEADLINE_EXCEEDED.value
_CORRELATION = UUID("2590d878-2d28-4784-ab87-c31c21fc3977")


def _config(**overrides: int) -> ModelDlqReplayEngineConfig:
    return ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_DLQ_TOPIC,
        max_replay_count=5,
        **overrides,
    )


def _dead_letter(
    *, failure_class: str | None, offset: int, retry_count: int = 0
) -> ModelDlqMessage:
    envelope: dict[str, object] = {
        "original_topic": _COMMAND_TOPIC,
        "original_message": {"value": '{"ticket": "OMN-17427"}', "offset": offset},
        "failure_reason": "handler exceeded the 600 s dispatch deadline",
        "failure_timestamp": "2026-11-25T19:33:00Z",
        "correlation_id": str(_CORRELATION if failure_class else uuid4()),
        "retry_count": retry_count,
        "error_type": "DispatchDeadlineExceededError",
    }
    if failure_class is not None:
        envelope["failure_class"] = failure_class
    return ModelDlqMessage.from_kafka_message(
        envelope, dlq_offset=offset, dlq_partition=0
    )


def test_contract_declares_the_deadline_replay_count() -> None:
    assert load_replay_policy().deadline_exceeded_max_replays == 0


def test_deadline_expiry_is_quarantined_by_name_under_the_declared_policy() -> None:
    eligible, reason = should_replay(
        _dead_letter(failure_class=_DEADLINE, offset=120620), _config()
    )
    assert eligible is False
    assert _DEADLINE in reason
    assert "OMN-26316" in reason


def test_transient_failure_is_still_replayed() -> None:
    """Positive control: only the deadline class is bounded this way."""
    message = _dead_letter(
        failure_class=EnumDlqFailureClass.CONSUMER_ERROR.value, offset=1
    )
    eligible, reason = should_replay(message, _config())
    assert eligible is True, reason


async def test_copies_with_fresh_counts_share_one_declared_count() -> None:
    """Each expiry writes a NEW dead letter with retry_count 0; the lineage bounds them."""
    copies = [
        _dead_letter(failure_class=_DEADLINE, offset=offset)
        for offset in (120620, 120621, 120622, 120624)
    ]
    assert {c.retry_count for c in copies} == {0}
    handler, _consumer, producer, quarantine = _handler(
        copies, _config(deadline_exceeded_max_replays=1)
    )

    result = await handler.run()

    assert result.completed == 1
    assert result.quarantined == 3
    assert len(producer.replayed) == 1
    assert all(_DEADLINE in reason for _m, reason, _p in quarantine.quarantined)


async def test_default_policy_replays_none_of_the_orphans() -> None:
    copies = [
        _dead_letter(failure_class=_DEADLINE, offset=offset)
        for offset in (120620, 120621)
    ]
    handler, _consumer, producer, _quarantine = _handler(copies, _config())

    result = await handler.run()

    assert result.completed == 0
    assert result.quarantined == 2
    assert producer.replayed == []
