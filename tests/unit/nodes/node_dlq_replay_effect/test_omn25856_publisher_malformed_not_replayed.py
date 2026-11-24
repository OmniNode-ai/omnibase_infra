# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25856 -- a command its publisher malformed is never replayed.

A dead letter classified ``failure_class=publisher_malformed`` carries a payload
that failed ``model_validate`` for every type-scoped candidate. The same bytes
meet the same validator on every replay, so each replay ends in a new failed
terminal. Measured on the dev lane over 24 h to 2026-11-24T17:28Z: 935 of 935
delegate-skill dead letters were ``publisher_malformed`` and 640 extra
``delegate-skill-failed`` events came from replaying them up to
``max_replay_count``.

The record must be counted as QUARANTINED, stay durable on the quarantine
topic with its reason, and never be published back to the original topic. A
transient failure (the classes a second attempt can fix) is still replayed.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus
from omnibase_infra.enums import EnumDlqFailureClass
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from tests.unit.nodes.node_dlq_replay_effect.test_handler_dlq_replay import (
    _FakeTracking,
    _handler,
)

pytestmark = pytest.mark.unit

# onex-topic-allow: quotes the live topic whose dead letters were measured
_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_DLQ_TOPIC = "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: live DLQ
_MALFORMED = EnumDlqFailureClass.PUBLISHER_MALFORMED.value


def _config() -> ModelDlqReplayEngineConfig:
    return ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_DLQ_TOPIC,
        max_replay_count=5,
    )


def _envelope(*, failure_class: str | None, failure_reason: str) -> dict[str, object]:
    """A DLQ envelope shaped as the dead-letter publisher writes it."""
    envelope: dict[str, object] = {
        "original_topic": _COMMAND_TOPIC,
        "original_message": {"value": '{"task_type": "x"}', "offset": 7},
        "failure_reason": failure_reason,
        "failure_timestamp": "2026-11-24T17:00:00Z",
        "correlation_id": str(uuid4()),
        "retry_count": 0,
        "error_type": "HandlerDispatchFailureError",
    }
    if failure_class is not None:
        envelope["failure_class"] = failure_class
    return envelope


def _parsed(*, failure_class: str | None, failure_reason: str) -> ModelDlqMessage:
    return ModelDlqMessage.from_kafka_message(
        _envelope(failure_class=failure_class, failure_reason=failure_reason),
        dlq_offset=3,
        dlq_partition=0,
    )


def test_publisher_malformed_dead_letter_is_not_replayed_on_first_attempt() -> None:
    message = _parsed(
        failure_class=_MALFORMED,
        failure_reason="payload failed model_validate: field text required",
    )
    eligible, reason = should_replay(message, _config())
    assert eligible is False
    assert _MALFORMED in reason
    assert "OMN-25856" in reason


def test_publisher_malformed_marker_in_failure_reason_is_not_replayed() -> None:
    """A record whose envelope lacks the key but whose reason names the class."""
    message = _parsed(
        failure_class=None,
        failure_reason=f"dispatch refused (failure_class={_MALFORMED})",
    )
    eligible, reason = should_replay(message, _config())
    assert eligible is False
    assert _MALFORMED in reason


@pytest.mark.parametrize(
    "failure_class",
    [
        EnumDlqFailureClass.CONSUMER_ERROR.value,
        EnumDlqFailureClass.NO_DISPATCHER.value,
        None,
    ],
)
def test_transient_failure_is_still_replayed(failure_class: str | None) -> None:
    """Positive control: the clause must refuse only the malformed class."""
    message = _parsed(
        failure_class=failure_class,
        failure_reason="HandlerDispatchFailureError: upstream timed out",
    )
    eligible, reason = should_replay(message, _config())
    assert eligible is True, reason


async def test_publisher_malformed_is_counted_and_kept_with_its_reason() -> None:
    message = _parsed(
        failure_class=_MALFORMED,
        failure_reason="payload failed model_validate: field text required",
    )
    tracking = _FakeTracking()
    handler, _consumer, producer, quarantine = _handler(
        [message], _config(), tracking=tracking
    )

    result = await handler.run()

    assert isinstance(handler, HandlerDlqReplay)
    assert result.quarantined == 1
    assert result.completed == 0
    assert producer.replayed == []
    assert len(quarantine.quarantined) == 1
    _msg, reason, payload = quarantine.quarantined[0]
    assert _MALFORMED in reason
    assert payload["reason"] == reason
    assert tracking.records[0].replay_status == EnumReplayStatus.QUARANTINED


async def test_transient_failure_is_replayed_by_the_handler() -> None:
    message = _parsed(
        failure_class=EnumDlqFailureClass.CONSUMER_ERROR.value,
        failure_reason="HandlerDispatchFailureError: upstream timed out",
    )
    handler, _consumer, producer, quarantine = _handler([message], _config())

    result = await handler.run()

    assert result.completed == 1
    assert result.quarantined == 0
    assert len(producer.replayed) == 1
    assert quarantine.quarantined == []
