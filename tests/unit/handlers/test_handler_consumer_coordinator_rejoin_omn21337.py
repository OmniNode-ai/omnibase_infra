# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure coordinator rejoin decisions and alarm invariants (OMN-21337)."""

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import (
    EnumConsumerCoordinatorRejoinReason,
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.handlers.handler_consumer_coordinator_rejoin import (
    HandlerConsumerCoordinatorRejoin,
)
from omnibase_infra.models.health.model_consumer_coordinator_rejoin_decision import (
    ModelConsumerCoordinatorRejoinDecision,
)
from omnibase_infra.models.health.model_consumer_coordinator_rejoin_request import (
    ModelConsumerCoordinatorRejoinRequest,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    (
        "known",
        "reachable",
        "seconds",
        "since",
        "failed",
        "raised",
        "reason",
        "rejoin",
        "alarm",
    ),
    [
        (True, True, 100, None, 1, False, "coordinator_known", False, False),
        (False, False, 100, None, 0, False, "broker_unreachable", False, False),
        (False, False, 100, None, 1, False, "broker_unreachable", False, True),
        (False, True, 59, None, 0, False, "within_bound", False, False),
        (False, True, 59, None, 1, False, "within_bound", False, True),
        (False, True, 60, 59, 0, False, "within_cooldown", False, False),
        (False, True, 60, 59, 1, False, "within_cooldown", False, True),
        (
            False,
            True,
            60,
            60,
            0,
            False,
            "coordinator_unavailable_past_bound",
            True,
            False,
        ),
        (
            False,
            True,
            60,
            None,
            0,
            False,
            "coordinator_unavailable_past_bound",
            True,
            False,
        ),
        (
            False,
            True,
            90,
            None,
            2,
            False,
            "coordinator_unavailable_past_bound",
            True,
            True,
        ),
        (
            False,
            True,
            90,
            None,
            2,
            True,
            "coordinator_unavailable_past_bound",
            True,
            False,
        ),
        (False, True, 90, 0, 2, True, "within_cooldown", False, False),
    ],
)
def test_coordinator_rejoin_decision(
    known: bool,
    reachable: bool,
    seconds: float,
    since: float | None,
    failed: int,
    raised: bool,
    reason: str,
    rejoin: bool,
    alarm: bool,
) -> None:
    request = ModelConsumerCoordinatorRejoinRequest(
        topic="topic",
        consumer_group="effective-group",
        broker="broker:9092",
        coordinator_known=known,
        broker_reachable=reachable,
        coordinator_unavailable_seconds=seconds,
        seconds_since_last_rejoin=since,
        failed_rejoins=failed,
        alarm_raised=raised,
        unavailable_bound_seconds=60,
        rejoin_cooldown_seconds=60,
    )
    handler = HandlerConsumerCoordinatorRejoin()
    decision = handler.handle(request)
    assert decision.reason.value == reason
    assert decision.rejoin is rejoin
    assert decision.alarm is alarm
    assert bool(decision.alarm_message) is alarm
    if alarm:
        for value in (
            request.topic,
            request.consumer_group,
            request.broker,
            str(failed),
        ):
            assert value in decision.alarm_message
        if reachable:
            assert f"has answered for {seconds:g}s" in decision.alarm_message
        else:
            # An unreachable broker never answered: the text must not say it did.
            assert "is unreachable" in decision.alarm_message
            assert "has answered" not in decision.alarm_message
            assert "coordinator" not in decision.alarm_message
    assert handler.handle(request) == decision
    assert handler.handler_type is EnumHandlerType.COMPUTE_HANDLER
    assert handler.handler_category is EnumHandlerTypeCategory.COMPUTE


@pytest.mark.parametrize(("alarm", "message"), [(True, ""), (False, "unexpected")])
def test_alarm_message_matches_alarm(alarm: bool, message: str) -> None:
    with pytest.raises(ValidationError):
        ModelConsumerCoordinatorRejoinDecision(
            reason=EnumConsumerCoordinatorRejoinReason.WITHIN_COOLDOWN,
            rejoin=False,
            alarm=alarm,
            alarm_message=message,
        )
