# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The build clock node turns container start and health readings into a verdict (OMN-26316).

Known good: healthy 14 seconds after the start passes. Known bad: healthy after
6 minutes fails the 5 minute limit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from omnibase_infra.handlers.handler_build_clock import HandlerBuildClock
from omnibase_infra.models.model_build_clock import ModelBuildClock
from omnibase_infra.models.model_build_clock_request import ModelBuildClockRequest

START = datetime(2026, 11, 21, 9, 10, 40, tzinfo=UTC)
DOCKER_START = "2026-11-21T09:10:40.123456789Z"


def _request(after: timedelta) -> ModelBuildClockRequest:
    return ModelBuildClockRequest(
        container_started_at=DOCKER_START,
        first_healthy_at=START + after,
    )


@pytest.mark.unit
def test_known_good_build_passes() -> None:
    clock = HandlerBuildClock().handle(_request(timedelta(seconds=14)))
    assert isinstance(clock, ModelBuildClock)
    assert clock.container_started_at == START.replace(microsecond=123456)
    assert 0 < clock.minutes_to_healthy < 1
    assert clock.limit_minutes == 5.0
    assert clock.exceeded is False
    assert clock.verdict == "pass"
    assert str(clock.minutes_to_healthy) in clock.detail


@pytest.mark.unit
def test_known_bad_build_health_after_six_minutes_fails() -> None:
    clock = HandlerBuildClock().handle(_request(timedelta(minutes=6)))
    assert clock.minutes_to_healthy > 5
    assert clock.exceeded is True
    assert clock.verdict == "fail"
    assert "5" in clock.detail


@pytest.mark.unit
def test_unreadable_start_is_unread_not_a_pass_or_fail() -> None:
    clock = HandlerBuildClock().handle(
        ModelBuildClockRequest(
            container_started_at="not a time", first_healthy_at=START
        )
    )
    assert clock.verdict == "unread"
    assert clock.container_started_at is None
    assert clock.exceeded is False
