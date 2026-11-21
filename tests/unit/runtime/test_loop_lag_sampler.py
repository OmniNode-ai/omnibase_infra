# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the event-loop lag sampler and its reading model (OMN-26249).

Validates:
- A handler that blocks the loop for 0.6 s reads as a gap of at least 0.48 s
- An idle loop reads as a gap under 0.1 s
- The reading carries every field and refuses a reading missing any one
- Tick lag is the newest tick's consume time minus its scheduled_at, and a
  member that does not subscribe reports not-applicable with the reason
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.handlers.handler_runtime_loop_lag_sampler import (
    NO_TICK_IN_WINDOW_REASON,
    NOT_A_TICK_SUBSCRIBER_REASON,
    RuntimeLoopLagSampler,
)
from omnibase_infra.models.health.model_runtime_loop_reading import (
    ModelRuntimeLoopReading,
)
from omnibase_infra.models.health.model_runtime_loop_window_request import (
    ModelRuntimeLoopWindowRequest,
)

pytestmark = pytest.mark.unit


def _request(abandoned: int = 0) -> ModelRuntimeLoopWindowRequest:
    return ModelRuntimeLoopWindowRequest(
        correlation_id=uuid4(),
        runtime_member="dev.omnibase-infra.runtime-main",
        host="host-a",
        window_end=datetime.now(UTC) + timedelta(seconds=1),
        abandoned_dispatches=abandoned,
    )


def _valid_reading_fields() -> dict[str, Any]:
    start = datetime(2026, 11, 21, 0, 0, 0, tzinfo=UTC)
    return {
        "correlation_id": uuid4(),
        "runtime_member": "dev.omnibase-infra.runtime-main",
        "host": "host-a",
        "window_start": start,
        "window_end": start + timedelta(seconds=60),
        "max_loop_gap_ms": 12.5,
        "tick_subscriber": True,
        "tick_lag_ms": 40.0,
        "tick_lag_not_applicable_reason": None,
        "abandoned_dispatches": 0,
        "contract_costs": (),
    }


async def test_block_reads_as_a_gap_of_at_least_0_48_seconds() -> None:
    sampler = RuntimeLoopLagSampler()
    await sampler.start()
    try:
        await asyncio.sleep(0.2)
        time.sleep(0.6)  # blocks the loop the way a stalled handler does
        await asyncio.sleep(0.2)
        reading = sampler.close_window(_request())
    finally:
        await sampler.stop()
    assert reading.max_loop_gap_ms >= 480.0


async def test_block_still_running_when_the_window_closes_is_counted() -> None:
    sampler = RuntimeLoopLagSampler()
    await sampler.start()
    try:
        await asyncio.sleep(0.2)
        time.sleep(0.6)
        # The sampler has not woken yet; closing the window must see the lateness.
        reading = sampler.close_window(_request())
    finally:
        await sampler.stop()
    assert reading.max_loop_gap_ms >= 480.0


async def test_idle_loop_reads_as_a_gap_under_0_1_seconds() -> None:
    sampler = RuntimeLoopLagSampler()
    await sampler.start()
    try:
        await asyncio.sleep(0.5)
        reading = sampler.close_window(_request())
    finally:
        await sampler.stop()
    assert reading.max_loop_gap_ms < 100.0


async def test_window_close_resets_the_gap() -> None:
    sampler = RuntimeLoopLagSampler()
    await sampler.start()
    try:
        await asyncio.sleep(0.1)
        time.sleep(0.3)
        await asyncio.sleep(0.1)
        first = sampler.close_window(_request())
        await asyncio.sleep(0.3)
        second = sampler.close_window(_request())
    finally:
        await sampler.stop()
    assert first.max_loop_gap_ms >= 240.0
    assert second.max_loop_gap_ms < 100.0
    assert second.window_start == first.window_end


async def test_stop_is_idempotent_and_cancels_the_task() -> None:
    sampler = RuntimeLoopLagSampler()
    await sampler.start()
    assert sampler.is_running
    await sampler.stop()
    await sampler.stop()
    assert not sampler.is_running


def test_non_subscriber_reports_not_applicable_with_the_reason() -> None:
    reading = RuntimeLoopLagSampler().close_window(_request(abandoned=3))
    assert reading.tick_subscriber is False
    assert reading.tick_lag_ms is None
    assert reading.tick_lag_not_applicable_reason == NOT_A_TICK_SUBSCRIBER_REASON
    assert reading.abandoned_dispatches == 3


def test_subscriber_without_a_tick_reports_a_different_reason() -> None:
    sampler = RuntimeLoopLagSampler()
    sampler.mark_tick_subscriber()
    reading = sampler.close_window(_request())
    assert reading.tick_subscriber is True
    assert reading.tick_lag_ms is None
    assert reading.tick_lag_not_applicable_reason == NO_TICK_IN_WINDOW_REASON


def test_tick_lag_is_the_newest_tick_consume_minus_scheduled() -> None:
    sampler = RuntimeLoopLagSampler()
    scheduled = datetime(2026, 11, 21, 0, 0, 0, tzinfo=UTC)
    sampler.record_tick_consumed(scheduled, scheduled + timedelta(milliseconds=900))
    sampler.record_tick_consumed(
        scheduled + timedelta(seconds=1),
        scheduled + timedelta(seconds=1, milliseconds=250),
    )
    reading = sampler.close_window(_request())
    assert reading.tick_subscriber is True
    assert reading.tick_lag_ms == pytest.approx(250.0)
    assert reading.tick_lag_not_applicable_reason is None
    # The next window starts with no tick.
    assert sampler.close_window(_request()).tick_lag_ms is None


@pytest.mark.parametrize("missing", sorted(_valid_reading_fields()))
def test_reading_refuses_each_missing_field(missing: str) -> None:
    fields = _valid_reading_fields()
    del fields[missing]
    with pytest.raises(ValidationError):
        ModelRuntimeLoopReading(**fields)


def test_reading_accepts_every_field() -> None:
    reading = ModelRuntimeLoopReading(**_valid_reading_fields())
    assert reading.max_loop_gap_ms == 12.5


def test_reading_needs_exactly_one_of_tick_lag_and_reason() -> None:
    both = {**_valid_reading_fields(), "tick_lag_not_applicable_reason": "why"}
    neither = {**_valid_reading_fields(), "tick_lag_ms": None}
    with pytest.raises(ValidationError):
        ModelRuntimeLoopReading(**both)
    with pytest.raises(ValidationError):
        ModelRuntimeLoopReading(**neither)


def test_non_subscriber_cannot_carry_a_tick_lag() -> None:
    fields = {**_valid_reading_fields(), "tick_subscriber": False}
    with pytest.raises(ValidationError):
        ModelRuntimeLoopReading(**fields)


def test_window_must_end_after_it_starts() -> None:
    fields = _valid_reading_fields()
    fields["window_end"] = fields["window_start"]
    with pytest.raises(ValidationError):
        ModelRuntimeLoopReading(**fields)
