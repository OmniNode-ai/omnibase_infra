# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the event-loop lag sampler and reading publisher (OMN-26249).

AC1: a 0.6 s block reads as a gap of at least 0.48 s, an idle loop under 0.1 s.
The publisher tests pin what a window's reading carries: the member identity,
contiguous windows, the tick lag or why it is absent, and the abandoned count.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from omnibase_infra.models.health.model_dispatch_deadline_status import (
    ModelDispatchDeadlineStatus,
)
from omnibase_infra.runtime.models.model_runtime_loop_reading import (
    ModelRuntimeLoopReading,
)
from omnibase_infra.runtime.util_loop_lag_sampler import (
    LoopLagSampler,
    RuntimeLoopReadingPublisher,
)
from omnibase_infra.topics import SUFFIX_RUNTIME_LOOP_READING_RECORDED

_INTERVAL = 0.02


@pytest.mark.unit
@pytest.mark.asyncio
async def test_block_reads_as_a_gap_of_at_least_480_ms() -> None:
    sampler = LoopLagSampler(interval_seconds=_INTERVAL)
    await sampler.start()
    try:
        await asyncio.sleep(_INTERVAL * 3)
        sampler.drain_max_gap_ms()
        time.sleep(0.6)  # a handler blocking the loop  # noqa: ASYNC251
        await asyncio.sleep(_INTERVAL * 3)
        assert sampler.drain_max_gap_ms() >= 480.0
    finally:
        await sampler.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_idle_loop_reads_under_100_ms() -> None:
    sampler = LoopLagSampler(interval_seconds=_INTERVAL)
    await sampler.start()
    try:
        await asyncio.sleep(0.5)
        assert sampler.drain_max_gap_ms() < 100.0
    finally:
        await sampler.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_drain_resets_the_window() -> None:
    sampler = LoopLagSampler(interval_seconds=_INTERVAL)
    await sampler.start()
    try:
        await asyncio.sleep(_INTERVAL * 3)
        time.sleep(0.3)  # noqa: ASYNC251
        await asyncio.sleep(_INTERVAL * 3)
        assert sampler.drain_max_gap_ms() >= 200.0
        await asyncio.sleep(0.2)
        assert sampler.drain_max_gap_ms() < 100.0
    finally:
        await sampler.stop()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_and_stop_are_idempotent() -> None:
    sampler = LoopLagSampler(interval_seconds=_INTERVAL)
    await sampler.start()
    await sampler.start()
    assert sampler.is_running
    await sampler.stop()
    await sampler.stop()
    assert not sampler.is_running


@pytest.mark.unit
def test_tick_lag_is_not_applicable_for_a_non_subscriber() -> None:
    sampler = LoopLagSampler()
    assert sampler.tick_lag() == (None, "not_a_tick_subscriber")


@pytest.mark.unit
def test_subscriber_with_no_tick_yet_is_not_the_not_applicable_case() -> None:
    sampler = LoopLagSampler()
    sampler.mark_tick_subscriber()
    assert sampler.tick_lag() == (None, "no_tick_consumed_yet")


@pytest.mark.unit
def test_tick_lag_is_consume_time_minus_scheduled_at_for_the_newest_tick() -> None:
    sampler = LoopLagSampler()
    t0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    sampler.record_tick(t0, t0 + timedelta(milliseconds=900))
    sampler.record_tick(t0 + timedelta(seconds=1), t0 + timedelta(seconds=1, milliseconds=40))
    assert sampler.tick_lag() == (pytest.approx(40.0), None)


@pytest.mark.unit
def test_a_tick_consumed_before_it_was_scheduled_reads_zero_not_negative() -> None:
    sampler = LoopLagSampler()
    t0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    sampler.record_tick(t0, t0 - timedelta(milliseconds=5))
    assert sampler.tick_lag() == (0.0, None)


class _BusWithDeadlines:
    """A transport with a consume loop: it reports what it abandoned."""

    def __init__(self, orphaned: int) -> None:
        self.publish_envelope = AsyncMock()
        self._orphaned = orphaned

    def dispatch_deadline_status(self) -> ModelDispatchDeadlineStatus:
        return ModelDispatchDeadlineStatus(
            deadline_seconds=30.0,
            orphan_limit=8,
            orphaned_dispatches=self._orphaned,
            degrading_orphans=0,
            deadline_expiries_total=self._orphaned,
            oldest_orphan_age_seconds=0.0,
            orphans=(),
        )


class _BusWithoutConsumeLoop:
    """The in-memory shape: nothing to abandon, no deadline status."""

    def __init__(self) -> None:
        self.publish_envelope = AsyncMock()


def _bus(orphaned: int | None) -> Any:
    return _BusWithoutConsumeLoop() if orphaned is None else _BusWithDeadlines(orphaned)


class _Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.t += timedelta(seconds=15)
        return self.t


def _publisher(bus: Any, sampler: LoopLagSampler) -> RuntimeLoopReadingPublisher:
    return RuntimeLoopReadingPublisher(
        sampler=sampler,
        event_bus=bus,
        member_id="dev.svc.node.v1:42",
        host="h201",
        window_seconds=15.0,
        now=_Clock(),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_windows_are_contiguous_and_carry_member_host_and_abandoned_count() -> None:
    bus = _bus(orphaned=3)
    publisher = _publisher(bus, LoopLagSampler())
    publisher._window_start = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)

    first = await publisher.publish_window()
    second = await publisher.publish_window()

    assert second.window_start == first.window_end
    assert first.member_id == "dev.svc.node.v1:42"
    assert first.host == "h201"
    assert first.abandoned_count == 3
    assert first.tick_lag_ms is None
    assert first.tick_lag_absent_reason == "not_a_tick_subscriber"
    assert bus.publish_envelope.await_count == 2
    kwargs = bus.publish_envelope.await_args.kwargs
    assert kwargs["topic"] == SUFFIX_RUNTIME_LOOP_READING_RECORDED
    assert isinstance(kwargs["envelope"].payload, ModelRuntimeLoopReading)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_bus_with_no_consume_loop_reports_zero_abandoned() -> None:
    publisher = _publisher(_bus(orphaned=None), LoopLagSampler())
    publisher._window_start = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    assert (await publisher.publish_window()).abandoned_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_failed_publish_does_not_stop_the_next_window() -> None:
    bus = _bus(orphaned=0)
    bus.publish_envelope.side_effect = [RuntimeError("broker down"), None]
    publisher = _publisher(bus, LoopLagSampler())
    publisher._window_start = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    await publisher.publish_window()
    await publisher.publish_window()
    assert bus.publish_envelope.await_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_window_loop_publishes_each_window() -> None:
    bus = _bus(orphaned=0)
    publisher = RuntimeLoopReadingPublisher(
        sampler=LoopLagSampler(interval_seconds=_INTERVAL),
        event_bus=bus,
        member_id="m",
        host="h",
        window_seconds=0.1,
    )
    await publisher.start()
    try:
        await asyncio.sleep(0.45)
    finally:
        await publisher.stop()
    assert bus.publish_envelope.await_count >= 3
    assert publisher.last_reading is not None
