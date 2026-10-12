# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20867: the lane's own tick producer stamps each tick with the slot it was due on.

The chain: ``start_lane_runtime_scheduler`` (main profile, lane switch on) runs its
tick loop against an event bus whose publish takes longer than one tick interval,
as the dev lane's did. Every tick that reaches the tick topic carries an
epoch-aligned ``scheduled_at`` no later than its ``now``, and the next tick is due
on the first slot after that ``now``: the spans ``(scheduled_at - interval, now]``
tile the clock, so a window gate on them fires each window exactly once.

RED on the parent: the loop stamped ``scheduled_at`` with ``now`` and waited a
whole interval after each publish, so consecutive ticks skipped slots.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.models import ModelRuntimeTick
from omnibase_infra.runtime.runtime_scheduler import start_lane_runtime_scheduler
from omnibase_infra.topics import SUFFIX_RUNTIME_TICK

_INTERVAL_MS = 100
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_SLOT = timedelta(milliseconds=_INTERVAL_MS)


def _slot_after(when: datetime) -> datetime:
    return _EPOCH + ((when - _EPOCH) // _SLOT + 1) * _SLOT


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lane_producer_ticks_tile_the_clock_when_publish_runs_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEX_RUNTIME_SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("ONEX_RUNTIME_SCHEDULER_PERSIST_SEQUENCE", "false")
    monkeypatch.setenv("ONEX_RUNTIME_SCHEDULER_TICK_INTERVAL_MS", str(_INTERVAL_MS))
    monkeypatch.setenv("ONEX_RUNTIME_SCHEDULER_MAX_JITTER_MS", "0")

    bus = EventBusInmemory(environment="test", group="runtime-tick-slot-alignment")
    await bus.start()
    publish_envelope = bus.publish_envelope

    async def _late_publish(
        envelope: object, topic: str, *, key: bytes | None = None
    ) -> None:
        # Each publish alone takes 1.12 intervals, the dev lane's median ratio.
        await asyncio.sleep(_INTERVAL_MS * 1.12 / 1000)
        await publish_envelope(envelope, topic, key=key)

    monkeypatch.setattr(bus, "publish_envelope", _late_publish)
    try:
        scheduler = await start_lane_runtime_scheduler(bus, "main")
        assert scheduler is not None
        try:

            async def _until(count: int) -> None:
                while (
                    len(await bus.get_event_history(topic=SUFFIX_RUNTIME_TICK)) < count
                ):
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(_until(6), timeout=10)
        finally:
            await scheduler.stop()

        history = await bus.get_event_history(topic=SUFFIX_RUNTIME_TICK)
        ticks = [
            ModelRuntimeTick.model_validate(
                ModelEventEnvelope[ModelRuntimeTick]
                .model_validate_json(message.value)
                .payload
            )
            for message in history
        ]
        assert len(ticks) >= 6
        for tick in ticks:
            assert (tick.scheduled_at - _EPOCH) % _SLOT == timedelta(0), tick
            assert tick.scheduled_at <= tick.now, tick
        for earlier, later in pairwise(ticks):
            assert later.scheduled_at == _slot_after(earlier.now), (earlier, later)
        # Positive control: each publish overran its interval, so the loop woke
        # past its slot (about 12 ms of a 100 ms slot) and still skipped none.
        assert any(
            t.now - t.scheduled_at > timedelta(milliseconds=5) for t in ticks[1:]
        )
    finally:
        await bus.close()
