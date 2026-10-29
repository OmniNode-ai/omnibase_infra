# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18087: the recovering runtime bus must not serialise consumer-group joins.

On the dev lane on 2026-10-28 the runtime joined 522 boot subscriptions
one at a time, about 3 s each (the broker's group_initial_rebalance_delay), so
plugin activation took 2207 s and delegation had no consumer for 37 minutes.
Auto-wiring attaches contracts concurrently; the serialisation was
``EventBusRecovering.subscribe`` holding its transport lock across the join.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from omnibase_infra.errors import InfraConnectionError, InfraUnavailableError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from tests.unit.runtime.test_runtime_bus_recovery import make_bus

pytestmark = pytest.mark.unit


async def _noop(_message: object) -> None:
    return None


async def test_kafka_joins_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three subscribes are all inside the broker join at the same time."""
    bus = make_bus()
    in_join = 0
    peak = 0
    all_joining = asyncio.Event()
    release = asyncio.Event()

    async def join(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        nonlocal in_join, peak
        in_join += 1
        peak = max(peak, in_join)
        if in_join == 3:
            all_joining.set()
        await release.wait()
        in_join -= 1
        return AsyncMock()

    monkeypatch.setattr(EventBusKafka, "subscribe", join)
    tasks = [
        asyncio.create_task(
            bus.subscribe(f"onex.evt.test.t{i}.v1", on_message=_noop, group_id="g")
        )
        for i in range(3)
    ]
    try:
        # Serialised joins never reach three at once: the first holds the lock
        # until ``release``, which is only set after all three are joining.
        await asyncio.wait_for(all_joining.wait(), timeout=1)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert peak == 3
    assert len(bus._attachments) == 3


async def test_fallback_during_a_join_moves_that_subscription_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A join overtaken by the fallback releases its stale consumer and attaches locally."""
    bus = make_bus()
    joining = asyncio.Event()
    release = asyncio.Event()
    stale = AsyncMock()

    async def join(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        joining.set()
        await release.wait()
        return stale

    monkeypatch.setattr(EventBusKafka, "subscribe", join)
    monkeypatch.setattr(bus, "_recover", AsyncMock())
    task = asyncio.create_task(
        bus.subscribe("onex.evt.test.race.v1", on_message=_noop, group_id="g")
    )
    await asyncio.wait_for(joining.wait(), timeout=1)

    async def lose_broker() -> None:
        async with bus._transport_lock:
            await bus._enter_fallback(InfraConnectionError("broker lost"))

    try:
        # A join that holds the transport lock blocks the fallback outright.
        await asyncio.wait_for(lose_broker(), timeout=1)
    finally:
        release.set()
    unsubscribe = await asyncio.wait_for(task, timeout=1)

    stale.assert_awaited_once()
    assert bus._fallback_active
    assert len(bus._attachments) == 1
    # The retained attachment is the local one: removing it tears down locally.
    await unsubscribe()
    assert bus._attachments == {}


async def test_a_failed_join_on_a_live_transport_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller still sees its own attach failure, and nothing is retained."""
    bus = make_bus()

    async def join(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        raise InfraConnectionError("topic not provisioned")

    monkeypatch.setattr(EventBusKafka, "subscribe", join)
    with pytest.raises(InfraConnectionError):
        await bus.subscribe("onex.evt.test.missing.v1", on_message=_noop, group_id="g")
    assert bus._attachments == {}


async def test_a_join_that_outlives_close_is_released_not_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer that finishes joining after close() is torn down, not kept."""
    bus = make_bus()
    joining = asyncio.Event()
    release = asyncio.Event()
    stale = AsyncMock()

    async def join(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        joining.set()
        await release.wait()
        return stale

    monkeypatch.setattr(EventBusKafka, "subscribe", join)
    task = asyncio.create_task(
        bus.subscribe("onex.evt.test.closing.v1", on_message=_noop, group_id="g")
    )
    await asyncio.wait_for(joining.wait(), timeout=1)
    await asyncio.wait_for(bus.close(), timeout=1)
    release.set()
    with pytest.raises(InfraUnavailableError):
        await asyncio.wait_for(task, timeout=1)
    stale.assert_awaited_once()
    assert bus._attachments == {}
