# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for OMN-7464 Kafka subscription cold-start concurrency."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from tests.conftest import make_test_node_identity

pytestmark = pytest.mark.integration


def _make_bus() -> EventBusKafka:
    return EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:19092",
            environment="test",
        )
    )


async def _handler(_message: ModelEventMessage) -> None:
    return None


@pytest.mark.asyncio
async def test_concurrent_distinct_subscriptions_start_consumers_in_parallel() -> None:
    """Distinct subscription keys must not serialize on EventBusKafka._lock."""
    bus = _make_bus()
    bus._started = True
    bus._validate_topic_name = MagicMock()  # type: ignore[method-assign]

    active_starts = 0
    max_active_starts = 0
    started: list[tuple[str, str]] = []

    async def fake_start(topic: str, group_id: str) -> None:
        nonlocal active_starts, max_active_starts
        active_starts += 1
        max_active_starts = max(max_active_starts, active_starts)
        await asyncio.sleep(0)
        bus._group_consumers[(topic, group_id)] = AsyncMock()
        bus._pending_consumer_keys.discard((topic, group_id))
        started.append((topic, group_id))
        active_starts -= 1

    bus._start_consumer_for_topic_unlocked = AsyncMock(  # type: ignore[method-assign]
        side_effect=fake_start
    )

    topics = tuple(f"onex.evt.omnibase-infra.cold-start-{idx}.v1" for idx in range(5))
    await asyncio.gather(
        *(
            bus.subscribe(topic, make_test_node_identity(str(idx)), _handler)
            for idx, topic in enumerate(topics)
        )
    )

    assert bus._start_consumer_for_topic_unlocked.await_count == len(topics)
    assert len(started) == len(topics)
    assert max_active_starts > 1


@pytest.mark.asyncio
async def test_concurrent_duplicate_subscription_starts_one_consumer() -> None:
    """Pending-key reservation prevents duplicate consumers for the same key."""
    bus = _make_bus()
    bus._started = True
    bus._validate_topic_name = MagicMock()  # type: ignore[method-assign]

    topic = "onex.evt.omnibase-infra.cold-start-shared.v1"
    identity = make_test_node_identity("shared")
    started: list[tuple[str, str]] = []

    async def fake_start(start_topic: str, group_id: str) -> None:
        await asyncio.sleep(0)
        bus._group_consumers[(start_topic, group_id)] = AsyncMock()
        bus._pending_consumer_keys.discard((start_topic, group_id))
        started.append((start_topic, group_id))

    bus._start_consumer_for_topic_unlocked = AsyncMock(  # type: ignore[method-assign]
        side_effect=fake_start
    )

    await asyncio.gather(*(bus.subscribe(topic, identity, _handler) for _ in range(5)))

    assert bus._start_consumer_for_topic_unlocked.await_count == 1
    assert len(started) == 1
    assert started[0][0] == topic
    assert len(bus._subscribers[topic]) == 5


@pytest.mark.asyncio
async def test_start_consuming_starts_distinct_consumers_in_parallel() -> None:
    """start_consuming reserves all keys and starts distinct consumers concurrently."""
    bus = _make_bus()
    bus._started = True

    active_starts = 0
    max_active_starts = 0
    started: list[tuple[str, str]] = []

    async def fake_start(topic: str, group_id: str) -> None:
        nonlocal active_starts, max_active_starts
        active_starts += 1
        max_active_starts = max(max_active_starts, active_starts)
        await asyncio.sleep(0)
        bus._group_consumers[(topic, group_id)] = AsyncMock()
        bus._pending_consumer_keys.discard((topic, group_id))
        started.append((topic, group_id))
        active_starts -= 1

    bus._start_consumer_for_topic_unlocked = AsyncMock(  # type: ignore[method-assign]
        side_effect=fake_start
    )

    group_a = "service-a"
    group_b = "service-b"
    async with bus._lock:
        bus._subscribers["onex.evt.omnibase-infra.start-consuming-a.v1"] = [
            (group_a, "sub-a-1", _handler),
            (group_b, "sub-a-2", _handler),
        ]
        bus._subscribers["onex.evt.omnibase-infra.start-consuming-b.v1"] = [
            (group_a, "sub-b-1", _handler),
        ]

    task = asyncio.create_task(bus.start_consuming())
    await asyncio.sleep(0.1)
    await bus.shutdown()
    await asyncio.wait_for(task, timeout=2.0)

    assert bus._start_consumer_for_topic_unlocked.await_count == 3
    assert len(started) == 3
    assert len(set(started)) == 3
    assert max_active_starts > 1


# ---------------------------------------------------------------------------
# OMN-12448: bounded consumer-startup concurrency + retry-then-raise.
# Unbounded asyncio.gather over ~870 group-joins stampedes the broker's group
# coordinator; one slow join blowing the timeout aborts the whole boot, which
# crash-loops the runtime. start_consuming must cap in-flight starts and retry
# transient failures before failing the boot.
# ---------------------------------------------------------------------------


def _make_bus_tuned(concurrency: int, max_retries: int) -> EventBusKafka:
    return EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:19092",
            environment="test",
            consumer_start_concurrency=concurrency,
            consumer_start_max_retries=max_retries,
        )
    )


async def _drive_start_consuming(bus: EventBusKafka) -> None:
    """Run start_consuming to completion of the startup fan-out, then stop."""
    task = asyncio.create_task(bus.start_consuming())
    await asyncio.sleep(0.2)
    await bus.shutdown()
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_start_consuming_caps_in_flight_consumer_starts() -> None:
    """In-flight consumer starts must never exceed consumer_start_concurrency."""
    concurrency = 4
    total = 20
    bus = _make_bus_tuned(concurrency=concurrency, max_retries=3)
    bus._started = True

    active = 0
    max_active = 0

    async def fake_start(topic: str, group_id: str) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        bus._group_consumers[(topic, group_id)] = AsyncMock()
        bus._pending_consumer_keys.discard((topic, group_id))
        active -= 1

    bus._start_consumer_for_topic_unlocked = AsyncMock(  # type: ignore[method-assign]
        side_effect=fake_start
    )

    async with bus._lock:
        for idx in range(total):
            bus._subscribers[f"onex.evt.omnibase-infra.bound-{idx}.v1"] = [
                ("service", f"sub-{idx}", _handler),
            ]

    await _drive_start_consuming(bus)

    assert bus._start_consumer_for_topic_unlocked.await_count == total
    assert max_active <= concurrency


@pytest.mark.asyncio
async def test_start_consuming_retries_transient_failure_without_aborting_boot() -> (
    None
):
    """A consumer that fails once then succeeds must not abort the boot."""
    bus = _make_bus_tuned(concurrency=4, max_retries=2)
    bus._started = True

    attempts: dict[str, int] = {}
    flaky = "onex.evt.omnibase-infra.flaky.v1"

    async def fake_start(topic: str, group_id: str) -> None:
        attempts[topic] = attempts.get(topic, 0) + 1
        if topic == flaky and attempts[topic] == 1:
            # Mirror the real failure path: pending key discarded before raising.
            bus._pending_consumer_keys.discard((topic, group_id))
            raise RuntimeError("transient group-join timeout")
        bus._group_consumers[(topic, group_id)] = AsyncMock()
        bus._pending_consumer_keys.discard((topic, group_id))

    bus._start_consumer_for_topic_unlocked = AsyncMock(  # type: ignore[method-assign]
        side_effect=fake_start
    )

    async with bus._lock:
        bus._subscribers[flaky] = [("service", "sub-flaky", _handler)]
        bus._subscribers["onex.evt.omnibase-infra.healthy.v1"] = [
            ("service", "sub-healthy", _handler),
        ]

    # Must NOT raise — the flaky consumer recovers on retry.
    await _drive_start_consuming(bus)

    assert attempts[flaky] == 2
    assert (flaky, "service.__t." + flaky) not in bus._pending_consumer_keys


@pytest.mark.asyncio
async def test_start_consuming_raises_after_retries_exhausted() -> None:
    """A consumer that always fails raises only after max_retries are spent."""
    max_retries = 2
    bus = _make_bus_tuned(concurrency=4, max_retries=max_retries)
    bus._started = True

    attempts: dict[str, int] = {}
    broken = "onex.evt.omnibase-infra.broken.v1"

    async def fake_start(topic: str, group_id: str) -> None:
        attempts[topic] = attempts.get(topic, 0) + 1
        if topic == broken:
            bus._pending_consumer_keys.discard((topic, group_id))
            raise RuntimeError("permanent group-join failure")
        bus._group_consumers[(topic, group_id)] = AsyncMock()
        bus._pending_consumer_keys.discard((topic, group_id))

    bus._start_consumer_for_topic_unlocked = AsyncMock(  # type: ignore[method-assign]
        side_effect=fake_start
    )

    async with bus._lock:
        bus._subscribers[broken] = [("service", "sub-broken", _handler)]
        bus._subscribers["onex.evt.omnibase-infra.ok.v1"] = [
            ("service", "sub-ok", _handler),
        ]

    with pytest.raises(RuntimeError, match="permanent group-join failure"):
        await bus.start_consuming()

    # Initial attempt + max_retries retries.
    assert attempts[broken] == max_retries + 1
