# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The periodic health check must not freeze the process that serves the gateway.

WHAT WAS WRONG. ``ServiceRuntimeHealthMonitor.run_once`` is a coroutine, and
it called contract discovery -- a fully synchronous scan of every
``onex.nodes`` entry point and the ``contract.yaml`` beside it -- directly.
A synchronous call inside a coroutine holds the event loop until it returns,
and on the .201 runtime that scan covers ~1,000 contracts and takes ABOUT
ELEVEN SECONDS. The loop it froze is the same loop that consumes
``gateway-heartbeat-request.v1`` and publishes ``gateway-session.v1``.

WHAT IT COST. ``onex-api`` waits 10.0s for that response and then answers the
customer 503. Eleven is larger than ten, so every heartbeat that arrived
inside a sweep was refused BY CONSTRUCTION -- not because anything failed, but
because the answer could not be computed until the sweep let go. Measured on
2026-09-24 against the ``onex-dev`` namespace on the staging node (the
instance id is denylisted in this repo and lives on OMN-19373 with the rest of
the capture): two sweeps, 02:12:59 -> 02:13:10 and 02:18:10 -> 02:18:21, each
~11s, and in BOTH cases
runtime-effects published the answer in the very second the sweep ended,
after the caller had already abandoned it. 2 of 14 heartbeats over one 3m42s
run at the designed 15s cadence; the healthy round trip is ~1s.

The check runs every 300s, so this recurred roughly every five minutes and no
client cadence could dodge it. It is also why OMN-15957's Test 3 could not
produce a clean trace on three attempts: that test needs ~20 consecutive
heartbeats over ~5 minutes, so it cannot avoid spanning a sweep.

WHAT THESE TESTS PIN. Not "the call was moved" -- that is an implementation
detail a refactor could undo while re-introducing the freeze. They pin the
BEHAVIOUR: while the check is discovering, another coroutine on the same loop
keeps getting scheduled. That is the property the gateway actually depends on,
and it is the one that was false.

No Kafka, no network: ``bootstrap_servers=""`` skips the admin dimensions, and
discovery is replaced with a fixture that blocks the way the real one does.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
)

pytestmark = pytest.mark.unit

#: Long enough that an inline call is unmistakable against the tick interval
#: below, short enough to keep the suite fast. The real scan is ~11s; the
#: ratio that matters is block >> tick, and 0.4s vs 0.01s is the same shape.
BLOCKING_SECONDS = 0.4
TICK_SECONDS = 0.01
#: How many times the ticker should wake during one block if the loop is free.
EXPECTED_TICKS = BLOCKING_SECONDS / TICK_SECONDS


def _manifest() -> MagicMock:
    manifest = MagicMock()
    manifest.total_discovered = 0
    manifest.total_errors = 0
    manifest.errors = ()
    manifest.all_subscribe_topics.return_value = ()
    return manifest


def _blocking_discover() -> MagicMock:
    """Discovery as it really behaves: synchronous, and slow."""
    time.sleep(BLOCKING_SECONDS)
    return _manifest()


class _SilentBus:
    """No consumer-sync surface. The dimensions under test do not need one.

    ``publish_envelope`` is a no-op rather than absent so the monitor's own
    emit path stays quiet; its failure is logged, not raised, and a stack
    trace in the captured log of every test here would bury a real one.
    """

    async def publish_envelope(self, *_args: object, **_kwargs: object) -> None:
        return None


async def _run_once_with(discover) -> tuple[object, list[float]]:
    """Run one health check while a ticker shares the loop with it.

    The ticker records the wall-clock gap between its own wake-ups. If the
    check blocks the loop, one of those gaps swallows the whole block.
    """
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=_SilentBus(),  # type: ignore[arg-type]
        bootstrap_servers="",
        check_interval_seconds=300.0,
        boot_grace_seconds=0.0,
    )
    gaps: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(TICK_SECONDS)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    with (
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            side_effect=discover,
        ),
        patch(
            "omnibase_infra.services.service_runtime_health_monitor"
            "._filter_manifest_for_runtime_profile",
            side_effect=lambda manifest: manifest,
        ),
    ):
        task = asyncio.create_task(ticker())
        try:
            event = await monitor.run_once()
        finally:
            stop.set()
            await task
    return event, gaps


def _dimension(event: object, name: str):
    for dimension in event.dimensions:  # type: ignore[attr-defined]
        if dimension.name == name:
            return dimension
    return None


# --- the property the gateway depends on -----------------------------------


@pytest.mark.asyncio
async def test_the_loop_keeps_running_while_the_check_discovers() -> None:
    """The freeze, stated as the thing a co-resident coroutine can observe.

    A gateway heartbeat is exactly such a co-resident coroutine. If the
    longest gap the ticker sees is on the order of the discovery block, the
    heartbeat handler was not scheduled either, and `onex-api` gets its 503.
    """
    _event, gaps = await _run_once_with(_blocking_discover)

    assert gaps, "the ticker never ran at all"
    worst = max(gaps)
    # Generous: the bar is "nowhere near the block", not "perfectly smooth".
    # A CI box under load can miss a 10ms wake-up; it cannot miss 400ms of
    # them unless the loop was actually held.
    assert worst < BLOCKING_SECONDS / 2, (
        f"the event loop stalled for {worst:.3f}s during the health check "
        f"(discovery blocks for {BLOCKING_SECONDS}s) -- a gateway heartbeat "
        f"arriving in that window cannot be answered inside onex-api's 10s "
        f"budget, which is OMN-19373"
    )
    # And it kept being scheduled throughout, not just before and after.
    assert len(gaps) >= EXPECTED_TICKS / 2, (
        f"only {len(gaps)} ticks during a {BLOCKING_SECONDS}s check -- the "
        "loop was starved even if no single gap crossed the bar"
    )


@pytest.mark.asyncio
async def test_the_positive_control_detects_an_inline_call() -> None:
    """The bar above is real: put the block back on the loop and it trips.

    Without this, a change that made discovery instant would make the test
    above pass for the wrong reason and stop guarding anything.
    """
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=_SilentBus(),  # type: ignore[arg-type]
        bootstrap_servers="",
        check_interval_seconds=300.0,
        boot_grace_seconds=0.0,
    )
    gaps: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(TICK_SECONDS)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    async def inline_block() -> None:
        # The pre-fix shape, reduced to one line: synchronous work awaited by
        # a coroutine on the shared loop.
        time.sleep(BLOCKING_SECONDS)

    task = asyncio.create_task(ticker())
    try:
        await inline_block()
    finally:
        stop.set()
        await task

    # Starvation shows up as ticks that never happened. On a hard block the
    # ticker does not run at all, so `gaps` is empty -- which is the strongest
    # form of the signal, not a missing measurement.
    assert len(gaps) < EXPECTED_TICKS / 2, (
        f"an inline synchronous block produced {len(gaps)} ticks, close to the "
        f"{EXPECTED_TICKS:.0f} a free loop gives -- the measurement in this "
        "file cannot detect the defect it exists to catch"
    )
    assert not gaps or max(gaps) >= BLOCKING_SECONDS / 2
    assert monitor is not None


# --- the check still answers the same question ------------------------------


@pytest.mark.asyncio
async def test_moving_it_off_the_loop_did_not_change_the_verdict() -> None:
    """Same manifest in, same discovery dimension out.

    Threading a call is only safe if the answer is unchanged, so this asserts
    the numbers the discovery dimension reports still come from the manifest
    discovery returned.
    """

    def discover() -> MagicMock:
        manifest = _manifest()
        manifest.total_discovered = 1009
        manifest.total_errors = 0
        return manifest

    event, _gaps = await _run_once_with(discover)
    found = _dimension(event, "discovery_errors")
    assert found is not None
    assert found.status == "HEALTHY"
    assert "1009 contracts loaded cleanly" in found.detail, (
        "the count the dimension reports must still be the one discovery "
        f"returned, got: {found.detail!r}"
    )


@pytest.mark.asyncio
async def test_a_discovery_error_is_still_reported_from_the_thread() -> None:
    """The failure path survives the move too, not just the happy one."""

    def discover() -> MagicMock:
        manifest = _manifest()
        manifest.total_discovered = 1009
        manifest.total_errors = 2
        manifest.errors = (
            MagicMock(entry_point_name="n1", package_name="p", error="boom"),
            MagicMock(entry_point_name="n2", package_name="p", error="boom"),
        )
        return manifest

    event, _gaps = await _run_once_with(discover)
    found = _dimension(event, "discovery_errors")
    assert found is not None
    assert found.status == "DEGRADED", (
        "a manifest carrying errors must still degrade the dimension -- "
        "moving the call to a thread must not swallow what it found"
    )


@pytest.mark.asyncio
async def test_a_raising_discovery_still_leaves_the_check_standing() -> None:
    """``to_thread`` re-raises in the awaiting coroutine, so the existing
    ``except`` around this block still catches it and the remaining dimensions
    are still reported. If the exception escaped instead, ``_loop`` would log
    and skip the whole check every interval."""

    def discover() -> MagicMock:
        raise RuntimeError("entry point scan exploded")

    event, _gaps = await _run_once_with(discover)
    found = _dimension(event, "discovery_errors")
    assert found is not None, "a discovery failure must not abort the check"
    assert found.status == "CRITICAL", (
        "the raise must surface as the CRITICAL branch of this dimension -- "
        "if it escaped instead, `_loop` would log and skip the whole check "
        "every interval and this surface would go silent"
    )
    assert "RuntimeError" in found.detail
