# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The REAL contract scan, and the loop that has to keep serving during it.

The unit half of OMN-19373 replaces discovery with a fixture that blocks for a
known time. That proves the wiring: whatever discovery does, it does it off
the event loop. It cannot prove the premise -- that the real scan is slow
enough to matter -- because it never runs the real scan.

This file runs it. No patching of ``_discover_contracts``: the monitor walks
the ``onex.nodes`` entry points actually installed in this environment, reads
the ``contract.yaml`` beside each, and builds a manifest, while a ticker
shares the loop and records whether it kept being scheduled.

WHY THE PREMISE NEEDS ITS OWN TEST. The measurement that produced this ticket
was taken on the .201 runtime, where the scan covers ~1,000 contracts and
takes about eleven seconds -- and ``onex-api`` gives a gateway heartbeat 10.0s
before it answers the customer 503. The number eleven is not a constant of the
code; it is a property of how many contracts are installed. A CI image with a
handful of contracts scans in milliseconds, and on such a box the freeze is
invisible. So this test does not assert a duration. It measures how long the
real scan takes HERE, and asserts the loop was not held for anything like it.

WHY THERE IS NO SKIP (OMN-19851). This test used to SKIP when the scan came in
under ``MEANINGFUL_SCAN_SECONDS``. On the CI runners the scan covers ~129
contracts and takes 48-55ms, astride the 50ms line, so the same test skipped on
one run and executed on the next. Every skip grew the skipped set the
skip-count ratchet (OMN-18776) refuses, and the shrink-only baseline gate
(OMN-19677) refuses recording it, so unrelated merge groups were dequeued on a
coin flip (omnibase_infra#4195, twice on 2026-09-27).

So the verdict no longer depends on runner speed. On EVERY runner the test
asserts the structural half: the real, unpatched discovery executed on a worker
thread, not on the thread running the event loop. That fails red on any box if
the ``asyncio.to_thread`` hop is removed, however fast the scan is, so a green
tick is never "discovery was instant here". Where the scan is also long enough
for timing to be meaningful, the loop-gap comparison runs on top of it. The
discovery memo is dropped first, so the scan measured is a real parse and not
a memo validation.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts_cache_clear,
)
from omnibase_infra.services import service_runtime_health_monitor as monitor_module
from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
)

pytestmark = pytest.mark.integration

TICK_SECONDS = 0.01

#: Below this, the real scan on this box is too quick to distinguish a held
#: loop from a free one by timing, so the loop-gap comparison is not made. The
#: thread assertion still is, on every runner.
MEANINGFUL_SCAN_SECONDS = 0.05


class _SilentBus:
    """No consumer-sync surface, and an emit that goes nowhere.

    The point of the run is the scan and the loop, not the event; a bus that
    raised on publish would bury the real signal under a stack trace.
    """

    async def publish_envelope(self, *_args: object, **_kwargs: object) -> None:
        return None


@pytest.mark.asyncio
async def test_the_real_contract_scan_does_not_hold_the_event_loop() -> None:
    """Run the actual discovery and watch a co-resident coroutine survive it.

    A gateway heartbeat handler is exactly such a coroutine. On the runtime
    this ticket was measured against, it is the one that stopped being
    scheduled for eleven seconds at a time.
    """
    scan_seconds = 0.0
    contracts_seen = 0
    scan_thread_ids: list[int] = []

    real_discover = monitor_module._discover_contracts

    def timed_discover():
        nonlocal scan_seconds, contracts_seen
        scan_thread_ids.append(threading.get_ident())
        started = time.monotonic()
        manifest = real_discover()
        scan_seconds = time.monotonic() - started
        contracts_seen = manifest.total_discovered
        return manifest

    # A memo left warm by an earlier test in this worker would make the
    # "real scan" a stat-only validation, which is neither the work the
    # runtime does on a cold sweep nor a fixed duration across runs.
    discover_contracts_cache_clear()
    loop_thread_id = threading.get_ident()

    monitor = ServiceRuntimeHealthMonitor(
        event_bus=_SilentBus(),  # type: ignore[arg-type]
        # Empty: the Kafka admin dimensions are a different concern and would
        # put a network call in the middle of the window being measured.
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

    # Only the timing wrapper is substituted. The scan underneath is the real
    # one, including the entry-point walk and every contract.yaml read.
    monitor_module._discover_contracts = timed_discover  # type: ignore[assignment]
    try:
        task = asyncio.create_task(ticker())
        try:
            event = await monitor.run_once()
        finally:
            stop.set()
            await task
    finally:
        monitor_module._discover_contracts = real_discover  # type: ignore[assignment]

    assert event is not None
    assert gaps, "the ticker never ran at all -- the loop was held throughout"

    # The structural half, asserted on every runner however fast the scan is.
    assert scan_thread_ids, (
        "run_once never called the real contract discovery, so this test "
        "measured nothing about it"
    )
    assert loop_thread_id not in scan_thread_ids, (
        "the real contract scan ran on the event-loop thread "
        f"({scan_seconds * 1000:.1f}ms over {contracts_seen} contracts here). "
        "On the .201 runtime that scan is ~1,000 contracts and ~11s, and a "
        "gateway heartbeat arriving in it cannot be answered inside onex-api's "
        "10.0s budget (OMN-19373)"
    )

    if scan_seconds < MEANINGFUL_SCAN_SECONDS:
        # Too fast here for timing to separate a held loop from a free one;
        # the thread assertion above is the verdict on this runner.
        return

    worst = max(gaps)
    assert worst < scan_seconds / 2, (
        f"the event loop stalled for {worst:.3f}s while the real contract scan "
        f"ran for {scan_seconds:.3f}s over {contracts_seen} contracts. A "
        "gateway heartbeat arriving in that window cannot be answered inside "
        "onex-api's 10.0s budget, and the customer gets a 503 for work that "
        "was merely late (OMN-19373)"
    )
