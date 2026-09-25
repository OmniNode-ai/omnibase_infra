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

Where the scan is too fast for that comparison to mean anything, the test
SKIPS with the measured number rather than passing. A green tick that only
means "discovery was instant on this runner" is the vacuous pass this ticket
exists to argue against.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from omnibase_infra.services import service_runtime_health_monitor as monitor_module
from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
)

pytestmark = pytest.mark.integration

TICK_SECONDS = 0.01

#: Below this, the real scan on this box is too quick to distinguish a held
#: loop from a free one, and the comparison below would be theatre.
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

    real_discover = monitor_module._discover_contracts

    def timed_discover():
        nonlocal scan_seconds, contracts_seen
        started = time.monotonic()
        manifest = real_discover()
        scan_seconds = time.monotonic() - started
        contracts_seen = manifest.total_discovered
        return manifest

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

    if scan_seconds < MEANINGFUL_SCAN_SECONDS:
        pytest.skip(
            "the real contract scan took "
            f"{scan_seconds * 1000:.1f}ms ({contracts_seen} contracts) on this "
            "runner -- too fast to tell a held loop from a free one. The "
            "defect is a function of how many contracts are installed: on the "
            ".201 runtime it is ~1,000 contracts and ~11s, against onex-api's "
            "10.0s heartbeat budget."
        )

    worst = max(gaps)
    assert worst < scan_seconds / 2, (
        f"the event loop stalled for {worst:.3f}s while the real contract scan "
        f"ran for {scan_seconds:.3f}s over {contracts_seen} contracts. A "
        "gateway heartbeat arriving in that window cannot be answered inside "
        "onex-api's 10.0s budget, and the customer gets a 503 for work that "
        "was merely late (OMN-19373)"
    )
