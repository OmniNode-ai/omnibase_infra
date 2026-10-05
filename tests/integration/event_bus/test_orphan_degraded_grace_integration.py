# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20464: an abandoned dispatch stops degrading the bus after a bounded grace.

Drives the real consumer loop through an in-memory partition log and reads the
bus health surface the runtime healthcheck consumes, so the grace is proven
end to end rather than on the status model alone.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.unit.event_bus.test_omn19355_dispatch_deadline import (
    HUNG,
    _config,
    _Handlers,
    _Harness,
    _PartitionLog,
    _settle,
)

GRACE_SECONDS = 0.3


@pytest.mark.asyncio
async def test_bus_health_recovers_after_grace_while_orphan_stays_tracked() -> None:
    handlers = _Handlers(hung={HUNG})
    harness = _Harness(
        _config(consumer_dispatch_orphan_degraded_grace_seconds=GRACE_SECONDS),
        _PartitionLog([HUNG, HUNG + 1]),
        handlers,
    )
    try:
        await harness.run()
        assert harness.bus is not None
        assert (await harness.bus.health_check())["degraded"] is True

        await asyncio.sleep(GRACE_SECONDS + 0.2)

        health = await harness.bus.health_check()
        assert health["degraded"] is False
        assert health["healthy"] is True
        assert harness.bus.dispatch_deadline_status().orphaned_dispatches == 1

        handlers.release.set()
        await _settle(harness.bus)
        assert harness.bus.dispatch_deadline_status().orphaned_dispatches == 0
    finally:
        handlers.release.set()
        await harness.close()
