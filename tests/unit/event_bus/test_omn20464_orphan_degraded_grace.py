# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20464: one stuck handler must not leave the runtime DEGRADED forever.

On the .201 dev lane on 2026-10-03 a delegation-inference-request dispatch was
abandoned past its 600s deadline (OMN-19355). The abandoned dispatch was
counted DEGRADED for as long as it ran, and the effects container's healthcheck
runs ``--degraded-policy fail``, so the container went unhealthy and stayed
that way until it was replaced.

The abandon path still must not cancel (a ``to_thread`` worker keeps running
under a cancelled awaiter and cancelling would release the projection gate slot
that thread still occupies). So the orphan stays tracked, and the gate-slot
accounting is untouched, but it stops counting toward DEGRADED after
``consumer_dispatch_orphan_degraded_grace_seconds``. The orphan limit still
flips the bus UNHEALTHY, however old the orphans are.
"""

from __future__ import annotations

import asyncio

import pytest

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from tests.unit.event_bus.test_omn19355_dispatch_deadline import (
    HUNG,
    _config,
    _Handlers,
    _Harness,
    _PartitionLog,
    _settle,
)

pytestmark = pytest.mark.unit

GRACE_SECONDS = 0.3


@pytest.mark.asyncio
async def test_an_old_orphan_stops_degrading_the_bus_but_stays_tracked() -> None:
    handlers = _Handlers(hung={HUNG})
    log = _PartitionLog([HUNG, HUNG + 1])
    harness = _Harness(
        _config(consumer_dispatch_orphan_degraded_grace_seconds=GRACE_SECONDS),
        log,
        handlers,
    )
    try:
        await harness.run()
        assert harness.bus is not None

        fresh = harness.bus.dispatch_deadline_status()
        assert fresh.orphaned_dispatches == 1
        assert fresh.status == "degraded"
        assert (await harness.bus.health_check())["degraded"] is True

        await asyncio.sleep(GRACE_SECONDS + 0.2)

        aged = harness.bus.dispatch_deadline_status()
        # Still tracked: the thread is running and holds its gate slot.
        assert aged.orphaned_dispatches == 1
        assert len(aged.orphans) == 1
        assert aged.status == "healthy"
        health = await harness.bus.health_check()
        assert health["degraded"] is False
        assert health["healthy"] is True

        handlers.release.set()
        await _settle(harness.bus)
        assert harness.bus.dispatch_deadline_status().orphaned_dispatches == 0
    finally:
        handlers.release.set()
        await harness.close()


@pytest.mark.asyncio
async def test_the_orphan_limit_still_flips_unhealthy_after_the_grace() -> None:
    handlers = _Handlers(hung={HUNG, HUNG + 1})
    log = _PartitionLog([HUNG, HUNG + 1, HUNG + 2])
    harness = _Harness(
        _config(
            consumer_dispatch_orphan_limit=2,
            consumer_dispatch_orphan_degraded_grace_seconds=GRACE_SECONDS,
        ),
        log,
        handlers,
    )
    try:
        await harness.run()
        assert harness.bus is not None
        await asyncio.sleep(GRACE_SECONDS + 0.2)
        status = harness.bus.dispatch_deadline_status()
        assert status.orphaned_dispatches == 2
        assert status.status == "unhealthy"
        assert (await harness.bus.health_check())["healthy"] is False
    finally:
        handlers.release.set()
        await harness.close()


def test_the_grace_is_a_typed_field_with_no_env_fallback() -> None:
    field = ModelKafkaEventBusConfig.model_fields[
        "consumer_dispatch_orphan_degraded_grace_seconds"
    ]
    assert field.default == 300.0
