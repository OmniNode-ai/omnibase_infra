# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316: an abandoned dispatch names the consumer group that holds it.

On a lab runtime on 2026-11-25 the health endpoint listed two
abandoned ``onex.intent.platform.runtime-tick.v1`` dispatches as::

    onex.intent.platform.runtime-tick.v1 partition=0 offset=744130
        subscription=343315bd-7a23-4192-805f-8d212f7962f3 age=5078s

The topic is read by about fifty handlers, one consumer group each, and the
``subscription`` value is a random id minted per subscribe call: it appears in
no contract, no log line outside this one and no DLQ record. The orphan carried
its consumer group (``ModelOrphanedDispatch.consumer_group``) but neither the
health line nor the ``dispatch_deadline_exceeded`` log line printed it, so the
question "which handler was the tick in" could not be answered from the runtime
once the container's log buffer had rotated past the subscribe line.
"""

from __future__ import annotations

import logging

import pytest

from tests.unit.event_bus.test_omn19355_dispatch_deadline import (
    GROUP,
    HUNG,
    _config,
    _Handlers,
    _Harness,
    _PartitionLog,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_the_health_line_for_an_orphan_names_its_consumer_group() -> None:
    handlers = _Handlers(hung={HUNG})
    log = _PartitionLog([HUNG, HUNG + 1])
    harness = _Harness(_config(), log, handlers)
    try:
        await harness.run()
        assert harness.bus is not None
        status = harness.bus.dispatch_deadline_status()
        assert status.orphaned_dispatches == 1
        assert f"group={GROUP}" in status.orphans[0]
    finally:
        handlers.release.set()
        await harness.close()


@pytest.mark.asyncio
async def test_the_deadline_error_line_names_the_consumer_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handlers = _Handlers(hung={HUNG})
    log = _PartitionLog([HUNG, HUNG + 1])
    harness = _Harness(_config(), log, handlers)
    try:
        with caplog.at_level(logging.ERROR, logger="omnibase_infra.event_bus"):
            await harness.run()
        lines = [
            r.getMessage()
            for r in caplog.records
            if "dispatch_deadline_exceeded" in r.getMessage()
        ]
        assert len(lines) == 1
        assert f"group={GROUP}" in lines[0]
    finally:
        handlers.release.set()
        await harness.close()
