# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration: an orphan read through the real consume loop names its group (OMN-26316).

The real ``EventBusKafka`` consume loop runs over a scripted partition log with a
handler parked past the dispatch deadline. The health status line for the
abandoned dispatch must print ``group=<consumer group>`` before
``subscription=<id>``.
"""

from __future__ import annotations

import pytest

from tests.unit.event_bus.test_omn19355_dispatch_deadline import (
    GROUP,
    HUNG,
    _config,
    _Handlers,
    _Harness,
    _PartitionLog,
)


@pytest.mark.asyncio
async def test_orphan_line_prints_group_before_subscription() -> None:
    handlers = _Handlers(hung={HUNG})
    log = _PartitionLog([HUNG, HUNG + 1])
    harness = _Harness(_config(), log, handlers)
    try:
        await harness.run()
        assert harness.bus is not None
        status = harness.bus.dispatch_deadline_status()
        assert status.orphaned_dispatches == 1
        line = status.orphans[0]
        assert f"group={GROUP}" in line
        assert line.index("group=") < line.index("subscription=")
    finally:
        handlers.release.set()
        await harness.close()
