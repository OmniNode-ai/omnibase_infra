# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request/reply over the real in-memory bus with group-less readers (OMN-17427).

Each caller reads its own reply through ``subscribe_ungrouped``, many callers
in a row leave no subscription behind, and a reply published before the reader
exists is never seen.
"""

from __future__ import annotations

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage

REPLY_TOPIC = "onex.evt.test.ungrouped-reply-integration.v1"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_many_one_shot_reply_reads_leave_no_subscription_behind() -> None:
    bus = EventBusInmemory(environment="test", group="g")
    await bus.start()
    await bus.publish(REPLY_TOPIC, None, b"stale-reply")
    for index in range(25):
        seen: list[bytes] = []

        async def on_message(
            message: ModelEventMessage, sink: list[bytes] = seen
        ) -> None:
            sink.append(message.value)

        unsubscribe = await bus.subscribe_ungrouped(REPLY_TOPIC, on_message)
        payload = f"reply-{index}".encode()
        await bus.publish(REPLY_TOPIC, None, payload)
        assert seen == [payload]
        await unsubscribe()
    assert not bus._subscribers.get(REPLY_TOPIC)
    await bus.close()
