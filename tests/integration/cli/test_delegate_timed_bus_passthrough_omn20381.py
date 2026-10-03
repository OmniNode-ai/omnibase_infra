# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The timing bus is call-compatible with the real in-memory bus (OMN-20381)."""

from __future__ import annotations

import pytest

from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_core.models.event_bus.model_event_message import ModelEventMessage
from omnibase_infra.cli.receipt_mode import (
    DelegatePhaseStopwatch,
    DelegatePhaseTimedBus,
)
from omnibase_infra.enums.enum_delegate_phase import EnumDelegatePhase

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_runtime_style_subscribe_and_inline_reentrant_publish() -> None:
    stopwatch = DelegatePhaseStopwatch()
    inner = EventBusInmemory()
    bus = DelegatePhaseTimedBus(inner, stopwatch)
    await bus.start()
    received: list[bytes] = []

    async def on_output(message: ModelEventMessage) -> None:
        received.append(message.value)

    async def on_command(_: object) -> None:
        # Delivered inline while the client PUBLISH span is open.
        await bus.publish("output", None, b"reply")

    await bus.subscribe("output", None, on_output, group_id="client")
    await bus.subscribe("command", None, on_command, group_id="handler")

    await bus.publish("command", None, b"go")

    assert received == [b"reply"]
    assert not stopwatch.is_running(EnumDelegatePhase.PUBLISH)
    assert stopwatch.durations().publish_seconds is not None
    await bus.close()
