# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: replay the cold stall from nightly run 40052484369."""

from __future__ import annotations

import time

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelPublishReceipt
from tests.performance.event_bus import test_event_bus_load

pytestmark = [pytest.mark.performance]

COLD_STALL_SECONDS = 1.2


@pytest.mark.asyncio
async def test_first_publish_stall_lands_in_warmup(
    monkeypatch: pytest.MonkeyPatch,
    sample_message_bytes: bytes,
) -> None:
    """Keep a one-off blocking publish stall out of measured throughput."""
    original_publish = EventBusInmemory.publish
    stall_injected = False

    async def publish_with_cold_stall(
        self: EventBusInmemory,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None = None,
    ) -> ModelPublishReceipt:
        nonlocal stall_injected
        if not stall_injected:
            stall_injected = True
            time.sleep(COLD_STALL_SECONDS)
        return await original_publish(self, topic, key, value, headers)

    monkeypatch.setattr(EventBusInmemory, "publish", publish_with_cold_stall)

    # Before the warm-up, the stall lands in measured interval 1 and the
    # variance assertion fails; with the warm-up it lands in the discarded
    # interval.
    await test_event_bus_load.TestSustainedLoad().test_sustained_5_second_load(
        sample_message_bytes
    )
    assert stall_injected
