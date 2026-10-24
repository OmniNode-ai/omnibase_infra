# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: replay the cold stall from nightly run 40052484369."""

from __future__ import annotations

import time

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelPublishReceipt
from tests.performance.event_bus.conftest import generate_unique_topic
from tests.performance.event_bus.test_event_bus_load import (
    SUSTAINED_VARIANCE_BOUND,
    WARMUP_KEY_PREFIX,
    SustainedLoadResult,
    max_throughput_variance,
    measure_sustained_load,
)

pytestmark = [pytest.mark.performance]

INTERVAL_SECONDS = 1.0
NUM_INTERVALS = 5
# Longer than one interval, as in the nightly run, so the stalled interval
# completes a single publish.
COLD_STALL_SECONDS = 1.2


async def _measure_with_cold_stall(
    monkeypatch: pytest.MonkeyPatch,
    value: bytes,
    *,
    warmup: bool,
) -> tuple[SustainedLoadResult, list[bytes]]:
    """Block the first publish, measure, and return the stalled publish key."""
    original_publish = EventBusInmemory.publish
    stalled_keys: list[bytes] = []

    async def publish_with_cold_stall(
        self: EventBusInmemory,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None = None,
    ) -> ModelPublishReceipt:
        if not stalled_keys:
            stalled_keys.append(key or b"")
            time.sleep(COLD_STALL_SECONDS)
        return await original_publish(self, topic, key, value, headers)

    monkeypatch.setattr(EventBusInmemory, "publish", publish_with_cold_stall)

    bus = EventBusInmemory(environment="sustained", group="load", max_history=10000)
    await bus.start()
    try:
        result = await measure_sustained_load(
            bus,
            generate_unique_topic(),
            value,
            interval_duration=INTERVAL_SECONDS,
            num_intervals=NUM_INTERVALS,
            warmup=warmup,
        )
    finally:
        await bus.close()
    return result, stalled_keys


@pytest.mark.asyncio
async def test_first_publish_stall_lands_in_warmup(
    monkeypatch: pytest.MonkeyPatch,
    sample_message_bytes: bytes,
) -> None:
    """A one-off blocking publish stall is absorbed by the discarded warm-up."""
    result, stalled_keys = await _measure_with_cold_stall(
        monkeypatch, sample_message_bytes, warmup=True
    )

    assert len(stalled_keys) == 1
    assert stalled_keys[0].startswith(f"{WARMUP_KEY_PREFIX}-".encode())
    # The stall outlasts the warm-up interval, so the warm-up completes only
    # the stalled publish.
    assert result.warmup_count == 1
    assert len(result.interval_counts) == NUM_INTERVALS
    assert min(result.interval_counts) > 1
    variance = max_throughput_variance(list(result.interval_counts))
    assert variance < SUSTAINED_VARIANCE_BOUND, (
        f"interval counts {result.interval_counts}, variance {variance:.1%}"
    )


@pytest.mark.asyncio
async def test_first_publish_stall_without_warmup_breaks_variance(
    monkeypatch: pytest.MonkeyPatch,
    sample_message_bytes: bytes,
) -> None:
    """Negative control: without the warm-up the stall lands in interval 1."""
    result, stalled_keys = await _measure_with_cold_stall(
        monkeypatch, sample_message_bytes, warmup=False
    )

    assert len(stalled_keys) == 1
    assert stalled_keys[0].startswith(b"sustained-0-")
    assert result.warmup_count == 0
    assert result.interval_counts[0] == 1
    variance = max_throughput_variance(list(result.interval_counts))
    assert variance >= SUSTAINED_VARIANCE_BOUND, (
        f"interval counts {result.interval_counts}, variance {variance:.1%}"
    )
