# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage: the loop reading reaches a subscriber of the contracted topic (OMN-26249).

Starts the real publisher against a real ``EventBusInmemory`` and subscribes to
the topic the on-disk node contract declares. Each reading a subscriber receives
must validate as ``ModelRuntimeLoopReading``, and a block on the loop must read
as a gap, so a stalled member is visible as a number on the bus.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.handlers import handler_runtime_loop_lag_sampler as handler_module
from omnibase_infra.handlers.handler_runtime_loop_lag_sampler import (
    RuntimeLoopLagSampler,
    RuntimeLoopReadingPublisher,
)
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.health.model_runtime_loop_reading import (
    ModelRuntimeLoopReading,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_runtime_loop_reading_effect/contract.yaml"
)
_IDENTITY = ModelNodeIdentity(
    env="dev", service="omnibase-infra", node_name="runtime-main", version="v1"
)


def _contracted_topic() -> str:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    topics = contract["event_bus"]["publish_topics"]
    assert len(topics) == 1
    return str(topics[0])


async def test_subscriber_of_contracted_topic_receives_a_reading_with_the_block_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(handler_module, "_PROCESS_SAMPLER", RuntimeLoopLagSampler())
    topic = _contracted_topic()
    bus = EventBusInmemory(environment="dev", group="loop-reading-it")
    await bus.start()
    received: list[ModelEventMessage] = []

    async def on_message(message: ModelEventMessage) -> None:
        received.append(message)

    await bus.subscribe(topic, _IDENTITY, on_message)
    publisher = RuntimeLoopReadingPublisher()
    await publisher.start(bus, _IDENTITY, MagicMock(), window_seconds=3600.0)
    try:
        await asyncio.sleep(0.05)
        time.sleep(0.6)  # handler-shaped block on the loop
        await publisher.publish_reading(bus, _IDENTITY)
    finally:
        await publisher.stop()
        await bus.close()

    assert len(received) == 1
    reading = ModelRuntimeLoopReading.model_validate(json.loads(received[0].value))
    assert reading.max_loop_gap_ms >= 480.0
    assert reading.runtime_member.startswith("dev.omnibase-infra.runtime-main.")
