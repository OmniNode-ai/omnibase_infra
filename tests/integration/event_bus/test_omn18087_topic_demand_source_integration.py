# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18087: the group-less demand reader against a real broker.

Splits a topic holding a record from an empty one, then publishes to the empty
one and expects the watch to report it.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable, Coroutine

import pytest
from aiokafka import AIOKafkaProducer

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_kafka_topic_demand_source import (
    HandlerKafkaTopicDemandSource,
)

pytestmark = [pytest.mark.integration, pytest.mark.kafka, pytest.mark.asyncio]


async def test_split_and_first_record_against_real_broker(
    ensure_test_topic: Callable[[str, int], Coroutine[object, object, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAFKA_TOPIC_NAMESPACE", raising=False)
    bootstrap = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    suffix = uuid.uuid4().hex[:12]
    busy = await ensure_test_topic(f"test.omn18087.busy.{suffix}", 1)
    idle = await ensure_test_topic(f"test.omn18087.idle.{suffix}", 1)

    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()
    try:
        await producer.send_and_wait(busy, b"already-there")

        started: list[str] = []

        async def on_first(topic: str) -> None:
            started.append(topic)

        source = HandlerKafkaTopicDemandSource(
            ModelKafkaEventBusConfig(bootstrap_servers=bootstrap)
        )
        in_use, stop = await source.watch_idle_topics({busy, idle}, on_first)
        try:
            assert in_use == {busy}
            await producer.send_and_wait(idle, b"first-request")
            async with asyncio.timeout(30):
                while not started:
                    await asyncio.sleep(0.1)
            assert started == [idle]
        finally:
            await stop()
    finally:
        await producer.stop()
