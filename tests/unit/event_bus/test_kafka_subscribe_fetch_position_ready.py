# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Subscription readiness includes the offset reset that can skip new records."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition

from omnibase_infra.errors import InfraTimeoutError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


@pytest.mark.asyncio
async def test_subscribe_waits_for_all_initial_fetch_positions() -> None:
    """A caller may publish immediately after subscribe without a late reset."""
    bus = EventBusKafka(config=ModelKafkaEventBusConfig())
    bus._started = True
    topic = "test-fetch-position-ready"
    partitions = {TopicPartition(topic, 0), TopicPartition(topic, 1)}
    consumer = MagicMock(spec=AIOKafkaConsumer)
    consumer.assignment.return_value = partitions
    reset_started = asyncio.Event()
    reset_allowed = asyncio.Event()
    resolved: set[TopicPartition] = set()

    async def position(partition: TopicPartition) -> int:
        reset_started.set()
        await reset_allowed.wait()
        resolved.add(partition)
        return 0

    consumer.position.side_effect = position
    # Keep the background poller idle while the test checks the startup boundary.
    poll_started = asyncio.Event()

    async def consume(*args: object) -> None:
        poll_started.set()
        await asyncio.Event().wait()

    with (
        patch.object(bus, "_build_consumer", return_value=consumer),
        patch.object(bus, "_consume_loop", side_effect=consume),
    ):
        subscription = asyncio.create_task(
            bus.subscribe(topic, on_message=AsyncMock(), group_id="readiness")
        )
        try:
            await asyncio.wait_for(reset_started.wait(), timeout=1)
            assert not subscription.done(), (
                "subscribe returned while the broker could still reset to latest"
            )
            assert not poll_started.is_set()
            reset_allowed.set()
            unsubscribe = await asyncio.wait_for(subscription, timeout=1)
            assert resolved == partitions
            await unsubscribe()
        finally:
            reset_allowed.set()
            if not subscription.done():
                subscription.cancel()
                await asyncio.gather(subscription, return_exceptions=True)
            await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_unresolved_position_cleans_up_startup(cancel: bool) -> None:
    """Timeout or cancellation cannot leave a consumer reserved or running."""
    bus = EventBusKafka(config=ModelKafkaEventBusConfig())
    bus._started = True
    bus._timeout_seconds = 1
    topic = "test-fetch-position-cleanup"
    consumer = MagicMock(spec=AIOKafkaConsumer)
    consumer.assignment.return_value = {TopicPartition(topic, 0)}
    reset_started = asyncio.Event()

    async def position(partition: TopicPartition) -> int:
        reset_started.set()
        await asyncio.Event().wait()
        return 0

    consumer.position.side_effect = position
    with patch.object(bus, "_build_consumer", return_value=consumer):
        subscription = asyncio.create_task(
            bus.subscribe(topic, on_message=AsyncMock(), group_id="readiness")
        )
        try:
            await asyncio.wait_for(reset_started.wait(), timeout=1)
            if cancel:
                subscription.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await subscription
            else:
                with pytest.raises(InfraTimeoutError):
                    await subscription
            consumer.stop.assert_awaited_once()
            assert not bus._pending_consumer_keys
            assert not bus._group_consumers
            assert not bus._group_consumer_tasks
        finally:
            if not subscription.done():
                subscription.cancel()
                await asyncio.gather(subscription, return_exceptions=True)
            await bus.close()
