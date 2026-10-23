# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consume-loop recovery, delivery and readiness after failed rejoins.

The bus, supervisor, rebuild path and subscriber dispatch run together. Only
Kafka transport and elapsed time are simulated; no shared broker is contacted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from itertools import pairwise
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import ConsumerStoppedError, KafkaConnectionError
from aiokafka.structs import ConsumerRecord, TopicPartition

from omnibase_infra.event_bus import consumer_rejoin_supervisor
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

TOPIC = (
    "onex.evt.omniclaude.hook-event.v1"  # onex-topic-allow: incident recovery replay
)
GROUP = "local.consumer-rejoin-recovery.consume.1.0.0"
PARTITION = TopicPartition(TOPIC, 0)
POLL_TIMEOUT_MS = 100


class _AsyncioWithVirtualSleep:
    """The real asyncio module, except that ``sleep`` advances a virtual clock."""

    def __init__(self, sleep: Callable[[float], Awaitable[None]]) -> None:
        self.sleep = sleep

    def __getattr__(self, name: str) -> object:
        return getattr(asyncio, name)


@pytest.mark.parametrize("poll_raises", [False, True], ids=["wedged", "poll-failed"])
async def test_consume_loop_delivers_after_two_failed_rejoins(
    monkeypatch: pytest.MonkeyPatch, poll_raises: bool
) -> None:
    """A failed replacement must not end the loop or clear failed readiness."""
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="test",
            consumer_stall_seconds=1.0,
            consumer_stall_required_confirmations=1,
            consumer_rejoin_cooldown_seconds=300.0,
            consumer_sync_unready_seconds=1.0,
            consumer_poll_timeout_ms=POLL_TIMEOUT_MS,
        )
    )
    now = [1_000.0]
    stopped = False
    stopped_polls = 0
    attempts: list[float] = []
    lost_readiness: list[bool] = []
    received: list[bytes] = []

    old = MagicMock(spec=AIOKafkaConsumer)
    old.assignment.return_value = {PARTITION}
    old.position = AsyncMock(return_value=0)
    old.end_offsets = AsyncMock(return_value={PARTITION: 1})

    async def poll_old(**kwargs: object) -> dict[TopicPartition, list[object]]:
        nonlocal stopped_polls
        now[0] += 2.0
        if stopped:
            stopped_polls += 1
            raise ConsumerStoppedError
        if poll_raises:
            raise KafkaConnectionError("broker connection lost during poll")
        return {}

    async def stop_old() -> None:
        nonlocal stopped
        stopped = True

    old.getmany = AsyncMock(side_effect=poll_old)
    old.stop = AsyncMock(side_effect=stop_old)
    bus._group_consumers[(TOPIC, GROUP)] = old
    supervisor = bus._rejoin_supervisor_for(TOPIC, GROUP)
    now[0] = supervisor._clock()
    monkeypatch.setattr(supervisor, "_clock", lambda: now[0])

    async def retry_sleep(seconds: float) -> None:
        now[0] += seconds
        if len(attempts) >= 2:
            lost_readiness.append(bus.consumer_sync_statuses()[0].ready)
        await asyncio.sleep(0)

    monkeypatch.setattr(
        consumer_rejoin_supervisor, "asyncio", _AsyncioWithVirtualSleep(retry_sleep)
    )

    async def start_replacement() -> None:
        attempts.append(now[0])
        if len(attempts) <= 2:
            raise KafkaConnectionError("broker still restarting")

    replacements = [MagicMock(spec=AIOKafkaConsumer) for _ in range(3)]
    for candidate in replacements:
        candidate.start = AsyncMock(side_effect=start_replacement)
    recovered = replacements[-1]
    record = ConsumerRecord(
        topic=TOPIC,
        partition=0,
        offset=0,
        timestamp=0,
        timestamp_type=0,
        key=None,
        value=b'{"event": "recovered"}',
        checksum=None,
        serialized_key_size=-1,
        serialized_value_size=22,
        headers=[],
    )
    recovered.getmany = AsyncMock(return_value={PARTITION: [record]})
    build = MagicMock(side_effect=replacements)
    monkeypatch.setattr(bus, "_build_consumer", build)

    async def subscriber(message: ModelEventMessage) -> None:
        received.append(message.value)
        bus._shutdown = True

    bus._subscribers[TOPIC] = [(GROUP, "recovery-subscriber", subscriber)]
    await asyncio.wait_for(bus._consume_loop(TOPIC, GROUP, uuid4()), timeout=5.0)

    assert received == [record.value], "the consume task ended before delivery"
    assert stopped_polls == 0, "the loop polled the stopped consumer"
    old.stop.assert_awaited_once()
    assert len(attempts) == 3
    # While lost, each next_batch sleeps one poll interval, so a retry lands
    # within one interval after the retry period has elapsed.
    retry = consumer_rejoin_supervisor.LOST_CONSUMER_RETRY_SECONDS
    assert all(
        retry <= later - earlier <= retry + POLL_TIMEOUT_MS / 1000
        for earlier, later in pairwise(attempts)
    )
    assert lost_readiness and not any(lost_readiness)
    assert bus._group_consumers[(TOPIC, GROUP)] is recovered
    assert [event.rejoin_succeeded for event in bus.consumer_rejoin_events()] == [
        False,
        False,
        True,
    ]
    assert bus.consumer_sync_statuses()[0].ready
    assert len({call.args[1] for call in build.call_args_list}) == 1
