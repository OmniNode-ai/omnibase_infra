# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``EventBusKafka``'s consume loop recovers a wedged group by itself (OMN-18640).

The supervisor is unit-tested in
``test_consumer_rejoin_on_coordinator_loss.py``. This file asserts the thing
that actually failed on 2026-09-17 and 2026-09-18: that the loop the effects
runtime runs is WIRED to it. Before this change ``_consume_loop`` iterated
``async for msg in consumer``, which never returns and never raises while
aiokafka retries a dead coordinator in the background, so the loop could not
have noticed a wedge however good the detector was.

RED before the fix: ``_consume_loop`` hangs on the wedged consumer and the
``asyncio.wait_for`` below times out with nothing recreated.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping, Sequence
from typing import Any
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)

pytestmark = pytest.mark.unit

TOPIC = "onex.cmd.omnimarket.occ-autobind.v1"  # onex-topic-allow: replay of a recorded incident
GROUP = "local.omnimarket.pr_lifecycle_fix_effect.consume.1.0.0"
PARTITION = TopicPartition(TOPIC, 0)
WEDGED_POSITION = 7473
LOG_END_OFFSET = 7489


class _WedgedConsumer:
    """Holds its assignment, answers the leaders, and never returns a record."""

    def __init__(self) -> None:
        self.stopped = False

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> Mapping[TopicPartition, Sequence[Any]]:
        # A real wedged aiokafka consumer returns from getmany at the deadline
        # having fetched nothing. Sleeping zero keeps the test fast while
        # preserving the yield point.
        await asyncio.sleep(0)
        return {}

    def assignment(self) -> set[TopicPartition]:
        return {PARTITION}

    async def position(self, partition: TopicPartition) -> int:
        return WEDGED_POSITION

    async def end_offsets(
        self, partitions: Collection[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        return {PARTITION: LOG_END_OFFSET}

    def seek(self, partition: TopicPartition, offset: int) -> None:
        return None

    async def stop(self) -> None:
        self.stopped = True

    def __aiter__(self) -> _WedgedConsumer:
        return self

    async def __anext__(self) -> Any:
        # The pre-fix loop's behaviour: wait forever, raise nothing.
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _bus() -> EventBusKafka:
    config = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092",
        environment="test",
        consumer_stall_seconds=1.0,
        consumer_stall_required_confirmations=1,
        consumer_rejoin_cooldown_seconds=0.0,
        consumer_poll_timeout_ms=100,
    )
    return EventBusKafka(config=config)


@pytest.mark.asyncio
async def test_consume_loop_rejoins_a_wedged_consumer() -> None:
    """The loop must close and rebuild a consumer that holds its group and stops fetching."""
    bus = _bus()
    wedged = _WedgedConsumer()
    replacement = _WedgedConsumer()
    recreated: list[_WedgedConsumer] = []

    bus._group_consumers[(TOPIC, GROUP)] = wedged  # type: ignore[assignment]

    async def fake_rebuild(topic: str, group_id: str) -> Any:
        recreated.append(replacement)
        bus._group_consumers[(topic, group_id)] = replacement  # type: ignore[assignment]
        return replacement

    bus._rebuild_consumer_for_rejoin = fake_rebuild  # type: ignore[assignment, method-assign]

    task = asyncio.create_task(bus._consume_loop(TOPIC, GROUP, uuid4()))
    try:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if recreated:
                break
        assert recreated, (
            "the consume loop never recreated the wedged consumer -- it is "
            "still waiting on an iterator that will not return (OMN-18640)"
        )
        assert wedged.stopped, "the wedged consumer must be closed"
        assert bus._group_consumers[(TOPIC, GROUP)] is replacement, (
            "the bus must publish the replacement handle so shutdown and "
            "health surfaces address the live consumer, not the dead one"
        )
    finally:
        bus._shutdown = True
        task.cancel()
        with pytest.raises((asyncio.CancelledError, TimeoutError)):
            await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_rejoin_records_are_exposed_for_a_readiness_probe() -> None:
    """The typed rejoin record must be readable in-process, with no feature flag."""
    bus = _bus()
    wedged = _WedgedConsumer()
    bus._group_consumers[(TOPIC, GROUP)] = wedged  # type: ignore[assignment]

    async def fake_rebuild(topic: str, group_id: str) -> Any:
        replacement = _WedgedConsumer()
        bus._group_consumers[(topic, group_id)] = replacement  # type: ignore[assignment]
        return replacement

    bus._rebuild_consumer_for_rejoin = fake_rebuild  # type: ignore[assignment, method-assign]

    task = asyncio.create_task(bus._consume_loop(TOPIC, GROUP, uuid4()))
    try:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if bus.consumer_rejoin_events():
                break
        events = bus.consumer_rejoin_events()
        assert events, "no typed rejoin event was recorded (OMN-18640)"
        assert events[0].topic == TOPIC
        assert events[0].rejoin_succeeded is True
    finally:
        bus._shutdown = True
        task.cancel()
        with pytest.raises((asyncio.CancelledError, TimeoutError)):
            await asyncio.wait_for(task, timeout=2.0)
