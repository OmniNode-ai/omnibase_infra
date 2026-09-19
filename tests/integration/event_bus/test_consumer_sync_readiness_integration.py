# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A wedged subscription turns the runtime's container unhealthy (OMN-18640 AC1).

The unit tests cover each link on its own: the supervisor's measurement, the
model's derived verdict, the monitor's dimension, the healthcheck's exit code.
This one drives them as one chain, over the surface a runtime actually uses --
``EventBusKafka.subscribe`` -- because every link being correct in isolation is
precisely the state the .201 dev lane was in at 2026-09-19T04:51:48Z. The
consumer measured nothing, the monitor published nothing about it, and the
container stayed green for fifty minutes.

What this adds over the unit coverage is the wiring: the real consumer
construction path, the real per-group supervisor registry, the real
health-monitor cycle, and the real ``/health`` body shape the container probe
parses. A dimension that is computed and then folded into nothing looks
identical to one that was never added, from outside.

**The recovery is made to fail here, on purpose.** ``omnibase_infra#3807``
makes the consumer rebuild itself and will handle most wedges before readiness
ever notices. This test is about the remainder: what the runtime reports when
the self-heal has run and the group is *still* not consuming. Every replacement
consumer below is wedged too.

The Kafka client is substituted at the ``AIOKafkaConsumer`` seam rather than
run against a live broker, for the same reason the sibling integration test
gives: the fault is a CLIENT state -- a coordinator marked dead while every
leader answers -- that no broker-side action produces on demand. Provoking it
for real means restarting a shared lane broker, which this ticket forbids.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime.health.container_healthcheck import (
    evaluate_health_response,
)
from omnibase_infra.runtime.health.runtime_health_block import (
    build_runtime_health_block,
    fold_runtime_verdict_into_status,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
)
from tests.conftest import make_test_node_identity

pytestmark = pytest.mark.integration

TOPIC = "onex.cmd.omnimarket.occ-autobind.v1"  # onex-topic-allow: replay of a recorded incident
PARTITION = TopicPartition(TOPIC, 0)

# The .201 dev lane at 2026-09-19T00:07:29Z: the fetch position had not moved
# since 23:52Z while the topic end kept advancing.
WEDGED_POSITION = 7473
LOG_END_OFFSET = 7489
BACKLOG = LOG_END_OFFSET - WEDGED_POSITION

# Compressed so the test asserts the mechanism rather than the clock. The
# shipped defaults, and the inequality between them that matters, are asserted
# in tests/unit/event_bus/test_consumer_sync_readiness_omn18640.py.
FAST_STALL_SECONDS = 0.3
FAST_UNREADY_SECONDS = 1.0


class _WedgedForeverConsumer:
    """An ``AIOKafkaConsumer`` stand-in that no rejoin can fix.

    Holds its assignment, answers an end-offset probe from the leaders with a
    log end well ahead of its own position, and returns nothing from every
    fetch -- and so does every replacement built for it.
    """

    instances: list[_WedgedForeverConsumer] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.group_id = kwargs.get("group_id", "")
        self.started = False
        self.stopped = False
        _WedgedForeverConsumer.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> Mapping[TopicPartition, Sequence[Any]]:
        # Return at the deadline having fetched nothing, forever -- and honour
        # the deadline, so the loop is paced as a real client paces it.
        await asyncio.sleep(max(timeout_ms, 1) / 1000.0)
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


class _CaughtUpConsumer(_WedgedForeverConsumer):
    """Quiet because there is nothing to consume. The positive control."""

    async def position(self, partition: TopicPartition) -> int:
        return LOG_END_OFFSET


@pytest.fixture(autouse=True)
def _reset_instances() -> None:
    _WedgedForeverConsumer.instances = []


def _bus() -> EventBusKafka:
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:19092",
            environment="test",
            consumer_poll_timeout_ms=100,
            consumer_stall_seconds=FAST_STALL_SECONDS,
            consumer_stall_required_confirmations=2,
            consumer_rejoin_cooldown_seconds=0.0,
            consumer_sync_unready_seconds=FAST_UNREADY_SECONDS,
        )
    )
    bus._started = True
    return bus


def _manifest() -> MagicMock:
    manifest = MagicMock()
    manifest.total_discovered = 0
    manifest.total_errors = 0
    manifest.errors = ()
    manifest.all_subscribe_topics.return_value = ()
    return manifest


async def _container_probe(
    bus: EventBusKafka, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, object]:
    """Run one real monitor cycle over ``bus`` and probe it as the container does.

    Returns the ``consumer_sync`` dimension's rendered detail and the verdict
    the container healthcheck reaches from the resulting ``/health`` body.
    """
    monkeypatch.setattr(
        "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
        _manifest,
    )
    monkeypatch.setattr(
        "omnibase_infra.services.service_runtime_health_monitor._filter_manifest_for_runtime_profile",
        lambda manifest: manifest,
    )
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=bus,
        bootstrap_servers="",
        check_interval_seconds=300.0,
        boot_grace_seconds=0.0,
    )
    event = await monitor.run_once()

    dimension = next(d for d in event.dimensions if d.name == "consumer_sync")
    payload = {
        "status": fold_runtime_verdict_into_status("healthy", event.status),
        "details": {"is_running": True, **build_runtime_health_block(event)},
    }
    # Round-tripped through JSON so this cannot pass on a shape the HTTP
    # surface would not actually serve.
    verdict = evaluate_health_response(
        http_status=200, payload=json.loads(json.dumps(payload))
    )
    return dimension.status, verdict


@pytest.mark.asyncio
async def test_a_wedge_the_rejoin_cannot_fix_turns_the_container_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """subscribe -> wedge -> failed self-heal -> readiness -> non-zero exit.

    The last step is the deliverable. `docker ps` reporting unhealthy is the
    single fact the deploy agent's AC7 force-recreate probes, and the only one
    an operator sees without reading JSON.
    """
    monkeypatch.setattr(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
        _WedgedForeverConsumer,
    )
    bus = _bus()

    async def handler(_message: ModelEventMessage) -> None:
        return None

    unsubscribe = await bus.subscribe(
        TOPIC, make_test_node_identity("omn18640-ac1"), handler
    )
    try:
        # Well past the readiness window, and past several rejoin attempts
        # that each land on another wedged consumer.
        await asyncio.sleep(3.0)

        statuses = bus.consumer_sync_statuses()
        assert len(statuses) == 1, (
            "one supervisor per (topic, group) -- the readiness surface must "
            f"see exactly the one subscription, saw {len(statuses)}"
        )
        status = statuses[0]
        assert not status.ready, (
            "the group has been behind the leaders and not advancing for "
            "longer than the declared window, through a self-heal that did "
            "not fix it -- this is the state that read green for 50 minutes"
        )
        assert status.backlog_records == BACKLOG
        assert status.stalled_seconds >= FAST_UNREADY_SECONDS
        assert status.topic == TOPIC

        assert len(_WedgedForeverConsumer.instances) > 1, (
            "the self-heal must have run; this test is about what readiness "
            "reports AFTER it has run and failed"
        )

        dimension_status, verdict = await _container_probe(bus, monkeypatch)
        assert dimension_status == "CRITICAL"
        assert verdict.exit_code != 0, (
            "the container healthcheck exited zero over a wedged consumer "
            "group -- `docker ps` stays green and AC7 never fires"
        )
    finally:
        await unsubscribe()
        bus._shutdown = True


@pytest.mark.asyncio
async def test_an_idle_subscription_keeps_the_container_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control on the same wiring, and the one that matters most.

    An idle topic is the common case. If this goes red the dimension is a
    pager that fires on a quiet Sunday, it gets turned off, and the surface
    that should have caught the outage stops existing -- which is a worse
    outcome than the outage.
    """
    monkeypatch.setattr(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
        _CaughtUpConsumer,
    )
    bus = _bus()

    async def handler(_message: ModelEventMessage) -> None:
        return None

    unsubscribe = await bus.subscribe(
        TOPIC, make_test_node_identity("omn18640-ac1-control"), handler
    )
    try:
        # Many multiples of both windows, all of them quiet.
        await asyncio.sleep(3.0)

        statuses = bus.consumer_sync_statuses()
        assert len(statuses) == 1
        assert statuses[0].ready
        assert statuses[0].backlog_records == 0
        assert statuses[0].stalled_seconds == 0.0

        dimension_status, verdict = await _container_probe(bus, monkeypatch)
        assert dimension_status == "HEALTHY"
        assert verdict.exit_code == 0, verdict.detail

        assert len(_WedgedForeverConsumer.instances) == 1, (
            "a caught-up consumer must never be recreated, however long it stays quiet"
        )
    finally:
        await unsubscribe()
        bus._shutdown = True
