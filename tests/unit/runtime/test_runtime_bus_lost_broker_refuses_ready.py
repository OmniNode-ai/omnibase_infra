# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A shared runtime bus refuses readiness while Kafka is lost and never falls back.

``refuse_ready`` is the shipped policy for a runtime that serves other
processes: no in-memory delivery, publishes refused, subscription intent kept
and Kafka retried. A runtime declared single-process (``local_fallback``)
keeps delivering in memory instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.enums import EnumEventBusLostBrokerPolicy
from omnibase_infra.errors import InfraConnectionError, InfraUnavailableError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.event_bus_recovering import EventBusRecovering
from omnibase_infra.models.model_event_bus_recovery_config import (
    ModelEventBusRecoveryConfig,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.services.health_checker import ServiceHealth
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.unit

RECOVERY_LOGGER = "omnibase_infra.handlers.event_bus_recovering"


def make_bus(
    policy: EnumEventBusLostBrokerPolicy = EnumEventBusLostBrokerPolicy.REFUSE_READY,
) -> tuple[EventBusRecovering, EventBusInmemory, AsyncMock, AsyncMock]:
    local_bus = EventBusInmemory(max_history=10)
    local_publish = AsyncMock(wraps=local_bus.publish)
    local_bus.publish = local_publish
    local_subscribe = AsyncMock(wraps=local_bus.subscribe)
    local_bus.subscribe = local_subscribe
    bus = EventBusRecovering(
        config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092"),
        recovery=ModelEventBusRecoveryConfig(
            initial_backoff_seconds=1,
            max_backoff_seconds=4,
            backoff_multiplier=2,
            attempt_timeout_seconds=1,
            # refuse_ready is the shipped default: a runtime serving other
            # processes must not route their traffic into a private bus.
            lost_broker_policy=policy,
        ),
        local_bus=local_bus,
        local_start=local_bus.start,
        local_close=local_bus.close,
        local_health_check=local_bus.health_check,
        alarm_topic="runtime-bus",
        consumer_group="test-runtime",
    )
    return bus, local_bus, local_publish, local_subscribe


def recovery_entries(caplog: pytest.LogCaptureFixture) -> list[tuple[int, str]]:
    """Interpolated messages from the bus logger as (level, message) pairs."""
    return [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == RECOVERY_LOGGER
    ]


async def test_broker_loss_in_a_shared_runtime_is_unready_and_retries_never_in_memory(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus, _local_bus, local_publish, local_subscribe = make_bus()
    starts = 0
    delays: asyncio.Queue[float] = asyncio.Queue()
    permits: asyncio.Queue[None] = asyncio.Queue()

    async def start_kafka(self: EventBusKafka) -> None:
        nonlocal starts
        starts += 1
        # One start per connect: #1 is the initial bus.start(), #2 and #3 fail
        # attempts 1 and 2, #4 is attempt 3.
        if starts <= 3:
            raise InfraConnectionError("broker unavailable")
        self._started = True
        self._shutdown = False
        self._producer = MagicMock(_closed=False, stop=AsyncMock())

    async def wait(delay: float) -> None:
        await delays.put(delay)
        await permits.get()

    monkeypatch.setattr(EventBusKafka, "start", start_kafka)
    monkeypatch.setattr(bus, "_sleep_before_retry", wait)
    caplog.set_level(logging.INFO, logger=RECOVERY_LOGGER)
    try:
        await bus.start()
        assert starts == 1  # The initial connect failed; nothing stands in.
        assert not (await bus.get_readiness_status()).is_ready
        health = await bus.health_check()
        assert health["degraded"] is True
        assert health["broker_lost"] is True
        with pytest.raises(InfraUnavailableError):
            await bus.publish("events", None, b"x")
        assert local_publish.await_count == 0
        assert local_subscribe.await_count == 0
        criticals = [
            message
            for level, message in recovery_entries(caplog)
            if level == logging.CRITICAL and "refusing readiness" in message
        ]
        assert len(criticals) == 1
        # Attempt 1 (start #2 fails), then attempt 2 (start #3 fails).
        assert await asyncio.wait_for(delays.get(), 1) == 1.0
        await permits.put(None)
        assert await asyncio.wait_for(delays.get(), 1) == 2.0
        assert not (await bus.get_readiness_status()).is_ready
        assert local_publish.await_count == 0
        assert local_subscribe.await_count == 0
        await permits.put(None)
        assert await asyncio.wait_for(delays.get(), 1) == 4.0
        assert not (await bus.get_readiness_status()).is_ready
        assert local_publish.await_count == 0
        assert local_subscribe.await_count == 0
        # Attempt 3 (start #4) reconnects Kafka.
        await permits.put(None)
        async with asyncio.timeout(1):
            while not (await bus.get_readiness_status()).is_ready:
                await asyncio.sleep(0)
        assert starts == 4
        assert local_publish.await_count == 0
        assert local_subscribe.await_count == 0
        # Still exactly one critical alarm for the whole episode.
        assert (
            len(
                [
                    message
                    for level, message in recovery_entries(caplog)
                    if level == logging.CRITICAL and "refusing readiness" in message
                ]
            )
            == 1
        )
        # Kafka carries the traffic again; still never the local bus.
        kafka_publish = AsyncMock(spec=EventBusKafka.publish, return_value=MagicMock())
        monkeypatch.setattr(EventBusKafka, "publish", kafka_publish)
        await bus.publish("events", None, b"x")
        assert kafka_publish.await_count == 1
        assert local_publish.await_count == 0
    finally:
        await bus.close()


async def test_subscription_during_the_loss_is_kept_and_attached_when_kafka_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _local_bus, local_publish, local_subscribe = make_bus()
    starts = 0
    kafka_topics: list[str] = []
    delays: asyncio.Queue[float] = asyncio.Queue()
    permits: asyncio.Queue[None] = asyncio.Queue()

    async def start_kafka(self: EventBusKafka) -> None:
        nonlocal starts
        starts += 1
        if starts == 1:
            raise InfraConnectionError("broker unavailable")
        self._started = True
        self._shutdown = False
        self._producer = MagicMock(_closed=False, stop=AsyncMock())

    async def subscribe_kafka(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        kafka_topics.append(str(kwargs["topic"]))
        return AsyncMock()

    async def wait(delay: float) -> None:
        await delays.put(delay)
        await permits.get()

    monkeypatch.setattr(EventBusKafka, "start", start_kafka)
    monkeypatch.setattr(EventBusKafka, "subscribe", subscribe_kafka)
    monkeypatch.setattr(bus, "_sleep_before_retry", wait)
    try:
        await bus.start()
        await bus.subscribe("events", group_id="g", on_message=AsyncMock())
        assert kafka_topics == []  # Kept as intent only while Kafka is lost.
        assert local_subscribe.await_count == 0
        assert not (await bus.get_readiness_status()).is_ready
        # One permitted attempt reconnects and attaches the retained intent.
        assert await asyncio.wait_for(delays.get(), 1) == 1.0
        await permits.put(None)
        async with asyncio.timeout(1):
            while not (await bus.get_readiness_status()).is_ready:
                await asyncio.sleep(0)
        assert kafka_topics == ["events"]
        assert starts == 2
        assert local_publish.await_count == 0
        assert local_subscribe.await_count == 0
    finally:
        await bus.close()


async def test_one_failed_join_keeps_the_rest_attached_and_kafka_carries_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _local_bus, local_publish, local_subscribe = make_bus()
    starts = 0
    joins: dict[str, int] = {"a": 0, "b": 0}
    delays: asyncio.Queue[float] = asyncio.Queue()
    permits: asyncio.Queue[None] = asyncio.Queue()

    async def start_kafka(self: EventBusKafka) -> None:
        nonlocal starts
        starts += 1
        if starts == 1:
            raise InfraConnectionError("broker unavailable")
        self._started = True
        self._shutdown = False
        self._producer = MagicMock(_closed=False, stop=AsyncMock())

    async def attach(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        topic = str(kwargs["topic"])
        joins[topic] += 1
        if topic == "b" and joins["b"] == 1:
            raise InfraConnectionError("join refused for b")
        return AsyncMock()

    async def wait(delay: float) -> None:
        await delays.put(delay)
        await permits.get()

    monkeypatch.setattr(EventBusKafka, "start", start_kafka)
    monkeypatch.setattr(EventBusKafka, "subscribe", attach)
    monkeypatch.setattr(bus, "_sleep_before_retry", wait)
    try:
        await bus.start()
        for topic in ("a", "b"):
            await bus.subscribe(topic, group_id="g", on_message=AsyncMock())
        # Attempt 1: both joins run; b fails on its own and a is kept.
        assert await asyncio.wait_for(delays.get(), 1) == 1.0
        await permits.put(None)
        assert await asyncio.wait_for(delays.get(), 1) == 2.0
        assert joins == {"a": 1, "b": 1}
        assert not (await bus.get_readiness_status()).is_ready
        assert (await bus.health_check())["pending_attachments"] == 1
        # Kafka answers again, so a publish reaches it instead of raising.
        kafka_publish = AsyncMock(spec=EventBusKafka.publish, return_value=MagicMock())
        monkeypatch.setattr(EventBusKafka, "publish", kafka_publish)
        await bus.publish("events", None, b"x")
        assert kafka_publish.await_count == 1
        assert local_publish.await_count == 0
        # Attempt 2: only b retries; a's live consumer is not re-joined.
        await permits.put(None)
        async with asyncio.timeout(1):
            while not (await bus.get_readiness_status()).is_ready:
                await asyncio.sleep(0)
        assert joins == {"a": 1, "b": 2}
        assert local_publish.await_count == 0
        assert local_subscribe.await_count == 0
    finally:
        await bus.close()


async def test_publish_connect_error_in_a_shared_runtime_raises_and_marks_the_broker_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _local_bus, local_publish, local_subscribe = make_bus()
    delays: asyncio.Queue[float] = asyncio.Queue()
    permits: asyncio.Queue[None] = asyncio.Queue()

    async def start_kafka(self: EventBusKafka) -> None:
        self._started = True
        self._shutdown = False
        self._producer = MagicMock(_closed=False, stop=AsyncMock())

    async def wait(delay: float) -> None:
        await delays.put(delay)
        await permits.get()

    monkeypatch.setattr(EventBusKafka, "start", start_kafka)
    monkeypatch.setattr(
        EventBusKafka,
        "publish",
        AsyncMock(side_effect=InfraConnectionError("produce failed: broker lost")),
    )
    monkeypatch.setattr(bus, "_sleep_before_retry", wait)
    try:
        await bus.start()
        # The failed produce degrades the bus to broker-lost, but the caller
        # still sees its own connect error; nothing is delivered locally.
        with pytest.raises(InfraConnectionError):
            await bus.publish("events", None, b"x")
        assert not (await bus.get_readiness_status()).is_ready
        assert (await bus.health_check())["broker_lost"] is True
        assert bus._recovery_task is not None  # Recovery has been scheduled.
        assert local_publish.await_count == 0
        assert local_subscribe.await_count == 0
    finally:
        await bus.close()


async def test_a_declared_single_process_runtime_still_delivers_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, _local_bus, local_publish, local_subscribe = make_bus(
        EnumEventBusLostBrokerPolicy.LOCAL_FALLBACK
    )
    starts = 0
    delays: asyncio.Queue[float] = asyncio.Queue()
    permits: asyncio.Queue[None] = asyncio.Queue()

    async def start_kafka(self: EventBusKafka) -> None:
        nonlocal starts
        starts += 1
        if starts == 1:
            raise InfraConnectionError("broker unavailable")
        self._started = True
        self._shutdown = False
        self._producer = MagicMock(_closed=False, stop=AsyncMock())

    async def wait(delay: float) -> None:
        await delays.put(delay)
        await permits.get()

    monkeypatch.setattr(EventBusKafka, "start", start_kafka)
    monkeypatch.setattr(bus, "_sleep_before_retry", wait)
    received = AsyncMock()
    try:
        await bus.start()  # Kafka is lost; local delivery holds the transport.
        await bus.subscribe("events", group_id="g", on_message=received)
        await bus.publish("events", None, b"x")
        assert received.await_count == 1
        assert local_publish.await_count == 1
        assert local_subscribe.await_count == 1
    finally:
        await bus.close()


async def test_shipped_contract_refuses_and_runtime_overlay_can_declare_single_process() -> (
    None
):
    contract = ModelEventBusRecoveryConfig.from_contract()
    assert contract.lost_broker_policy is EnumEventBusLostBrokerPolicy.REFUSE_READY
    overlay = ModelEventBusRecoveryConfig.from_runtime_config(
        {
            "event_bus": {
                "recovery": {
                    "initial_backoff_seconds": 1,
                    "max_backoff_seconds": 4,
                    "backoff_multiplier": 2,
                    "attempt_timeout_seconds": 1,
                    "lost_broker_policy": "local_fallback",
                }
            }
        }
    )
    assert overlay.lost_broker_policy is EnumEventBusLostBrokerPolicy.LOCAL_FALLBACK


async def test_runtime_serves_unready_to_clients_while_the_broker_is_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seam: runtime bus to broker, and readiness to clients (/ready)."""
    bus, _local_bus, local_publish, local_subscribe = make_bus()
    starts = 0
    delays: asyncio.Queue[float] = asyncio.Queue()
    permits: asyncio.Queue[None] = asyncio.Queue()

    async def start_kafka(self: EventBusKafka) -> None:
        nonlocal starts
        starts += 1
        if starts < 3:
            raise InfraConnectionError("broker unavailable")
        self._started = True
        self._shutdown = False
        self._producer = MagicMock(_closed=False, stop=AsyncMock())

    async def wait(delay: float) -> None:
        await delays.put(delay)
        await permits.get()

    monkeypatch.setattr(EventBusKafka, "start", start_kafka)
    monkeypatch.setattr(bus, "_sleep_before_retry", wait)
    process = RuntimeHostProcess(
        container=MagicMock(service_registry=None),
        event_bus=bus,
        config=make_runtime_config(),
    )
    seed_mock_handlers(process)
    for method in (
        "_discover_or_wire_handlers",
        "_populate_handlers_from_registry",
        "_wire_package_node_subscriptions",
    ):
        monkeypatch.setattr(process, method, AsyncMock())
    monkeypatch.setattr(EventBusKafka, "subscribe", AsyncMock(return_value=AsyncMock()))
    try:
        await process.start()
        # The process keeps running (liveness 200) but refuses readiness (503).
        assert process.is_running
        assert (await process.readiness_check())["ready"] is False
        health_server = ServiceHealth(runtime=process)
        liveness = await health_server._handle_health(MagicMock())
        assert liveness.status == 200
        assert json.loads(liveness.text)["status"] == "degraded"
        assert (await health_server._handle_readiness(MagicMock())).status == 503
        assert local_publish.await_count == 0
        assert local_subscribe.await_count == 0
        for expected in (1, 2):
            assert await asyncio.wait_for(delays.get(), 1) == expected
            await permits.put(None)
        async with asyncio.timeout(1):
            while not (await process.readiness_check())["ready"]:
                await asyncio.sleep(0)
        assert (await health_server._handle_readiness(MagicMock())).status == 200
        assert local_publish.await_count == 0
    finally:
        await process.stop()


async def test_real_unreachable_broker_refuses_ready_without_patching_kafka() -> None:
    """Seam: the real Kafka connect error (nothing listening) takes the refusal path."""
    local_bus = EventBusInmemory(max_history=10)
    local_publish = AsyncMock(wraps=local_bus.publish)
    local_bus.publish = local_publish
    bus = EventBusRecovering(
        config=ModelKafkaEventBusConfig(bootstrap_servers="127.0.0.1:1"),
        recovery=ModelEventBusRecoveryConfig(
            initial_backoff_seconds=30,
            max_backoff_seconds=30,
            backoff_multiplier=1,
            attempt_timeout_seconds=5,
        ),
        local_bus=local_bus,
        local_start=local_bus.start,
        local_close=local_bus.close,
        local_health_check=local_bus.health_check,
        alarm_topic="runtime-bus",
        consumer_group="test-runtime",
    )
    try:
        await bus.start()
        assert not (await bus.get_readiness_status()).is_ready
        assert (await bus.health_check())["broker_lost"] is True
        with pytest.raises(InfraUnavailableError):
            await bus.publish("events", None, b"x")
        assert local_publish.await_count == 0
        assert bus._recovery_task is not None
    finally:
        await bus.close()
