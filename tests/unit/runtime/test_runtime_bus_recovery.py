# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime bus outages degrade locally and recover without restarting the process."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import (
    EventBusKafka,
    EventBusRecovering,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.model_event_bus_recovery_config import (
    ModelEventBusRecoveryConfig,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.services.health_checker import ServiceHealth
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.unit


def make_bus() -> EventBusRecovering:
    local_bus = EventBusInmemory(max_history=10)
    return EventBusRecovering(
        config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092"),
        recovery=ModelEventBusRecoveryConfig(
            initial_backoff_seconds=1,
            max_backoff_seconds=4,
            backoff_multiplier=2,
            attempt_timeout_seconds=1,
        ),
        local_bus=local_bus,
        local_start=local_bus.start,
        local_close=local_bus.close,
        local_health_check=local_bus.health_check,
        alarm_topic="runtime-bus",
        consumer_group="test-runtime",
    )


async def test_failed_start_retries_with_cap_and_runtime_becomes_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = make_bus()
    attempts = 0
    delays: asyncio.Queue[float] = asyncio.Queue()
    permits: asyncio.Queue[None] = asyncio.Queue()

    async def start_kafka(self: EventBusKafka) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 5:
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

    # Keep the real process startup/liveness/readiness lifecycle. Handler discovery
    # and package scanning are orthogonal to this injected transport seam.
    for method in (
        "_discover_or_wire_handlers",
        "_populate_handlers_from_registry",
        "_wire_package_node_subscriptions",
    ):
        monkeypatch.setattr(process, method, AsyncMock())
    monkeypatch.setattr(EventBusKafka, "subscribe", AsyncMock(return_value=AsyncMock()))
    try:
        await process.start()
        assert process.is_running
        assert (await process.health_check())["degraded"] is True
        assert (await process.readiness_check())["ready"] is False
        health_server = ServiceHealth(runtime=process)
        liveness = await health_server._handle_health(MagicMock())
        assert liveness.status == 200
        assert json.loads(liveness.text)["status"] == "degraded"
        assert (await health_server._handle_readiness(MagicMock())).status == 503
        await bus.start()  # RuntimeHostProcess starts the already injected bus again.
        assert attempts == 1
        assert len(bus.consumer_rejoin_alarms()) == 1
        for expected in (1, 2, 4, 4):
            assert await asyncio.wait_for(delays.get(), timeout=1) == expected
            await permits.put(None)
        async with asyncio.timeout(1):
            while not (await process.readiness_check())["ready"]:
                await asyncio.sleep(0)
        assert process.is_running
        assert attempts == 5
        assert (await process.health_check())["degraded"] is False
        assert len(bus.consumer_rejoin_alarms()) == 1
    finally:
        await process.stop()


async def test_fallback_delivers_locally_then_restores_live_subscriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = make_bus()
    broker_up = False
    permit = asyncio.Event()
    attached: list[dict[str, object]] = []
    received: list[bytes] = []
    kafka_unsubscribe = AsyncMock(spec=EventBusKafka.close)

    async def start_kafka(self: EventBusKafka) -> None:
        if not broker_up:
            raise InfraConnectionError("broker unavailable")
        self._started = True
        self._shutdown = False

    async def subscribe_kafka(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        attached.append(kwargs)
        return kafka_unsubscribe

    async def receive(message: object) -> None:
        received.append(message.value)

    monkeypatch.setattr(EventBusKafka, "start", start_kafka)
    monkeypatch.setattr(EventBusKafka, "subscribe", subscribe_kafka)
    monkeypatch.setattr(bus, "_sleep_before_retry", lambda delay: permit.wait())
    try:
        await bus.start()
        unsubscribe = await bus.subscribe(
            "events",
            group_id="group",
            on_message=receive,
            required_for_readiness=True,
            auto_offset_reset="earliest",
        )
        removed = await bus.subscribe("removed", group_id="gone", on_message=receive)
        await removed()
        receipt = await bus.publish("events", None, b"local")
        assert receipt.transport.value == "inmemory"
        assert received == [b"local"]
        assert not (await bus.get_readiness_status()).is_ready
        broker_up = True
        permit.set()
        async with asyncio.timeout(1):
            while (await bus.health_check())["degraded"]:
                await asyncio.sleep(0)
        assert len(attached) == 1
        assert attached[0]["topic"] == "events"
        assert attached[0]["required_for_readiness"] is True
        assert attached[0]["auto_offset_reset"] == "earliest"
        assert received == [b"local"]  # Local delivery is never replayed to Kafka.
        await unsubscribe()
        kafka_unsubscribe.assert_awaited_once()
    finally:
        await bus.close()


async def test_shutdown_cancels_recovery_and_never_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = make_bus()
    start = AsyncMock(side_effect=InfraConnectionError("down"))
    waiting = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait(delay: float) -> None:
        waiting.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(EventBusKafka, "start", start)
    monkeypatch.setattr(bus, "_sleep_before_retry", wait)
    await bus.start()
    await waiting.wait()
    await bus.close()
    await bus.close()
    assert cancelled.is_set()
    start.assert_awaited_once()
    assert (await bus.health_check())["healthy"] is False


async def test_programming_errors_remain_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = make_bus()
    monkeypatch.setattr(
        EventBusKafka, "start", AsyncMock(side_effect=ValueError("bad config"))
    )
    with pytest.raises(ValueError, match="bad config"):
        await bus.start()
    assert not bus.consumer_rejoin_alarms()
    await bus.close()


def test_recovery_configuration_comes_from_contract_and_rejects_invalid_bounds() -> (
    None
):
    policy = ModelEventBusRecoveryConfig.from_contract()
    assert policy.initial_backoff_seconds <= policy.max_backoff_seconds
    with pytest.raises(ValueError):
        ModelEventBusRecoveryConfig(
            initial_backoff_seconds=4,
            max_backoff_seconds=1,
            backoff_multiplier=2,
            attempt_timeout_seconds=1,
        )


async def test_real_kafka_start_path_recovers_and_alarms_once_per_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiokafka.errors import KafkaConnectionError

    from omnibase_infra.event_bus import event_bus_kafka

    broker_up = False
    producers: list[MagicMock] = []
    permit = asyncio.Event()
    bus = make_bus()
    bus._config = bus.config.model_copy(update={"max_retry_attempts": 0})
    bus._max_retry_attempts = 0

    def producer_factory(**kwargs: object) -> MagicMock:
        producer = MagicMock(_closed=False)
        producer.stop = AsyncMock()
        producer.start = AsyncMock(
            side_effect=None if broker_up else KafkaConnectionError("broker down")
        )
        producers.append(producer)
        return producer

    monkeypatch.setattr(event_bus_kafka, "AIOKafkaProducer", producer_factory)
    monkeypatch.setattr(bus, "_sleep_before_retry", lambda delay: permit.wait())
    try:
        await bus.start()
        assert (await bus.health_check())["degraded"]
        producers[0].stop.assert_awaited_once()
        assert len(bus.consumer_rejoin_alarms()) == 1
        broker_up = True
        permit.set()
        async with asyncio.timeout(1):
            while not (await bus.get_readiness_status()).is_ready:
                await asyncio.sleep(0)
        assert len(producers) == 2
        assert len(bus.consumer_rejoin_alarms()) == 1
        # A later transport outage starts a new episode, retaining the old alarm.
        mirror = AsyncMock()
        bus._health_emitter = MagicMock(emit_event=mirror)
        permit.clear()
        monkeypatch.setattr(
            EventBusKafka,
            "publish",
            AsyncMock(side_effect=InfraConnectionError("down again")),
        )
        await bus.publish("events", None, b"second outage")
        await bus.publish("events", None, b"still down")
        assert len(bus.consumer_rejoin_alarms()) == 2
        mirror.assert_awaited_once()
        assert all(
            a.reason == "runtime_event_bus_unavailable"
            for a in bus.consumer_rejoin_alarms()
        )
        assert (await bus.health_check())["degraded"]
    finally:
        await bus.close()


async def test_hung_connect_is_bounded_and_recovery_cancels_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = make_bus()
    bus._recovery_policy = bus._recovery_policy.model_copy(
        update={"attempt_timeout_seconds": 0.01, "initial_backoff_seconds": 0.001}
    )
    cancelled = asyncio.Event()

    async def hanging_start(self: EventBusKafka) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(EventBusKafka, "start", hanging_start)
    try:
        async with asyncio.timeout(1):
            await bus.start()
        assert cancelled.is_set()
        assert (await bus.health_check())["degraded"]
        assert len(bus.consumer_rejoin_alarms()) == 1
    finally:
        await bus.close()


async def test_failed_consumer_restore_keeps_fallback_until_all_attach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = make_bus()
    permits: asyncio.Queue[None] = asyncio.Queue()
    attempts: asyncio.Queue[float] = asyncio.Queue()
    starts = 0
    attaches = 0

    async def start(self: EventBusKafka) -> None:
        nonlocal starts
        starts += 1
        if starts == 1:
            raise InfraConnectionError("down")
        self._started = True
        self._shutdown = False

    async def attach(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        nonlocal attaches
        attaches += 1
        if attaches == 1:
            raise InfraConnectionError("coordinator unavailable")
        return AsyncMock()

    async def wait(delay: float) -> None:
        await attempts.put(delay)
        await permits.get()

    monkeypatch.setattr(EventBusKafka, "start", start)
    monkeypatch.setattr(EventBusKafka, "subscribe", attach)
    monkeypatch.setattr(bus, "_sleep_before_retry", wait)
    try:
        await bus.start()
        received = AsyncMock()
        await bus.subscribe("events", group_id="g", on_message=received)
        assert await asyncio.wait_for(attempts.get(), 1) == 1
        await permits.put(None)
        assert await asyncio.wait_for(attempts.get(), 1) == 2
        assert not (await bus.get_readiness_status()).is_ready
        await bus.publish("events", None, b"still local")
        received.assert_awaited_once()
        await permits.put(None)
        async with asyncio.timeout(1):
            while not (await bus.get_readiness_status()).is_ready:
                await asyncio.sleep(0)
        assert attaches == 2
        assert len(bus.consumer_rejoin_alarms()) == 1
    finally:
        await bus.close()


async def test_slow_reconnect_does_not_block_local_subscribe_or_unsubscribe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = make_bus()
    connecting = asyncio.Event()
    connected = asyncio.Event()
    attempts = 0
    attached: list[str] = []

    async def start(self: EventBusKafka) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InfraConnectionError("down")
        connecting.set()
        await connected.wait()
        self._started = True
        self._shutdown = False

    async def attach(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        attached.append(str(kwargs["topic"]))
        return AsyncMock()

    monkeypatch.setattr(EventBusKafka, "start", start)
    monkeypatch.setattr(EventBusKafka, "subscribe", attach)
    monkeypatch.setattr(bus, "_sleep_before_retry", AsyncMock())
    try:
        await bus.start()
        await asyncio.wait_for(connecting.wait(), 1)
        received = AsyncMock()
        async with asyncio.timeout(1):
            removed = await bus.subscribe("removed", group_id="g", on_message=received)
            await removed()
            await bus.subscribe("live", group_id="g", on_message=received)
            await bus.publish("live", None, b"while connecting")
        received.assert_awaited_once()
        connected.set()
        async with asyncio.timeout(1):
            while not (await bus.get_readiness_status()).is_ready:
                await asyncio.sleep(0)
        assert attached == ["live"]
    finally:
        await bus.close()


async def test_subscription_changes_during_join_are_reconciled_before_cutover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = make_bus()
    joining = asyncio.Event()
    joined = asyncio.Event()
    attempts = 0
    attached: list[str] = []
    removed_broker_subscription = AsyncMock(spec=EventBusKafka.close)

    async def start(self: EventBusKafka) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InfraConnectionError("down")
        self._started = True
        self._shutdown = False

    async def attach(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        topic = str(kwargs["topic"])
        attached.append(topic)
        if topic == "old":
            joining.set()
            await joined.wait()
            return removed_broker_subscription
        return AsyncMock(spec=EventBusKafka.close)

    monkeypatch.setattr(EventBusKafka, "start", start)
    monkeypatch.setattr(EventBusKafka, "subscribe", attach)
    monkeypatch.setattr(bus, "_sleep_before_retry", AsyncMock())
    try:
        await bus.start()
        removed = await bus.subscribe("old", group_id="g", on_message=AsyncMock())
        await asyncio.wait_for(joining.wait(), 1)
        async with asyncio.timeout(1):
            await removed()
            await bus.subscribe("new", group_id="g", on_message=AsyncMock())
        assert not (await bus.get_readiness_status()).is_ready
        joined.set()
        async with asyncio.timeout(1):
            while not (await bus.get_readiness_status()).is_ready:
                await asyncio.sleep(0)
        assert attached == ["old", "new"]
        removed_broker_subscription.assert_awaited_once()
    finally:
        await bus.close()


async def test_hung_alarm_mirror_preserves_local_alarm_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = make_bus()
    bus._recovery_policy = bus._recovery_policy.model_copy(
        update={"attempt_timeout_seconds": 0.01}
    )
    mirror_cancelled = asyncio.Event()

    async def mirror(**kwargs: object) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            mirror_cancelled.set()

    bus._health_emitter = MagicMock(emit_event=mirror)
    monkeypatch.setattr(
        EventBusKafka, "start", AsyncMock(side_effect=InfraConnectionError("down"))
    )
    try:
        async with asyncio.timeout(1):
            await bus.start()
        assert mirror_cancelled.is_set()
        assert len(bus.consumer_rejoin_alarms()) == 1
        assert (await bus.health_check())["degraded"]
        assert (await bus.health_check())["recovery_running"]
    finally:
        await bus.close()


def test_runtime_overlay_replaces_policy_and_nonfinite_delays_are_refused() -> None:
    from omnibase_infra.runtime.models.model_event_bus_config import ModelEventBusConfig

    override = {
        "initial_backoff_seconds": 3,
        "max_backoff_seconds": 9,
        "backoff_multiplier": 3,
        "attempt_timeout_seconds": 10,
    }
    assert ModelEventBusConfig(recovery=override).recovery.model_dump() == override
    for value in (float("inf"), float("nan"), 0, -1):
        with pytest.raises(ValueError):
            ModelEventBusRecoveryConfig(
                **{**override, "initial_backoff_seconds": value}
            )


async def test_shutdown_finishes_teardown_already_started_by_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = make_bus()
    close_entered = asyncio.Event()
    allow_close = asyncio.Event()
    closed = asyncio.Event()
    cancelled = asyncio.Event()

    async def close(self: EventBusKafka) -> None:
        # The real transport also marks shutdown before awaiting cleanup.
        if self._shutdown:
            return
        self._shutdown = True
        close_entered.set()
        try:
            await allow_close.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        closed.set()

    monkeypatch.setattr(EventBusKafka, "close", close)
    monkeypatch.setattr(
        EventBusKafka, "start", AsyncMock(side_effect=InfraConnectionError("down"))
    )
    monkeypatch.setattr(bus, "_sleep_before_retry", AsyncMock())
    await bus.start()
    await asyncio.wait_for(close_entered.wait(), 1)
    shutdown = asyncio.create_task(bus.close())
    try:
        await asyncio.sleep(0)
        allow_close.set()
        await asyncio.wait_for(shutdown, 1)
        assert closed.is_set()
        assert not cancelled.is_set()
    finally:
        allow_close.set()
        await shutdown


async def test_one_failed_subscription_on_live_kafka_does_not_degrade_the_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unprovisioned topic fails its own subscribe; Kafka stays the transport."""
    bus = make_bus()

    async def start(self: EventBusKafka) -> None:
        self._started = True
        self._shutdown = False
        self._producer = MagicMock(_closed=False, stop=AsyncMock())

    async def attach(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        if kwargs["topic"] == "unprovisioned":
            raise InfraConnectionError("topic metadata unavailable after all retries")
        return AsyncMock()

    monkeypatch.setattr(EventBusKafka, "start", start)
    monkeypatch.setattr(EventBusKafka, "close", AsyncMock())
    monkeypatch.setattr(EventBusKafka, "subscribe", attach)
    monkeypatch.setattr(
        EventBusKafka, "health_check", AsyncMock(return_value={"healthy": True})
    )
    try:
        await bus.start()
        await bus.subscribe("events", group_id="g", on_message=AsyncMock())
        with pytest.raises(InfraConnectionError, match="metadata unavailable"):
            await bus.subscribe("unprovisioned", group_id="g", on_message=AsyncMock())
        health = await bus.health_check()
        assert health == {"healthy": True}
        assert "transport" not in health
        assert not bus.consumer_rejoin_alarms()
        assert bus._recovery_task is None
    finally:
        await bus.close()
