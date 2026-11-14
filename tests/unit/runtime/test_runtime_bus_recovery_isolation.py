# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime bus recovery logs every attempt and isolates one failed join."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.event_bus_recovering import EventBusRecovering
from omnibase_infra.models.model_event_bus_recovery_config import (
    ModelEventBusRecoveryConfig,
)

pytestmark = pytest.mark.unit

RECOVERY_LOGGER = "omnibase_infra.handlers.event_bus_recovering"


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


def recovery_entries(caplog: pytest.LogCaptureFixture) -> list[tuple[int, str]]:
    """Interpolated messages from the bus logger as (level, message) pairs."""
    return [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == RECOVERY_LOGGER
    ]


async def test_recovery_logs_every_attempt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = make_bus()
    starts = 0
    delays: asyncio.Queue[float] = asyncio.Queue()
    permits: asyncio.Queue[None] = asyncio.Queue()

    async def start_kafka(self: EventBusKafka) -> None:
        nonlocal starts
        starts += 1
        # One start per recovery attempt's reconnect: #1 is the initial
        # bus.start(), #2 and #3 fail attempts 1 and 2, #4 is attempt 3.
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
        assert starts == 1  # Initial connect failed; fallback holds the transport.
        for expected in (1, 2, 4):  # Two failed attempts, then the successful one.
            assert await asyncio.wait_for(delays.get(), 1) == expected
            await permits.put(None)
        async with asyncio.timeout(1):
            while not (await bus.get_readiness_status()).is_ready:
                await asyncio.sleep(0)
        assert starts == 4
    finally:
        await bus.close()
    entries = recovery_entries(caplog)
    for attempt in (1, 2, 3):
        assert any(
            level == logging.INFO
            and message.startswith(
                f"Runtime event bus recovery attempt {attempt} starting"
            )
            for level, message in entries
        ), entries
    for attempt in (1, 2):
        assert any(
            level == logging.WARNING and f"recovery attempt {attempt} failed" in message
            for level, message in entries
        ), entries
    recovered = [
        message
        for level, message in entries
        if level == logging.INFO and "Runtime event bus recovered" in message
    ]
    assert len(recovered) == 1
    assert "attempt=3" in recovered[0]


async def test_one_join_timeout_does_not_fail_the_other_attachments(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = make_bus()
    starts = 0
    joins: dict[str, int] = {"a": 0, "b": 0, "c": 0}
    delays: asyncio.Queue[float] = asyncio.Queue()
    permits: asyncio.Queue[None] = asyncio.Queue()

    async def start_kafka(self: EventBusKafka) -> None:
        nonlocal starts
        starts += 1
        if starts == 1:
            raise InfraConnectionError("broker unavailable")
        self._started = True
        self._shutdown = False

    async def attach(self: EventBusKafka, **kwargs: object) -> AsyncMock:
        topic = str(kwargs["topic"])
        joins[topic] += 1
        if topic == "b" and joins["b"] == 1:
            await asyncio.sleep(10)  # The 1s attempt timeout times this join out.
        return AsyncMock()

    async def wait(delay: float) -> None:
        await delays.put(delay)
        await permits.get()

    monkeypatch.setattr(EventBusKafka, "start", start_kafka)
    monkeypatch.setattr(EventBusKafka, "subscribe", attach)
    monkeypatch.setattr(bus, "_sleep_before_retry", wait)
    caplog.set_level(logging.INFO, logger=RECOVERY_LOGGER)
    try:
        await bus.start()
        for topic in ("a", "b", "c"):
            await bus.subscribe(topic, group_id="g", on_message=AsyncMock())
        # Attempt 1: every join runs, b hangs until its own deadline.
        assert await asyncio.wait_for(delays.get(), 1) == 1.0
        await permits.put(None)
        assert await asyncio.wait_for(delays.get(), 5) == 2.0
        assert not (await bus.get_readiness_status()).is_ready
        assert joins == {"a": 1, "b": 1, "c": 1}
        assert any(
            record.levelno == logging.WARNING
            and "subscription restore failed" in record.getMessage()
            and "(topic=b," in record.getMessage()
            for record in caplog.records
        )
        # Attempt 2: only b retries; the transport from attempt 1 is kept.
        await permits.put(None)
        async with asyncio.timeout(1):
            while not (await bus.get_readiness_status()).is_ready:
                await asyncio.sleep(0)
        assert joins == {"a": 1, "b": 2, "c": 1}
        assert starts == 2  # First failure plus one reconnect; no teardown retry.
    finally:
        await bus.close()


async def test_unexpected_error_in_an_attempt_is_logged_and_the_loop_keeps_retrying(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = make_bus()
    starts = 0
    delays: asyncio.Queue[float] = asyncio.Queue()
    permits: asyncio.Queue[None] = asyncio.Queue()

    async def start_kafka(self: EventBusKafka) -> None:
        nonlocal starts
        starts += 1
        if starts == 1:
            raise InfraConnectionError("broker unavailable")
        if starts == 2:
            raise ValueError("boom")
        self._started = True
        self._shutdown = False

    async def wait(delay: float) -> None:
        await delays.put(delay)
        await permits.get()

    monkeypatch.setattr(EventBusKafka, "start", start_kafka)
    monkeypatch.setattr(bus, "_sleep_before_retry", wait)
    caplog.set_level(logging.INFO, logger=RECOVERY_LOGGER)
    try:
        await bus.start()
        assert starts == 1
        assert await asyncio.wait_for(delays.get(), 1) == 1.0
        await permits.put(None)  # Attempt 1: the unexpected ValueError.
        assert await asyncio.wait_for(delays.get(), 1) == 2.0
        assert any(
            level == logging.ERROR
            and "recovery attempt 1 failed" in message
            and "ValueError" in message
            for level, message in recovery_entries(caplog)
        )
        await permits.put(None)  # The loop kept retrying and now reconnects.
        async with asyncio.timeout(1):
            while not (await bus.get_readiness_status()).is_ready:
                await asyncio.sleep(0)
        assert starts == 3
    finally:
        await bus.close()
