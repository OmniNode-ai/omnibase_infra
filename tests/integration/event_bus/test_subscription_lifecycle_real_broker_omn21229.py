# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21229 on a real broker: start-failure cleanup and the SIGTERM commit.

1. A subscribe whose consumer started and then failed leaves no callback; the
   retry on the same broker dispatches the published record once.
2. A runtime process holding a fetched batch of five records is sent SIGTERM
   while the first callback runs. The callback is not cancelled, the rest of
   the batch is released, and the group's committed offset on the broker
   covers exactly the completed records.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import uuid
from collections.abc import Callable, Coroutine
from pathlib import Path
from unittest.mock import patch

import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import TopicPartition

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from tests.helpers.service_env import require_service_env

from .conftest import BUS_TEST_ENVIRONMENT

pytestmark = [pytest.mark.integration, pytest.mark.kafka]

RECORDS = 5
CALLBACK_SECONDS = 1.5


@pytest.fixture(autouse=True, scope="module")
def _require_kafka() -> None:
    require_service_env(
        opt_in="KAFKA_INTEGRATION_TESTS",
        endpoint="KAFKA_BOOTSTRAP_SERVERS",
        workflow=".github/workflows/ci.yml (service-integration-suites)",
        service="Kafka/Redpanda",
    )


@pytest.fixture
def bootstrap() -> str:
    return os.environ["KAFKA_BOOTSTRAP_SERVERS"]


@pytest.fixture
async def topic(
    ensure_test_topic: Callable[[str, int], Coroutine[None, None, str]],
) -> str:
    return await ensure_test_topic(
        f"onex.evt.omnimarket.omn21229-lifecycle-{uuid.uuid4().hex[:10]}.v1", 1
    )


async def _publish(bootstrap: str, topic: str, count: int) -> None:
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()
    try:
        for index in range(count):
            await producer.send_and_wait(
                topic, b'{"payload": "omn21229"}', key=str(index).encode()
            )
    finally:
        await producer.stop()


@pytest.mark.asyncio
async def test_start_failure_after_callback_insertion_then_retry_dispatches_once(
    bootstrap: str, topic: str
) -> None:
    group = f"omn21229.{uuid.uuid4().hex[:8]}.consume.v1"
    dispatched: list[str] = []

    async def _callback(message: ModelEventMessage) -> None:
        dispatched.append(str(message.offset))

    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=bootstrap,
            environment=BUS_TEST_ENVIRONMENT,
            timeout_seconds=30,
            auto_offset_reset="earliest",
        )
    )
    await bus.start()
    real_start = bus._start_consumer_for_topic_unlocked
    attempts = 0

    async def _fail_after_start(
        topic_name: str,
        group_id: str,
        *,
        auto_offset_reset_override: str | None = None,
    ) -> None:
        nonlocal attempts
        attempts += 1
        await real_start(
            topic_name, group_id, auto_offset_reset_override=auto_offset_reset_override
        )
        if attempts == 1:
            # The consumer joined and its loop is running; now the start fails.
            await bus._stop_consumer_for_topic(topic_name, group_id)
            raise InfraConnectionError("injected start failure after the join")

    try:
        with patch.object(bus, "_start_consumer_for_topic_unlocked", _fail_after_start):
            with pytest.raises(InfraConnectionError):
                await bus.subscribe(topic, group_id=group, on_message=_callback)
            assert bus._subscribers.get(topic, []) == [], "a callback was left behind"

            await _publish(bootstrap, topic, 1)
            handle = await bus.subscribe(topic, group_id=group, on_message=_callback)
            async with asyncio.timeout(30):
                while not dispatched:
                    await asyncio.sleep(0.1)
            await asyncio.sleep(1.0)
            await handle.stop()
    finally:
        await bus.close()

    assert dispatched == ["0"], f"the record was dispatched {dispatched}"


async def _read_until(
    stream: asyncio.StreamReader, prefix: str, seen: list[str]
) -> str:
    async with asyncio.timeout(60):
        while True:
            line = (await stream.readline()).decode().strip()
            assert line, f"the runtime exited before {prefix}: {seen}"
            seen.append(line)
            if line.startswith(prefix):
                return line.split("=", 1)[1]


@pytest.mark.asyncio
async def test_sigterm_with_a_fetched_batch_commits_exactly_the_completed_records(
    bootstrap: str, topic: str
) -> None:
    group = f"omn21229.{uuid.uuid4().hex[:8]}.consume.v1"
    await _publish(bootstrap, topic, RECORDS)
    helper = Path(__file__).with_name("omn21229_sigterm_runtime.py")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(helper),
        bootstrap,
        topic,
        group,
        str(CALLBACK_SECONDS),
        stdout=asyncio.subprocess.PIPE,
        env={**os.environ, "KAFKA_BROKER_ALLOWLIST": bootstrap},
    )
    assert process.stdout is not None
    seen: list[str] = []
    try:
        consumer_group = await _read_until(process.stdout, "group=", seen)
        await _read_until(process.stdout, "started=", seen)
        process.send_signal(signal.SIGTERM)
        async with asyncio.timeout(60):
            await process.wait()
        rest = (await process.stdout.read()).decode().split()
    finally:
        if process.returncode is None:
            process.kill()
    seen.extend(rest)

    completed = [
        int(line.split("=")[1]) for line in seen if line.startswith("completed=")
    ]
    started = [int(line.split("=")[1]) for line in seen if line.startswith("started=")]
    assert process.returncode == 0, seen
    assert completed, f"the in-flight callback was cancelled: {seen}"
    assert started == completed, f"a started callback never completed: {seen}"
    assert len(completed) < RECORDS, (
        f"the whole batch ran, nothing was in flight: {seen}"
    )

    reader = AIOKafkaConsumer(
        bootstrap_servers=bootstrap, group_id=consumer_group, enable_auto_commit=False
    )
    await reader.start()
    try:
        committed = await reader.committed(TopicPartition(topic, 0))
    finally:
        await reader.stop()
    assert committed == len(completed), (
        f"committed offset {committed} is not the {len(completed)} completed records: "
        f"{seen}"
    )
