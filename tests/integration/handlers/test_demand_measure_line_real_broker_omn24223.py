# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24223: the demand source's boot measure line counts the topics of a real broker.

Real Redpanda broker (testcontainers); the demand source is the production class.
The seam under test is the one between the broker's own state and the measure
line the boot path logs: the broker is read, not the code, for which topics
hold retained records, which are missing, and which are idle and watched.

Golden chain: one topic holds a retained record, one is empty and idle, and one
name is never created; the measure line counts all three and the in-use set
holds the retained and the missing topics. Error chain: a boot set of one
missing topic counts as unreadable and in use, with nothing idle and watched.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator

import pytest
from aiokafka import AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import KafkaError
from testcontainers.core.container import DockerContainer
from testcontainers.core.docker_client import DockerClient

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_kafka_topic_demand_source import (
    HandlerKafkaTopicDemandSource,
)

pytestmark = [pytest.mark.integration, pytest.mark.kafka]

BROKER_IMAGE = "redpandadata/redpanda:v24.2.7"
BROKER_READY_SECONDS = 90.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


async def _wait_for_broker(bootstrap: str) -> None:
    deadline = time.monotonic() + BROKER_READY_SECONDS
    last: Exception | None = None
    while time.monotonic() < deadline:
        admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
        try:
            await admin.start()
            await admin.describe_cluster()
            return
        except (KafkaError, OSError) as exc:
            last = exc
            await asyncio.sleep(1.0)
        finally:
            await admin.close()
    raise AssertionError(f"broker at {bootstrap} not ready: {last!r}")


@pytest.fixture(scope="module")
def broker_bootstrap() -> Iterator[str]:
    """A throwaway Redpanda broker on a free port, advertised at that port."""
    port = _free_port()
    host = DockerClient().host()
    container = (
        DockerContainer(BROKER_IMAGE)
        .with_bind_ports(port, port)
        .with_command(
            "redpanda start "
            f"--kafka-addr PLAINTEXT://0.0.0.0:{port} "
            f"--advertise-kafka-addr PLAINTEXT://{host}:{port} "
            "--mode dev-container --smp 1 --memory 1G --reserve-memory 0M "
            "--reactor-backend=epoll --default-log-level=warn"
        )
    )
    with container:
        bootstrap = f"{host}:{port}"
        asyncio.run(_wait_for_broker(bootstrap))
        yield bootstrap


async def _create_topic(bootstrap: str, logical: str) -> None:
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        await admin.create_topics(
            [
                NewTopic(
                    name=logical,
                    num_partitions=1,
                    replication_factor=1,
                )
            ]
        )
    finally:
        await admin.close()


async def _write_record(bootstrap: str, logical: str) -> None:
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()
    try:
        await producer.send_and_wait(
            logical, key=b"k", value=b"{}", headers=[("retained", b"true")]
        )
    finally:
        await producer.stop()


class _Names:
    """Topic names of one scenario; the suffix keeps scenarios apart on one broker."""

    def __init__(self) -> None:
        suffix = uuid.uuid4().hex[:8]
        self.retained = f"onex.evt.platform.demand-measure-retained-{suffix}.v1"
        self.idle = f"onex.evt.platform.demand-measure-idle-{suffix}.v1"
        self.absent = f"onex.evt.platform.demand-measure-absent-{suffix}.v1"


async def _on_first(topic: str) -> None:
    return None


@pytest.mark.asyncio
async def test_the_measure_line_counts_the_brokers_topics(
    broker_bootstrap: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("KAFKA_TOPIC_NAMESPACE", raising=False)
    names = _Names()
    await _create_topic(broker_bootstrap, names.retained)
    await _write_record(broker_bootstrap, names.retained)
    await _create_topic(broker_bootstrap, names.idle)

    source = HandlerKafkaTopicDemandSource(
        ModelKafkaEventBusConfig(bootstrap_servers=broker_bootstrap)
    )
    stop: Callable[[], Awaitable[None]] = _noop_stop
    try:
        with caplog.at_level(
            logging.INFO,
            logger="omnibase_infra.handlers.handler_kafka_topic_demand_source",
        ):
            in_use, stop = await source.watch_idle_topics(
                {names.retained, names.idle, names.absent}, _on_first
            )
        assert in_use == frozenset({names.retained, names.absent})
        measure = [
            record.message
            for record in caplog.records
            if record.message.startswith("Boot subscription demand measure")
        ]
        assert len(measure) == 1
        assert measure[0] == (
            "Boot subscription demand measure: 3 topic(s) measured; 1 hold retained "
            "records, 1 are missing from the broker, 0 exist but are unreadable, "
            "1 are idle and watched (OMN-24223)"
        )
    finally:
        await stop()


async def _noop_stop() -> None:
    return None


@pytest.mark.asyncio
async def test_the_measure_line_counts_a_boot_set_of_missing_topics(
    broker_bootstrap: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("KAFKA_TOPIC_NAMESPACE", raising=False)
    names = _Names()

    source = HandlerKafkaTopicDemandSource(
        ModelKafkaEventBusConfig(bootstrap_servers=broker_bootstrap)
    )
    stop: Callable[[], Awaitable[None]] = _noop_stop
    try:
        with caplog.at_level(
            logging.INFO,
            logger="omnibase_infra.handlers.handler_kafka_topic_demand_source",
        ):
            in_use, stop = await source.watch_idle_topics({names.absent}, _on_first)
        assert in_use == frozenset({names.absent})
        measure = [
            record.message
            for record in caplog.records
            if record.message.startswith("Boot subscription demand measure")
        ]
        assert len(measure) == 1
        assert measure[0] == (
            "Boot subscription demand measure: 1 topic(s) measured; 0 hold retained "
            "records, 1 are missing from the broker, 0 exist but are unreadable, "
            "0 are idle and watched (OMN-24223)"
        )
    finally:
        await stop()
