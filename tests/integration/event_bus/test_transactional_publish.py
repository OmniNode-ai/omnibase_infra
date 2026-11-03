# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21228: transactional publish with a caller-fixed ``transactional.id``.

A real Redpanda broker is started by the test module (testcontainers) and the
bus under test talks to it through ``EventBusKafka.transactional_producer``.
The reader is built here, independently of the bus: a plain ``AIOKafkaConsumer``
with ``isolation_level="read_committed"``.

The acceptance criterion: after a second producer with the same id initializes,
a publish from the first fails, and nothing the first sent is visible to the
independent ``read_committed`` consumer. Two shapes of it are pinned:

* the fence lands between two publishes (the first producer's whole batch is
  refused);
* the fence lands in the middle of a transaction, after the first record of the
  batch is already in the log (the aborted record must stay invisible).

The second shape carries a positive control: a ``read_uncommitted`` consumer
DOES see the aborted record, so the ``read_committed`` silence is the broker
filtering a record that exists, not a record that was never written.
"""

from __future__ import annotations

import asyncio
import socket
import time
import uuid
from collections.abc import AsyncGenerator, Iterator
from typing import Any
from unittest.mock import patch

import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import KafkaError
from testcontainers.core.container import DockerContainer
from testcontainers.core.docker_client import DockerClient

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.transactional_producer_fencing import (
    InfraProducerFencedError,
)
from omnibase_infra.models.event_bus.model_transactional_record import (
    ModelTransactionalRecord,
)
from omnibase_infra.protocols.protocol_transactional_producer import (
    ProtocolTransactionalProducer,
)

pytestmark = [pytest.mark.integration, pytest.mark.kafka]

BROKER_IMAGE = "redpandadata/redpanda:v24.2.7"
BROKER_READY_SECONDS = 90.0
BUS_ENVIRONMENT = "local"


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


@pytest.fixture
async def topic(broker_bootstrap: str) -> str:
    name = f"onex.evt.platform.txn-probe-{uuid.uuid4().hex[:12]}.v1"
    admin = AIOKafkaAdminClient(bootstrap_servers=broker_bootstrap)
    await admin.start()
    try:
        await admin.create_topics(
            [NewTopic(name=name, num_partitions=1, replication_factor=1)]
        )
    finally:
        await admin.close()
    return name


async def _started_bus(bootstrap: str) -> EventBusKafka:
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=bootstrap,
            environment=BUS_ENVIRONMENT,
            timeout_seconds=30,
            max_retry_attempts=0,
        )
    )
    await bus.start()
    return bus


@pytest.fixture
async def buses(
    broker_bootstrap: str,
) -> AsyncGenerator[tuple[EventBusKafka, ...], None]:
    started = (
        await _started_bus(broker_bootstrap),
        await _started_bus(broker_bootstrap),
    )
    yield started
    for bus in started:
        await bus.close()


async def _read(bootstrap: str, topic: str, *, isolation_level: str) -> list[bytes]:
    """Everything an independent consumer sees on ``topic`` at this isolation level."""
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=None,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        isolation_level=isolation_level,
    )
    await consumer.start()
    values: list[bytes] = []
    try:
        idle = 0
        while idle < 3:
            batches = await consumer.getmany(timeout_ms=1500)
            polled = [r.value for records in batches.values() for r in records]
            values.extend(polled)
            idle = 0 if polled else idle + 1
    finally:
        await consumer.stop()
    return values


def _record(topic: str, value: bytes) -> ModelTransactionalRecord:
    return ModelTransactionalRecord(topic=topic, key=b"k", value=value)


async def test_fenced_producer_publish_fails_and_nothing_it_sent_is_visible(
    broker_bootstrap: str,
    topic: str,
    buses: tuple[EventBusKafka, ...],
) -> None:
    older_bus, newer_bus = buses
    transactional_id = f"omn21228-fence-{uuid.uuid4().hex[:8]}"

    older = await older_bus.transactional_producer(transactional_id)
    receipts = await older.publish([_record(topic, b"older-before-fence")])
    assert receipts[0].topic == topic

    newer = await newer_bus.transactional_producer(transactional_id)

    with pytest.raises(InfraProducerFencedError):
        await older.publish(
            [_record(topic, b"older-fenced-1"), _record(topic, b"older-fenced-2")]
        )
    await newer.publish([_record(topic, b"newer-after-fence")])

    committed = await _read(broker_bootstrap, topic, isolation_level="read_committed")
    assert committed == [b"older-before-fence", b"newer-after-fence"]


async def test_record_sent_inside_the_fenced_transaction_stays_invisible(
    broker_bootstrap: str,
    topic: str,
    buses: tuple[EventBusKafka, ...],
) -> None:
    older_bus, newer_bus = buses
    transactional_id = f"omn21228-inflight-{uuid.uuid4().hex[:8]}"

    older = await older_bus.transactional_producer(transactional_id)
    newer_holder: list[ProtocolTransactionalProducer] = []
    original_send = AIOKafkaProducer.send

    async def send_then_fence(
        producer: AIOKafkaProducer, *args: Any, **kwargs: Any
    ) -> Any:
        future = await original_send(producer, *args, **kwargs)
        if not newer_holder:
            newer_holder.append(
                await newer_bus.transactional_producer(transactional_id)
            )
        return future

    with (
        patch.object(AIOKafkaProducer, "send", send_then_fence),
        pytest.raises(InfraProducerFencedError),
    ):
        await older.publish(
            [_record(topic, b"inflight-1"), _record(topic, b"inflight-2")]
        )

    assert newer_holder, "the second producer must have initialized mid-transaction"
    await newer_holder[0].publish([_record(topic, b"newer-after-fence")])

    uncommitted = await _read(
        broker_bootstrap, topic, isolation_level="read_uncommitted"
    )
    assert b"inflight-1" in uncommitted, (
        "positive control: the aborted record must be in the log"
    )
    committed = await _read(broker_bootstrap, topic, isolation_level="read_committed")
    assert committed == [b"newer-after-fence"]
