# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read-only mirror sources must never initialize a Kafka producer (OMN-20867)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from aiokafka.errors import ClusterAuthorizationFailedError
from aiokafka.structs import TopicPartition

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus import kafka_transport
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayForwarderConfig,
    ModelGatewayForwarderRuntimeConfig,
    ModelGatewayLaneMirrorConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.runtime import gateway_forwarder
from omnibase_infra.topics.topic_namespace import TOPIC_NAMESPACE_ENV_VAR

pytestmark = pytest.mark.integration

CAPACITY_TOPIC = (
    "onex.evt.omnimarket.lab-host-capacity-advertised.v1"  # onex-topic-allow: OMN-20867
)
PR_STATE_TOPIC = (
    "onex.evt.omnimarket.pr-state-observed.v1"  # onex-topic-allow: OMN-20867
)
SOURCE = "source.invalid:9092"
DESTINATION = "destination.invalid:9092"


@dataclass
class _Clients:
    read_only: set[str] = field(default_factory=lambda: {SOURCE})
    producers: list[_Producer] = field(default_factory=list)
    consumers: list[_Consumer] = field(default_factory=list)
    pending: dict[str, list[SimpleNamespace]] = field(default_factory=dict)
    delivered: asyncio.Event = field(default_factory=asyncio.Event)


class _Producer:
    clients: _Clients

    def __init__(self, *, bootstrap_servers: str, **kwargs: object) -> None:
        self.bootstrap = bootstrap_servers
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.sent: list[tuple[str, bytes]] = []
        self.clients.producers.append(self)

    async def start(self) -> None:
        if self.bootstrap in self.clients.read_only:
            raise ClusterAuthorizationFailedError
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send_and_wait(
        self, topic: str, *, value: bytes, **kwargs: object
    ) -> SimpleNamespace:
        assert self.started
        self.sent.append((topic, value))
        return SimpleNamespace(topic=topic, partition=0, offset=len(self.sent) + 10)


class _Consumer:
    clients: _Clients

    def __init__(self, *topics: str, bootstrap_servers: str, **kwargs: object) -> None:
        self.topics = topics
        self.bootstrap = bootstrap_servers
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.committed: list[Mapping[TopicPartition, int]] = []
        self.clients.consumers.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def assignment(self) -> set[TopicPartition]:
        return {TopicPartition(topic, 0) for topic in self.topics}

    async def getmany(
        self, **kwargs: object
    ) -> dict[TopicPartition, list[SimpleNamespace]]:
        records = self.clients.pending.pop(self.bootstrap, [])
        await asyncio.sleep(0)
        return {TopicPartition(record.topic, 0): [record] for record in records}

    async def commit(self, offsets: Mapping[TopicPartition, int]) -> None:
        self.committed.append(offsets)
        self.clients.delivered.set()


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> _Clients:
    state = _Clients()
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    monkeypatch.setattr(_Producer, "clients", state, raising=False)
    monkeypatch.setattr(_Consumer, "clients", state, raising=False)
    monkeypatch.setattr(kafka_transport, "AIOKafkaProducer", _Producer)
    monkeypatch.setattr(kafka_transport, "AIOKafkaConsumer", _Consumer)
    return state


def _bus(bootstrap: str) -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers=bootstrap,
        environment="test",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )


def _runtime(tmp_path: Path, topic: str) -> ModelGatewayForwarderRuntimeConfig:
    return ModelGatewayForwarderRuntimeConfig(
        forwarder=ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("79afa726-3852-464f-b7a4-d4b8b9c75ee7"),
                tenant_slug="mirror-source-test",
                principal_id="t-79afa7263852464fb7a4d4b8b9c75ee7",
            ),
            local_transport_flavor="containerized",
            dedupe_store_path=tmp_path / "dedupe.sqlite3",
            lane_mirror=ModelGatewayLaneMirrorConfig(
                source_lane="lane-src",
                mirror_lanes=("lane-dst",),
                declared_lanes=("lane-src", "lane-dst"),
                topics=(topic,),
            ),
        ),
        lane_mirror_source_bus=_bus(SOURCE),
        lane_mirror_buses={"lane-dst": _bus(DESTINATION)},
    )


@pytest.mark.asyncio
async def test_consume_only_starts_and_restarts_consumer_without_constructing_producer(
    clients: _Clients,
) -> None:
    transport = KafkaTransport(
        config=_bus(SOURCE), topics=(CAPACITY_TOPIC,), consume_only=True
    )
    try:
        await transport.start()
        await transport.start()
        assert len(clients.consumers) == 1
        assert clients.consumers[0].started
        assert clients.consumers[0].kwargs["enable_auto_commit"] is False
        assert clients.producers == []
        await transport.restart_consumer()
        assert clients.consumers[0].stopped
        assert len(clients.consumers) == 2
        assert clients.consumers[1].started
        assert clients.producers == []
    finally:
        await transport.close()
    assert all(consumer.stopped for consumer in clients.consumers)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send", "send_with_coordinate"])
@pytest.mark.parametrize("started", [False, True])
async def test_consume_only_publish_is_a_typed_configuration_error(
    clients: _Clients, method: str, started: bool
) -> None:
    transport = KafkaTransport(
        config=_bus(SOURCE), topics=(CAPACITY_TOPIC,), consume_only=True
    )
    try:
        if started:
            await transport.start()
        with pytest.raises(ProtocolConfigurationError, match="consume-only"):
            await getattr(transport, method)(CAPACITY_TOPIC, None, b"{}", {})
        assert clients.producers == []
    finally:
        await transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", [CAPACITY_TOPIC, PR_STATE_TOPIC])
async def test_mirror_only_entrypoint_delivers_from_read_only_source(
    clients: _Clients, tmp_path: Path, topic: str
) -> None:
    config = _runtime(tmp_path, topic)
    payload = b'{"payload": {}}'
    clients.pending[SOURCE] = [
        SimpleNamespace(
            topic=topic,
            partition=0,
            offset=0,
            key=None,
            value=payload,
            headers=[
                ("message_id", str(uuid4()).encode()),
                ("content_type", b"application/json"),
            ],
        )
    ]
    shutdown = asyncio.Event()
    ready = tmp_path / "ready"

    async def no_secret(reference: str) -> str | None:
        raise AssertionError("mirror-only startup must not resolve cloud secrets")

    task = asyncio.create_task(
        gateway_forwarder.run_gateway_forwarder(
            config,
            shutdown_event=shutdown,
            ready_path=ready,
            resolve_secret=no_secret,
        )
    )
    delivered = asyncio.create_task(clients.delivered.wait())
    try:
        done, _ = await asyncio.wait(
            {task, delivered}, timeout=5, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            await task
        assert delivered in done, "mirror did not commit its source record"
        assert ready.exists()
        assert [p.bootstrap for p in clients.producers] == [DESTINATION]
        assert clients.producers[0].started
        assert clients.producers[0].kwargs["enable_idempotence"] is True
        assert clients.producers[0].sent == [(topic, payload)]
        assert len(clients.consumers) == 1
        assert clients.consumers[0].bootstrap == SOURCE
        assert clients.consumers[0].committed == [{TopicPartition(topic, 0): 1}]
    finally:
        shutdown.set()
        delivered.cancel()
        await asyncio.gather(delivered, return_exceptions=True)
        await asyncio.wait_for(task, timeout=5)
    assert all(p.stopped for p in clients.producers)
    assert all(c.stopped for c in clients.consumers)
    assert not ready.exists()


@pytest.mark.asyncio
async def test_default_transport_still_starts_a_producer(clients: _Clients) -> None:
    transport = KafkaTransport(config=_bus(DESTINATION))
    try:
        await transport.start()
        await transport.send(CAPACITY_TOPIC, None, b"{}", {})
        assert len(clients.producers) == 1
        assert clients.producers[0].started
        assert clients.producers[0].sent == [(CAPACITY_TOPIC, b"{}")]
        assert clients.consumers == []
    finally:
        await transport.close()
    assert clients.producers[0].stopped


@pytest.mark.asyncio
async def test_cloud_forwarder_transports_keep_their_producers(
    clients: _Clients, tmp_path: Path
) -> None:
    clients.read_only.clear()
    config = _runtime(tmp_path, PR_STATE_TOPIC)
    config = ModelGatewayForwarderRuntimeConfig(
        forwarder=config.forwarder.model_copy(
            update={
                "cloud_bus": ModelGatewayCloudBusConfig(
                    broker_provider_id=uuid4(),
                    cloud_broker_ref="cloud.kafka.broker",
                    cloud_auth_ref="cloud.kafka.auth",
                    acl_provisioner_ref="cloud.kafka.acl",
                    msk_region_ref="cloud.kafka.region",
                    sasl_mechanism="AWS_MSK_IAM",
                ),
                "mirror_topics": ModelGatewayMirrorTopics(
                    outbound=(
                        CAPACITY_TOPIC,
                        "onex.evt.omnibase-infra.gateway-heartbeat.v1",
                    ),  # onex-topic-allow: OMN-20867
                    inbound=(PR_STATE_TOPIC,),
                ),
                "canary": ModelGatewayCanaryConfig(
                    topic="onex.evt.omnibase-infra.gateway-canary.v1",  # onex-topic-allow: OMN-20867
                    cadence_seconds=30,
                    produce_deadline_seconds=8,
                    readback_deadline_seconds=12,
                ),
            }
        ),
        local_bus=_bus("local.invalid:9092"),
        cloud_bus=ModelKafkaEventBusConfig(
            bootstrap_servers="cloud.invalid:9092",
            environment="test",
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            security_protocol="SASL_SSL",
            sasl_mechanism="AWS_MSK_IAM",
            msk_region="us-east-1",
        ),
        lane_mirror_source_bus=config.lane_mirror_source_bus,
        lane_mirror_buses=config.lane_mirror_buses,
    )
    _, transports = gateway_forwarder.build_gateway_transports(config)
    try:
        for transport in transports.values():
            await transport.start()
        assert len(clients.producers) == len(transports) == 4
        assert all(p.started for p in clients.producers)
    finally:
        for transport in transports.values():
            await transport.close()
