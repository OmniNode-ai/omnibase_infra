# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: quarantine untyped inbound records and recover failed consumers."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from aiokafka.errors import KafkaConnectionError
from aiokafka.structs import TopicPartition

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.event_bus import kafka_transport
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.topic_constants import get_dlq_topic_for_original
from omnibase_infra.idempotency.store_sqlite import StoreIdempotencySqlite
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayForwarderConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_delivery import (
    NodeGatewayDelivery,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ServiceGatewayForwarder,
)
from omnibase_infra.topics.topic_namespace import TOPIC_NAMESPACE_ENV_VAR

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
PRINCIPAL_ID = "t-33333333333333333333333333333333"
INBOUND_TOPIC = "onex.cmd.omnibase-infra.delegation-request.v1"
WIRE_INBOUND_TOPIC = f"tenant-acme.{INBOUND_TOPIC}"


class _RecordingBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes | None, bytes]] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.published.append((topic, key, value))


class _MemoryConsumer:
    def __init__(self) -> None:
        self.committed: list[object] = []
        self.nacked: list[object] = []
        self.sent: list[tuple[str, bytes | None, bytes, dict[str, bytes]]] = []

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> list[ModelTransportMessage]:
        return []

    async def commit(self, message: object) -> None:
        self.committed.append(message)

    async def nack(self, message: object) -> None:
        self.nacked.append(message)

    async def send(
        self, topic: str, key: bytes | None, value: bytes, headers: dict[str, bytes]
    ) -> None:
        self.sent.append((topic, key, value, headers))


def _config(store_path: Path) -> ModelGatewayForwarderConfig:
    return ModelGatewayForwarderConfig(
        tenant_identity=ModelGatewayTenantIdentity(
            tenant_id=TENANT_ID,
            tenant_slug="acme",
            principal_id=PRINCIPAL_ID,
        ),
        cloud_bus=ModelGatewayCloudBusConfig(
            broker_provider_id=UUID("22222222-2222-2222-2222-222222222222"),
            cloud_broker_ref="gateway.cloud.kafka.broker",
            cloud_auth_ref="gateway.cloud.kafka.msk_iam",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            msk_region_ref="gateway.cloud.kafka.msk_region",
            sasl_mechanism="AWS_MSK_IAM",
        ),
        local_transport_flavor="containerized",
        dedupe_store_path=store_path,
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=(INBOUND_TOPIC,),
            outbound=("onex.evt.omnibase-infra.inference-response.v1",),
        ),
        canary=ModelGatewayCanaryConfig(
            topic="onex.evt.omnibase-infra.gateway-canary.v1",
            cadence_seconds=30,
            produce_deadline_seconds=8,
            readback_deadline_seconds=12,
        ),
    )


def _inbound(event_type: str | None, *, offset: int) -> ModelTransportMessage:
    envelope = ModelEventEnvelope[dict[str, object]](
        envelope_id=uuid4(),
        correlation_id=uuid4(),
        event_type=event_type,
        payload={"prompt": "hello"},
        metadata=ModelEnvelopeMetadata(
            tags={
                "source_tenant_id": str(TENANT_ID),
                "source_tenant_principal_id": PRINCIPAL_ID,
            }
        ),
    )
    return ModelTransportMessage(
        topic=WIRE_INBOUND_TOPIC,
        partition=0,
        offset=offset,
        key=b"key-1",
        value=envelope.model_dump_json().encode("utf-8"),
        headers={},
        ack_token=(TopicPartition(WIRE_INBOUND_TOPIC, 0), offset),
    )


async def test_untyped_record_is_quarantined_without_blocking_typed_record(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "gateway-test.sqlite3")
    local_bus = _RecordingBus()
    cloud_consumer = _MemoryConsumer()
    forwarder = ServiceGatewayForwarder(
        config=config, local_bus=local_bus, cloud_bus=_RecordingBus()
    )
    store = StoreIdempotencySqlite(config.dedupe_store_path)
    delivery = NodeGatewayDelivery(
        config=config,
        forwarder=forwarder,
        local_consumer=_MemoryConsumer(),
        cloud_consumer=cloud_consumer,
        idempotency_store=store,
    )
    untyped_record = _inbound(None, offset=10)
    typed_record = _inbound("DelegationRequest", offset=11)
    await store.start()
    try:
        await delivery.deliver_messages(
            "inbound", cloud_consumer, [untyped_record, typed_record]
        )

        assert cloud_consumer.committed == [untyped_record, typed_record]
        assert cloud_consumer.nacked == []
        assert len(cloud_consumer.sent) == 1
        topic, key, value, headers = cloud_consumer.sent[0]
        expected_dlq = get_dlq_topic_for_original(WIRE_INBOUND_TOPIC)
        assert expected_dlq is not None
        assert topic == expected_dlq
        assert key == untyped_record.key
        assert headers == {"original_topic": WIRE_INBOUND_TOPIC.encode("utf-8")}
        quarantine = json.loads(value)
        assert quarantine["original_offset"] == untyped_record.offset
        assert quarantine["failure_class"] == "gateway_refused_record"
        assert quarantine["error_type"] == "GatewayRecordRefusedError"

        assert len(local_bus.published) == 1
        local_topic, local_key, local_value = local_bus.published[0]
        assert local_topic == INBOUND_TOPIC
        assert local_key == typed_record.key
        published = ModelEventEnvelope[dict[str, object]].model_validate_json(
            local_value
        )
        typed_envelope = forwarder.decode_message(typed_record)
        assert published.envelope_id == typed_envelope.envelope_id
        assert published.event_type == "DelegationRequest"
        domain = f"gateway:{config.tenant_identity.tenant_slug}"
        assert await store.is_processed(typed_envelope.envelope_id, domain=domain)
        assert not await store.is_processed(
            forwarder.decode_message(untyped_record).envelope_id, domain=domain
        )
    finally:
        await store.close()


@dataclass
class _KafkaRecord:
    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes
    headers: tuple[tuple[str, bytes], ...] = ()


@dataclass
class _Broker:
    fail_next_start: BaseException | None = None
    consumers: list[_FakeConsumer] = field(default_factory=list)
    records: list[_KafkaRecord] = field(default_factory=list)


class _FakeConsumer:
    """Like aiokafka, fetching before start finishes has no coordinator."""

    broker: _Broker

    def __init__(self, *topics: str, **_kwargs: object) -> None:
        self.topics = tuple(topics)
        self.started = False
        self.stopped = False
        self.broker.consumers.append(self)

    async def start(self) -> None:
        failure = self.broker.fail_next_start
        if failure is not None:
            self.broker.fail_next_start = None
            raise failure
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def getmany(
        self, timeout_ms: int = 0, max_records: int | None = None
    ) -> dict[TopicPartition, list[_KafkaRecord]]:
        if not self.started:
            raise AttributeError("'NoneType' object has no attribute 'check_errors'")
        assert not self.stopped
        records = self.broker.records[:max_records]
        del self.broker.records[: len(records)]
        return {TopicPartition(self.topics[0], 0): records} if records else {}

    def assignment(self) -> set[TopicPartition]:
        return {TopicPartition(t, 0) for t in self.topics} if self.started else set()


class _FakeProducer:
    def __init__(self, **_kwargs: object) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.mark.parametrize(
    "failure",
    [KafkaConnectionError("bootstrap timed out"), asyncio.CancelledError()],
    ids=["start-raised", "start-cancelled"],
)
async def test_failed_consumer_restart_recovers_records_on_next_poll(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    broker = _Broker()
    consumer_cls = type("_BoundFakeConsumer", (_FakeConsumer,), {"broker": broker})
    monkeypatch.setattr(kafka_transport, "AIOKafkaConsumer", consumer_cls)
    monkeypatch.setattr(kafka_transport, "AIOKafkaProducer", _FakeProducer)
    transport = KafkaTransport(
        config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092"),
        group="tenant-t-gateway-forwarder-inbound",
        topics=(WIRE_INBOUND_TOPIC,),
    )
    try:
        await transport.start()
        original = broker.consumers[0]
        broker.fail_next_start = failure
        record = _inbound("DelegationRequest", offset=12)
        broker.records.append(
            _KafkaRecord(
                topic=record.topic,
                partition=record.partition,
                offset=record.offset,
                key=record.key,
                value=record.value,
            )
        )

        with pytest.raises(type(failure)):
            await transport.restart_consumer()

        assert original.stopped
        failed = broker.consumers[-1]
        assert not failed.started
        assert failed.stopped
        assert transport._consumer is None
        assert not transport.has_group_membership()

        assert await transport.poll(max_messages=1, timeout_ms=1) == [record]
        assert len(broker.consumers) == 3
        fresh = broker.consumers[-1]
        assert fresh is not failed
        assert fresh.started
        assert not fresh.stopped
        assert fresh.topics == (WIRE_INBOUND_TOPIC,)
        assert transport.has_group_membership()
    finally:
        await transport.close()
