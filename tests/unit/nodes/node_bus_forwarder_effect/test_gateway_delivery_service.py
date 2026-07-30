# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.idempotency import StoreIdempotencyInmemory, StoreIdempotencySqlite
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
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

pytestmark = pytest.mark.asyncio

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
PRINCIPAL_ID = "t-33333333333333333333333333333333"
OUTBOUND_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
WIRE_OUTBOUND_TOPIC = f"tenant-acme.{OUTBOUND_TOPIC}"


class _RecordingBus:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.sent: list[tuple[str, bytes]] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.events.append("destination_ack")
        self.sent.append((topic, value))


class _BlockingBus(_RecordingBus):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.publish_calls = 0

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.publish_calls += 1
        self.entered.set()
        await self.release.wait()
        await super().publish(topic, key, value, headers)


class _Source:
    def __init__(self, events: list[str], *, fail_commit: bool = False) -> None:
        self.events = events
        self.fail_commit = fail_commit
        self.committed: list[object] = []
        self.nacked: list[object] = []

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ModelTransportMessage]:
        return []

    async def commit(self, message: object) -> None:
        self.events.append("source_commit")
        if self.fail_commit:
            raise RuntimeError("commit unavailable")
        self.committed.append(message)

    async def nack(self, message: object) -> None:
        self.events.append("source_nack")
        self.nacked.append(message)


class _RecordingStore(StoreIdempotencyInmemory):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def mark_processed(self, *args: object, **kwargs: object) -> None:
        self.events.append("durable_marker")
        await super().mark_processed(*args, **kwargs)  # type: ignore[arg-type]


def _config() -> ModelGatewayForwarderConfig:
    return ModelGatewayForwarderConfig(
        tenant_identity=ModelGatewayTenantIdentity(
            tenant_id=TENANT_ID,
            tenant_slug="acme",
            principal_id=PRINCIPAL_ID,
        ),
        cloud_bus=ModelGatewayCloudBusConfig(
            broker_provider_id=UUID("22222222-2222-2222-2222-222222222222"),
            cloud_broker_ref="gateway.cloud.kafka.broker",
            cloud_auth_ref="gateway.cloud.kafka.oauth",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            client_id_ref="gateway.cloud.kafka.oauth.client_id",
            client_secret_api_key_ref="infisical://gateway/redpanda-events",
        ),
        local_transport_flavor="containerized",
        dedupe_store_path=Path.cwd() / "gateway-test.sqlite3",
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
            outbound=(OUTBOUND_TOPIC,),
        ),
    )


def _message(envelope_id: UUID | None = None) -> ModelTransportMessage:
    identity = envelope_id or uuid4()
    envelope = ModelEventEnvelope[dict[str, object]](
        envelope_id=identity,
        correlation_id=identity,
        event_type="LlmInferenceResponse",
        payload={"ok": True},
        metadata=ModelEnvelopeMetadata(tags={}),
    )
    return ModelTransportMessage(
        topic=OUTBOUND_TOPIC,
        partition=0,
        offset=7,
        key=b"tenant-key",
        value=envelope.model_dump_json().encode("utf-8"),
        headers={"traceparent": b"00-test"},
        ack_token=(OUTBOUND_TOPIC, 0, 7),
    )


def _delivery(
    events: list[str],
    source: _Source,
    store: StoreIdempotencyInmemory | StoreIdempotencySqlite,
) -> tuple[NodeGatewayDelivery, _RecordingBus]:
    local_bus = _RecordingBus(events)
    cloud_bus = _RecordingBus(events)
    forwarder = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,  # type: ignore[arg-type]
        cloud_bus=cloud_bus,  # type: ignore[arg-type]
    )
    return (
        NodeGatewayDelivery(
            config=_config(),
            forwarder=forwarder,
            local_consumer=source,  # type: ignore[arg-type]
            cloud_consumer=source,  # type: ignore[arg-type]
            idempotency_store=store,
        ),
        cloud_bus,
    )


async def test_destination_ack_precedes_marker_and_source_commit() -> None:
    events: list[str] = []
    source = _Source(events)
    store = _RecordingStore(events)
    delivery, cloud_bus = _delivery(events, source, store)
    message = _message()

    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert events == ["destination_ack", "durable_marker", "source_commit"]
    assert len(cloud_bus.sent) == 1
    assert cloud_bus.sent[0][0] == WIRE_OUTBOUND_TOPIC
    assert source.committed == [message]


async def test_restart_after_commit_failure_suppresses_republish(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gateway-delivery.sqlite3"
    envelope_id = uuid4()
    message = _message(envelope_id)
    first_events: list[str] = []
    first_source = _Source(first_events, fail_commit=True)
    first_store = StoreIdempotencySqlite(path)
    await first_store.start()
    first_delivery, first_cloud = _delivery(first_events, first_source, first_store)

    with pytest.raises(RuntimeError, match="commit unavailable"):
        await first_delivery.deliver_message(
            "outbound",
            first_source,
            message,  # type: ignore[arg-type]
        )
    await first_store.close()

    restarted_events: list[str] = []
    restarted_source = _Source(restarted_events)
    restarted_store = StoreIdempotencySqlite(path)
    await restarted_store.start()
    restarted_delivery, restarted_cloud = _delivery(
        restarted_events,
        restarted_source,
        restarted_store,
    )
    await restarted_delivery.deliver_message(
        "outbound",
        restarted_source,
        message,  # type: ignore[arg-type]
    )

    assert len(first_cloud.sent) == 1
    assert restarted_cloud.sent == []
    assert restarted_events == ["source_commit"]
    assert restarted_source.committed == [message]


async def test_concurrent_duplicate_is_published_once() -> None:
    events: list[str] = []
    source = _Source(events)
    store = StoreIdempotencyInmemory()
    local_bus = _RecordingBus(events)
    cloud_bus = _BlockingBus(events)
    config = _config()
    delivery = NodeGatewayDelivery(
        config=config,
        forwarder=ServiceGatewayForwarder(
            config=config,
            local_bus=local_bus,  # type: ignore[arg-type]
            cloud_bus=cloud_bus,  # type: ignore[arg-type]
        ),
        local_consumer=source,  # type: ignore[arg-type]
        cloud_consumer=source,  # type: ignore[arg-type]
        idempotency_store=store,
    )
    message = _message()

    first = asyncio.create_task(
        delivery.deliver_message("outbound", source, message),  # type: ignore[arg-type]
    )
    await asyncio.wait_for(cloud_bus.entered.wait(), timeout=1)
    duplicate = asyncio.create_task(
        delivery.deliver_message("outbound", source, message),  # type: ignore[arg-type]
    )
    for _ in range(5):
        await asyncio.sleep(0)

    assert cloud_bus.publish_calls == 1
    cloud_bus.release.set()
    await asyncio.gather(first, duplicate)

    assert len(cloud_bus.sent) == 1
    assert source.committed == [message, message]


async def test_store_failure_nacks_without_destination_dispatch() -> None:
    events: list[str] = []
    source = _Source(events)
    unavailable_store = StoreIdempotencySqlite(Path("/unused/not-started.sqlite3"))
    delivery, cloud_bus = _delivery(events, source, unavailable_store)
    message = _message()

    with pytest.raises(RuntimeError, match="not started"):
        await delivery.deliver_message(
            "outbound",
            source,
            message,  # type: ignore[arg-type]
        )

    assert cloud_bus.sent == []
    assert source.committed == []
    assert source.nacked == [message]
    assert events == ["source_nack"]
