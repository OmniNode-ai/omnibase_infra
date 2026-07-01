# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest

from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCloudBusConfig,
    ModelGatewayEnvelope,
    ModelGatewayForwarderConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ServiceGatewayForwarder,
)

pytestmark = pytest.mark.asyncio

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
BROKER_PROVIDER_ID = UUID("22222222-2222-2222-2222-222222222222")
PRINCIPAL_ID = UUID("33333333-3333-3333-3333-333333333333")
CORRELATION_ID = UUID("44444444-4444-4444-4444-444444444444")
INBOUND_TOPIC = "onex.cmd.omnibase-infra.delegation-inference-request.v1"
OUTBOUND_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
WIRE_INBOUND_TOPIC = f"tenant-acme.{INBOUND_TOPIC}"
WIRE_OUTBOUND_TOPIC = f"tenant-acme.{OUTBOUND_TOPIC}"


@dataclass(frozen=True)
class _Message:
    topic: str
    key: bytes | None
    value: bytes
    headers: object | None = None


class _MockGatewayBus:
    def __init__(self) -> None:
        self.subscriptions: dict[str, Callable[[Any], Awaitable[None]]] = {}
        self.subscription_groups: dict[str, str] = {}
        self.published: list[_Message] = []
        self.unsubscribe_count = 0

    async def subscribe(
        self,
        topic: str,
        node_identity: object | None = None,
        on_message: Callable[[Any], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        **_kwargs: object,
    ) -> Callable[[], Awaitable[None]]:
        assert on_message is not None
        assert group_id is not None
        self.subscriptions[topic] = on_message
        self.subscription_groups[topic] = group_id

        async def _unsubscribe() -> None:
            self.unsubscribe_count += 1
            self.subscriptions.pop(topic, None)
            self.subscription_groups.pop(topic, None)

        return _unsubscribe

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.published.append(_Message(topic, key, value, headers))

    async def emit(self, topic: str, envelope: ModelGatewayEnvelope) -> None:
        await self.subscriptions[topic](
            _Message(
                topic=topic,
                key=b"key-1",
                value=envelope.model_dump_json().encode("utf-8"),
                headers={"trace": "preserved"},
            )
        )


def _config() -> ModelGatewayForwarderConfig:
    return ModelGatewayForwarderConfig(
        tenant_identity=ModelGatewayTenantIdentity(
            tenant_id=TENANT_ID,
            tenant_slug="acme",
            principal_id=PRINCIPAL_ID,
        ),
        cloud_bus=ModelGatewayCloudBusConfig(
            broker_provider_id=BROKER_PROVIDER_ID,
            cloud_broker_ref="gateway.cloud.kafka.broker",
            cloud_auth_ref="gateway.cloud.kafka.oauth",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            client_id_ref="gateway.cloud.kafka.oauth.client_id",
            client_secret_api_key_ref="infisical://gateway/redpanda-events",
        ),
        local_transport_flavor="containerized",
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=(INBOUND_TOPIC,),
            outbound=(OUTBOUND_TOPIC,),
        ),
    )


def _envelope(**overrides: object) -> ModelGatewayEnvelope:
    values = {
        "tenant_id": TENANT_ID,
        "tenant_slug": "acme",
        "envelope_id": uuid4(),
        "correlation_id": CORRELATION_ID,
        "causation_id": "cause-1",
        "event_type": "LlmInferenceResponse",
        "source_topic": OUTBOUND_TOPIC,
        "wire_topic": "",
        "canonical_topic": OUTBOUND_TOPIC,
        "payload": {"ok": True},
    }
    values.update(overrides)
    return ModelGatewayEnvelope(**values)


async def test_start_subscribes_declared_topics_only() -> None:
    local_bus = _MockGatewayBus()
    cloud_bus = _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )

    await service.start()

    assert set(local_bus.subscriptions) == {OUTBOUND_TOPIC}
    assert set(cloud_bus.subscriptions) == {WIRE_INBOUND_TOPIC}
    assert local_bus.subscription_groups[OUTBOUND_TOPIC].endswith("-outbound")
    assert cloud_bus.subscription_groups[WIRE_INBOUND_TOPIC].endswith("-inbound")


async def test_outbound_local_message_is_published_to_cloud_wire_topic() -> None:
    local_bus = _MockGatewayBus()
    cloud_bus = _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    await service.start()

    await local_bus.emit(OUTBOUND_TOPIC, _envelope())

    assert len(cloud_bus.published) == 1
    published = cloud_bus.published[0]
    assert published.topic == WIRE_OUTBOUND_TOPIC
    assert published.key == b"key-1"
    assert published.headers == {"trace": "preserved"}
    forwarded = ModelGatewayEnvelope.model_validate_json(published.value)
    assert forwarded.wire_topic == WIRE_OUTBOUND_TOPIC
    assert forwarded.canonical_topic == OUTBOUND_TOPIC
    assert forwarded.correlation_id == CORRELATION_ID
    assert forwarded.event_type == "LlmInferenceResponse"


async def test_inbound_cloud_message_is_published_to_local_canonical_topic() -> None:
    local_bus = _MockGatewayBus()
    cloud_bus = _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    await service.start()

    await cloud_bus.emit(
        WIRE_INBOUND_TOPIC,
        _envelope(
            event_type="DelegationInferenceRequest",
            source_topic=WIRE_INBOUND_TOPIC,
            wire_topic=WIRE_INBOUND_TOPIC,
            canonical_topic=INBOUND_TOPIC,
        ),
    )

    assert len(local_bus.published) == 1
    published = local_bus.published[0]
    assert published.topic == INBOUND_TOPIC
    forwarded = ModelGatewayEnvelope.model_validate_json(published.value)
    assert forwarded.source_topic == WIRE_INBOUND_TOPIC
    assert forwarded.canonical_topic == INBOUND_TOPIC
    assert forwarded.correlation_id == CORRELATION_ID
    assert forwarded.event_type == "DelegationInferenceRequest"


async def test_outbound_rejects_undeclared_topic_before_cloud_publish() -> None:
    local_bus = _MockGatewayBus()
    cloud_bus = _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    await service.start()

    with pytest.raises(ValueError, match="not declared"):
        await local_bus.emit(
            OUTBOUND_TOPIC,
            _envelope(canonical_topic="onex.evt.omnibase-infra.not-declared.v1"),
        )

    assert cloud_bus.published == []


async def test_stop_unsubscribes_both_legs() -> None:
    local_bus = _MockGatewayBus()
    cloud_bus = _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    await service.start()

    await service.stop()

    assert local_bus.subscriptions == {}
    assert cloud_bus.subscriptions == {}
    assert local_bus.unsubscribe_count == 1
    assert cloud_bus.unsubscribe_count == 1
