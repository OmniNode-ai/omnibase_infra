# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
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

pytestmark = pytest.mark.integration

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
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


class _RecordingBus:
    def __init__(self) -> None:
        self.subscriptions: dict[str, Callable[[Any], Awaitable[None]]] = {}
        self.subscription_groups: dict[str, str] = {}
        self.published: list[_Message] = []

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
                key=b"tenant-key",
                value=envelope.model_dump_json().encode("utf-8"),
                headers={"traceparent": "00-test"},
            )
        )


def _config() -> ModelGatewayForwarderConfig:
    return ModelGatewayForwarderConfig(
        tenant_identity=ModelGatewayTenantIdentity(
            tenant_id=TENANT_ID,
            tenant_slug="acme",
            principal_id=f"tenant:{TENANT_ID}",
        ),
        cloud_bus=ModelGatewayCloudBusConfig(
            broker_provider_id="redpanda-dogfood",
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
        "envelope_id": str(uuid4()),
        "correlation_id": "corr-1",
        "causation_id": None,
        "event_type": "LlmInferenceResponse",
        "source_topic": OUTBOUND_TOPIC,
        "wire_topic": "",
        "canonical_topic": OUTBOUND_TOPIC,
        "payload": {"ok": True},
    }
    values.update(overrides)
    return ModelGatewayEnvelope(**values)


@pytest.mark.asyncio
async def test_gateway_forwarder_preserves_envelope_across_both_bus_legs() -> None:
    local_bus = _RecordingBus()
    cloud_bus = _RecordingBus()
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )

    await service.start()
    await local_bus.emit(OUTBOUND_TOPIC, _envelope())
    await cloud_bus.emit(
        WIRE_INBOUND_TOPIC,
        _envelope(
            event_type="DelegationInferenceRequest",
            source_topic=WIRE_INBOUND_TOPIC,
            wire_topic=WIRE_INBOUND_TOPIC,
            canonical_topic=INBOUND_TOPIC,
        ),
    )

    assert set(local_bus.subscriptions) == {OUTBOUND_TOPIC}
    assert set(cloud_bus.subscriptions) == {WIRE_INBOUND_TOPIC}
    assert local_bus.subscription_groups[OUTBOUND_TOPIC] == (
        "tenant-acme-gateway-forwarder-outbound"
    )
    assert cloud_bus.subscription_groups[WIRE_INBOUND_TOPIC] == (
        "tenant-acme-gateway-forwarder-inbound"
    )

    outbound = cloud_bus.published[0]
    assert outbound.topic == WIRE_OUTBOUND_TOPIC
    assert outbound.key == b"tenant-key"
    assert outbound.headers == {"traceparent": "00-test"}
    outbound_envelope = ModelGatewayEnvelope.model_validate_json(outbound.value)
    assert outbound_envelope.wire_topic == WIRE_OUTBOUND_TOPIC
    assert outbound_envelope.canonical_topic == OUTBOUND_TOPIC
    assert outbound_envelope.tenant_id == TENANT_ID

    inbound = local_bus.published[0]
    assert inbound.topic == INBOUND_TOPIC
    assert inbound.key == b"tenant-key"
    assert inbound.headers == {"traceparent": "00-test"}
    inbound_envelope = ModelGatewayEnvelope.model_validate_json(inbound.value)
    assert inbound_envelope.source_topic == WIRE_INBOUND_TOPIC
    assert inbound_envelope.canonical_topic == INBOUND_TOPIC
    assert inbound_envelope.tenant_slug == "acme"
