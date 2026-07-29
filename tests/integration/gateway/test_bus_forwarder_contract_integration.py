# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCloudBusConfig,
    ModelGatewayForwarderConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ServiceGatewayForwarder,
)

pytestmark = pytest.mark.integration

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
BROKER_PROVIDER_ID = UUID("22222222-2222-2222-2222-222222222222")
PRINCIPAL_ID = "t-33333333333333333333333333333333"
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

    async def emit(
        self,
        topic: str,
        envelope: ModelEventEnvelope[dict[str, object]],
    ) -> None:
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


def _envelope(**overrides: object) -> ModelEventEnvelope[dict[str, object]]:
    values = {
        "envelope_id": uuid4(),
        "correlation_id": CORRELATION_ID,
        "event_type": "LlmInferenceResponse",
        "payload": {"ok": True},
        "metadata": ModelEnvelopeMetadata(
            tags={
                "source_tenant_id": str(TENANT_ID),
                "source_tenant_principal_id": PRINCIPAL_ID,
            }
        ),
    }
    values.update(overrides)
    return ModelEventEnvelope[dict[str, object]](**values)


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
    outbound_envelope = ModelEventEnvelope[dict[str, object]].model_validate_json(
        outbound.value
    )
    assert outbound_envelope.metadata.tags["gateway_wire_topic"] == (
        WIRE_OUTBOUND_TOPIC
    )
    assert outbound_envelope.metadata.tags["gateway_canonical_topic"] == OUTBOUND_TOPIC
    assert outbound_envelope.metadata.tags["source_tenant_id"] == str(TENANT_ID)

    inbound = local_bus.published[0]
    assert inbound.topic == INBOUND_TOPIC
    assert inbound.key == b"tenant-key"
    assert inbound.headers == {"traceparent": "00-test"}
    inbound_envelope = ModelEventEnvelope[dict[str, object]].model_validate_json(
        inbound.value
    )
    assert inbound_envelope.metadata.tags["gateway_wire_topic"] == WIRE_INBOUND_TOPIC
    assert inbound_envelope.metadata.tags["gateway_canonical_topic"] == INBOUND_TOPIC
    assert inbound_envelope.payload["tenant_id"] == "acme"
