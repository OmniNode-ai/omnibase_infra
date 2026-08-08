# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15740 seam test: ServiceGatewayForwarder must actually call the
contract-declared COMPUTE handlers, not re-derive the tenant-prefix transform
inline.

Prior to this fix, ``_prepare_inbound``/``_prepare_outbound`` called
``prefix_topic``/``strip_topic_prefix`` directly and never touched
``HandlerForwardOutbound``/``HandlerConsumeInbound`` -- the handlers had zero
production call sites (see
``docs/design/2026-08-08-gateway-node-architecture-lift.md`` axis #3). This
test spies on the REAL handler instances the service constructs and wires
internally (``wraps=``, not a replacement mock) so a regression that reverts
to the inline duplicate transform is caught: the spy's call count would drop
to zero while the assertions on the published output would still pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_bus_forwarder_effect.handlers import (
    HandlerConsumeInbound,
    HandlerForwardOutbound,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCloudBusConfig,
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


class _MockGatewayBus:
    def __init__(self) -> None:
        self.published: list[_Message] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.published.append(_Message(topic, key, value, headers))

    def message(
        self,
        topic: str,
        envelope: ModelEventEnvelope[dict[str, object]],
    ) -> _Message:
        return _Message(
            topic=topic,
            key=b"key-1",
            value=envelope.model_dump_json().encode("utf-8"),
            headers=None,
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
        dedupe_store_path=Path.cwd() / "gateway-test-seam.sqlite3",
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=(INBOUND_TOPIC,),
            outbound=(OUTBOUND_TOPIC,),
        ),
    )


def _envelope(**overrides: object) -> ModelEventEnvelope[dict[str, object]]:
    values: dict[str, object] = {
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


async def test_forward_outbound_message_actually_calls_handler_forward_outbound() -> (
    None
):
    """The service's own ``HandlerForwardOutbound`` instance is invoked, not bypassed."""
    local_bus = _MockGatewayBus()
    cloud_bus = _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    # Spy on the REAL handler instance the service constructed internally --
    # `wraps=` preserves the real implementation, it does not replace it.
    with patch.object(
        service._forward_outbound_handler,
        "forward_outbound",
        wraps=service._forward_outbound_handler.forward_outbound,
    ) as spy:
        await service.forward_outbound_message(
            local_bus.message(OUTBOUND_TOPIC, _envelope())
        )

    assert spy.call_count == 1
    called_envelope = spy.call_args.args[0]
    assert called_envelope.canonical_topic == OUTBOUND_TOPIC
    assert called_envelope.tenant_id == TENANT_ID

    assert len(cloud_bus.published) == 1
    published = cloud_bus.published[0]
    assert published.topic == WIRE_OUTBOUND_TOPIC
    forwarded = ModelEventEnvelope[dict[str, object]].model_validate_json(
        published.value
    )
    assert forwarded.metadata.tags["gateway_wire_topic"] == WIRE_OUTBOUND_TOPIC


async def test_consume_inbound_message_actually_calls_handler_consume_inbound() -> None:
    """The service's own ``HandlerConsumeInbound`` instance is invoked, not bypassed."""
    local_bus = _MockGatewayBus()
    cloud_bus = _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    with patch.object(
        service._consume_inbound_handler,
        "consume_inbound",
        wraps=service._consume_inbound_handler.consume_inbound,
    ) as spy:
        await service.consume_inbound_message(
            cloud_bus.message(
                WIRE_INBOUND_TOPIC,
                _envelope(event_type="DelegationInferenceRequest"),
            )
        )

    assert spy.call_count == 1
    called_envelope = spy.call_args.args[0]
    assert called_envelope.wire_topic == WIRE_INBOUND_TOPIC
    assert called_envelope.canonical_topic == INBOUND_TOPIC

    assert len(local_bus.published) == 1
    published = local_bus.published[0]
    assert published.topic == INBOUND_TOPIC
    forwarded = ModelEventEnvelope[dict[str, object]].model_validate_json(
        published.value
    )
    assert forwarded.payload["tenant_id"] == "acme"


async def test_service_wires_the_real_handler_types_not_a_stand_in() -> None:
    """Belt-and-suspenders: the service constructs the actual contract handlers."""
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=_MockGatewayBus(),
        cloud_bus=_MockGatewayBus(),
    )
    assert isinstance(service._forward_outbound_handler, HandlerForwardOutbound)
    assert isinstance(service._consume_inbound_handler, HandlerConsumeInbound)
