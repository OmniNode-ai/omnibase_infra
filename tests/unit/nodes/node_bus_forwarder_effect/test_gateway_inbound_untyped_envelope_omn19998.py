# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: an inbound record with no ``event_type`` is a refusal, not a crash.

``ModelEventEnvelope.event_type`` is optional; ``ModelGatewayEnvelope.event_type``
is required. Measured on the lab gateway forwarder on 2026-10-16: one cloud
record with no ``event_type`` raised a pydantic ``ValidationError`` out of
``_prepare_inbound`` from 13:59:22Z, 91 times. That is neither a decode failure
nor a ``GatewayRecordRefusedError``, so the delivery node nacked and re-raised
it, the whole delivery loop restarted every cycle, and from 14:46:03Z the
inbound leg delivered nothing at all. The verdict is the same on every
redelivery of those bytes, so it belongs on the per-record refusal path:
quarantine and commit past it.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayForwarderConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    GatewayRecordRefusedError,
    ServiceGatewayForwarder,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
PRINCIPAL_ID = "t-33333333333333333333333333333333"
INBOUND_TOPIC = "onex.cmd.omnibase-infra.delegation-request.v1"
WIRE_INBOUND_TOPIC = f"tenant-acme.{INBOUND_TOPIC}"


class _Message:
    def __init__(self, topic: str, value: bytes) -> None:
        self.topic = topic
        self.key = b"key-1"
        self.value = value
        self.headers = None


class _RecordingBus:
    def __init__(self) -> None:
        self.published: list[str] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.published.append(topic)


def _service(local_bus: _RecordingBus) -> ServiceGatewayForwarder:
    config = ModelGatewayForwarderConfig(
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
        dedupe_store_path=Path.cwd() / "gateway-test.sqlite3",
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
    return ServiceGatewayForwarder(
        config=config, local_bus=local_bus, cloud_bus=_RecordingBus()
    )


def _inbound(event_type: str | None) -> _Message:
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
    return _Message(WIRE_INBOUND_TOPIC, envelope.model_dump_json().encode("utf-8"))


async def test_untyped_inbound_envelope_is_refused_on_validate() -> None:
    local_bus = _RecordingBus()
    service = _service(local_bus)

    with pytest.raises(GatewayRecordRefusedError, match="event_type"):
        service.validate_inbound_message(_inbound(None))

    assert local_bus.published == []


async def test_untyped_inbound_envelope_is_refused_on_consume() -> None:
    local_bus = _RecordingBus()
    service = _service(local_bus)

    with pytest.raises(GatewayRecordRefusedError, match="event_type"):
        await service.consume_inbound_message(_inbound(None))

    assert local_bus.published == []


async def test_typed_inbound_envelope_still_crosses() -> None:
    """Positive control: the same record with an event_type is carried."""
    local_bus = _RecordingBus()
    service = _service(local_bus)

    service.validate_inbound_message(_inbound("DelegationRequest"))
    await service.consume_inbound_message(_inbound("DelegationRequest"))

    assert local_bus.published == [INBOUND_TOPIC]
