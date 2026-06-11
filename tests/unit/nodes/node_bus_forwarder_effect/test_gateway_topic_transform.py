# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from omnibase_infra.nodes.node_bus_forwarder_effect.handlers import (
    HandlerConsumeInbound,
    HandlerForwardOutbound,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCloudBusConfig,
    ModelGatewayEnvelope,
    ModelGatewayForwarderConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
BROKER_PROVIDER_ID = UUID("22222222-2222-2222-2222-222222222222")
PRINCIPAL_ID = UUID("33333333-3333-3333-3333-333333333333")
CORRELATION_ID = UUID("44444444-4444-4444-4444-444444444444")
INBOUND_TOPIC = "onex.cmd.omnibase-infra.delegation-inference-request.v1"
OUTBOUND_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"


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
        "causation_id": None,
        "event_type": "LlmInferenceResponse",
        "source_topic": OUTBOUND_TOPIC,
        "wire_topic": "",
        "canonical_topic": OUTBOUND_TOPIC,
        "payload": {"ok": True},
    }
    values.update(overrides)
    return ModelGatewayEnvelope(**values)


def test_outbound_adds_tenant_wire_prefix_and_preserves_event_fields() -> None:
    envelope = _envelope()

    result = HandlerForwardOutbound(_config()).forward_outbound(envelope)

    assert result.wire_topic == f"tenant-acme.{OUTBOUND_TOPIC}"
    assert result.canonical_topic == OUTBOUND_TOPIC
    assert result.event_type == envelope.event_type
    assert result.correlation_id == envelope.correlation_id


def test_outbound_rejects_undeclared_topic() -> None:
    envelope = _envelope(canonical_topic="onex.evt.omnibase-infra.not-declared.v1")

    with pytest.raises(ValueError, match="not declared"):
        HandlerForwardOutbound(_config()).forward_outbound(envelope)


def test_inbound_strips_tenant_prefix_to_bare_contract_topic() -> None:
    envelope = _envelope(
        event_type="DelegationInferenceRequest",
        source_topic=f"tenant-acme.{INBOUND_TOPIC}",
        wire_topic=f"tenant-acme.{INBOUND_TOPIC}",
        canonical_topic=INBOUND_TOPIC,
    )

    result = HandlerConsumeInbound(_config()).consume_inbound(envelope)

    assert result.source_topic == f"tenant-acme.{INBOUND_TOPIC}"
    assert result.canonical_topic == INBOUND_TOPIC
    assert result.event_type == envelope.event_type
    assert result.correlation_id == envelope.correlation_id


def test_inbound_rejects_cross_tenant_wire_prefix() -> None:
    envelope = _envelope(
        source_topic=f"tenant-other.{INBOUND_TOPIC}",
        wire_topic=f"tenant-other.{INBOUND_TOPIC}",
        canonical_topic=INBOUND_TOPIC,
    )

    with pytest.raises(ValueError, match="tenant prefix"):
        HandlerConsumeInbound(_config()).consume_inbound(envelope)


def test_inbound_rejects_envelope_tenant_mismatch() -> None:
    envelope = _envelope(
        tenant_id=UUID("22222222-2222-2222-2222-222222222222"),
        source_topic=f"tenant-acme.{INBOUND_TOPIC}",
        wire_topic=f"tenant-acme.{INBOUND_TOPIC}",
        canonical_topic=INBOUND_TOPIC,
    )

    with pytest.raises(ValueError, match="tenant_id"):
        HandlerConsumeInbound(_config()).consume_inbound(envelope)
