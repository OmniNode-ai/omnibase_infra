# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for OMN-13546: HandlerForwardOutbound and HandlerConsumeInbound handle() dispatch entrypoints.

Confirms that both handlers expose callable handle() methods and return a valid
ModelHandlerOutput so the runtime dispatcher does not raise AttributeError on dispatch.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_infra.nodes.node_bus_forwarder_effect.handlers.handler_consume_inbound import (
    HandlerConsumeInbound,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.handlers.handler_forward_outbound import (
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
INBOUND_TOPIC = "onex.cmd.omnibase-infra.delegation-inference-request.v1"
OUTBOUND_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
WIRE_INBOUND_TOPIC = f"tenant-acme.{INBOUND_TOPIC}"
WIRE_OUTBOUND_TOPIC = f"tenant-acme.{OUTBOUND_TOPIC}"


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


def _outbound_gateway_envelope() -> ModelGatewayEnvelope:
    return ModelGatewayEnvelope(
        tenant_id=TENANT_ID,
        tenant_slug="acme",
        envelope_id=uuid4(),
        correlation_id=uuid4(),
        causation_id="cause-1",
        event_type="LlmInferenceResponse",
        source_topic=OUTBOUND_TOPIC,
        wire_topic="",
        canonical_topic=OUTBOUND_TOPIC,
        payload={"ok": True},
    )


def _inbound_gateway_envelope() -> ModelGatewayEnvelope:
    return ModelGatewayEnvelope(
        tenant_id=TENANT_ID,
        tenant_slug="acme",
        envelope_id=uuid4(),
        correlation_id=uuid4(),
        causation_id="cause-2",
        event_type="DelegationInferenceRequest",
        source_topic=WIRE_INBOUND_TOPIC,
        wire_topic=WIRE_INBOUND_TOPIC,
        canonical_topic=INBOUND_TOPIC,
        payload={"request": True},
    )


def _make_dispatch_envelope(gateway_envelope: ModelGatewayEnvelope) -> SimpleNamespace:
    """Minimal stand-in for ModelEventEnvelope as seen by the auto-wiring engine."""
    return SimpleNamespace(
        envelope_id=gateway_envelope.envelope_id,
        correlation_id=gateway_envelope.correlation_id,
        payload=gateway_envelope.model_dump(),
    )


@pytest.mark.asyncio
async def test_handler_forward_outbound_handle_returns_model_handler_output() -> None:
    """HandlerForwardOutbound.handle() returns a valid ModelHandlerOutput (no AttributeError)."""
    handler = HandlerForwardOutbound(config=_config())
    gw_envelope = _outbound_gateway_envelope()
    dispatch_envelope = _make_dispatch_envelope(gw_envelope)

    result = await handler.handle(dispatch_envelope)

    assert isinstance(result, ModelHandlerOutput)
    assert result.result is not None
    assert isinstance(result.result, ModelGatewayEnvelope)
    assert result.result.wire_topic == WIRE_OUTBOUND_TOPIC
    assert result.result.source_topic == OUTBOUND_TOPIC
    assert result.correlation_id == gw_envelope.correlation_id


@pytest.mark.asyncio
async def test_handler_consume_inbound_handle_returns_model_handler_output() -> None:
    """HandlerConsumeInbound.handle() returns a valid ModelHandlerOutput (no AttributeError)."""
    handler = HandlerConsumeInbound(config=_config())
    gw_envelope = _inbound_gateway_envelope()
    dispatch_envelope = _make_dispatch_envelope(gw_envelope)

    result = await handler.handle(dispatch_envelope)

    assert isinstance(result, ModelHandlerOutput)
    assert result.result is not None
    assert isinstance(result.result, ModelGatewayEnvelope)
    assert result.result.canonical_topic == INBOUND_TOPIC
    assert result.result.source_topic == WIRE_INBOUND_TOPIC
    assert result.correlation_id == gw_envelope.correlation_id


def test_handler_forward_outbound_exposes_callable_handle() -> None:
    """HandlerForwardOutbound.handle is callable (satisfies ProtocolHandleable check)."""
    handler = HandlerForwardOutbound(config=_config())
    assert callable(getattr(handler, "handle", None))


def test_handler_consume_inbound_exposes_callable_handle() -> None:
    """HandlerConsumeInbound.handle is callable (satisfies ProtocolHandleable check)."""
    handler = HandlerConsumeInbound(config=_config())
    assert callable(getattr(handler, "handle", None))
