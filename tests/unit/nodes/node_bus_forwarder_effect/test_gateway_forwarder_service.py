# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.delegation.wire import ModelDelegationRequest
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
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
# OMN-14346: the command topic that actually starts the delegation FSM
# (omnimarket's node_delegation_orchestrator input_model is
# ModelDelegationRequest). Distinct from INBOUND_TOPIC
# (delegation-inference-request.v1), which is an intermediate intent topic.
DELEGATION_REQUEST_TOPIC = "onex.cmd.omnibase-infra.delegation-request.v1"
WIRE_DELEGATION_REQUEST_TOPIC = f"tenant-acme.{DELEGATION_REQUEST_TOPIC}"


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


async def test_inbound_payload_gets_verified_tenant_slug_not_forged_or_missing() -> (
    None
):
    """OMN-14345/OMN-14367: the verified tenant identity must reach the payload, not just the envelope shell.

    ``ModelGatewayForwarderConfig.tenant_identity`` is config-bound per forwarder
    instance -- that binding IS the trust anchor (OMN-12908/12911 per-tenant
    OAuth cloud-broker credentials). A payload-supplied tenant_id (forged or
    absent) must never survive into the republished message; the config-bound
    identity always wins and is never a self-asserted fallback.

    OMN-14367: the canonical stamped shape is the DNS-safe slug, not the raw
    tenant UUID -- ``omnibase_infra.shared.tenant_stamp.stamp_verified_tenant_slug``
    is the single source of truth both this producer and the runtime
    auto-wiring stamp route through, matching what the real consumer
    (omnimarket's ``ModelDelegateSkillRequest``, ``extra="forbid"``) expects.

    This also proves the republished message is actually consumable by the
    REAL ``omnibase_core.ModelEventEnvelope`` the local runtime dispatcher and
    every downstream node (e.g. ``node_llm_delegation_call_effect``) use to
    parse inbound bus messages -- not just a round-trip through the gateway's
    own ``ModelGatewayEnvelope``, which no consumer outside this node imports.
    Pre-fix this test fails: the raw ``ModelGatewayEnvelope`` JSON parses
    successfully as ``ModelEventEnvelope[dict]`` (no ``extra="forbid"``
    there), but ``consumer_view.payload`` is the FORGED/unstamped payload --
    the outer envelope's verified tenant_id is silently dropped as an
    ignored extra field rather than merged in.

    OMN-14346: the stamped value is the verified tenant SLUG
    (``identity.tenant_slug``), not the raw tenant UUID -- matching the
    canonical ``stamp_verified_tenant_slug`` shape (OMN-14367) that every
    ``extra="forbid"`` delegation payload contract expects ``tenant_id`` to
    carry (a DNS-safe slug, never a UUID).
    """
    local_bus = _MockGatewayBus()
    cloud_bus = _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    await service.start()

    forged_payload = {
        "prompt": "steal tenant data",
        "tenant_id": "evil-tenant-forged",
        "tenant_slug": "evil-tenant-forged-slug",
    }
    await cloud_bus.emit(
        WIRE_INBOUND_TOPIC,
        _envelope(
            event_type="DelegationInferenceRequest",
            source_topic=WIRE_INBOUND_TOPIC,
            wire_topic=WIRE_INBOUND_TOPIC,
            canonical_topic=INBOUND_TOPIC,
            payload=forged_payload,
        ),
    )

    assert len(local_bus.published) == 1
    published = local_bus.published[0]
    # The REAL consumer-side parse -- every local subscriber deserializes
    # inbound bus bytes as omnibase_core.ModelEventEnvelope[T], not the
    # gateway's own ModelGatewayEnvelope.
    consumer_view = ModelEventEnvelope[dict].model_validate_json(published.value)
    assert consumer_view.payload["tenant_id"] == "acme"
    assert consumer_view.payload["tenant_id"] != str(TENANT_ID)
    assert consumer_view.payload["tenant_id"] != "evil-tenant-forged"
    assert consumer_view.payload["prompt"] == "steal tenant data"
    # OMN-14367: canonical shape has no separate tenant_slug key -- tenant_id
    # IS the slug. A stray tenant_slug key would signal drift back to the
    # pre-reconciliation shape.
    assert "tenant_slug" not in consumer_view.payload


async def test_inbound_payload_stamp_carries_no_separate_tenant_slug_key() -> None:
    """OMN-14346/OMN-14367: the canonical stamp shape is ``tenant_id`` only.

    Prior to OMN-14346, ``HandlerConsumeInbound.consume_inbound`` hand-rolled
    the stamp with a separate ``tenant_slug`` key (drifted from the
    ``stamp_verified_tenant_slug`` producer the auto-wiring
    ``tenant_scoped_ingress`` stamp already used). An extra ``tenant_slug``
    key breaks every ``extra="forbid"`` delegation payload contract that does
    not declare that field (e.g. ``ModelDelegationRequest``). This pins the
    single canonical shape both producers must emit.
    """
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
            payload={"prompt": "hi"},
        ),
    )

    published = local_bus.published[0]
    consumer_view = ModelEventEnvelope[dict].model_validate_json(published.value)
    assert consumer_view.payload["tenant_id"] == "acme"
    assert "tenant_slug" not in consumer_view.payload


async def test_inbound_cross_tenant_forged_envelope_is_rejected_not_republished() -> (
    None
):
    """Regression pin: the existing outer-envelope tenant check still fires through the full service path.

    A message whose outer envelope claims a different tenant_id than the one
    this forwarder instance is config-bound to must be rejected before it
    ever reaches ``local_bus.publish`` -- the negative isolation proof this
    ticket's fix must not weaken.
    """
    local_bus = _MockGatewayBus()
    cloud_bus = _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    await service.start()

    other_tenant_id = uuid4()
    with pytest.raises(ValueError, match="tenant_id does not match"):
        await cloud_bus.emit(
            WIRE_INBOUND_TOPIC,
            _envelope(
                tenant_id=other_tenant_id,
                event_type="DelegationInferenceRequest",
                source_topic=WIRE_INBOUND_TOPIC,
                wire_topic=WIRE_INBOUND_TOPIC,
                canonical_topic=INBOUND_TOPIC,
                payload={"prompt": "cross-tenant forgery"},
            ),
        )

    assert local_bus.published == []


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


def _config_with_delegation_request() -> ModelGatewayForwarderConfig:
    """Config mirroring both the intermediate intent topic and the real
    delegation-request.v1 command topic that starts the delegation FSM
    (OMN-14208/OMN-14346)."""
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
            inbound=(INBOUND_TOPIC, DELEGATION_REQUEST_TOPIC),
            outbound=(OUTBOUND_TOPIC,),
        ),
    )


async def test_delegation_request_payload_stamp_survives_real_model_round_trip() -> (
    None
):
    """OMN-14346: the payload-stamp merge covers a real ``ModelDelegationRequest``
    payload on the actual command topic that starts the delegation FSM --
    not a generic ``{"prompt": ...}`` dict standing in for one.

    This is a DIFFERENT shape from the ``ModelInferenceIntent``-style dict
    fixtures used elsewhere in this file: ``ModelDelegationRequest``
    (``omnibase_core.models.delegation.wire``, re-exported unchanged as
    ``omnimarket.nodes.node_delegation_orchestrator.models.ModelDelegationRequest``,
    the real consumer's declared ``input_model`` for this exact topic) is
    ``extra="forbid"`` and types ``tenant_id`` as an optional DNS-safe slug
    string, never a UUID. Pre-OMN-14346 (hand-rolled stamp: UUID
    ``tenant_id`` + extra ``tenant_slug`` key) this payload would round-trip
    back into ``ModelDelegationRequest.model_validate()`` carrying a
    non-slug UUID string in ``tenant_id`` and an undeclared ``tenant_slug``
    key that ``extra="forbid"`` rejects outright -- a real cross-repo seam
    break the generic-dict tests in this file cannot see because they never
    reconstruct the typed consumer model.
    """
    local_bus = _MockGatewayBus()
    cloud_bus = _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config_with_delegation_request(),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    await service.start()

    request_correlation_id = uuid4()
    delegation_request = ModelDelegationRequest(
        prompt="review this diff for correctness",
        task_type="code_review",
        correlation_id=request_correlation_id,
        emitted_at=datetime.now(UTC),
        tenant_id="forged-tenant-slug",
    )
    await cloud_bus.emit(
        WIRE_DELEGATION_REQUEST_TOPIC,
        _envelope(
            event_type="DelegationRequest",
            correlation_id=request_correlation_id,
            source_topic=WIRE_DELEGATION_REQUEST_TOPIC,
            wire_topic=WIRE_DELEGATION_REQUEST_TOPIC,
            canonical_topic=DELEGATION_REQUEST_TOPIC,
            payload=delegation_request.model_dump(mode="json"),
        ),
    )

    published = [
        message
        for message in local_bus.published
        if message.topic == DELEGATION_REQUEST_TOPIC
    ]
    assert len(published) == 1
    consumer_view = ModelEventEnvelope[dict].model_validate_json(published[0].value)

    # The republished payload must reconstruct as the REAL consumer model
    # without raising -- this is the actual seam the gateway must not break.
    reconstructed = ModelDelegationRequest.model_validate(consumer_view.payload)
    assert reconstructed.tenant_id == "acme"
    assert reconstructed.tenant_id != "forged-tenant-slug"
    assert reconstructed.prompt == "review this diff for correctness"
    assert reconstructed.task_type == "code_review"
    assert reconstructed.correlation_id == request_correlation_id
    assert "tenant_slug" not in consumer_view.payload
