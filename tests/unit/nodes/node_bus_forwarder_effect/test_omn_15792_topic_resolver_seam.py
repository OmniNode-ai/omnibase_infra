# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary seam tests for the single runtime topic resolver (OMN-15792).

2026-08-09 operator addressing ruling: physical topic addressing is a RUNTIME
concern resolved from contract-declared canonical topic + optional tenant
execution context through ONE shared resolver, consulted by BOTH the publish
path and the subscribe/dispatch path. ``tenant_wire_topic`` resolution
(``resolve_physical_topic`` / ``resolve_tenant_from_wire_topic`` in
``service_gateway_topic_transform``) is now that sole path.

These tests drive the REAL publish-side call site
(``HandlerForwardOutbound.forward_outbound`` -- the gateway forwarder's
outbound COMPUTE handler) and the REAL subscribe-side call site
(``_make_event_bus_callback`` -> ``_stamp_tenant_id_from_topic_prefix`` in
``handler_wiring.py``) and asserts they agree byte-for-byte on the physical
wire topic / tenant slug for the same ``(tenant_slug, canonical_topic)``
input. This is the exact seam that broke twice (OMN-15757 -> OMN-15778): two
independent implementations silently diverging. This test file does NOT
reimplement the transform -- it calls the two real production call sites and
compares their outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_bus_forwarder_effect.handlers import (
    HandlerForwardOutbound,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayEnvelope,
    ModelGatewayForwarderConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    resolve_physical_topic,
    resolve_tenant_from_wire_topic,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_event_bus_callback
from omnibase_infra.utils.util_onex_topic_format import (
    TopicValidationResult,
    validate_onex_topic_format,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
BROKER_PROVIDER_ID = UUID("22222222-2222-2222-2222-222222222222")
PRINCIPAL_ID = "t-33333333333333333333333333333333"
CORRELATION_ID = UUID("44444444-4444-4444-4444-444444444444")
CANONICAL_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
TENANT_SLUG = "acme"


def _forwarder_config() -> ModelGatewayForwarderConfig:
    return ModelGatewayForwarderConfig(
        tenant_identity=ModelGatewayTenantIdentity(
            tenant_id=TENANT_ID,
            tenant_slug=TENANT_SLUG,
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
        dedupe_store_path=Path.cwd() / "gateway-test-omn15792.sqlite3",
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=("onex.cmd.omnibase-infra.delegation-inference-request.v1",),
            outbound=(CANONICAL_TOPIC,),
        ),
        canary=ModelGatewayCanaryConfig(
            topic="onex.evt.omnibase-infra.gateway-canary.v1",
            cadence_seconds=30,
            produce_deadline_seconds=8,
            readback_deadline_seconds=12,
        ),
    )


def _outbound_envelope() -> ModelGatewayEnvelope:
    return ModelGatewayEnvelope(
        tenant_id=TENANT_ID,
        tenant_slug=TENANT_SLUG,
        envelope_id=uuid4(),
        correlation_id=CORRELATION_ID,
        causation_id=None,
        event_type="LlmInferenceResponse",
        source_topic=CANONICAL_TOPIC,
        wire_topic="",
        canonical_topic=CANONICAL_TOPIC,
        payload={"ok": True},
    )


def test_publish_side_wire_topic_matches_shared_resolver_output() -> None:
    """The real publish call site and the shared resolver must agree exactly."""
    result = HandlerForwardOutbound(_forwarder_config()).forward_outbound(
        _outbound_envelope()
    )

    assert result.wire_topic == resolve_physical_topic(
        CANONICAL_TOPIC, tenant_slug=TENANT_SLUG
    )


def test_publish_then_reverse_resolve_round_trips_to_the_same_seam_fields() -> None:
    """Publish-side output, fed through the reverse resolver, matches the input.

    Drives the real publish call site (``HandlerForwardOutbound``), then feeds
    its output wire_topic through ``resolve_tenant_from_wire_topic`` -- the
    same function the subscribe/dispatch path (``handler_wiring.py``) calls.
    Field-by-field seam match: tenant_slug and canonical_topic must survive
    the round trip unchanged.
    """
    published = HandlerForwardOutbound(_forwarder_config()).forward_outbound(
        _outbound_envelope()
    )

    slug, canonical_topic = resolve_tenant_from_wire_topic(published.wire_topic)

    assert slug == TENANT_SLUG
    assert canonical_topic == CANONICAL_TOPIC


@dataclass
class _FakeDispatchEngine:
    calls: list[tuple[str, ModelEventEnvelope[object]]] = field(default_factory=list)
    is_frozen: bool = True

    async def dispatch(self, topic: str, envelope: ModelEventEnvelope[object]) -> None:
        self.calls.append((topic, envelope))

    async def dispatch_scoped(
        self,
        topic: str,
        envelope: ModelEventEnvelope[object],
        *,
        allowed_dispatcher_ids: object,
    ) -> None:
        del allowed_dispatcher_ids
        await self.dispatch(topic, envelope)

    async def dispatch_with_transaction(
        self, *, topic: str, envelope: ModelEventEnvelope[object], tx: object
    ) -> None:
        del tx
        await self.dispatch(topic, envelope)


@dataclass(frozen=True)
class _Message:
    value: bytes


def _raw_message(payload: dict[str, object]) -> _Message:
    body: dict[str, object] = {
        "payload": payload,
        "correlation_id": str(uuid4()),
        "envelope_timestamp": datetime.now(UTC).isoformat(),
        "event_type": "omnibase-infra.inference-response",
        "source_tool": "test-adapter",
    }
    return _Message(value=json.dumps(body).encode("utf-8"))


@pytest.mark.asyncio
async def test_publish_and_subscribe_call_sites_agree_on_tenant_via_shared_resolver() -> (
    None
):
    """The actual OMN-15757/15778 failure class, closed: both real call sites agree.

    1. Publish side: ``HandlerForwardOutbound.forward_outbound`` (gateway
       forwarder outbound COMPUTE handler) resolves the physical wire topic
       for ``(TENANT_SLUG, CANONICAL_TOPIC)``.
    2. Subscribe side: that wire topic is handed to
       ``_make_event_bus_callback`` (the real local-runtime auto-wiring
       dispatch construction path) with ``tenant_scoped=True``; the real
       dispatch callback derives the tenant slug from the topic and stamps
       it into the payload before the handler ever sees it.

    Both sides must resolve to the same tenant slug -- neither reimplements
    the transform in this test; both go through
    ``service_gateway_topic_transform`` primitives.
    """
    published = HandlerForwardOutbound(_forwarder_config()).forward_outbound(
        _outbound_envelope()
    )
    wire_topic = published.wire_topic

    engine = _FakeDispatchEngine()
    callback = _make_event_bus_callback(
        wire_topic,
        engine,
        tenant_scoped=True,
        allowed_dispatcher_ids={"tenant-test-dispatcher"},
    )

    await callback(_raw_message({"ok": True}))

    assert len(engine.calls) == 1
    dispatched_topic, envelope = engine.calls[0]
    assert dispatched_topic == wire_topic
    assert envelope.payload["tenant_id"] == TENANT_SLUG


def test_no_tenant_context_resolves_bare_on_both_directions() -> None:
    """No tenant context -> bare canonical topic, symmetric both ways."""
    assert resolve_physical_topic(CANONICAL_TOPIC, tenant_slug=None) == CANONICAL_TOPIC
    assert resolve_tenant_from_wire_topic(CANONICAL_TOPIC) == (None, CANONICAL_TOPIC)


def test_reverse_resolver_rejects_reserved_slug_embedded_in_wire_prefix() -> None:
    """A ``tenant-system.`` prefix is structurally slug-shaped but reserved.

    The old ad hoc regex in ``handler_wiring.py`` extracted the slug without
    validating it against ``RESERVED_TENANT_SLUGS``. Routing through the
    shared resolver's ``validate_tenant_slug`` now rejects it instead of
    silently stamping an invalid tenant.
    """
    with pytest.raises(ValueError, match="reserved"):
        resolve_tenant_from_wire_topic(f"tenant-system.{CANONICAL_TOPIC}")


# ---------------------------------------------------------------------------
# OMN-15792 corrective verify pass: the THIRD resolver on the live Kafka
# publish path (util_onex_topic_format.py, called from event_bus_kafka.py's
# _enforce_onex_topic_format on every outbound publish) previously hand-rolled
# its own tenant-wire regex with no RESERVED_TENANT_SLUGS awareness --
# publish-ALLOWING topics the subscribe side rejected. These tests prove the
# publish-enforcer and the subscribe-resolver now agree.
# ---------------------------------------------------------------------------

_PUBLISH_ALLOWED_RESULTS = (
    TopicValidationResult.VALID,
    TopicValidationResult.VALID_TENANT_WIRE,
    TopicValidationResult.VALID_LEGACY_DLQ,
    TopicValidationResult.SKIPPED_INTERNAL,
)


def _publish_allows(topic: str) -> bool:
    """THE third resolver on the live Kafka publish path.

    ``validate_onex_topic_format`` is exactly what ``event_bus_kafka.py``'s
    ``_enforce_onex_topic_format`` calls on every outbound publish.
    """
    result, _reason = validate_onex_topic_format(topic)
    return result in _PUBLISH_ALLOWED_RESULTS


def _subscribe_allows(topic: str) -> bool:
    try:
        resolve_tenant_from_wire_topic(topic)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(
    "wire_topic",
    [
        pytest.param(f"tenant-system.{CANONICAL_TOPIC}", id="reserved-slug-system"),
        pytest.param(f"tenant-ab.{CANONICAL_TOPIC}", id="too-short-slug"),
        pytest.param(f"tenant-Acme.{CANONICAL_TOPIC}", id="uppercase-slug"),
    ],
)
def test_publish_enforcer_and_subscribe_resolver_agree_on_probe_cases(
    wire_topic: str,
) -> None:
    """The three probe cases from the OMN-15792 corrective verify pass.

    Before this fix, ``util_onex_topic_format.py`` (the live Kafka publish
    gate) hand-rolled its own tenant-wire regex with no reserved-slug
    awareness: it publish-ALLOWED ``tenant-system.<canonical>`` while the
    subscribe-side resolver REJECTED it as reserved -- the exact
    publish-allowed / subscribe-dropped divergence class of
    OMN-15757/OMN-15778. Both sides now delegate to the same
    ``resolve_tenant_from_wire_topic`` primitive and MUST agree -- both
    accept or both reject -- on every case, never one without the other.
    """
    publish_allowed = _publish_allows(wire_topic)
    subscribe_allowed = _subscribe_allows(wire_topic)
    assert publish_allowed == subscribe_allowed, (
        f"publish/subscribe disagree on {wire_topic!r}: "
        f"publish_allowed={publish_allowed} subscribe_allowed={subscribe_allowed}"
    )
    # All three probe cases are invalid tenant slugs (reserved, too short,
    # wrong case) -- both directions must reject, not merely agree by both
    # silently accepting.
    assert publish_allowed is False
    assert subscribe_allowed is False


@pytest.mark.asyncio
async def test_reserved_slug_topic_routes_to_swallow_boundary_not_stamped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Drives the REAL subscribe seam end-to-end for a reserved-slug topic.

    ``test_reverse_resolver_rejects_reserved_slug_embedded_in_wire_prefix``
    above only proves the resolver function itself raises in isolation. This
    test proves the *observable* production behavior: a message that arrives
    on a ``tenant-system.`` topic is never dispatched to the handler and
    never gets a tenant stamped -- it is routed through the existing
    swallowed-exception boundary (``_route_swallowed_exception``), the same
    path a raised handler exception takes (OMN-14507).
    """
    wire_topic = f"tenant-system.{CANONICAL_TOPIC}"
    engine = _FakeDispatchEngine()
    callback = _make_event_bus_callback(
        wire_topic,
        engine,
        tenant_scoped=True,
        allowed_dispatcher_ids={"tenant-test-dispatcher"},
    )

    with caplog.at_level("ERROR"):
        await callback(_raw_message({"prompt": "hi"}))

    # Not dispatched: the handler never sees this message.
    assert engine.calls == []
    # Observably routed to the swallow boundary, not silently dropped with no
    # trace (OMN-14507 metric-shaped log line).
    assert any(
        "boundary_swallow_observed" in record.message for record in caplog.records
    )
