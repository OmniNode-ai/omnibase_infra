# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17981: the outbound trust-boundary leg must read the wire shape the hook edge publishes.

OMN-17919 diagnosed this exact wire-shape mismatch and fixed the LANE MIRROR leg
(#3205) by keying it on the ``message_id`` wire header instead of a decoded body.
It scoped the trust-boundary legs out on purpose -- they genuinely need a parsed
envelope for the tenant transform. That was right for the mirror and wrong to
stop there: the OUTBOUND trust-boundary leg is the hop the hook records take one
step later, and it was left with the same incompatibility.

Measured live 2026-09-06T07:09-07:20Z, compose project ``omninode-gateway`` on
the lab host, over the container log since the redeploy that carried #3221:

    onex.evt.omniclaude.tool-executed.v1
       61  Gateway undecodable record quarantined   direction=outbound
        0  Gateway delivery acknowledged

    positive control, onex.evt.omnibase-infra.inference-response.v1
      154  Gateway duplicate suppressed             direction=outbound
       82  Gateway delivery acknowledged            direction=outbound

100% quarantine on the hook topic against a control topic that delivers, and
across the container's entire prior life there was never one outbound ack on any
of the four omniclaude hook topics.

The cause is a fixture-vs-wire mismatch, the same one OMN-17919 named: every
gateway-leg unit fixture in this repo constructs a ``ModelEventEnvelope``, which
is the one shape the hook edge does not publish. The hook edge publishes a FLAT
hook payload with the envelope metadata in Kafka HEADERS, and
``ModelEventEnvelope`` is ``extra="forbid"`` with ``payload`` required, so
``ServiceGatewayForwarder._decode_message`` rejects every one of them and
``NodeGatewayDelivery`` quarantines it.

The fix synthesises an envelope around the flat record on the outbound leg only,
using the mandatory ``message_id`` / ``correlation_id`` wire headers as its
identity -- the same headers #3205 keys the mirror on. It does NOT weaken the
trust boundary, and these tests are what holds that:

* the synthesised envelope carries NO ``source_tenant_id`` /
  ``source_tenant_principal_id`` metadata tag, so ``_prepare_outbound``'s tenant
  checks run exactly as they do today and stamp the identity themselves;
* a flat record whose payload ``tenant_id`` disagrees with the attach slug is
  still REFUSED (and quarantined, post-OMN-17382);
* a record whose headers cannot identify it is still QUARANTINED, not defaulted;
* the fail-closed OMN-16979 egress redaction gate is untouched -- a governed
  record with no admitted ``redaction_state`` is still dropped at the boundary;
* the INBOUND leg does not synthesise at all. Inbound records arrive from cloud
  across the tenant trust boundary and the tenant tags on the envelope are the
  thing being validated; a synthesised envelope there would have no tags to
  check, which is the definition of weakening the boundary.

AC4 requires the fixture be a record captured off the lane, not one this repo
made up. Both captured records below were read read-only off the stability lane
on 2026-09-06 with ``rpk topic consume``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import UUID

import pytest

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.idempotency import StoreIdempotencyInmemory
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayEgressRedaction,
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
BROKER_PROVIDER_ID = UUID("22222222-2222-2222-2222-222222222222")
PRINCIPAL_ID = "t-33333333333333333333333333333333"
TENANT_SLUG = "acme"

INBOUND_TOPIC = "onex.cmd.omnibase-infra.delegation-request.v1"
SESSION_STARTED_TOPIC = "onex.evt.omniclaude.session-started.v1"
TOOL_EXECUTED_TOPIC = "onex.evt.omniclaude.tool-executed.v1"

# ---------------------------------------------------------------------------
# The captured records (AC4)
# ---------------------------------------------------------------------------
# Read read-only off the live stability lane 2026-09-06:
#
#   docker exec omnibase-infra-stability-test-redpanda rpk topic consume \
#     onex.evt.omniclaude.session-started.v1 -p 0 -o 63816330 -n 1 --format json
#   docker exec omnibase-infra-stability-test-redpanda rpk topic consume \
#     onex.evt.omniclaude.tool-executed.v1  -p 0 -o 34710    -n 1 --format json
#
# Verbatim except for one substitution: each record's live agent-session UUID
# appears three times in the value (session_id, correlation_id, entity_id) and
# once more as the `correlation_id` header, and is replaced by a fixed
# placeholder UUID of identical form. Nothing else is altered -- key order,
# spacing, the explicit `"causation_id": null`, and the header set and its order
# are all as the broker held them. Neither record carries credential or token
# material; `duration_ms` is a count and `working_directory` is a bare repo name.
_SESSION_UUID = "00000000-0000-4000-8000-000000000001"

_SESSION_STARTED_MESSAGE_ID = "fce95306-15b9-4f5d-9d38-ae93785dd856"
_SESSION_STARTED_VALUE: bytes = (
    '{"hook_source": "startup", '
    f'"session_id": "{_SESSION_UUID}", '
    '"working_directory": "omni_home", '
    f'"correlation_id": "{_SESSION_UUID}", '
    '"causation_id": null, '
    '"emitted_at": "2026-09-05T08:15:33.461665+00:00", '
    f'"entity_id": "{_SESSION_UUID}", '
    '"schema_version": "1.0.0"}'
).encode()
_SESSION_STARTED_HEADERS: dict[str, bytes] = {
    "content_type": b"application/json",
    "correlation_id": _SESSION_UUID.encode("utf-8"),
    "message_id": _SESSION_STARTED_MESSAGE_ID.encode("utf-8"),
    "timestamp": b"2026-09-05T08:15:33.690171+00:00",
    "source": b"node_event_emit_effect",
    "event_type": SESSION_STARTED_TOPIC.encode("utf-8"),
    "schema_version": b"1.0.0",
    "priority": b"normal",
    "retry_count": b"0",
    "max_retries": b"3",
}

_TOOL_EXECUTED_MESSAGE_ID = "ab950a67-4a9a-45af-a46f-7ed4391b4ef4"
_TOOL_EXECUTED_VALUE: bytes = (
    '{"duration_ms": 361, "hook_source": "post_tool_use", "interrupted": false, '
    f'"session_id": "{_SESSION_UUID}", '
    '"tool_name": "Bash", "working_directory": "omni_home", '
    f'"correlation_id": "{_SESSION_UUID}", '
    '"causation_id": null, '
    '"emitted_at": "2026-09-06T07:47:38.731279+00:00", '
    f'"entity_id": "{_SESSION_UUID}", '
    '"schema_version": "1.0.0"}'
).encode()
_TOOL_EXECUTED_HEADERS: dict[str, bytes] = {
    **_SESSION_STARTED_HEADERS,
    "message_id": _TOOL_EXECUTED_MESSAGE_ID.encode("utf-8"),
    "timestamp": b"2026-09-06T07:47:38.763958+00:00",
    "event_type": TOOL_EXECUTED_TOPIC.encode("utf-8"),
}


def _record(
    *,
    topic: str,
    value: bytes,
    headers: Mapping[str, bytes],
    partition: int = 0,
    offset: int = 34710,
) -> ModelTransportMessage:
    return ModelTransportMessage(
        topic=topic,
        partition=partition,
        offset=offset,
        key=None,
        value=value,
        headers=dict(headers),
        ack_token=(topic, partition, offset),
    )


def _session_started_record(
    headers: Mapping[str, bytes] | None = None,
    value: bytes | None = None,
) -> ModelTransportMessage:
    return _record(
        topic=SESSION_STARTED_TOPIC,
        value=_SESSION_STARTED_VALUE if value is None else value,
        headers=_SESSION_STARTED_HEADERS if headers is None else headers,
        offset=63816330,
    )


def _tool_executed_record(
    value: bytes | None = None,
) -> ModelTransportMessage:
    return _record(
        topic=TOOL_EXECUTED_TOPIC,
        value=_TOOL_EXECUTED_VALUE if value is None else value,
        headers=_TOOL_EXECUTED_HEADERS,
    )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _RecordingBus:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes, object]] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.sent.append((topic, value, headers))


class _Source:
    def __init__(self) -> None:
        self.committed: list[object] = []
        self.nacked: list[object] = []
        self.dlq: list[tuple[str, bytes]] = []

    async def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Mapping[str, bytes],
    ) -> None:
        self.dlq.append((topic, value))

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ModelTransportMessage]:
        return []

    async def commit(self, message: object) -> None:
        self.committed.append(message)

    async def nack(self, message: object) -> None:
        self.nacked.append(message)


def _config(
    *,
    outbound: tuple[str, ...],
    egress_redaction: ModelGatewayEgressRedaction | None = None,
) -> ModelGatewayForwarderConfig:
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
        dedupe_store_path=Path.cwd() / "gateway-omn17981.sqlite3",
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=(INBOUND_TOPIC,),
            outbound=outbound,
        ),
        egress_redaction=egress_redaction,
        canary=ModelGatewayCanaryConfig(
            topic="onex.evt.omnibase-infra.gateway-canary.v1",
            cadence_seconds=30,
            produce_deadline_seconds=8,
            readback_deadline_seconds=12,
        ),
    )


# The live contract's policy, verbatim from node_bus_forwarder_effect/contract.yaml.
def _governed_config() -> ModelGatewayForwarderConfig:
    return _config(
        outbound=(SESSION_STARTED_TOPIC, TOOL_EXECUTED_TOPIC),
        egress_redaction=ModelGatewayEgressRedaction(
            state_field="redaction_state",
            admitted_states=("redacted", "restricted", "secret_detected"),
            governed_topics=(TOOL_EXECUTED_TOPIC,),
        ),
    )


def _delivery(
    config: ModelGatewayForwarderConfig, source: _Source
) -> tuple[NodeGatewayDelivery, _RecordingBus]:
    local_bus = _RecordingBus()
    cloud_bus = _RecordingBus()
    forwarder = ServiceGatewayForwarder(
        config=config,
        local_bus=local_bus,  # type: ignore[arg-type]
        cloud_bus=cloud_bus,  # type: ignore[arg-type]
    )
    return (
        NodeGatewayDelivery(
            config=config,
            forwarder=forwarder,
            local_consumer=source,  # type: ignore[arg-type]
            cloud_consumer=source,  # type: ignore[arg-type]
            idempotency_store=StoreIdempotencyInmemory(),
        ),
        cloud_bus,
    )


# ---------------------------------------------------------------------------
# (a) the captured flat record with headers is delivered and acknowledged
# ---------------------------------------------------------------------------


async def test_captured_flat_hook_record_is_delivered_and_acknowledged() -> None:
    """The line class that had never once appeared: an outbound ack on a hook topic.

    ``session-started.v1`` is the un-governed content-free hook class (operator
    OD-9 ruling 2026-08-18), so it exercises the decode fix alone with no
    redaction gate in the way. Before the fix this record quarantined.
    """
    source = _Source()
    delivery, cloud_bus = _delivery(_config(outbound=(SESSION_STARTED_TOPIC,)), source)
    message = _session_started_record()

    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert source.nacked == []
    assert source.dlq == []
    assert source.committed == [message]
    assert len(cloud_bus.sent) == 1

    wire_topic, wire_value, _ = cloud_bus.sent[0]
    assert wire_topic == f"tenant-{TENANT_SLUG}.{SESSION_STARTED_TOPIC}"

    published = ModelEventEnvelope[dict[str, object]].model_validate_json(wire_value)
    # Identity comes off the wire headers, exactly as #3205 keys the mirror.
    assert published.envelope_id == UUID(_SESSION_STARTED_MESSAGE_ID)
    assert published.correlation_id == UUID(_SESSION_UUID)
    # The flat record crosses intact as the payload -- nothing is dropped.
    assert published.payload == json.loads(_SESSION_STARTED_VALUE)
    # The trust-boundary stamp is _prepare_outbound's, unchanged.
    tags = published.metadata.tags
    assert tags["source_tenant_id"] == str(TENANT_ID)
    assert tags["source_tenant_principal_id"] == PRINCIPAL_ID
    assert tags["gateway_tenant_slug"] == TENANT_SLUG
    assert tags["gateway_direction"] == "local-to-cloud"
    # And the record is marked as synthesised so a reader can tell it apart
    # from a producer-minted envelope.
    assert tags["gateway_synthesized_envelope"] == "true"
    assert tags["gateway_synthesized_source_topic"] == SESSION_STARTED_TOPIC
    assert tags["gateway_synthesized_source_partition"] == "0"
    assert tags["gateway_synthesized_source_offset"] == "63816330"


async def test_synthesized_envelope_carries_the_attach_config_tenant() -> None:
    """Tenant identity on a synthesised envelope comes from the attach config.

    The flat hook record carries no tenant of its own, so the only admissible
    source is the config the gateway is attached as -- never the untrusted
    record.
    """
    source = _Source()
    delivery, cloud_bus = _delivery(_config(outbound=(SESSION_STARTED_TOPIC,)), source)

    await delivery.deliver_message("outbound", source, _session_started_record())  # type: ignore[arg-type]

    published = ModelEventEnvelope[dict[str, object]].model_validate_json(
        cloud_bus.sent[0][1]
    )
    assert published.tenant_id == TENANT_SLUG


# ---------------------------------------------------------------------------
# (b) the same record with no usable headers stays quarantined
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dropped",
    [
        pytest.param(("message_id",), id="no-message-id"),
        pytest.param(("correlation_id",), id="no-correlation-id"),
        pytest.param(("message_id", "correlation_id"), id="neither"),
    ],
)
async def test_flat_record_without_identity_headers_is_quarantined(
    dropped: tuple[str, ...],
) -> None:
    """No identity on the wire, no synthesis. Minting one would make the
    durable dedupe marker meaningless, which is exactly the reasoning
    ``NodeLaneMirror._record_identity`` records for the mirror leg.
    """
    headers = {
        key: value
        for key, value in _SESSION_STARTED_HEADERS.items()
        if key not in dropped
    }
    source = _Source()
    delivery, cloud_bus = _delivery(_config(outbound=(SESSION_STARTED_TOPIC,)), source)
    message = _session_started_record(headers=headers)

    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert cloud_bus.sent == []
    # Quarantine, not nack: a record that can never be identified would be
    # refused again on every redelivery (OMN-15748 poison-pill DoS).
    assert source.nacked == []
    assert source.committed == [message]


async def test_flat_record_with_non_uuid_message_id_is_quarantined() -> None:
    headers = {**_SESSION_STARTED_HEADERS, "message_id": b"not-a-uuid"}
    source = _Source()
    delivery, cloud_bus = _delivery(_config(outbound=(SESSION_STARTED_TOPIC,)), source)
    message = _session_started_record(headers=headers)

    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert cloud_bus.sent == []
    assert source.nacked == []
    assert source.committed == [message]


async def test_non_object_json_body_is_quarantined_even_with_headers() -> None:
    """A JSON array or scalar is not a hook payload; synthesis refuses it."""
    source = _Source()
    delivery, cloud_bus = _delivery(_config(outbound=(SESSION_STARTED_TOPIC,)), source)
    message = _session_started_record(value=b'["not", "a", "payload"]')

    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert cloud_bus.sent == []
    assert source.nacked == []
    assert source.committed == [message]


# ---------------------------------------------------------------------------
# (c) an envelope record is unchanged
# ---------------------------------------------------------------------------


async def test_real_envelope_record_is_unchanged_by_the_synthesis_path() -> None:
    """The positive-control shape. Its identity is the envelope's own, not a header.

    The headers deliberately carry a DIFFERENT ``message_id`` than the envelope
    body: if synthesis ever ran on a decodable record, this assertion is what
    catches it.
    """
    envelope_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    correlation_id = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    envelope = ModelEventEnvelope[dict[str, object]](
        envelope_id=envelope_id,
        correlation_id=correlation_id,
        event_type="omniclaude.session-started",
        payload={"ok": True},
        metadata=ModelEnvelopeMetadata(tags={}),
    )
    source = _Source()
    delivery, cloud_bus = _delivery(_config(outbound=(SESSION_STARTED_TOPIC,)), source)
    message = _record(
        topic=SESSION_STARTED_TOPIC,
        value=envelope.model_dump_json().encode("utf-8"),
        headers={**_SESSION_STARTED_HEADERS},
    )

    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert source.committed == [message]
    published = ModelEventEnvelope[dict[str, object]].model_validate_json(
        cloud_bus.sent[0][1]
    )
    assert published.envelope_id == envelope_id
    assert published.correlation_id == correlation_id
    assert published.payload == {"ok": True}
    assert "gateway_synthesized_envelope" not in published.metadata.tags


# ---------------------------------------------------------------------------
# (d) the tenant checks still refuse a synthesised record
# ---------------------------------------------------------------------------


async def test_synthesized_record_with_foreign_payload_tenant_is_refused() -> None:
    """AC5: ``_prepare_outbound``'s tenant checks are unchanged in strictness.

    Synthesis puts the untrusted flat record in ``payload``, which is exactly
    where ``_prepare_outbound`` reads ``tenant_id`` from. A record claiming a
    tenant that is not the attached one must still refuse, and (post-OMN-17382)
    that refusal is a quarantine rather than a wedge.
    """
    body = json.loads(_SESSION_STARTED_VALUE)
    body["tenant_id"] = "not-acme"
    source = _Source()
    delivery, cloud_bus = _delivery(_config(outbound=(SESSION_STARTED_TOPIC,)), source)
    message = _session_started_record(value=json.dumps(body).encode("utf-8"))

    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert cloud_bus.sent == []
    assert source.nacked == []
    assert source.committed == [message]
    assert [topic for topic, _ in source.dlq] != []


async def test_synthesized_record_with_matching_payload_tenant_crosses() -> None:
    """The positive control for the refusal above -- otherwise a refusal that
    fires on everything would read identically to a working check."""
    body = json.loads(_SESSION_STARTED_VALUE)
    body["tenant_id"] = TENANT_SLUG
    source = _Source()
    delivery, cloud_bus = _delivery(_config(outbound=(SESSION_STARTED_TOPIC,)), source)

    await delivery.deliver_message(  # type: ignore[arg-type]
        "outbound",
        source,
        _session_started_record(value=json.dumps(body).encode("utf-8")),
    )

    assert len(cloud_bus.sent) == 1
    assert source.dlq == []


# ---------------------------------------------------------------------------
# The fail-closed OMN-16979 egress gate is untouched
# ---------------------------------------------------------------------------


async def test_governed_flat_record_without_redaction_state_still_drops() -> None:
    """The captured ``tool-executed`` record decodes now, and is STILL dropped.

    Measured 2026-09-06 on the stability lane: 0 of 200 consecutive
    ``tool-executed`` records carry ``redaction_state`` (positive control: 200
    of 200 carry ``duration_ms``, so the probe reads real records). Until the
    upstream omnimarket emit seam (OMN-17209 / OMN-16019) stamps the field,
    this class decodes but does not cross. That is the gate working, not a
    regression -- and it is why fixing the decode does not on its own make the
    tenant wire topic for this class advance.
    """
    source = _Source()
    delivery, cloud_bus = _delivery(_governed_config(), source)
    message = _tool_executed_record()

    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert cloud_bus.sent == []
    assert source.nacked == []
    assert source.committed == [message]


async def test_governed_flat_record_with_admitted_redaction_state_crosses() -> None:
    """Positive control for the drop above: the shape OMN-17209's seam will emit."""
    body = json.loads(_TOOL_EXECUTED_VALUE)
    body["redaction_state"] = "redacted"
    source = _Source()
    delivery, cloud_bus = _delivery(_governed_config(), source)

    await delivery.deliver_message(  # type: ignore[arg-type]
        "outbound",
        source,
        _tool_executed_record(value=json.dumps(body).encode("utf-8")),
    )

    assert len(cloud_bus.sent) == 1
    assert cloud_bus.sent[0][0] == f"tenant-{TENANT_SLUG}.{TOOL_EXECUTED_TOPIC}"


# ---------------------------------------------------------------------------
# The inbound leg does NOT synthesise
# ---------------------------------------------------------------------------


async def test_inbound_flat_record_is_still_quarantined() -> None:
    """Synthesis is outbound-only, on purpose.

    An inbound record crosses INTO this cluster from cloud, and what
    ``_prepare_inbound`` validates is the ``source_tenant_id`` /
    ``source_tenant_principal_id`` tags on the envelope the cloud sent. A
    synthesised envelope has no such tags to check, so synthesising here would
    convert a validated record into an unvalidated one -- weakening the exact
    boundary this node exists to hold.
    """
    source = _Source()
    delivery, local_bus = _delivery(_config(outbound=(SESSION_STARTED_TOPIC,)), source)
    message = _record(
        topic=f"tenant-{TENANT_SLUG}.{INBOUND_TOPIC}",
        value=_SESSION_STARTED_VALUE,
        headers=_SESSION_STARTED_HEADERS,
    )

    await delivery.deliver_message("inbound", source, message)  # type: ignore[arg-type]

    assert local_bus.sent == []
    assert source.nacked == []
    assert source.committed == [message]
