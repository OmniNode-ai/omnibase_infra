# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Executable bus-to-bus gateway forwarder service.

The wire on both broker legs is the platform canonical
``ModelEventEnvelope``.  ``ModelGatewayEnvelope`` is a transform-boundary
model used by the node handlers; it is not a second event-bus envelope and
must never be required from ordinary runtime producers or consumers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID, uuid4

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.nodes.node_bus_forwarder_effect.handlers import (
    HandlerConsumeInbound,
    HandlerForwardOutbound,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayEgressRedaction,
    ModelGatewayEnvelope,
    ModelGatewayForwarderConfig,
    ModelGatewayHeartbeat,
    ModelGatewayPublishReceipt,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    strip_topic_prefix,
)

logger = logging.getLogger(__name__)

# ``gateway_direction`` tag values that mark an envelope as local-bus-only --
# the forwarder's own outbound consumer loop (NodeGatewayDelivery polling the
# SAME transport local_bus publishes into, see runtime/gateway_forwarder.py)
# must skip these rather than re-forward them to cloud (OMN-15570/OMN-15742
# reconciliation finding D1). "cloud-to-local" is an inbound-transformed
# envelope publish_status/publish_heartbeat never emit and needs the same
# skip already established for the inbound leg; "local-mirror" marks a
# direct local-only publish (DEGRADED status, and the G3 heartbeat local
# mirror) that was never meant to leave this cluster at all.
_LOCAL_ONLY_DIRECTIONS = frozenset({"cloud-to-local", "local-mirror"})

# OMN-17981: the two mandatory identity headers ``ModelEventHeaders`` declares
# and ``event_bus_kafka._model_headers_to_kafka`` stamps on every ONEX publish.
# These are the SAME headers #3205 keys the lane mirror on -- reading them here
# rather than re-deriving an identity is what keeps the two legs' idempotency
# keys equal for one record, so a record that crossed the mirror and the cloud
# leg is deduped on the same value on both.
_MESSAGE_ID_HEADER = "message_id"
_CORRELATION_ID_HEADER = "correlation_id"
_EVENT_TYPE_HEADER = "event_type"
_TIMESTAMP_HEADER = "timestamp"


def _header_text(headers: object, name: str) -> str | None:
    """One wire header as text, or None when it is absent or unreadable."""
    if not isinstance(headers, Mapping):
        return None
    raw = headers.get(name)
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(raw, str):
        return raw
    return None


def _header_uuid(headers: object, name: str) -> UUID | None:
    """One wire header as a UUID, or None when absent or not a UUID.

    Refused rather than defaulted: minting an identity for a record that does
    not carry one would make the durable dedupe marker meaningless and turn the
    leg's at-least-once promise into a coin flip on redelivery. That is the
    same reasoning ``NodeLaneMirror._record_identity`` records for the mirror.
    """
    text = _header_text(headers, name)
    if text is None:
        return None
    try:
        return UUID(text)
    except ValueError:
        return None


def decode_envelope_strict(
    message: object,
) -> ModelEventEnvelope[dict[str, object]]:
    """Decode the canonical envelope, raising when the record is not one.

    Module-level (not a method) for the same reason ``_stamp_local_only`` is:
    ``ServiceGatewayForwarder`` sits at the ONEX pattern validator's
    method-count bound, and the OMN-17981 outbound seam below needs a method
    slot more than this does.
    """
    value = getattr(message, "value", message)
    if isinstance(value, ModelEventEnvelope):
        return ModelEventEnvelope[dict[str, object]].model_validate(value)
    if isinstance(value, str):
        value = value.encode("utf-8")
    if not isinstance(value, bytes):
        raise TypeError("gateway bus message value must be bytes or string")
    return ModelEventEnvelope[dict[str, object]].model_validate_json(value)


def synthesize_outbound_envelope(
    message: object,
    identity: ModelGatewayTenantIdentity,
) -> ModelEventEnvelope[dict[str, object]] | None:
    """Wrap a flat outbound record in an envelope, or return None to quarantine.

    OMN-17981, measured on the ``omninode-gateway`` lane 2026-09-06: the
    omniclaude hook edge publishes a FLAT hook payload with the envelope
    metadata in Kafka HEADERS, and ``ModelEventEnvelope`` is ``extra="forbid"``
    with ``payload`` required, so ``decode_envelope_strict`` rejected 61 of 61 records
    on ``onex.evt.omniclaude.tool-executed.v1`` while the ``inference-response``
    control topic delivered 82. Across the container's entire prior life there
    was never one outbound acknowledgement on any of the four hook topics.

    OMN-17919 fixed the same wire-shape mismatch on the lane mirror (#3205) by
    keying it on the ``message_id`` header, and scoped the trust-boundary legs
    out because they "genuinely do need the envelope, for the tenant
    transform". They do -- so this builds the envelope the transform needs
    instead of relaxing the transform.

    This does NOT weaken the trust boundary, in three specific ways:

    * The flat record becomes the ``payload`` verbatim, which is exactly where
      :meth:`ServiceGatewayForwarder._prepare_outbound` reads ``tenant_id``
      from. A record claiming a foreign tenant still refuses.
    * No ``source_tenant_id`` / ``source_tenant_principal_id`` metadata tag is
      stamped here. ``_prepare_outbound``'s two tag checks therefore run
      against absence, exactly as they do today for every real
      ``inference-response`` record, and ``_prepare_outbound`` remains the only
      writer of those tags.
    * The tenant DIMENSION on the envelope comes from the attach config
      (``identity.tenant_slug``) and never from the untrusted record.

    Returns ``None`` -- never a partial envelope -- when the record carries no
    usable identity, so the caller re-raises the original decode failure and
    the existing quarantine path handles it unchanged.
    """
    headers = getattr(message, "headers", None)
    envelope_id = _header_uuid(headers, _MESSAGE_ID_HEADER)
    correlation_id = _header_uuid(headers, _CORRELATION_ID_HEADER)
    if envelope_id is None or correlation_id is None:
        return None

    value = getattr(message, "value", None)
    if isinstance(value, str):
        value = value.encode("utf-8")
    if not isinstance(value, bytes):
        return None
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    # A JSON array or scalar is not a hook payload. ``payload`` is typed
    # ``dict[str, object]`` downstream and ``_prepare_outbound`` does a
    # ``.get("tenant_id")`` on it, so anything else is refused rather than
    # coerced into a shape the tenant check cannot read.
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        return None

    source_topic = getattr(message, "topic", None)
    source_partition = getattr(message, "partition", None)
    source_offset = getattr(message, "offset", None)
    metadata = ModelEnvelopeMetadata(
        tags={
            # The marker exists so a downstream reader can tell a synthesised
            # envelope from a producer-minted one without guessing, and so the
            # class is countable in the log rather than invisible.
            "gateway_synthesized_envelope": "true",
            "gateway_synthesized_source_topic": str(source_topic),
            "gateway_synthesized_source_partition": str(source_partition),
            "gateway_synthesized_source_offset": str(source_offset),
        }
    )
    return ModelEventEnvelope[dict[str, object]](
        payload=payload,
        envelope_id=envelope_id,
        correlation_id=correlation_id,
        envelope_timestamp=_header_timestamp(headers),
        event_type=_header_text(headers, _EVENT_TYPE_HEADER),
        metadata=metadata,
        tenant_id=identity.tenant_slug,
    )


def _header_timestamp(headers: object) -> datetime:
    """The producer's own emit time when the wire carries a readable one.

    Falls back to now: the timestamp is provenance, not identity, so an
    unparseable one must not cost the record its crossing the way a missing
    ``message_id`` does.
    """
    text = _header_text(headers, _TIMESTAMP_HEADER)
    if text is not None:
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
    return datetime.now(UTC)


def _stamp_local_only(
    envelope: ModelEventEnvelope[dict[str, object]],
) -> ModelEventEnvelope[dict[str, object]]:
    """Tag a local-bus-only publish so the outbound consumer never re-forwards it.

    Both the DEGRADED status publish (``ServiceGatewayForwarder.publish_status``)
    and the G3 heartbeat local mirror (``publish_heartbeat``) go directly onto
    the local bus's canonical outbound topic -- exactly the topic the
    forwarder's own outbound consumer polls (``NodeGatewayDelivery`` on
    ``local_consumer``, the SAME transport object ``local_bus`` publishes into
    in the real runtime). Without this tag, ``_forward_outbound_message``'s
    loopback skip does not match an untagged envelope, so it falls through to
    ``_prepare_outbound`` -- a second cloud publish per heartbeat tick, or a
    DEGRADED status leak to cloud (OMN-15570/OMN-15742 reconciliation finding
    D1). Module-level (not a method) to stay under the class's method-count
    pattern threshold.
    """
    metadata = envelope.metadata.model_copy(
        update={
            "tags": {
                **envelope.metadata.tags,
                "gateway_direction": "local-mirror",
            }
        }
    )
    return envelope.model_copy(update={"metadata": metadata})


class ProtocolGatewayPublisher(Protocol):
    """Destination publish boundary shared by push and pull transports."""

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> ModelGatewayPublishReceipt | None:
        """Publish bytes to a topic and report where the destination put them.

        OMN-17201 widened the return from ``None``. Every caller that only
        needs "the broker took it" ignores the value and is unaffected; the
        lane mirror needs the destination COORDINATE, because an
        acknowledgement proves a broker took the record and never that the
        intended broker did. ``None`` is a legitimate answer from a transport
        that has no broker coordinates to report (the HTTPS ingest route), and
        a publisher must return ``None`` rather than invent one.
        """


class GatewayRecordRefusedError(ValueError):
    """One record is not this gateway's to carry -- a DROP, never a wedge.

    OMN-17382, measured on this exact leg. A single foreign-tenant record on
    the dev bus raised ``outbound payload tenant_id does not match attached
    tenant`` out of :meth:`ServiceGatewayForwarder._prepare_outbound`. The
    delivery loop nacked, seeked back to the same offset, and re-read the same
    record: 925 consecutive failures over 7h45m with 177 real records stuck
    behind it, and on 2026-09-05 the same signature again at reconnect attempt
    295 after 8799s, while the container reported healthy throughout.

    Redelivery cannot change the verdict. Tenant binding is a property of the
    RECORD, so a refusal is permanent for that offset, which makes it the same
    poison-pill class the undecodable path (OMN-15748) already quarantines:
    log, best-effort dead-letter, commit past it, keep the bridge alive.
    ``egress_admits`` already returns ``False`` rather than raising for exactly
    this reason (see its docstring); this type extends that rule to the
    remaining per-record trust-boundary refusals, which were the ones actually
    wedging the leg.

    NOT used for infrastructure faults. A broker timeout, a connection or
    SASL-handshake failure, or a serialization bug is transient or global, and
    MUST keep raising so the loop retries -- silently committing past those is
    data loss. The distinction is exactly "is this verdict a property of the
    record".

    OMN-17201 narrows one word of the paragraph above. It previously said "an
    auth failure" without qualification, which conflated two different
    verdicts. A connection/handshake auth failure IS global and transient and
    still retries. A per-topic AUTHORIZATION DENIAL is not: the broker
    answered, and it answered about (topic, principal). The destination topic
    is derived from the record, so the denial is reached identically on every
    redelivery -- see :class:`GatewayEgressDeniedError`.
    """


class GatewayEgressDeniedError(GatewayRecordRefusedError):
    """The destination broker denied this record's topic for our identity.

    OMN-17201, measured live on the lab forwarder (compose project
    ``omninode-gateway``, image revision ``2ea74bc4de76``) on 2026-09-06:
    aiokafka logged ``Topic
    tenant-beta-gateway-canary-...onex.evt.omniclaude.session-started.v1 is
    not authorized for this client`` and ``TransportGatewayBus.publish``
    blanket-mapped every ``KafkaError`` -- this one included -- onto
    ``InfraUnavailableError``. That type means "transient", so
    ``_publish_with_delivery_retry`` retained the record and retried it
    forever (observed at attempt=7, delay_seconds=30.0). The single outbound
    direction task is serialised behind that one record, so outbound
    TOTAL-LAG on the local group climbed 0 -> 284 in ~35 min (523 by the time
    the fix was written, 510 of it on ``onex.evt.omniclaude.tool-executed.v1``
    -- a topic that was never denied, merely stuck behind one that was) while
    the container reported ``healthy``.

    A ``TOPIC_AUTHORIZATION_FAILED`` is a broker ANSWER, not a broker outage.
    It is a verdict about ``(topic, principal)``, and the destination topic is
    a pure function of the record's source topic and the attached tenant
    slug. Redelivery therefore reaches the identical verdict, which makes this
    the same poison-pill class the undecodable path (OMN-15748) and the
    foreign-tenant refusal (OMN-17382) already quarantine.

    Dropping is not free: a record quarantined here would have crossed if the
    ACL were later granted. That trade is deliberate and is why this class is
    counted rather than only logged -- see
    ``service_gateway_egress_health.ServiceGatewayEgressHealth``. A leg whose
    records are all being denied MUST read unhealthy; silently draining lag
    into a dead letter queue while reporting healthy would be a worse failure
    than the wedge it replaces.
    """

    def __init__(
        self,
        *,
        topic: str,
        tenant_id: str,
        principal_id: str,
    ) -> None:
        super().__init__(
            "destination broker denied topic authorization for this identity: "
            f"topic={topic} tenant_id={tenant_id} principal_id={principal_id}"
        )
        self.topic = topic
        self.tenant_id = tenant_id
        self.principal_id = principal_id


def egress_admits(
    policy: ModelGatewayEgressRedaction | None,
    envelope: ModelEventEnvelope[dict[str, object]],
    canonical_topic: str,
) -> bool:
    """OMN-16979: fail-closed redaction admission for the widened hook classes.

    Module-level rather than a method so both the forward path and any
    validate/canary path resolve the SAME decision, and so the service class
    does not grow past the OMN pattern-validator's method bound.

    The redaction itself is produced upstream at omnimarket's emit seam
    (OMN-16019 / OMN-17209), which is the only place that knows tool semantics.
    This boundary does not re-derive that judgement; it refuses to cross
    anything the upstream seam did not stamp.

    A refusal is a DROP, never a raise. OMN-17382 is the live proof: on
    2026-09-05 one foreign-tenant probe record raised out of
    ``_prepare_outbound`` and wedged this same outbound leg for 7h45m over 925
    consecutive retries, with 177 real records stuck behind it. A per-record
    policy decision must not be able to stop the bridge.
    """
    if policy is None or not policy.governs(canonical_topic):
        return True
    if policy.admits(envelope.payload):
        return True
    logger.warning(
        "Dropping outbound record: topic %s is redaction-governed and the "
        "payload carries no admitted %s (envelope_id=%s). The upstream emit "
        "seam did not stamp it; nothing crosses the boundary.",
        canonical_topic,
        policy.state_field,
        envelope.envelope_id,
    )
    return False


class ServiceGatewayForwarder:
    """Validate, transform, and republish explicitly polled gateway envelopes."""

    def __init__(
        self,
        *,
        config: ModelGatewayForwarderConfig,
        local_bus: ProtocolGatewayPublisher,
        cloud_bus: ProtocolGatewayPublisher,
        retry_sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._local_bus = local_bus
        self._cloud_bus = cloud_bus
        if retry_sleep is None:
            import asyncio

            retry_sleep = asyncio.sleep
        self._retry_sleep = retry_sleep
        # OMN-15740: the tenant-prefix transform is owned by the contract-declared
        # COMPUTE handlers, not re-derived here. The service's job is the trust
        # boundary (tag/tenant validation against untrusted bus input) plus the
        # I/O; the pure prefix/strip transform and payload stamp are delegated.
        self._forward_outbound_handler = HandlerForwardOutbound(config)
        self._consume_inbound_handler = HandlerConsumeInbound(config)

    async def _forward_outbound_message(self, message: object) -> None:
        source_topic = self._message_topic(message)
        envelope = self.decode_outbound_message(message)
        direction = envelope.metadata.tags.get("gateway_direction")
        if direction in _LOCAL_ONLY_DIRECTIONS:
            logger.debug(
                "Skipping gateway loopback on local topic %s (direction=%s)",
                source_topic,
                direction,
            )
            return
        if not egress_admits(self._config.egress_redaction, envelope, source_topic):
            return
        transformed, wire_topic = self._prepare_outbound(envelope, source_topic)
        await self._publish_with_delivery_retry(
            bus=self._cloud_bus,
            topic=wire_topic,
            key=getattr(message, "key", None),
            value=self._encode_envelope(transformed),
            headers=getattr(message, "headers", None),
        )

    async def forward_outbound_message(self, message: object) -> None:
        """Validate, transform, and broker-acknowledge one outbound message."""
        await self._forward_outbound_message(message)

    def validate_outbound_message(self, message: object) -> None:
        """Validate an outbound trust-boundary message without publishing it."""
        source_topic = self._message_topic(message)
        envelope = self.decode_outbound_message(message)
        direction = envelope.metadata.tags.get("gateway_direction")
        if direction not in _LOCAL_ONLY_DIRECTIONS:
            self._prepare_outbound(envelope, source_topic)

    async def _consume_inbound_message(self, message: object) -> None:
        wire_topic = self._message_topic(message)
        envelope = decode_envelope_strict(message)
        if envelope.metadata.tags.get("gateway_direction") == "local-to-cloud":
            logger.debug(
                "Skipping gateway loopback on cloud topic %s",
                wire_topic,
            )
            return
        transformed, canonical_topic = self._prepare_inbound(envelope, wire_topic)
        await self._publish_with_delivery_retry(
            bus=self._local_bus,
            topic=canonical_topic,
            key=getattr(message, "key", None),
            value=self._encode_envelope(transformed),
            headers=getattr(message, "headers", None),
        )

    async def consume_inbound_message(self, message: object) -> None:
        """Validate, transform, and broker-acknowledge one inbound message."""
        await self._consume_inbound_message(message)

    def validate_inbound_message(self, message: object) -> None:
        """Validate an inbound trust-boundary message without publishing it."""
        wire_topic = self._message_topic(message)
        envelope = decode_envelope_strict(message)
        if envelope.metadata.tags.get("gateway_direction") != "local-to-cloud":
            self._prepare_inbound(envelope, wire_topic)

    @classmethod
    def decode_message(
        cls,
        message: object,
    ) -> ModelEventEnvelope[dict[str, object]]:
        """Decode the canonical envelope used as the durable dedupe key source."""
        return decode_envelope_strict(message)

    def decode_outbound_message(
        self,
        message: object,
    ) -> ModelEventEnvelope[dict[str, object]]:
        """Decode an outbound record, synthesising an envelope for a flat one.

        Strict decode first, so a real producer-minted envelope is completely
        untouched by this path and keeps its own ``envelope_id`` -- synthesis
        is a fallback for the shape that would otherwise be quarantined, never
        an alternative representation of a record that already decodes.

        Outbound ONLY, deliberately. An inbound record arrives from cloud
        across the tenant trust boundary, and what ``_prepare_inbound``
        validates is the ``source_tenant_id`` / ``source_tenant_principal_id``
        tags the cloud side stamped on the envelope. A synthesised envelope has
        no such tags to check, so synthesising there would convert a validated
        record into an unvalidated one -- which is the boundary this node
        exists to hold. Inbound keeps the strict decode and keeps quarantining.
        """
        try:
            return decode_envelope_strict(message)
        except Exception:
            synthesized = synthesize_outbound_envelope(
                message,
                self._config.tenant_identity,
            )
            if synthesized is None:
                # No usable identity on the wire: re-raise the ORIGINAL decode
                # failure so the caller's quarantine reason still names the
                # real cause instead of a synthesis-specific one.
                raise
            return synthesized

    def _build_status_envelope(
        self,
        status: Literal["active", "degraded"],
        *,
        consecutive_failures: int = 0,
        detail: str = "",
    ) -> tuple[ModelEventEnvelope[dict[str, object]], str]:
        """Build the heartbeat/status envelope plus its canonical topic."""
        identity = self._config.tenant_identity
        now = datetime.now(UTC)
        envelope_id = uuid4()
        heartbeat = ModelGatewayHeartbeat(
            tenant_id=identity.tenant_slug,
            principal_id=identity.principal_id,
            status=status,
            emitted_at=now,
            local_transport_flavor=self._config.local_transport_flavor,
            consecutive_failures=consecutive_failures,
            detail=detail,
        )
        envelope = ModelEventEnvelope[dict[str, object]](
            envelope_id=envelope_id,
            envelope_timestamp=now,
            correlation_id=envelope_id,
            source_tool="gateway-forwarder",
            event_type="omnibase-infra.gateway-heartbeat",
            payload=heartbeat.model_dump(mode="json"),
            metadata=ModelEnvelopeMetadata(
                tags={
                    "source_tenant_id": str(identity.tenant_id),
                    "source_tenant_principal_id": identity.principal_id,
                }
            ),
        )
        canonical_topic = next(
            topic
            for topic in self._config.mirror_topics.outbound
            if topic.endswith(".gateway-heartbeat.v1")
        )
        return envelope, canonical_topic

    async def publish_heartbeat(self) -> None:
        """Publish one tenant-scoped liveness event onto the cloud wire topic
        and mirror the same, untransformed envelope onto the local bus's
        canonical topic.

        OMN-15570 (G3): the local mirror exists because
        NodeGatewayLinkHealthProjectionCompute subscribes to
        ``onex.evt.omnibase-infra.gateway-heartbeat.v1`` on the LOCAL bus
        (bus-is-transport: in-cluster consumers read local canonical
        topics, never the tenant-prefixed cloud wire topic) -- before this
        fix, heartbeats only ever reached the cloud leg, so the projection
        never saw a live event. This is the minimal doctrinally-correct
        fix: it reuses the exact dual-publish shape ``publish_status``
        (OMN-15742/G2) already established for the DEGRADED transition,
        rather than inventing a second mechanism, and publishes the SAME
        envelope object both places so envelope_id/correlation_id stay
        identical across legs instead of minting two envelopes for one
        liveness tick.
        """
        identity = self._config.tenant_identity
        envelope, canonical_topic = self._build_status_envelope("active")
        # OMN-15740: _prepare_outbound returns (transformed_envelope, wire_topic) --
        # the transform-seam handler delegation G0 wired in, not the pre-G0 single
        # return this call site used to unpack. Keep G0's tuple return; G2 only
        # adds the reconnect-supervision status publish below.
        transformed, wire_topic = self._prepare_outbound(envelope, canonical_topic)
        await self._publish_with_delivery_retry(
            bus=self._cloud_bus,
            topic=wire_topic,
            key=str(identity.tenant_id).encode("utf-8"),
            value=self._encode_envelope(transformed),
        )
        await self._publish_with_delivery_retry(
            bus=self._local_bus,
            topic=canonical_topic,
            key=str(identity.tenant_id).encode("utf-8"),
            value=self._encode_envelope(envelope),
        )

    async def publish_status(
        self,
        status: Literal["active", "degraded"],
        *,
        consecutive_failures: int = 0,
        detail: str = "",
    ) -> None:
        """Publish a reconnect-supervision status event onto the LOCAL bus.

        Unlike ``publish_heartbeat`` (which crosses the cloud wire),
        reconnect-supervision status -- most importantly a ``DEGRADED``
        transition -- must stay observable while the cloud leg that caused
        it is itself unreachable, so this publishes on the local bus using
        the same heartbeat topic instead of the cloud leg.
        """
        identity = self._config.tenant_identity
        envelope, canonical_topic = self._build_status_envelope(
            status,
            consecutive_failures=consecutive_failures,
            detail=detail,
        )
        local_only = _stamp_local_only(envelope)
        await self._publish_with_delivery_retry(
            bus=self._local_bus,
            topic=canonical_topic,
            key=str(identity.tenant_id).encode("utf-8"),
            value=self._encode_envelope(local_only),
        )

    async def _publish_with_delivery_retry(
        self,
        *,
        bus: ProtocolGatewayPublisher,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        """Block source acknowledgement until a transient destination recovers.

        ``KafkaTransport.send`` awaits the destination broker acknowledgement.
        The gateway adds a process-lifetime retry around that boundary so the
        delivery node cannot record its durable marker or commit the source
        offset while the message exists on only one broker. Cancellation during
        shutdown is deliberately not caught.
        """
        delay = self._config.forward_retry_initial_seconds
        attempt = 0
        while True:
            try:
                await bus.publish(
                    topic=topic,
                    key=key,
                    value=value,
                    headers=headers,
                )
                return
            except RuntimeHostError as exc:
                attempt += 1
                logger.warning(
                    "Gateway destination unavailable; retaining source message "
                    "and retrying topic=%s attempt=%d delay_seconds=%.1f "
                    "error_type=%s",
                    topic,
                    attempt,
                    delay,
                    type(exc).__name__,
                )
                await self._retry_sleep(delay)
                delay = min(delay * 2, self._config.forward_retry_max_seconds)

    @staticmethod
    def _encode_envelope(
        envelope: ModelEventEnvelope[dict[str, object]],
    ) -> bytes:
        return envelope.model_dump_json(exclude_none=True).encode("utf-8")

    @staticmethod
    def _message_topic(message: object) -> str:
        topic = getattr(message, "topic", None)
        if not isinstance(topic, str) or not topic:
            raise TypeError("gateway bus message must carry its source topic")
        return topic

    def _prepare_inbound(
        self,
        envelope: ModelEventEnvelope[dict[str, object]],
        wire_topic: str,
    ) -> tuple[ModelEventEnvelope[dict[str, object]], str]:
        """Validate a cloud command and stamp its config-bound local tenant.

        Trust-boundary validation (topic declared, tags match the config-bound
        identity) stays here against the untrusted bus input. The prefix-strip
        transform and payload stamp themselves are delegated to the
        contract-declared ``HandlerConsumeInbound`` COMPUTE handler (OMN-15740)
        so there is exactly one implementation of that transform.
        """
        identity = self._config.tenant_identity
        canonical_topic = strip_topic_prefix(identity.tenant_slug, wire_topic)
        if canonical_topic not in self._config.mirror_topics.inbound:
            raise GatewayRecordRefusedError(
                "canonical_topic is not declared for inbound mirroring"
            )

        tags = envelope.metadata.tags
        if tags.get("source_tenant_id") != str(identity.tenant_id):
            raise GatewayRecordRefusedError(
                "envelope tenant_id does not match attached tenant"
            )
        if tags.get("source_tenant_principal_id") != str(identity.principal_id):
            raise GatewayRecordRefusedError(
                "envelope principal_id does not match attached tenant"
            )

        gateway_envelope = ModelGatewayEnvelope(
            tenant_id=identity.tenant_id,
            tenant_slug=identity.tenant_slug,
            envelope_id=envelope.envelope_id,
            correlation_id=envelope.correlation_id,
            event_type=envelope.event_type,
            source_topic=wire_topic,
            wire_topic=wire_topic,
            canonical_topic=canonical_topic,
            payload=envelope.payload,
        )
        transformed_gateway = self._consume_inbound_handler.consume_inbound(
            gateway_envelope
        )

        metadata = envelope.metadata.model_copy(
            update={
                "tags": {
                    **tags,
                    "gateway_tenant_id": str(identity.tenant_id),
                    "gateway_tenant_slug": identity.tenant_slug,
                    "gateway_principal_id": str(identity.principal_id),
                    "gateway_wire_topic": wire_topic,
                    "gateway_canonical_topic": canonical_topic,
                    "gateway_direction": "cloud-to-local",
                }
            }
        )
        return envelope.model_copy(
            update={"payload": transformed_gateway.payload, "metadata": metadata}
        ), canonical_topic

    def _prepare_outbound(
        self,
        envelope: ModelEventEnvelope[dict[str, object]],
        canonical_topic: str,
    ) -> tuple[ModelEventEnvelope[dict[str, object]], str]:
        """Validate a local event and bind it to the attached tenant.

        Trust-boundary validation stays here; the wire-topic prefix transform
        is delegated to the contract-declared ``HandlerForwardOutbound``
        COMPUTE handler (OMN-15740).
        """
        identity = self._config.tenant_identity
        if canonical_topic not in self._config.mirror_topics.outbound:
            raise GatewayRecordRefusedError(
                "canonical_topic is not declared for outbound mirroring"
            )

        payload_tenant = envelope.payload.get("tenant_id")
        if payload_tenant is not None and payload_tenant != identity.tenant_slug:
            raise GatewayRecordRefusedError(
                "outbound payload tenant_id does not match attached tenant"
            )

        tags = envelope.metadata.tags
        existing_tenant_id = tags.get("source_tenant_id")
        if existing_tenant_id is not None and existing_tenant_id != str(
            identity.tenant_id
        ):
            raise GatewayRecordRefusedError(
                "outbound envelope tenant_id does not match attached tenant"
            )
        existing_principal_id = tags.get("source_tenant_principal_id")
        if existing_principal_id is not None and existing_principal_id != str(
            identity.principal_id
        ):
            raise GatewayRecordRefusedError(
                "outbound envelope principal_id does not match attached tenant"
            )

        gateway_envelope = ModelGatewayEnvelope(
            tenant_id=identity.tenant_id,
            tenant_slug=identity.tenant_slug,
            envelope_id=envelope.envelope_id,
            correlation_id=envelope.correlation_id,
            event_type=envelope.event_type,
            source_topic=canonical_topic,
            wire_topic=canonical_topic,
            canonical_topic=canonical_topic,
            payload=envelope.payload,
        )
        transformed_gateway = self._forward_outbound_handler.forward_outbound(
            gateway_envelope
        )
        wire_topic = transformed_gateway.wire_topic

        metadata = envelope.metadata.model_copy(
            update={
                "tags": {
                    **tags,
                    "source_tenant_id": str(identity.tenant_id),
                    "source_tenant_principal_id": str(identity.principal_id),
                    "gateway_tenant_slug": identity.tenant_slug,
                    "gateway_wire_topic": wire_topic,
                    "gateway_canonical_topic": canonical_topic,
                    "gateway_direction": "local-to-cloud",
                }
            }
        )
        return envelope.model_copy(update={"metadata": metadata}), wire_topic
