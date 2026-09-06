# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17201: a destination authorization denial must quarantine, not wedge.

Measured on the lab forwarder (compose project ``omninode-gateway``, image
revision ``2ea74bc4de76``, read 2026-09-06T11:07Z):

    ERROR aiokafka.cluster Topic tenant-beta-gateway-canary-...
      .onex.evt.omniclaude.session-started.v1 is not authorized for this client
    WARNING ...service_gateway_forwarder Gateway destination unavailable;
      retaining source message and retrying topic=... attempt=7
      delay_seconds=30.0 error_type=InfraUnavailableError

``TransportGatewayBus.publish`` mapped every ``KafkaError`` onto
``InfraUnavailableError`` -- "transient" -- so ``_publish_with_delivery_retry``
retained the record and retried it forever. The outbound direction is a single
task, so everything queued behind that record stopped: outbound TOTAL-LAG on
``tenant-beta-gateway-canary-79afa7263852-gateway-forwarder-outbound`` was 523,
510 of it on ``onex.evt.omniclaude.tool-executed.v1`` -- a topic that was never
denied. The container reported ``healthy`` throughout, ``RestartCount 0``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from aiokafka.errors import (
    KafkaConnectionError,
    TopicAuthorizationFailedError,
)

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.idempotency import StoreIdempotencyInmemory
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayEgressHealth,
    ModelGatewayForwarderConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_delivery import (
    NodeGatewayDelivery,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_egress_health import (
    EGRESS_DENIAL_WINDOW_SECONDS,
    evaluate_egress_health,
    load_egress_health,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    GatewayEgressDeniedError,
    ServiceGatewayForwarder,
)
from omnibase_infra.runtime.gateway_forwarder import TransportGatewayBus

# asyncio_mode = "auto" (pyproject) already collects the async tests below;
# a module-level asyncio mark would additionally be applied to the four
# synchronous health-verdict tests and warn on each of them.
pytestmark = pytest.mark.unit

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
PRINCIPAL_ID = "t-33333333333333333333333333333333"
OUTBOUND_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
WIRE_OUTBOUND_TOPIC = f"tenant-acme.{OUTBOUND_TOPIC}"


def _identity() -> ModelGatewayTenantIdentity:
    return ModelGatewayTenantIdentity(
        tenant_id=TENANT_ID,
        tenant_slug="acme",
        principal_id=PRINCIPAL_ID,
    )


def _config() -> ModelGatewayForwarderConfig:
    return ModelGatewayForwarderConfig(
        tenant_identity=_identity(),
        cloud_bus=ModelGatewayCloudBusConfig(
            broker_provider_id=UUID("22222222-2222-2222-2222-222222222222"),
            cloud_broker_ref="gateway.cloud.kafka.broker",
            cloud_auth_ref="gateway.cloud.kafka.oauth",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            client_id_ref="gateway.cloud.kafka.oauth.client_id",
            client_secret_api_key_ref="vault://gateway/redpanda-events",
        ),
        local_transport_flavor="containerized",
        dedupe_store_path=Path.cwd() / "gateway-egress-denial-test.sqlite3",
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
            outbound=(OUTBOUND_TOPIC,),
        ),
        canary=ModelGatewayCanaryConfig(
            topic="onex.evt.omnibase-infra.gateway-canary.v1",
            cadence_seconds=30,
            produce_deadline_seconds=8,
            readback_deadline_seconds=12,
        ),
    )


def _message(offset: int = 7) -> ModelTransportMessage:
    identity = uuid4()
    envelope = ModelEventEnvelope[dict[str, object]](
        envelope_id=identity,
        correlation_id=identity,
        event_type="LlmInferenceResponse",
        payload={"ok": True},
        metadata=ModelEnvelopeMetadata(tags={}),
    )
    return ModelTransportMessage(
        topic=OUTBOUND_TOPIC,
        partition=0,
        offset=offset,
        key=b"tenant-key",
        value=envelope.model_dump_json().encode("utf-8"),
        headers={},
        ack_token=(OUTBOUND_TOPIC, 0, offset),
    )


class _Source:
    """Consumer surface plus the ``send`` the quarantine path dead-letters to."""

    def __init__(self) -> None:
        self.committed: list[object] = []
        self.nacked: list[object] = []
        self.dlq: list[tuple[str, bytes]] = []

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ModelTransportMessage]:
        return []

    async def commit(self, message: object) -> None:
        self.committed.append(message)

    async def nack(self, message: object) -> None:
        self.nacked.append(message)

    async def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Mapping[str, bytes],
    ) -> None:
        self.dlq.append((topic, value))


class _DenyingProducer:
    """A producer whose broker answers TOPIC_AUTHORIZATION_FAILED, as MSK did."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    async def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Mapping[str, bytes] | None = None,
    ) -> None:
        self.attempts.append(topic)
        raise TopicAuthorizationFailedError(
            f"Topic {topic} is not authorized for this client"
        )


class _FlakyProducer:
    """The positive control: a REAL transient broker fault, then success."""

    def __init__(self, *, failures: int) -> None:
        self.remaining = failures
        self.attempts: list[str] = []
        self.delivered: list[str] = []

    async def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Mapping[str, bytes] | None = None,
    ) -> None:
        self.attempts.append(topic)
        if self.remaining > 0:
            self.remaining -= 1
            raise KafkaConnectionError("SYNTHETIC broker connection lost")
        self.delivered.append(topic)


async def _no_sleep(_seconds: float) -> None:
    return None


def _delivery(
    *,
    cloud_bus: object,
    source: _Source,
    egress_health_path: Path | None = None,
) -> NodeGatewayDelivery:
    forwarder = ServiceGatewayForwarder(
        config=_config(),
        local_bus=TransportGatewayBus(_FlakyProducer(failures=0), identity=_identity()),  # type: ignore[arg-type]
        cloud_bus=cloud_bus,  # type: ignore[arg-type]
        retry_sleep=_no_sleep,
    )
    return NodeGatewayDelivery(
        config=_config(),
        forwarder=forwarder,
        local_consumer=source,  # type: ignore[arg-type]
        cloud_consumer=source,  # type: ignore[arg-type]
        idempotency_store=StoreIdempotencyInmemory(),
        egress_health_path=egress_health_path,
    )


# --------------------------------------------------------------------------
# The wedge itself
# --------------------------------------------------------------------------


async def test_topic_authorization_denial_raises_a_permanent_typed_refusal() -> None:
    """The classification, at the boundary that owns the broker taxonomy.

    Before OMN-17201 this raised ``InfraUnavailableError`` -- the type that
    means "the destination is down, retain the record" -- for an answer the
    broker gave about (topic, principal).
    """
    bus = TransportGatewayBus(_DenyingProducer(), identity=_identity())  # type: ignore[arg-type]

    with pytest.raises(GatewayEgressDeniedError) as raised:
        await bus.publish(WIRE_OUTBOUND_TOPIC, b"k", b"v", None)

    denial = raised.value
    assert denial.topic == WIRE_OUTBOUND_TOPIC
    assert denial.tenant_id == str(TENANT_ID)
    assert denial.principal_id == PRINCIPAL_ID
    # The reason must name both, or the operator cannot tell which ACL is
    # missing for which principal from the log line alone.
    assert WIRE_OUTBOUND_TOPIC in str(denial)
    assert PRINCIPAL_ID in str(denial)
    assert str(TENANT_ID) in str(denial)
    assert not isinstance(denial, InfraUnavailableError)


async def test_denied_record_is_quarantined_and_committed_not_retried() -> None:
    """The outage, restated as an assertion: attempt=7 delay=30s, forever."""
    producer = _DenyingProducer()
    source = _Source()
    delivery = _delivery(
        cloud_bus=TransportGatewayBus(producer, identity=_identity()),
        source=source,
    )
    message = _message()

    # Must not raise and must not hang: raising nacks and seeks back, hanging
    # is the retain-and-retry loop that produced attempt=7.
    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert producer.attempts == [WIRE_OUTBOUND_TOPIC], (
        "a denial must be produced exactly once -- retrying re-derives the "
        "same destination topic and is refused identically"
    )
    assert source.committed == [message], (
        "a denied record must be committed past, or the leg re-reads it forever"
    )
    assert source.nacked == []


async def test_denied_record_is_dead_lettered_with_a_naming_reason() -> None:
    """A dropped record must leave forensics, and they must name the class."""
    source = _Source()
    delivery = _delivery(
        cloud_bus=TransportGatewayBus(_DenyingProducer(), identity=_identity()),
        source=source,
    )

    await delivery.deliver_message("outbound", source, _message())  # type: ignore[arg-type]

    assert len(source.dlq) == 1
    payload = json.loads(source.dlq[0][1])
    assert payload["failure_class"] == "gateway_egress_denied_record"
    assert payload["error_type"] == "GatewayEgressDeniedError"
    assert payload["original_topic"] == OUTBOUND_TOPIC
    # The reason is carried as typed fields, not scraped out of the message
    # string: ``sanitize_error_message`` matches the substring "authorization"
    # and collapses any reason containing it to its own type name. The
    # redactor is deliberately left alone -- see _build_quarantine_payload.
    assert payload["denied_topic"] == WIRE_OUTBOUND_TOPIC
    assert payload["denied_principal_id"] == PRINCIPAL_ID
    assert payload["denied_tenant_id"] == str(TENANT_ID)
    assert payload["error_message"] == (
        "GatewayEgressDeniedError: [REDACTED - potentially sensitive data]"
    ), (
        "positive control on the redactor: it still fires on this message, so "
        "the typed fields above are the only reason the operator can read the "
        "verdict -- do not 'fix' this by rewording the exception"
    )


async def test_a_denial_does_not_block_the_next_deliverable_record() -> None:
    """The half that matters operationally: 510 records were stuck behind one.

    ``onex.evt.omniclaude.tool-executed.v1`` was never denied; it was queued
    behind ``session-started``, which was. Asserting only that the denial is
    committed would pass even with the loop left unusable.
    """

    class _DenyOnceProducer:
        def __init__(self) -> None:
            self.denied = False
            self.delivered: list[str] = []

        async def send(
            self,
            topic: str,
            key: bytes | None,
            value: bytes,
            headers: Mapping[str, bytes] | None = None,
        ) -> None:
            if not self.denied:
                self.denied = True
                raise TopicAuthorizationFailedError(
                    f"Topic {topic} is not authorized for this client"
                )
            self.delivered.append(topic)

    producer = _DenyOnceProducer()
    source = _Source()
    delivery = _delivery(
        cloud_bus=TransportGatewayBus(producer, identity=_identity()),
        source=source,
    )

    await delivery.deliver_message("outbound", source, _message(offset=7))  # type: ignore[arg-type]
    good = _message(offset=8)
    await delivery.deliver_message("outbound", source, good)  # type: ignore[arg-type]

    assert producer.delivered == [WIRE_OUTBOUND_TOPIC]
    assert good in source.committed


# --------------------------------------------------------------------------
# Positive control: what must NOT change
# --------------------------------------------------------------------------


async def test_a_transient_broker_fault_still_retries_and_still_delivers() -> None:
    """The scope guard. Without this, the fix reads as "stop retrying".

    A connection-level ``KafkaError`` is the case the retain-and-retry loop
    exists for. It must still be retained, retried, and eventually delivered
    -- never quarantined, because committing past a real broker fault is
    silent data loss.
    """
    producer = _FlakyProducer(failures=3)
    source = _Source()
    delivery = _delivery(
        cloud_bus=TransportGatewayBus(producer, identity=_identity()),
        source=source,
    )
    message = _message()

    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert len(producer.attempts) == 4, "a transient fault must be retried"
    assert producer.delivered == [WIRE_OUTBOUND_TOPIC]
    assert source.committed == [message]
    assert source.dlq == [], "a transient fault must never be dead-lettered"


async def test_a_transient_broker_fault_is_still_typed_transient() -> None:
    """The classification boundary's own positive control."""
    bus = TransportGatewayBus(_FlakyProducer(failures=1), identity=_identity())  # type: ignore[arg-type]

    with pytest.raises(InfraUnavailableError):
        await bus.publish(WIRE_OUTBOUND_TOPIC, b"k", b"v", None)


# --------------------------------------------------------------------------
# The health surface the quarantine makes mandatory
# --------------------------------------------------------------------------


async def test_denials_and_deliveries_are_counted_and_published(
    tmp_path: Path,
) -> None:
    """Quarantining silently would be a worse failure than the wedge."""
    state_path = tmp_path / "egress-health.json"
    source = _Source()
    delivery = _delivery(
        cloud_bus=TransportGatewayBus(_DenyingProducer(), identity=_identity()),
        source=source,
        egress_health_path=state_path,
    )

    await delivery.deliver_message("outbound", source, _message())  # type: ignore[arg-type]

    published = load_egress_health(state_path)
    assert published is not None
    assert published.denied_total == 1
    assert published.delivered_total == 0
    assert published.last_denied_topic == WIRE_OUTBOUND_TOPIC
    assert published.last_denied_principal_id == PRINCIPAL_ID
    assert published.last_denied_tenant_id == str(TENANT_ID)


async def test_a_successful_outbound_delivery_is_counted(tmp_path: Path) -> None:
    state_path = tmp_path / "egress-health.json"
    source = _Source()
    delivery = _delivery(
        cloud_bus=TransportGatewayBus(_FlakyProducer(failures=0), identity=_identity()),
        source=source,
        egress_health_path=state_path,
    )

    await delivery.deliver_message("outbound", source, _message())  # type: ignore[arg-type]

    published = load_egress_health(state_path)
    assert published is not None
    assert published.delivered_total == 1
    assert published.denied_total == 0


def test_health_is_unhealthy_when_everything_is_denied() -> None:
    """The condition the container must not report healthy on."""
    now = datetime.now(UTC)
    state = ModelGatewayEgressHealth(
        denied_total=284,
        delivered_total=0,
        last_denied_at=now - timedelta(seconds=5),
        last_denied_topic=WIRE_OUTBOUND_TOPIC,
        last_denied_tenant_id=str(TENANT_ID),
        last_denied_principal_id=PRINCIPAL_ID,
    )

    passed, detail = evaluate_egress_health(state, now=now)

    assert passed is False
    assert WIRE_OUTBOUND_TOPIC in detail
    assert PRINCIPAL_ID in detail


def test_health_passes_while_records_are_still_crossing() -> None:
    """A partial ACL gap is degraded, not dead -- and must not flap."""
    now = datetime.now(UTC)
    state = ModelGatewayEgressHealth(
        denied_total=12,
        delivered_total=900,
        last_denied_at=now - timedelta(seconds=1),
        last_denied_topic=WIRE_OUTBOUND_TOPIC,
        last_denied_tenant_id=str(TENANT_ID),
        last_denied_principal_id=PRINCIPAL_ID,
        # Older than the denial: a strict last-writer comparison would call
        # this unhealthy and flip back on the next record.
        last_delivered_at=now - timedelta(seconds=3),
    )

    passed, detail = evaluate_egress_health(state, now=now)

    assert passed is True
    assert "denied" in detail, "a degraded leg must still say so on the PASS line"


def test_health_recovers_once_denials_stop() -> None:
    """A granted ACL must clear the verdict without a restart."""
    now = datetime.now(UTC)
    state = ModelGatewayEgressHealth(
        denied_total=284,
        delivered_total=0,
        last_denied_at=now - timedelta(seconds=EGRESS_DENIAL_WINDOW_SECONDS + 1),
        last_denied_topic=WIRE_OUTBOUND_TOPIC,
        last_denied_tenant_id=str(TENANT_ID),
        last_denied_principal_id=PRINCIPAL_ID,
    )

    passed, _ = evaluate_egress_health(state, now=now)

    assert passed is True


def test_absent_state_is_not_a_failure(tmp_path: Path) -> None:
    """Failing closed here would recreate the sentinel-file coupling."""
    assert load_egress_health(tmp_path / "missing.json") is None
    passed, detail = evaluate_egress_health(None, now=datetime.now(UTC))
    assert passed is True
    assert "no denial state" in detail
