# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.idempotency import StoreIdempotencyInmemory, StoreIdempotencySqlite
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
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
from omnibase_infra.runtime.gateway_forwarder import TransportGatewayBus

pytestmark = pytest.mark.asyncio

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
PRINCIPAL_ID = "t-33333333333333333333333333333333"
OUTBOUND_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
HEARTBEAT_TOPIC = "onex.evt.omnibase-infra.gateway-heartbeat.v1"
WIRE_OUTBOUND_TOPIC = f"tenant-acme.{OUTBOUND_TOPIC}"
WIRE_HEARTBEAT_TOPIC = f"tenant-acme.{HEARTBEAT_TOPIC}"


class _RecordingBus:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.sent: list[tuple[str, bytes]] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.events.append("destination_ack")
        self.sent.append((topic, value))


class _BlockingBus(_RecordingBus):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.publish_calls = 0

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.publish_calls += 1
        self.entered.set()
        await self.release.wait()
        await super().publish(topic, key, value, headers)


class _Source:
    def __init__(self, events: list[str], *, fail_commit: bool = False) -> None:
        self.events = events
        self.fail_commit = fail_commit
        self.committed: list[object] = []
        self.nacked: list[object] = []
        self.sent: list[tuple[str, bytes]] = []

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Mapping[str, bytes],
    ) -> None:
        self.events.append("dlq_send")
        self.sent.append((topic, value))

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ModelTransportMessage]:
        # A real transport's poll always blocks up to timeout_ms (or yields
        # on an actual socket read) before returning empty -- it never
        # returns synchronously with zero awaits inside. A poll that never
        # suspends starves the asyncio event loop when driven by
        # NodeGatewayDelivery's real `_run_direction` task loop (a bare
        # `while True: await source.poll(...)` with nothing else to yield
        # on): `task.cancel()` only takes effect at the next real suspension
        # point, so a non-yielding poll makes the loop uncancellable and
        # `delivery.stop()` hangs forever awaiting it. Only
        # test_real_outbound_consumer_loop_does_not_reforward_degraded_status
        # drives this fake through the real task loop (every other _Source
        # usage in this file calls deliver_message directly, never through
        # poll()), so this sleep costs that one test ~poll_timeout_ms and
        # nothing else.
        await asyncio.sleep(timeout_ms / 1000)
        return []

    async def commit(self, message: object) -> None:
        self.events.append("source_commit")
        if self.fail_commit:
            raise RuntimeError("commit unavailable")
        self.committed.append(message)

    async def nack(self, message: object) -> None:
        self.events.append("source_nack")
        self.nacked.append(message)


class _RecordingStore(StoreIdempotencyInmemory):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def mark_processed(self, *args: object, **kwargs: object) -> None:
        self.events.append("durable_marker")
        await super().mark_processed(*args, **kwargs)  # type: ignore[arg-type]


def _config() -> ModelGatewayForwarderConfig:
    return ModelGatewayForwarderConfig(
        tenant_identity=ModelGatewayTenantIdentity(
            tenant_id=TENANT_ID,
            tenant_slug="acme",
            principal_id=PRINCIPAL_ID,
        ),
        cloud_bus=ModelGatewayCloudBusConfig(
            broker_provider_id=UUID("22222222-2222-2222-2222-222222222222"),
            cloud_broker_ref="gateway.cloud.kafka.broker",
            cloud_auth_ref="gateway.cloud.kafka.oauth",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            client_id_ref="gateway.cloud.kafka.oauth.client_id",
            client_secret_api_key_ref="infisical://gateway/redpanda-events",
        ),
        local_transport_flavor="containerized",
        dedupe_store_path=Path.cwd() / "gateway-test.sqlite3",
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


def _message(envelope_id: UUID | None = None) -> ModelTransportMessage:
    identity = envelope_id or uuid4()
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
        offset=7,
        key=b"tenant-key",
        value=envelope.model_dump_json().encode("utf-8"),
        headers={"traceparent": b"00-test"},
        ack_token=(OUTBOUND_TOPIC, 0, 7),
    )


def _delivery(
    events: list[str],
    source: _Source,
    store: StoreIdempotencyInmemory | StoreIdempotencySqlite,
) -> tuple[NodeGatewayDelivery, _RecordingBus]:
    local_bus = _RecordingBus(events)
    cloud_bus = _RecordingBus(events)
    forwarder = ServiceGatewayForwarder(
        config=_config(),
        local_bus=local_bus,  # type: ignore[arg-type]
        cloud_bus=cloud_bus,  # type: ignore[arg-type]
    )
    return (
        NodeGatewayDelivery(
            config=_config(),
            forwarder=forwarder,
            local_consumer=source,  # type: ignore[arg-type]
            cloud_consumer=source,  # type: ignore[arg-type]
            idempotency_store=store,
        ),
        cloud_bus,
    )


async def test_destination_ack_precedes_marker_and_source_commit() -> None:
    events: list[str] = []
    source = _Source(events)
    store = _RecordingStore(events)
    delivery, cloud_bus = _delivery(events, source, store)
    message = _message()

    await delivery.deliver_message("outbound", source, message)  # type: ignore[arg-type]

    assert events == ["destination_ack", "durable_marker", "source_commit"]
    assert len(cloud_bus.sent) == 1
    assert cloud_bus.sent[0][0] == WIRE_OUTBOUND_TOPIC
    assert source.committed == [message]


async def test_restart_after_commit_failure_suppresses_republish(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gateway-delivery.sqlite3"
    envelope_id = uuid4()
    message = _message(envelope_id)
    first_events: list[str] = []
    first_source = _Source(first_events, fail_commit=True)
    first_store = StoreIdempotencySqlite(path)
    await first_store.start()
    first_delivery, first_cloud = _delivery(first_events, first_source, first_store)

    with pytest.raises(RuntimeError, match="commit unavailable"):
        await first_delivery.deliver_message(
            "outbound",
            first_source,
            message,  # type: ignore[arg-type]
        )
    await first_store.close()

    restarted_events: list[str] = []
    restarted_source = _Source(restarted_events)
    restarted_store = StoreIdempotencySqlite(path)
    await restarted_store.start()
    restarted_delivery, restarted_cloud = _delivery(
        restarted_events,
        restarted_source,
        restarted_store,
    )
    await restarted_delivery.deliver_message(
        "outbound",
        restarted_source,
        message,  # type: ignore[arg-type]
    )

    assert len(first_cloud.sent) == 1
    assert restarted_cloud.sent == []
    assert restarted_events == ["source_commit"]
    assert restarted_source.committed == [message]


async def test_concurrent_duplicate_is_published_once() -> None:
    events: list[str] = []
    source = _Source(events)
    store = StoreIdempotencyInmemory()
    local_bus = _RecordingBus(events)
    cloud_bus = _BlockingBus(events)
    config = _config()
    delivery = NodeGatewayDelivery(
        config=config,
        forwarder=ServiceGatewayForwarder(
            config=config,
            local_bus=local_bus,  # type: ignore[arg-type]
            cloud_bus=cloud_bus,  # type: ignore[arg-type]
        ),
        local_consumer=source,  # type: ignore[arg-type]
        cloud_consumer=source,  # type: ignore[arg-type]
        idempotency_store=store,
    )
    message = _message()

    first = asyncio.create_task(
        delivery.deliver_message("outbound", source, message),  # type: ignore[arg-type]
    )
    await asyncio.wait_for(cloud_bus.entered.wait(), timeout=1)
    duplicate = asyncio.create_task(
        delivery.deliver_message("outbound", source, message),  # type: ignore[arg-type]
    )
    for _ in range(5):
        await asyncio.sleep(0)

    assert cloud_bus.publish_calls == 1
    cloud_bus.release.set()
    await asyncio.gather(first, duplicate)

    assert len(cloud_bus.sent) == 1
    assert source.committed == [message, message]


async def test_store_failure_nacks_without_destination_dispatch() -> None:
    events: list[str] = []
    source = _Source(events)
    unavailable_store = StoreIdempotencySqlite(Path("/unused/not-started.sqlite3"))
    delivery, cloud_bus = _delivery(events, source, unavailable_store)
    message = _message()

    with pytest.raises(RuntimeError, match="not started"):
        await delivery.deliver_message(
            "outbound",
            source,
            message,  # type: ignore[arg-type]
        )

    assert cloud_bus.sent == []
    assert source.committed == []
    assert source.nacked == [message]
    assert events == ["source_nack"]


class _SharedLocalTransport:
    """A single fake object playing BOTH runtime roles ``local_transport`` plays.

    ``runtime/gateway_forwarder.py:run_gateway_forwarder`` builds exactly one
    ``KafkaTransport`` as ``local_transport``, wraps it in
    ``TransportGatewayBus`` for ``local_bus`` (what ``publish_status`` and the
    heartbeat local mirror publish INTO), and passes the SAME object as
    ``local_consumer`` to ``NodeGatewayDelivery`` (what the outbound direction
    polls FROM). Every other test in this module uses two separate fakes
    (``local_bus`` / ``local_consumer``) that never actually connect, so none
    of them can observe a message published locally being re-polled by the
    forwarder's own outbound consumer loop and re-forwarded to cloud
    (OMN-15570/OMN-15742 reconciliation finding D1). This fake connects
    ``send`` (the ``ProtocolTransportProducer`` surface ``TransportGatewayBus``
    calls) directly to ``poll``/``commit``/``nack`` (the
    ``ProtocolGatewayConsumer`` surface ``NodeGatewayDelivery`` polls) through
    one in-memory queue, reproducing the real wiring.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[ModelTransportMessage] = asyncio.Queue()
        self.sent: list[tuple[str, bytes]] = []
        self.committed: list[object] = []
        self._offset = 0

    async def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Mapping[str, bytes] | None = None,
    ) -> None:
        self._offset += 1
        message = ModelTransportMessage(
            topic=topic,
            partition=0,
            offset=self._offset,
            key=key,
            value=value,
            headers=dict(headers or {}),
            ack_token=(topic, 0, self._offset),
        )
        self.sent.append((topic, value))
        await self._queue.put(message)

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ModelTransportMessage]:
        try:
            message = await asyncio.wait_for(
                self._queue.get(), timeout=timeout_ms / 1000
            )
        except TimeoutError:
            return []
        return [message]

    async def commit(self, message: object) -> None:
        self.committed.append(message)

    async def nack(self, message: object) -> None:  # pragma: no cover - defensive
        pass


async def test_real_outbound_consumer_loop_does_not_reforward_degraded_status() -> None:
    """OMN-15742 reconciliation D1 (real-dispatch-path regression).

    ``publish_status`` publishes DEGRADED straight onto the local bus's
    canonical outbound topic. That topic is exactly what the forwarder's own
    outbound consumer (``NodeGatewayDelivery`` polling ``local_consumer``,
    the SAME transport object ``local_bus`` publishes into in the real
    runtime -- see ``_SharedLocalTransport`` above) is subscribed to. Every
    existing DEGRADED test (``test_publish_status_degraded_goes_to_local_bus_not_cloud``
    in ``test_gateway_forwarder_service.py``) uses a bare ``_MockGatewayBus``
    with no consumer loop at all, so it cannot see the DEGRADED status get
    picked back up and re-forwarded to cloud. This test wires the real
    ``NodeGatewayDelivery`` consumer loop against a transport that actually
    connects publish to poll, and drives it end to end.

    Pre-fix: FAILS -- the untagged DEGRADED envelope round-trips through the
    outbound consumer loop, ``_forward_outbound_message``'s loopback skip
    (``gateway_direction == "cloud-to-local"``) does not match, and the
    envelope reaches ``cloud_bus`` a second time carrying the DEGRADED
    payload.
    """
    events: list[str] = []
    shared_local_transport = _SharedLocalTransport()
    cloud_bus = _RecordingBus(events)
    idle_cloud_consumer = _Source(events)
    config = _config().model_copy(
        update={
            "mirror_topics": ModelGatewayMirrorTopics(
                inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
                outbound=(OUTBOUND_TOPIC, HEARTBEAT_TOPIC),
            )
        }
    )
    local_bus = TransportGatewayBus(shared_local_transport)  # type: ignore[arg-type]
    forwarder = ServiceGatewayForwarder(
        config=config,
        local_bus=local_bus,
        cloud_bus=cloud_bus,  # type: ignore[arg-type]
    )
    delivery = NodeGatewayDelivery(
        config=config,
        forwarder=forwarder,
        local_consumer=shared_local_transport,  # type: ignore[arg-type]
        cloud_consumer=idle_cloud_consumer,  # type: ignore[arg-type]
        idempotency_store=StoreIdempotencyInmemory(),
        poll_timeout_ms=50,
    )

    await delivery.start()
    try:
        await forwarder.publish_status(
            "degraded",
            consecutive_failures=3,
            detail="InfraUnavailableError: boom",
        )
        for _ in range(100):
            if shared_local_transport.committed:
                break
            await asyncio.sleep(0.01)
    finally:
        await delivery.stop()

    assert shared_local_transport.committed, (
        "outbound consumer loop never polled the local-bus DEGRADED publish "
        "-- test wiring is broken, not proving anything"
    )
    degraded_on_cloud = [
        value for _topic, value in cloud_bus.sent if b'"degraded"' in value
    ]
    assert degraded_on_cloud == [], (
        "DEGRADED status leaked to the cloud leg via the outbound consumer "
        f"loop re-forwarding the local-bus mirror: {degraded_on_cloud!r}"
    )


# --- OMN-15748: poison-pill quarantine + membership-loss watchdog ----------


async def test_undecodable_record_is_quarantined_not_crashed() -> None:
    """decode_message() failure must be quarantined, never propagate raw.

    Pre-fix: ``_deliver_message_locked`` calls ``decode_message()`` BEFORE
    the try/except that nacks, so a malformed record raises straight out of
    ``deliver_message`` -- exactly the live 2026-08-08T16:50Z crash trigger
    (verbatim pydantic ``ValidationError`` traceback ending in
    ``sys.exit(main())``). Nacking it (the ordinary failure path) would also
    be wrong: a permanently malformed record redelivers via nack's ``seek``
    and re-crashes forever. It must be logged, dead-lettered, and committed
    past (skipped) so the consumer loop keeps making forward progress.
    """
    events: list[str] = []
    source = _Source(events)
    store = _RecordingStore(events)
    delivery, cloud_bus = _delivery(events, source, store)
    poison = ModelTransportMessage(
        topic=OUTBOUND_TOPIC,
        partition=0,
        offset=3,
        key=b"tenant-key",
        value=b"SYNTHETIC-not-valid-json{{{",
        headers={},
        ack_token=(OUTBOUND_TOPIC, 0, 3),
    )

    # Must not raise -- this is the process-crash regression.
    await delivery.deliver_message("outbound", source, poison)  # type: ignore[arg-type]

    assert source.committed == [poison], (
        "poison record must be committed (skipped), not left uncommitted"
    )
    assert source.nacked == [], (
        "must not nack a permanently undecodable record -- nack seeks back "
        "to the same offset and re-crashes forever (poison-loop)"
    )
    assert cloud_bus.sent == [], (
        "an undecodable record must never reach the destination"
    )
    assert source.sent, "undecodable record must be dead-lettered, not silently dropped"
    dlq_topic, dlq_value = source.sent[0]
    assert dlq_topic == "onex.dlq.omnibase-infra.events.v1"
    payload = json.loads(dlq_value)
    assert payload["original_topic"] == OUTBOUND_TOPIC
    assert payload["original_partition"] == 0
    assert payload["original_offset"] == 3
    assert payload["direction"] == "outbound"
    assert payload["failure_class"] == "gateway_undecodable_record"


class _StallThenRecoverSource:
    """Reproduces the live 2026-08-09T10:03:07Z silent-eviction mechanism.

    aiokafka's client-side ``max_poll_interval_ms`` idle-eviction fires from
    a SEPARATE asyncio task (``GroupCoordinator._heartbeat_routine``) than
    the poll loop -- it force-leaves the group and the poll loop's own
    ``poll()`` call simply never returns and never raises. The only way out
    is a brand-new client instance (a fresh group join), which is exactly
    what ``close()`` + ``start()`` on a real ``KafkaTransport`` does. This
    fake models that: the first ``poll()`` hangs forever with zero
    exception; only after ``close()``/``start()`` are called does it behave
    like a healthy transport again (real suspension via ``asyncio.sleep``,
    not a busy-spin return -- the same discipline
    ``test_real_outbound_consumer_loop_does_not_reforward_degraded_status``
    documents above).
    """

    def __init__(self) -> None:
        self.poll_calls = 0
        self.start_calls = 0
        self.close_calls = 0
        self.committed: list[object] = []
        self.nacked: list[object] = []
        self._recreated = False
        self._stuck: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        self.start_calls += 1
        self._recreated = True

    async def close(self) -> None:
        self.close_calls += 1

    def has_group_membership(self) -> bool:
        return self._recreated

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ModelTransportMessage]:
        self.poll_calls += 1
        if not self._recreated:
            await self._stuck.wait()  # never set: permanent stall, no exception
            return []  # pragma: no cover - unreachable in this test
        await asyncio.sleep(timeout_ms / 1000)
        return []

    async def commit(self, message: object) -> None:
        self.committed.append(message)

    async def nack(self, message: object) -> None:
        self.nacked.append(message)


async def test_membership_loss_watchdog_recreates_and_recovers() -> None:
    """OMN-15748/OMN-15690: silent LeaveGroup (zero exception) must recover.

    Pre-fix: there is no watchdog at all. The only recovery mechanism is
    ``runtime/gateway_forwarder.py``'s ``_supervise_gateway_delivery``, which
    is exception-triggered off ``delivery.wait()`` -- structurally unable to
    observe a task that is alive and hung, never done, never exceptioned.
    That is the adjudicated live failure mode: the process is left with
    MEMBERS=0 on one direction forever, no rejoin, no restart, no alert.
    """
    local_bus = _RecordingBus([])
    cloud_bus = _RecordingBus([])
    config = _config().model_copy(
        update={
            "heartbeat_interval_seconds": 1,
            "max_silence_window_seconds": 2,
            "mirror_topics": ModelGatewayMirrorTopics(
                inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
                outbound=(OUTBOUND_TOPIC, HEARTBEAT_TOPIC),
            ),
        }
    )
    forwarder = ServiceGatewayForwarder(
        config=config,
        local_bus=local_bus,  # type: ignore[arg-type]
        cloud_bus=cloud_bus,  # type: ignore[arg-type]
    )
    stalled_outbound = _StallThenRecoverSource()
    idle_inbound = _Source([])
    delivery = NodeGatewayDelivery(
        config=config,
        forwarder=forwarder,
        local_consumer=stalled_outbound,  # type: ignore[arg-type]
        cloud_consumer=idle_inbound,  # type: ignore[arg-type]
        idempotency_store=StoreIdempotencyInmemory(),
        poll_timeout_ms=50,
    )

    await delivery.start()
    try:
        for _ in range(200):
            if stalled_outbound.start_calls:
                break
            await asyncio.sleep(0.02)
        assert stalled_outbound.start_calls >= 1, (
            "watchdog never force-recreated the stalled transport -- the "
            "silent-eviction mode was never detected"
        )
        assert stalled_outbound.close_calls >= 1

        # Prove actual recovery (forward progress resumes), not merely a
        # crash-free wait.
        polls_at_recreate = stalled_outbound.poll_calls
        for _ in range(100):
            if stalled_outbound.poll_calls > polls_at_recreate:
                break
            await asyncio.sleep(0.02)
        assert stalled_outbound.poll_calls > polls_at_recreate, (
            "recreated direction never resumed polling"
        )

        degraded = [value for _topic, value in local_bus.sent if b'"degraded"' in value]
        assert degraded, "membership-loss recovery must publish a DEGRADED status"

        # No task death: the exception-triggered supervision surface
        # (delivery.wait()) must never see this recovery -- it is handled
        # entirely inside the watchdog.
        wait_task = asyncio.create_task(delivery.wait())
        done, _pending = await asyncio.wait({wait_task}, timeout=0.2)
        assert wait_task not in done, "delivery loop task must not have died"
        wait_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await wait_task
    finally:
        await delivery.stop()
