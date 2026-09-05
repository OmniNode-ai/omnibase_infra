# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane-to-lane mirror leg for the gateway forwarder (OMN-17034).

Named ``NodeLaneMirror``, matching its sibling ``NodeGatewayDelivery`` in this
same node, because ``Service*`` is a hard-fail shape under the OMN-14350
non-canonical lifecycle-class ratchet. The one ``Service*`` class here
(``ServiceGatewayForwarder``) is a pre-existing residual in that ratchet's
frozen allowlist, which may only SHRINK -- adding brand-new code to it would
be using an allowlist as a fix.

This is the third leg of ``node_bus_forwarder_effect``, and it is deliberately
NOT part of ``NodeGatewayDelivery``. That service owns the two *trust-boundary*
directions (local->cloud outbound, cloud->local inbound) whose defining work is
the tenant prefix transform and the tenant stamp. A lane mirror crosses no trust
boundary: both endpoints are lane brokers on one operator-controlled host, and
the record is republished byte-for-byte under its own canonical topic name. It
carries the same durable-marker ordering guarantee as the trust-boundary legs
and reuses the same store, but folding it into ``NodeGatewayDelivery`` would
have meant adding a third member to ``_DIRECTIONS`` and then branching every
transform, validation, quarantine and watchdog path on "except for this one".

It declares NO protocols of its own: the source is the node's existing
``ProtocolGatewayConsumer`` (identical poll/commit/nack surface) and each mirror
is its existing ``ProtocolGatewayPublisher``, reached through the same
``TransportGatewayBus`` adapter the trust-boundary legs already use. A first
revision declared two new ones and the OMN-12912 protocol-ownership gate caught
it -- correctly: infra should implement protocols, not mint near-duplicates.

It does NOT decode the record (OMN-17919). A first revision obtained its
idempotency key by round-tripping the value through ``ModelEventEnvelope``, and
that assumption is what made this leg inert on deploy: the Claude Code hook edge
publishes a FLAT hook payload with the envelope metadata in Kafka HEADERS, so
``ModelEventEnvelope`` -- ``extra="forbid"``, ``payload`` required -- rejected
every single record. 261 of 261 records were refused, the mirror task exited,
and the consumer group went ``Dead`` while the dev high-water marks never moved.

The envelope round-trip was never needed. A byte-for-byte republisher has no use
for a parsed body; the only thing it must read is the identity it keys its
durable marker on, and that is already on the wire as the mandatory
``message_id`` header ``event_bus_kafka._model_headers_to_kafka`` stamps on
every publish. Reading the header instead of the body is what makes this leg
agnostic to the body shape, which is the property a byte-for-byte mirror should
have had from the start. The trust-boundary legs still decode, and must: the
tenant prefix transform and the tenant stamp are operations on a parsed
envelope, not on opaque bytes.

Ordering contract, identical in shape to ``NodeGatewayDelivery``'s and for the
same reason (cross-cluster Kafka has no distributed transaction):

1. read the source record's ``message_id`` header as its canonical identity;
2. if the durable marker already exists, publish nothing and commit the source;
3. otherwise publish to EVERY declared mirror lane and await each broker ack;
4. durably record the identity;
5. commit the source offset.

A failure at step 3 or 4 nacks the source, so the record is redelivered and the
marker check at step 2 makes the retry safe. The residual window is a crash
between the last mirror ack and the marker write: that redelivers and
republishes, which is at-least-once and is exactly what the two existing legs
already promise (``delivery.semantics: at_least_once`` in the node contract).
Downstream consumers key on the same ``envelope_id``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from uuid import UUID

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.errors import LaneMirrorRecordRefusedError
from omnibase_infra.idempotency import ProtocolIdempotencyStore
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayLaneMirrorConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_delivery import (
    ProtocolGatewayConsumer,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ProtocolGatewayPublisher,
)
from omnibase_infra.utils import sanitize_error_message

_MESSAGE_ID_HEADER = "message_id"
_CORRELATION_ID_HEADER = "correlation_id"

logger = logging.getLogger(__name__)


class NodeLaneMirror:
    """Mirror the contract-declared topic set from one lane broker to others."""

    def __init__(
        self,
        *,
        config: ModelGatewayLaneMirrorConfig,
        source_consumer: ProtocolGatewayConsumer,
        mirror_producers: Mapping[str, ProtocolGatewayPublisher],
        idempotency_store: ProtocolIdempotencyStore,
    ) -> None:
        missing = set(config.mirror_lanes) - set(mirror_producers)
        if missing:
            raise ValueError(
                "every contract-declared mirror lane requires a producer; "
                f"missing: {sorted(missing)}"
            )
        self._config = config
        self._source = source_consumer
        self._producers = dict(mirror_producers)
        self._store = idempotency_store
        self._allowed_topics = frozenset(config.topics)
        # OMN-17919: a record the mirror cannot identify is refused, and the
        # refusal is COUNTED. Silence is the failure mode this leg already had
        # once -- a mirror that commits past everything it cannot read looks
        # identical to a mirror that is working and simply has no traffic.
        self._refused_record_count = 0
        # Namespaced away from the trust-boundary legs' ``gateway:<slug>``
        # domain on purpose: one envelope can legitimately be both mirrored
        # across lanes AND forwarded to cloud, and a shared domain would make
        # whichever leg ran first suppress the other.
        self._domain = f"lane-mirror:{config.source_lane}"

    @property
    def refused_record_count(self) -> int:
        """Records refused for carrying no usable identity, this process.

        A non-zero and rising value means a producer on the source lane is
        emitting records this leg cannot key, which is exactly the condition
        that previously presented as a silently inert mirror (OMN-17919).
        """
        return self._refused_record_count

    async def drain_once(self) -> int:
        """Poll one batch from the source and mirror it. Returns records handled."""
        batch = await self._source.poll(
            max_messages=self._config.max_messages_per_poll,
            timeout_ms=self._config.poll_timeout_ms,
        )
        handled = 0
        for message in batch:
            await self._mirror_one(message)
            handled += 1
        return handled

    async def run(self, shutdown_event: asyncio.Event) -> None:
        """Drain until shutdown is signalled."""
        while not shutdown_event.is_set():
            drained = await self.drain_once()
            if drained == 0:
                await asyncio.sleep(0)

    def _record_identity(
        self, message: ModelTransportMessage
    ) -> tuple[UUID, UUID | None]:
        """Read the record's canonical identity from its Kafka headers.

        ``message_id`` is a MANDATORY header: ``ModelEventHeaders`` declares it
        and ``event_bus_kafka._model_headers_to_kafka`` stamps it on every
        publish, so its absence means the record did not come from the ONEX
        event bus at all. It is refused rather than defaulted -- minting a key
        for an unidentifiable record would make the durable marker meaningless
        and turn the leg's exactly-once promise into a coin flip on redelivery.

        ``correlation_id`` is advisory here: it is carried into the marker for
        chain tracing only, and a missing or malformed one does not stop a
        record that is otherwise identifiable from crossing.
        """
        raw_message_id = message.headers.get(_MESSAGE_ID_HEADER)
        if raw_message_id is None:
            raise LaneMirrorRecordRefusedError(
                "lane mirror record carries no message_id header",
                topic=message.topic,
                partition=message.partition,
                offset=message.offset,
                header_keys=sorted(message.headers),
            )
        try:
            envelope_id = UUID(raw_message_id.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as parse_error:
            raise LaneMirrorRecordRefusedError(
                "lane mirror record message_id header is not a UUID",
                topic=message.topic,
                partition=message.partition,
                offset=message.offset,
            ) from parse_error

        raw_correlation_id = message.headers.get(_CORRELATION_ID_HEADER)
        correlation_id: UUID | None = None
        if raw_correlation_id is not None:
            try:
                correlation_id = UUID(raw_correlation_id.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                logger.warning(
                    "Lane mirror record correlation_id header is not a UUID "
                    "topic=%s partition=%s offset=%s -- mirroring without it",
                    message.topic,
                    message.partition,
                    message.offset,
                )
        return envelope_id, correlation_id

    async def _mirror_one(self, message: ModelTransportMessage) -> None:
        if message.topic not in self._allowed_topics:
            # Contract-declared topics only -- the same rule the cloud legs
            # obey. A subscription can drift wider than the contract (a
            # wildcard, a hand-edited group); the contract is what decides
            # what crosses, not what happened to arrive.
            logger.warning(
                "Lane mirror skipped undeclared topic=%s partition=%s offset=%s",
                message.topic,
                message.partition,
                message.offset,
            )
            await self._source.commit(message)
            return

        try:
            envelope_id, correlation_id = self._record_identity(message)
        except LaneMirrorRecordRefusedError as refusal:
            # Same policy as NodeGatewayDelivery's quarantine path: a record
            # that can never be identified would be refused again on every
            # redelivery (OMN-15748 poison-pill DoS), so commit past it rather
            # than nack. The counter above is what keeps that from being a
            # silent drop.
            self._refused_record_count += 1
            # ``exception`` (ERROR + traceback), matching the sibling
            # quarantine path in service_gateway_delivery.py: a refusal here
            # means a producer on the source lane is emitting records this leg
            # cannot key, and the stack is what says which check refused it.
            logger.exception(
                "Lane mirror REFUSED record topic=%s partition=%s offset=%s "
                "error=%s refused_record_count=%d -- committing past it "
                "without mirroring",
                message.topic,
                message.partition,
                message.offset,
                sanitize_error_message(refusal),
                self._refused_record_count,
            )
            await self._source.commit(message)
            return

        try:
            if await self._store.is_processed(envelope_id, domain=self._domain):
                logger.info(
                    "Lane mirror duplicate suppressed envelope_id=%s topic=%s "
                    "partition=%s offset=%s",
                    envelope_id,
                    message.topic,
                    message.partition,
                    message.offset,
                )
                await self._source.commit(message)
                return

            for lane in self._config.mirror_lanes:
                await self._producers[lane].publish(
                    message.topic,
                    message.key,
                    message.value,
                    dict(message.headers),
                )

            await self._store.mark_processed(
                envelope_id,
                domain=self._domain,
                correlation_id=correlation_id,
            )
            await self._source.commit(message)
            logger.info(
                "Lane mirror delivered envelope_id=%s topic=%s source_lane=%s "
                "mirror_lanes=%s source_partition=%s source_offset=%s",
                envelope_id,
                message.topic,
                self._config.source_lane,
                ",".join(self._config.mirror_lanes),
                message.partition,
                message.offset,
            )
        except asyncio.CancelledError:
            raise
        except Exception as mirror_error:
            logger.exception(
                "Lane mirror failed envelope_id=%s topic=%s partition=%s "
                "offset=%s error=%s -- source offset left uncommitted",
                envelope_id,
                message.topic,
                message.partition,
                message.offset,
                sanitize_error_message(mirror_error),
            )
            await self._source.nack(message)


__all__ = ["NodeLaneMirror"]
