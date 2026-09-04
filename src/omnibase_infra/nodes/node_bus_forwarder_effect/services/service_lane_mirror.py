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

Ordering contract, identical in shape to ``NodeGatewayDelivery``'s and for the
same reason (cross-cluster Kafka has no distributed transaction):

1. decode the source record and read its canonical ``envelope_id``;
2. if the durable marker already exists, publish nothing and commit the source;
3. otherwise publish to EVERY declared mirror lane and await each broker ack;
4. durably record the envelope id;
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
from omnibase_infra.idempotency import ProtocolIdempotencyStore
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayLaneMirrorConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_delivery import (
    ProtocolGatewayConsumer,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ProtocolGatewayPublisher,
    ServiceGatewayForwarder,
)
from omnibase_infra.utils import sanitize_error_message

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
        # Namespaced away from the trust-boundary legs' ``gateway:<slug>``
        # domain on purpose: one envelope can legitimately be both mirrored
        # across lanes AND forwarded to cloud, and a shared domain would make
        # whichever leg ran first suppress the other.
        self._domain = f"lane-mirror:{config.source_lane}"

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
            envelope = ServiceGatewayForwarder.decode_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as decode_error:
            # Same policy as NodeGatewayDelivery's quarantine path: a record
            # that can never decode would re-crash on every redelivery
            # (OMN-15748 poison-pill DoS), so commit past it rather than nack.
            logger.exception(
                "Lane mirror could not decode record topic=%s partition=%s "
                "offset=%s error=%s -- committing past it without mirroring",
                message.topic,
                message.partition,
                message.offset,
                sanitize_error_message(decode_error),
            )
            await self._source.commit(message)
            return

        envelope_id: UUID = envelope.envelope_id
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
                correlation_id=envelope.correlation_id,
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
