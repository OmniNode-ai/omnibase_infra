# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-owned delivery and source acknowledgement for the gateway edge."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Literal, Protocol

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.idempotency import ProtocolIdempotencyStore
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ServiceGatewayForwarder,
)

logger = logging.getLogger(__name__)


class ProtocolGatewayConsumer(Protocol):
    """Concrete pull surface used by the Kafka-backed gateway runtime."""

    async def poll(
        self,
        *,
        max_messages: int,
        timeout_ms: int,
    ) -> Sequence[ModelTransportMessage]: ...

    async def commit(self, message: object) -> None: ...

    async def nack(self, message: object) -> None: ...


class NodeGatewayDelivery:
    """Poll both legs and acknowledge only after durable destination delivery.

    Cross-cluster Kafka cannot make the destination publish and source offset
    commit one atomic transaction.  The explicit order is therefore:

    1. reject or transform the source envelope at the gateway trust boundary;
    2. await the destination broker acknowledgement;
    3. durably record the envelope ID in the edge-local store;
    4. commit the source offset.

    If step 4 fails, a restarted consumer sees the durable marker, skips the
    destination publish, and commits the redelivered source record.  The
    irreducible publish-to-marker crash window remains at-least-once and is
    observable through the stable envelope ID; downstream execution must use
    that same ID as its idempotency key.
    """

    def __init__(
        self,
        *,
        config: ModelGatewayForwarderConfig,
        forwarder: ServiceGatewayForwarder,
        local_consumer: ProtocolGatewayConsumer,
        cloud_consumer: ProtocolGatewayConsumer,
        idempotency_store: ProtocolIdempotencyStore,
        poll_timeout_ms: int = 1_000,
    ) -> None:
        self._config = config
        self._forwarder = forwarder
        self._local_consumer = local_consumer
        self._cloud_consumer = cloud_consumer
        self._idempotency_store = idempotency_store
        self._poll_timeout_ms = poll_timeout_ms
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        """Start one bounded pull loop per direction."""
        if self._tasks:
            return
        await self._idempotency_store.cleanup_expired(self._retention_seconds)
        self._tasks = [
            asyncio.create_task(
                self._run_direction("outbound", self._local_consumer),
                name="gateway-delivery-outbound",
            ),
            asyncio.create_task(
                self._run_direction("inbound", self._cloud_consumer),
                name="gateway-delivery-inbound",
            ),
            asyncio.create_task(
                self._run_cleanup_loop(),
                name="gateway-delivery-dedupe-cleanup",
            ),
        ]

    async def wait(self) -> None:
        """Propagate a delivery-loop failure to the composition root."""
        if not self._tasks:
            raise RuntimeError("gateway delivery node is not started")
        await asyncio.gather(*self._tasks)

    async def stop(self) -> None:
        """Cancel pull loops without acknowledging any in-flight source record."""
        tasks = list(self._tasks)
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def deliver_message(
        self,
        direction: Literal["outbound", "inbound"],
        source: ProtocolGatewayConsumer,
        message: ModelTransportMessage,
    ) -> None:
        """Deliver one record and commit its source offset in the safe order."""
        envelope = self._forwarder.decode_message(message)
        domain = f"gateway:{self._config.tenant_identity.tenant_slug}"
        try:
            if direction == "outbound":
                self._forwarder.validate_outbound_message(message)
            else:
                self._forwarder.validate_inbound_message(message)

            if await self._idempotency_store.is_processed(
                envelope.envelope_id,
                domain=domain,
            ):
                logger.info(
                    "Gateway duplicate suppressed envelope_id=%s direction=%s "
                    "source_topic=%s source_partition=%s source_offset=%s",
                    envelope.envelope_id,
                    direction,
                    message.topic,
                    message.partition,
                    message.offset,
                )
                await source.commit(message)
                return

            if direction == "outbound":
                await self._forwarder.forward_outbound_message(message)
            else:
                await self._forwarder.consume_inbound_message(message)

            await self._idempotency_store.mark_processed(
                envelope.envelope_id,
                domain=domain,
                correlation_id=envelope.correlation_id,
            )
            await source.commit(message)
            logger.info(
                "Gateway delivery acknowledged envelope_id=%s direction=%s "
                "source_topic=%s source_partition=%s source_offset=%s",
                envelope.envelope_id,
                direction,
                message.topic,
                message.partition,
                message.offset,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            try:
                await source.nack(message)
            except Exception:
                logger.exception(
                    "Gateway source nack failed direction=%s source_topic=%s "
                    "source_partition=%s source_offset=%s",
                    direction,
                    message.topic,
                    message.partition,
                    message.offset,
                )
            raise

    async def _run_direction(
        self,
        direction: Literal["outbound", "inbound"],
        source: ProtocolGatewayConsumer,
    ) -> None:
        while True:
            messages = await source.poll(
                max_messages=1,
                timeout_ms=self._poll_timeout_ms,
            )
            for message in messages:
                await self.deliver_message(direction, source, message)

    async def _run_cleanup_loop(self) -> None:
        cleanup_interval_seconds = min(self._retention_seconds / 4, 3_600)
        while True:
            await asyncio.sleep(cleanup_interval_seconds)
            removed = await self._idempotency_store.cleanup_expired(
                self._retention_seconds
            )
            if removed:
                logger.info("Gateway dedupe cleanup removed=%d", removed)

    @property
    def _retention_seconds(self) -> int:
        return self._config.dedupe_retention_hours * 60 * 60


__all__ = ["NodeGatewayDelivery"]
