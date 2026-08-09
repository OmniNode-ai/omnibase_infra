# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-owned delivery and source acknowledgement for the gateway edge."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from typing import Literal, Protocol

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.event_bus.topic_constants import get_dlq_topic_for_original
from omnibase_infra.idempotency import ProtocolIdempotencyStore
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ServiceGatewayForwarder,
)
from omnibase_infra.utils import sanitize_error_message

logger = logging.getLogger(__name__)

_DIRECTIONS: tuple[Literal["outbound", "inbound"], ...] = ("outbound", "inbound")


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


# The delivery loop's real sources (``KafkaTransport``) carry three optional
# capabilities beyond the ``ProtocolGatewayConsumer`` pull surface above:
# ``start``/``close`` (recreate the underlying client -- a fresh group join),
# ``has_group_membership`` (a best-effort local assignment probe), and
# ``send`` (the same object is also a producer, for best-effort
# dead-lettering). None of these are declared as their own ``Protocol`` --
# the architecture validator enforces one ``Protocol`` class per file, and
# ``ProtocolGatewayConsumer`` above already occupies this file's slot -- so
# they are duck-typed via ``getattr``/``callable`` instead. Absence of any of
# them (e.g. the unit-test fakes that only implement poll/commit/nack)
# degrades gracefully: no dead-letter, no forced recreate, no membership
# signal, never an error.


def _build_quarantine_payload(
    *,
    direction: Literal["outbound", "inbound"],
    message: ModelTransportMessage,
    error: Exception,
) -> bytes:
    """Serialize an undecodable record's forensic context for the DLQ."""
    payload: dict[str, object] = {
        "original_topic": message.topic,
        "original_partition": message.partition,
        "original_offset": message.offset,
        "direction": direction,
        "failure_class": "gateway_undecodable_record",
        "error_type": type(error).__name__,
        "error_message": sanitize_error_message(error),
    }
    return json.dumps(payload).encode("utf-8")


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
        watchdog_stale_seconds: float | None = None,
    ) -> None:
        self._config = config
        self._forwarder = forwarder
        self._local_consumer = local_consumer
        self._cloud_consumer = cloud_consumer
        self._idempotency_store = idempotency_store
        self._poll_timeout_ms = poll_timeout_ms
        # The two directional loops share one process and one durable marker
        # namespace. Serialize the check -> publish -> marker -> commit sequence
        # so the same envelope cannot race through both directions before either
        # loop records it.
        self._delivery_lock = asyncio.Lock()
        self._tasks: list[asyncio.Task[None]] = []
        self._direction_sources: dict[
            Literal["outbound", "inbound"], ProtocolGatewayConsumer
        ] = {"outbound": local_consumer, "inbound": cloud_consumer}
        self._direction_tasks: dict[
            Literal["outbound", "inbound"], asyncio.Task[None]
        ] = {}
        # Membership-loss watchdog state (OMN-15748/OMN-15690). A silent
        # aiokafka max_poll_interval_ms idle-eviction raises zero exception,
        # so the exception-triggered reconnect supervision in
        # runtime/gateway_forwarder.py structurally cannot observe it -- this
        # watchdog is the independent detection path. Reuses the
        # already-contract-declared but previously unwired
        # ``max_silence_window_seconds`` as its staleness threshold.
        self._last_progress_monotonic: dict[Literal["outbound", "inbound"], float] = {}
        self._membership_lost_streak: dict[Literal["outbound", "inbound"], int] = {}
        self._watchdog_degraded: set[Literal["outbound", "inbound"]] = set()
        self._watchdog_stale_seconds = (
            watchdog_stale_seconds
            if watchdog_stale_seconds is not None
            else float(config.max_silence_window_seconds)
        )
        self._watchdog_tick_seconds = min(self._watchdog_stale_seconds / 4, 30.0)

    async def start(self) -> None:
        """Start one bounded pull loop per direction."""
        if self._tasks:
            return
        await self._idempotency_store.cleanup_expired(self._retention_seconds)
        now = time.monotonic()
        self._last_progress_monotonic = {"outbound": now, "inbound": now}
        self._membership_lost_streak = {"outbound": 0, "inbound": 0}
        self._watchdog_degraded.clear()
        self._tasks = [
            self._spawn_direction_task("outbound"),
            self._spawn_direction_task("inbound"),
            asyncio.create_task(
                self._run_cleanup_loop(),
                name="gateway-delivery-dedupe-cleanup",
            ),
            asyncio.create_task(
                self._run_watchdog_loop(),
                name="gateway-delivery-watchdog",
            ),
        ]

    def _spawn_direction_task(
        self, direction: Literal["outbound", "inbound"]
    ) -> asyncio.Task[None]:
        source = self._direction_sources[direction]
        task = asyncio.create_task(
            self._run_direction(direction, source),
            name=f"gateway-delivery-{direction}",
        )
        self._direction_tasks[direction] = task
        return task

    async def wait(self) -> None:
        """Propagate a delivery-loop failure to the composition root."""
        if not self._tasks:
            raise RuntimeError("gateway delivery node is not started")
        await asyncio.gather(*self._tasks)

    async def stop(self) -> None:
        """Cancel pull loops without acknowledging any in-flight source record."""
        tasks = list(self._tasks)
        self._tasks.clear()
        self._direction_tasks.clear()
        self._watchdog_degraded.clear()
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
        """Serialize and deliver one record in the durable acknowledgement order."""
        async with self._delivery_lock:
            await self._deliver_message_locked(direction, source, message)

    async def _deliver_message_locked(
        self,
        direction: Literal["outbound", "inbound"],
        source: ProtocolGatewayConsumer,
        message: ModelTransportMessage,
    ) -> None:
        """Deliver one record while holding the process-wide delivery lock."""
        try:
            envelope = self._forwarder.decode_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as decode_error:  # noqa: BLE001 — boundary: any decode failure is quarantined, never a bare swallow
            # A permanently malformed record can never decode no matter how
            # many times it is redelivered -- routing it through the nack
            # path below would seek back to the same offset and re-crash
            # forever (OMN-15748 poison-pill DoS). Quarantine instead: log,
            # best-effort dead-letter, commit past it, keep the loop alive.
            await self._quarantine_undecodable_message(
                direction, source, message, decode_error
            )
            return
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

    async def _quarantine_undecodable_message(
        self,
        direction: Literal["outbound", "inbound"],
        source: ProtocolGatewayConsumer,
        message: ModelTransportMessage,
        error: Exception,
    ) -> None:
        """Dead-letter an undecodable record and commit past it (never redeliver).

        Commit failure still propagates (uncaught): that is a broker-level
        fault, a different failure class already handled by the existing
        reconnect-supervision loop in ``runtime/gateway_forwarder.py``.
        """
        logger.error(
            "Gateway undecodable record quarantined direction=%s source_topic=%s "
            "source_partition=%s source_offset=%s error_type=%s",
            direction,
            message.topic,
            message.partition,
            message.offset,
            type(error).__name__,
            exc_info=error,
        )
        sender = getattr(source, "send", None)
        if callable(sender):
            dlq_topic = get_dlq_topic_for_original(message.topic)
            if dlq_topic is not None:
                try:
                    await sender(
                        dlq_topic,
                        message.key,
                        _build_quarantine_payload(
                            direction=direction, message=message, error=error
                        ),
                        {"original_topic": message.topic.encode("utf-8")},
                    )
                except Exception:
                    logger.exception(
                        "Gateway quarantine DLQ publish failed direction=%s "
                        "source_topic=%s source_partition=%s source_offset=%s",
                        direction,
                        message.topic,
                        message.partition,
                        message.offset,
                    )
        await source.commit(message)

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
            self._last_progress_monotonic[direction] = time.monotonic()
            for message in messages:
                await self.deliver_message(direction, source, message)
                self._last_progress_monotonic[direction] = time.monotonic()

    async def _run_watchdog_loop(self) -> None:
        """Detect a stalled direction independent of task exceptions.

        Two triggers, either sufficient: (1) no poll/deliver progress for
        ``_watchdog_stale_seconds`` -- catches a task alive but hung inside a
        stuck ``poll()`` (the live-observed 2026-08-09T10:03:07Z mechanism:
        aiokafka's client-side idle-eviction fires a clean LeaveGroup with no
        exception, then ``ensure_active_group`` refuses to rejoin while the
        idle clock stays past threshold -- which it does forever once the
        underlying poll never returns); (2) two consecutive lost-membership
        probes -- catches membership silently lost even while poll keeps
        returning. Recovery force-recreates just the affected direction's
        transport (a fresh client instance re-joins the group from zero) and
        publishes a DEGRADED status; a later tick with progress republishes
        recovery.
        """
        while True:
            await asyncio.sleep(self._watchdog_tick_seconds)
            now = time.monotonic()
            for direction in _DIRECTIONS:
                source = self._direction_sources[direction]
                elapsed = now - self._last_progress_monotonic.get(direction, now)
                membership_lost = self._probe_membership_lost(source)
                self._membership_lost_streak[direction] = (
                    self._membership_lost_streak.get(direction, 0) + 1
                    if membership_lost
                    else 0
                )
                stale_by_time = elapsed >= self._watchdog_stale_seconds
                stale_by_membership = self._membership_lost_streak[direction] >= 2
                if stale_by_time or stale_by_membership:
                    await self._recover_stalled_direction(
                        direction,
                        elapsed_seconds=elapsed,
                        membership_lost=membership_lost,
                    )
                elif direction in self._watchdog_degraded:
                    await self._mark_direction_recovered(direction)

    @staticmethod
    def _probe_membership_lost(source: ProtocolGatewayConsumer) -> bool:
        prober = getattr(source, "has_group_membership", None)
        if not callable(prober):
            return False
        try:
            return not bool(prober())
        except Exception:  # noqa: BLE001 — boundary: probe failure counts as lost, fail closed
            return True

    async def _recover_stalled_direction(
        self,
        direction: Literal["outbound", "inbound"],
        *,
        elapsed_seconds: float,
        membership_lost: bool,
    ) -> None:
        logger.warning(
            "Gateway %s delivery loop stalled elapsed_seconds=%.1f "
            "membership_lost=%s; forcing recreate + rejoin",
            direction,
            elapsed_seconds,
            membership_lost,
        )
        stale_task = self._direction_tasks.get(direction)
        if stale_task is not None and not stale_task.done():
            stale_task.cancel()
            await asyncio.gather(stale_task, return_exceptions=True)

        source = self._direction_sources[direction]
        closer = getattr(source, "close", None)
        starter = getattr(source, "start", None)
        if callable(closer) and callable(starter):
            try:
                await closer()
                await starter()
            except Exception:
                logger.exception(
                    "Gateway %s transport recreate failed; will retry next "
                    "watchdog tick",
                    direction,
                )
                return

        self._last_progress_monotonic[direction] = time.monotonic()
        self._membership_lost_streak[direction] = 0
        new_task = self._spawn_direction_task(direction)
        self._tasks = [new_task if t is stale_task else t for t in self._tasks]
        if direction not in self._watchdog_degraded:
            self._watchdog_degraded.add(direction)
            await self._publish_watchdog_status(
                "degraded",
                detail=(
                    f"{direction} delivery loop membership-loss recovery "
                    f"elapsed_seconds={elapsed_seconds:.1f} "
                    f"membership_lost={membership_lost}"
                ),
            )

    async def _mark_direction_recovered(
        self, direction: Literal["outbound", "inbound"]
    ) -> None:
        self._watchdog_degraded.discard(direction)
        await self._publish_watchdog_status(
            "active", detail=f"{direction} delivery loop recovered"
        )

    async def _publish_watchdog_status(
        self, status: Literal["active", "degraded"], *, detail: str
    ) -> None:
        """Best-effort status publish -- must never itself take down the watchdog."""
        try:
            await self._forwarder.publish_status(status, detail=detail)
        except Exception:
            logger.exception("Gateway watchdog %s status publish failed", status)

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
