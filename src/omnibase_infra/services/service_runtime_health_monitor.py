# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime health monitor service.

Runs every ``check_interval_seconds`` (default: 300) inside the runtime
container and checks:

1. **Consumer group coverage** — for every topic declared as a ``subscribe``
   target in any discovered contract, verifies that a non-empty consumer group
   exists on the broker.
2. **Discovery errors** — if ``discover_contracts()`` found errors, the
   dimension is DEGRADED.
3. **Topic coverage** — every subscribe topic should have at least one
   non-empty consumer group.

Results are emitted to ``onex.evt.omnibase-infra.runtime-health-check.v1``.

The service is intentionally **best-effort**: any failure during a check
cycle is logged and the next cycle proceeds normally.

.. versionadded:: 0.39.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.health.model_runtime_health_check_event import (
    ModelRuntimeHealthCheckEvent,
)
from omnibase_infra.models.health.model_runtime_health_dimension import (
    ModelRuntimeHealthDimension,
)
from omnibase_infra.protocols import ProtocolTopicRegistry
from omnibase_infra.protocols.protocol_auto_wiring_manifest_like import (
    ProtocolAutoWiringManifestLike,
)
from omnibase_infra.protocols.protocol_kafka_admin_like import ProtocolKafkaAdminLike
from omnibase_infra.topics import topic_keys
from omnibase_infra.utils.correlation import generate_correlation_id

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike

logger = logging.getLogger(__name__)


def _discover_contracts() -> ProtocolAutoWiringManifestLike:
    """Module-level shim — allows tests to patch without lazy-import complications.

    The return type is declared as ``ProtocolAutoWiringManifestLike`` to avoid a
    circular import of ``ModelAutoWiringManifest`` at module parse time while
    still providing accurate type information to callers.
    """
    from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts

    return discover_contracts()  # type: ignore[return-value]


def _get_kafka_admin_client(
    bootstrap_servers: str, request_timeout_ms: int
) -> ProtocolKafkaAdminLike:
    """Module-level shim for AIOKafkaAdminClient — patchable in tests.

    The return type is declared as ``ProtocolKafkaAdminLike`` to avoid a hard
    import of ``AIOKafkaAdminClient`` at module import time while still providing
    accurate type information to callers.
    """
    from aiokafka.admin import AIOKafkaAdminClient

    return AIOKafkaAdminClient(  # type: ignore[return-value]
        bootstrap_servers=bootstrap_servers,
        request_timeout_ms=request_timeout_ms,
    )


_HealthStatus = Literal["HEALTHY", "DEGRADED", "CRITICAL"]

_DEFAULT_CHECK_INTERVAL: float = 300.0  # 5 minutes
_DEFAULT_BOOT_GRACE: float = 120.0  # 2 minutes — covers typical topic provisioning time
_KAFKA_ADMIN_TIMEOUT_MS: int = 5_000


def _worst(statuses: list[_HealthStatus]) -> _HealthStatus:
    """Return the worst status from a list."""
    if "CRITICAL" in statuses:
        return "CRITICAL"
    if "DEGRADED" in statuses:
        return "DEGRADED"
    return "HEALTHY"


class ServiceRuntimeHealthMonitor:
    """Periodic runtime health monitor.

    Usage::

        monitor = ServiceRuntimeHealthMonitor(
            event_bus=bus,
            bootstrap_servers="redpanda:9092",
            check_interval_seconds=300.0,
        )
        await monitor.start()
        ...
        await monitor.stop()
    """

    def __init__(
        self,
        event_bus: ProtocolEventBusLike | None = None,
        bootstrap_servers: str | None = None,
        check_interval_seconds: float = _DEFAULT_CHECK_INTERVAL,
        topic_registry: ProtocolTopicRegistry | None = None,
        boot_grace_seconds: float = _DEFAULT_BOOT_GRACE,
    ) -> None:
        """Initialize the health monitor.

        Args:
            event_bus: Optional event bus for emitting health events.
            bootstrap_servers: Kafka bootstrap servers string. Defaults to
                ``KAFKA_BOOTSTRAP_SERVERS`` env var.
            check_interval_seconds: How often to run a full check.
            topic_registry: Optional topic registry. Defaults to
                ``ServiceTopicRegistry.from_defaults()``.
        """
        if check_interval_seconds <= 0:
            raise ValueError(
                f"check_interval_seconds must be positive, got {check_interval_seconds}"
            )

        if topic_registry is None:
            from omnibase_infra.topics.service_topic_registry import (
                ServiceTopicRegistry,
            )

            topic_registry = ServiceTopicRegistry.from_defaults()

        self._health_topic = topic_registry.resolve(topic_keys.RUNTIME_HEALTH_CHECK)
        self._event_bus = event_bus
        # Only fall back to env var when caller passes None (not when they pass "").
        if bootstrap_servers is None:
            self._bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
        else:
            self._bootstrap_servers = bootstrap_servers
        self._check_interval = check_interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._boot_grace_seconds = boot_grace_seconds
        self._started_at: float = time.monotonic()  # grace window starts at construction

    async def start(self) -> None:
        """Start the background health check loop. Idempotent.

        Runs the first health check immediately before entering the periodic loop
        so callers get an initial signal without waiting one full interval.
        """
        if self._running:
            return
        self._running = True
        # Run the first check immediately so callers get an initial health signal.
        try:
            await self.run_once()
        except Exception:  # noqa: BLE001 — best-effort; don't block startup
            logger.warning(
                "ServiceRuntimeHealthMonitor: initial check failed", exc_info=True
            )
        self._task = asyncio.create_task(self._loop(), name="runtime-health-monitor")
        logger.info(
            "ServiceRuntimeHealthMonitor started (interval=%ds)",
            int(self._check_interval),
        )

    async def stop(self) -> None:
        """Stop the background health check loop. Idempotent."""
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                # Expected: task.cancel() raises CancelledError on the awaiter.
                pass
            self._task = None
        logger.info("ServiceRuntimeHealthMonitor stopped")

    async def run_once(self) -> ModelRuntimeHealthCheckEvent:
        """Run a single health check cycle and return the event.

        This is the core method used by both the background loop and direct
        one-shot callers (e.g. tests).

        Returns:
            A ``ModelRuntimeHealthCheckEvent`` with the current health state.
        """
        correlation_id = generate_correlation_id()
        dimensions: list[ModelRuntimeHealthDimension] = []

        # --- Dimension 1: Contract discovery health -------------------------
        contract_count = 0
        discovery_error_count = 0
        subscribe_topics: set[str] = set()
        try:
            manifest = _discover_contracts()
            contract_count = manifest.total_discovered
            discovery_error_count = manifest.total_errors
            subscribe_topics = set(manifest.all_subscribe_topics())

            if discovery_error_count > 0:
                dimensions.append(
                    ModelRuntimeHealthDimension(
                        name="discovery_errors",
                        status="DEGRADED",
                        detail=(f"{discovery_error_count} contract(s) failed to load"),
                    )
                )
            else:
                dimensions.append(
                    ModelRuntimeHealthDimension(
                        name="discovery_errors",
                        status="HEALTHY",
                        detail=f"{contract_count} contracts loaded cleanly",
                    )
                )
        except Exception as exc:  # noqa: BLE001 — boundary: dimension degrades
            logger.warning(
                "Runtime health: discovery check failed — %s (correlation_id=%s)",
                type(exc).__name__,
                correlation_id,
            )
            dimensions.append(
                ModelRuntimeHealthDimension(
                    name="discovery_errors",
                    status="CRITICAL",
                    detail=f"discover_contracts() raised: {type(exc).__name__}",
                )
            )

        # --- Dimension 2 & 3: Consumer group coverage -----------------------
        consumer_group_count = 0
        empty_consumer_group_count = 0
        uncovered_topic_count = 0

        if self._bootstrap_servers:
            try:
                admin = None
                try:
                    admin = _get_kafka_admin_client(
                        self._bootstrap_servers, _KAFKA_ADMIN_TIMEOUT_MS
                    )
                    await admin.start()  # type: ignore[union-attr]

                    # list_consumer_groups() returns list of (group_id, protocol_type) tuples
                    all_groups_raw = await admin.list_consumer_groups()  # type: ignore[union-attr]
                    all_group_ids = [g[0] for g in all_groups_raw]
                    consumer_group_count = len(all_group_ids)

                    # describe_consumer_groups to find empty ones
                    described: dict[str, object] = {}
                    describe_failed = False
                    if all_group_ids:
                        try:
                            raw_described = await admin.describe_consumer_groups(  # type: ignore[union-attr]
                                all_group_ids
                            )
                            described = dict(
                                zip(all_group_ids, raw_described, strict=False)
                            )
                        except Exception as exc:  # noqa: BLE001
                            describe_failed = True
                            logger.warning(
                                "Runtime health: describe_consumer_groups failed — %s; "
                                "consumer group states unknown",
                                exc,
                            )

                    empty_groups: set[str] = set()
                    for group_id, group_meta in described.items():
                        # GroupMetadata.state is a string e.g. "Empty", "Stable"
                        state = getattr(group_meta, "state", None)
                        if state == "Empty" or not getattr(group_meta, "members", None):
                            empty_groups.add(group_id)
                    empty_consumer_group_count = len(empty_groups)

                    # Check which subscribe topics have a matching non-empty group
                    # Consumer groups in ONEX typically encode the topic in the group ID
                    # (e.g. "onex-runtime-consumer-<topic-slug>"). We consider a topic
                    # "covered" if at least one non-empty group ID contains the full
                    # topic name, ensuring exact-identity matching rather than a loose
                    # suffix match that could yield false positives on version tokens
                    # like "v1" (which appear in every consumer group name).
                    if describe_failed:
                        # Consumer group states are unknown — do not compute coverage
                        # from unreliable data; both dimensions degrade together.
                        non_empty_groups: set[str] = set()
                    else:
                        non_empty_groups = set(all_group_ids) - empty_groups
                    uncovered: list[str] = []
                    for topic in sorted(subscribe_topics):
                        covered = any(topic in grp for grp in non_empty_groups)
                        if not covered:
                            uncovered.append(topic)
                    uncovered_topic_count = len(uncovered)

                    if describe_failed:
                        dimensions.append(
                            ModelRuntimeHealthDimension(
                                name="empty_consumer_groups",
                                status="DEGRADED",
                                detail=(
                                    f"describe_consumer_groups failed; "
                                    f"{consumer_group_count} groups listed but states unknown"
                                ),
                            )
                        )
                        dimensions.append(
                            ModelRuntimeHealthDimension(
                                name="topic_coverage",
                                status="DEGRADED",
                                detail="Consumer group states unknown — topic coverage cannot be verified",
                            )
                        )
                    elif empty_consumer_group_count > 0:
                        dimensions.append(
                            ModelRuntimeHealthDimension(
                                name="empty_consumer_groups",
                                status="DEGRADED",
                                detail=(
                                    f"{empty_consumer_group_count}/{consumer_group_count}"
                                    " consumer groups are Empty"
                                ),
                            )
                        )
                    else:
                        dimensions.append(
                            ModelRuntimeHealthDimension(
                                name="empty_consumer_groups",
                                status="HEALTHY",
                                detail=(
                                    f"All {consumer_group_count} consumer groups active"
                                ),
                            )
                        )

                    if not describe_failed:
                        if uncovered_topic_count > 0:
                            detail_topics = ", ".join(uncovered[:5])
                            if len(uncovered) > 5:
                                detail_topics += f" … +{len(uncovered) - 5} more"
                            dimensions.append(
                                ModelRuntimeHealthDimension(
                                    name="topic_coverage",
                                    status="CRITICAL"
                                    if uncovered_topic_count > 10
                                    else "DEGRADED",
                                    detail=(
                                        f"{uncovered_topic_count} subscribe topic(s) have"
                                        f" no active consumer group: {detail_topics}"
                                    ),
                                )
                            )
                        else:
                            dimensions.append(
                                ModelRuntimeHealthDimension(
                                    name="topic_coverage",
                                    status="HEALTHY",
                                    detail=(
                                        f"All {len(subscribe_topics)} subscribe topics covered"
                                    ),
                                )
                            )

                finally:
                    if admin is not None:
                        try:
                            await admin.close()
                        except Exception:  # noqa: BLE001 — best-effort admin close
                            logger.debug(
                                "Runtime health: failed to close Kafka admin client",
                                exc_info=True,
                            )

            except ImportError:
                logger.debug(
                    "Runtime health: aiokafka not available — consumer checks skipped"
                )
                dimensions.append(
                    ModelRuntimeHealthDimension(
                        name="consumer_coverage",
                        status="HEALTHY",
                        detail="aiokafka not installed — consumer checks skipped",
                    )
                )
            except Exception as exc:  # noqa: BLE001 — boundary: dimension degrades
                logger.warning(
                    "Runtime health: consumer group check failed — %s (correlation_id=%s)",
                    type(exc).__name__,
                    correlation_id,
                )
                dimensions.append(
                    ModelRuntimeHealthDimension(
                        name="consumer_coverage",
                        status="DEGRADED",
                        detail=f"Admin client error: {type(exc).__name__}",
                    )
                )
        else:
            dimensions.append(
                ModelRuntimeHealthDimension(
                    name="consumer_coverage",
                    status="HEALTHY",
                    detail="No bootstrap_servers configured — consumer checks skipped",
                )
            )

        aggregate_status: _HealthStatus = _worst([d.status for d in dimensions])

        event = ModelRuntimeHealthCheckEvent(
            correlation_id=correlation_id,
            timestamp=datetime.now(UTC),
            status=aggregate_status,
            dimensions=tuple(dimensions),
            contract_count=contract_count,
            discovery_error_count=discovery_error_count,
            consumer_group_count=consumer_group_count,
            empty_consumer_group_count=empty_consumer_group_count,
            subscribe_topic_count=len(subscribe_topics),
            uncovered_topic_count=uncovered_topic_count,
        )

        logger.info(
            "Runtime health check: status=%s contracts=%d errors=%d "
            "consumer_groups=%d empty=%d uncovered_topics=%d",
            aggregate_status,
            contract_count,
            discovery_error_count,
            consumer_group_count,
            empty_consumer_group_count,
            uncovered_topic_count,
        )

        if aggregate_status != "HEALTHY":
            for dim in dimensions:
                if dim.status == "CRITICAL":
                    logger.error(
                        "Runtime health CRITICAL dimension=%s status=%s detail=%s",
                        dim.name,
                        dim.status,
                        dim.detail,
                    )
                elif dim.status == "DEGRADED":
                    logger.warning(
                        "Runtime health DEGRADED dimension=%s status=%s detail=%s",
                        dim.name,
                        dim.status,
                        dim.detail,
                    )

        await self._emit(event)
        return event

    # -- Internal -------------------------------------------------------------

    async def _loop(self) -> None:
        """Background loop that runs health checks at the configured interval."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001 — never crash the loop
                logger.warning(
                    "Runtime health monitor loop iteration failed (will retry in %ds)",
                    int(self._check_interval),
                    exc_info=True,
                )

    async def _emit(self, event: ModelRuntimeHealthCheckEvent) -> None:
        """Emit the health event to the event bus. Best-effort fire-and-forget."""
        if self._event_bus is None:
            return

        # Suppress emit during boot grace window — a not-yet-provisioned health-check
        # topic must not trip the circuit breaker on first boot. (OMN-9552)
        elapsed = time.monotonic() - self._started_at
        if elapsed < self._boot_grace_seconds:
            logger.debug(
                "ServiceRuntimeHealthMonitor: suppressing emit during boot grace "
                "(elapsed=%.1fs grace=%.1fs)",
                elapsed,
                self._boot_grace_seconds,
            )
            return

        envelope: ModelEventEnvelope[ModelRuntimeHealthCheckEvent] = ModelEventEnvelope(
            payload=event,
            correlation_id=event.correlation_id,
            event_type="runtime-health-check",
            source_tool="ServiceRuntimeHealthMonitor",
        )
        try:
            await self._event_bus.publish_envelope(
                envelope=envelope,
                topic=self._health_topic,
            )
        except Exception:
            logger.exception(
                "Failed to emit runtime health check event",
                extra={"correlation_id": str(event.correlation_id)},
            )


__all__: list[str] = ["ServiceRuntimeHealthMonitor"]
