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
import inspect
import logging
import math
import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, NamedTuple

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.models import ModelNodeIdentity
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
from omnibase_infra.topics import topic_keys
from omnibase_infra.utils import (
    apply_instance_discriminator,
    compute_consumer_group_id,
)
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
    from omnibase_infra.runtime.auto_wiring.profile_ownership import (
        filter_manifest_for_runtime_profile,
    )

    manifest = discover_contracts()
    runtime_profile = os.getenv("RUNTIME_PROFILE", "main")
    ownership_result = filter_manifest_for_runtime_profile(
        manifest=manifest,
        runtime_profile=runtime_profile,
    )
    return ownership_result.manifest  # type: ignore[return-value]


def _filter_manifest_for_runtime_profile(
    manifest: ProtocolAutoWiringManifestLike,
) -> ProtocolAutoWiringManifestLike:
    """Apply the same runtime-profile ownership filter used by service startup."""
    from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
        ModelAutoWiringManifest,
    )
    from omnibase_infra.runtime.auto_wiring.profile_ownership import (
        filter_manifest_for_runtime_profile,
    )

    if not isinstance(manifest, ModelAutoWiringManifest):
        return manifest

    ownership_result = filter_manifest_for_runtime_profile(
        manifest=manifest,
        runtime_profile=os.environ.get("RUNTIME_PROFILE", "main"),
    )
    if ownership_result.skipped_contracts:
        logger.debug(
            "Runtime health monitor profile ownership: profile=%s owned=%d skipped=%d",
            ownership_result.runtime_profile,
            ownership_result.manifest.total_discovered,
            len(ownership_result.skipped_contracts),
        )
    return ownership_result.manifest


class ConsumerGroupSnapshot(NamedTuple):
    """Minimal consumer group state used by runtime health checks."""

    group_id: str
    state: str


class ExpectedConsumerGroup(NamedTuple):
    """Expected per-topic consumer group derived from discovered contracts."""

    topic: str
    group_id: str
    node_name: str
    package_name: str


def _consumer_group_state_name(state: object) -> str:
    """Normalize confluent-kafka consumer group states to plain uppercase names."""
    enum_name = getattr(state, "name", None)
    raw = str(enum_name if enum_name else state)
    return raw.rsplit(".", maxsplit=1)[-1].upper()


def _list_consumer_group_snapshots(
    bootstrap_servers: str, request_timeout_ms: int
) -> list[ConsumerGroupSnapshot]:
    """List consumer groups via confluent-kafka without decoding member metadata.

    aiokafka's ``describe_consumer_groups`` path decodes Redpanda member
    metadata as UTF-8 and can raise ``UnicodeDecodeError`` on valid binary
    payloads. The confluent client exposes group state from list metadata, which
    is sufficient for health coverage and avoids the brittle member decode path.
    """
    from confluent_kafka.admin import AdminClient

    timeout_seconds = max(request_timeout_ms / 1000.0, 1.0)
    admin = AdminClient(
        {
            "bootstrap.servers": bootstrap_servers,
            "socket.timeout.ms": request_timeout_ms,
            "request.timeout.ms": request_timeout_ms,
        }
    )
    result = admin.list_consumer_groups(request_timeout=timeout_seconds).result(
        timeout=timeout_seconds + 1.0
    )

    errors = getattr(result, "errors", None) or []
    if errors:
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="list_consumer_groups",
        )
        raise InfraConnectionError(
            f"list_consumer_groups returned {len(errors)} error(s)",
            context=context,
        )

    snapshots: list[ConsumerGroupSnapshot] = []
    for group in getattr(result, "valid", None) or []:
        group_id = str(getattr(group, "group_id", ""))
        if not group_id:
            continue
        snapshots.append(
            ConsumerGroupSnapshot(
                group_id=group_id,
                state=_consumer_group_state_name(getattr(group, "state", "UNKNOWN")),
            )
        )
    return snapshots


def _expected_consumer_groups_from_manifest(
    manifest: ProtocolAutoWiringManifestLike, instance_id: str | None = None
) -> list[ExpectedConsumerGroup]:
    """Derive the exact Kafka group IDs runtime wiring should create.

    Runtime auto-wiring subscribes each contract topic with a node identity based
    group and ``EventBusKafka`` appends the per-topic ``.__t.<topic>`` suffix.
    The health monitor must compare against that exact shape; topic substring
    matching lets unrelated groups mask missing runtime consumers.
    """
    expected: list[ExpectedConsumerGroup] = []
    contracts = getattr(manifest, "contracts", ())
    for contract in contracts:
        event_bus = getattr(contract, "event_bus", None)
        if event_bus is None:
            continue
        topics = tuple(getattr(event_bus, "subscribe_topics", ()) or ())
        if not topics:
            continue

        node_name = str(getattr(contract, "name", "unknown"))
        package_name = str(getattr(contract, "package_name", "unknown"))
        version = str(getattr(contract, "contract_version", "0.0.0"))
        identity = ModelNodeIdentity(
            env=os.environ.get("ONEX_ENVIRONMENT", "local"),
            service=package_name,
            node_name=node_name,
            version=version,
        )
        base_group_id = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.CONSUME
        )
        discriminated_group_id = apply_instance_discriminator(
            base_group_id, instance_id
        )
        for topic in topics:
            topic_str = str(topic)
            expected.append(
                ExpectedConsumerGroup(
                    topic=topic_str,
                    group_id=f"{discriminated_group_id}.__t.{topic_str}",
                    node_name=node_name,
                    package_name=package_name,
                )
            )
    return expected


def _expected_consumer_groups_from_event_bus(
    event_bus: ProtocolEventBusLike | None,
) -> list[ExpectedConsumerGroup]:
    """Return live expected groups from the runtime event bus, when available.

    ``discover_contracts()`` sees every installed contract, including contracts
    skipped by auto-wiring because they lack handler routing or are superseded by
    dedicated runtime wiring. The event bus registry is the authoritative source
    for subscriptions that this runtime actually attempted to start.
    """
    if event_bus is None:
        return []

    get_consumer_groups = getattr(event_bus, "get_consumer_groups", None)
    if not callable(get_consumer_groups):
        return []

    groups = get_consumer_groups()
    if inspect.isawaitable(groups):
        close = getattr(groups, "close", None)
        if callable(close):
            close()
        return []
    if not isinstance(groups, Mapping):
        return []

    expected: list[ExpectedConsumerGroup] = []
    for key, effective_group_id in groups.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(effective_group_id, str)
            or not effective_group_id
        ):
            continue
        topic, base_group_id = key
        if not isinstance(topic, str) or not isinstance(base_group_id, str):
            continue
        expected.append(
            ExpectedConsumerGroup(
                topic=topic,
                group_id=effective_group_id,
                node_name=base_group_id,
                package_name="event_bus",
            )
        )
    return expected


def _topic_is_covered_by_legacy_group(topic: str, group_id: str) -> bool:
    """Best-effort coverage check for manifests without contract records."""
    return (
        f".__t.{topic}" in group_id
        or f"-{topic}-" in group_id
        or group_id.endswith(f"-{topic}")
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
            boot_grace_seconds: Seconds to suppress event emission after the
                monitor starts, protecting boot from not-yet-provisioned topics.
        """
        if check_interval_seconds <= 0:
            raise ValueError(
                f"check_interval_seconds must be positive, got {check_interval_seconds}"
            )
        if not math.isfinite(boot_grace_seconds) or boot_grace_seconds < 0:
            raise ValueError(
                "boot_grace_seconds must be finite and non-negative, "
                f"got {boot_grace_seconds}"
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
        self._started_at: float | None = None
        self._boot_grace_complete_logged = boot_grace_seconds == 0

    async def start(self) -> None:
        """Start the background health check loop. Idempotent.

        The first full health check is intentionally deferred to the background
        loop. Contract discovery and Kafka group inspection are expensive on the
        full runtime image; running them synchronously here delays health-server
        binding and can make a healthy runtime appear stuck in Docker
        ``starting``.
        """
        if self._running:
            return
        self._running = True
        self._started_at = time.monotonic()
        self._boot_grace_complete_logged = self._boot_grace_seconds == 0
        if self._boot_grace_seconds > 0:
            logger.info(
                "ServiceRuntimeHealthMonitor boot grace active for %.1fs",
                self._boot_grace_seconds,
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
        expected_groups: list[ExpectedConsumerGroup] = []
        try:
            manifest = _filter_manifest_for_runtime_profile(_discover_contracts())
            contract_count = manifest.total_discovered
            discovery_error_count = manifest.total_errors
            subscribe_topics = set(manifest.all_subscribe_topics())
            expected_groups = _expected_consumer_groups_from_manifest(
                manifest, os.environ.get("KAFKA_INSTANCE_ID")
            )
            live_expected_groups = _expected_consumer_groups_from_event_bus(
                self._event_bus
            )
            if live_expected_groups:
                expected_groups = live_expected_groups
                subscribe_topics = {expected.topic for expected in live_expected_groups}

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
                group_snapshots = await asyncio.to_thread(
                    _list_consumer_group_snapshots,
                    self._bootstrap_servers,
                    _KAFKA_ADMIN_TIMEOUT_MS,
                )
                all_group_ids = [g.group_id for g in group_snapshots]
                consumer_group_count = len(all_group_ids)

                empty_states = {"DEAD", "EMPTY", "UNKNOWN"}
                empty_group_ids = {
                    g.group_id for g in group_snapshots if g.state in empty_states
                }

                # Check which subscribe topics have a matching non-empty group.
                non_empty_groups = set(all_group_ids) - empty_group_ids
                if expected_groups:
                    expected_group_ids = {
                        expected.group_id for expected in expected_groups
                    }
                    empty_groups = empty_group_ids & expected_group_ids
                    missing_expected = [
                        expected
                        for expected in expected_groups
                        if expected.group_id not in non_empty_groups
                    ]
                    uncovered_topic_count = len(missing_expected)
                    uncovered_details = [
                        f"{expected.topic} ({expected.node_name}: {expected.group_id})"
                        for expected in missing_expected
                    ]
                else:
                    empty_groups = empty_group_ids
                    # Test doubles and older manifest protocols may only expose
                    # topic names. Keep a bounded fallback, but production
                    # manifests use exact expected group IDs above.
                    uncovered_topics: list[str] = []
                    for topic in sorted(subscribe_topics):
                        covered = any(
                            _topic_is_covered_by_legacy_group(topic, group_id)
                            for group_id in non_empty_groups
                        )
                        if not covered:
                            uncovered_topics.append(topic)
                    uncovered_topic_count = len(uncovered_topics)
                    uncovered_details = uncovered_topics
                empty_consumer_group_count = len(empty_groups)

                if empty_consumer_group_count > 0:
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
                            detail=f"All {consumer_group_count} consumer groups active",
                        )
                    )

                if uncovered_topic_count > 0:
                    detail_topics = ", ".join(uncovered_details[:5])
                    if len(uncovered_details) > 5:
                        detail_topics += f" … +{len(uncovered_details) - 5} more"
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
                    coverage_target_count = (
                        len(expected_groups)
                        if expected_groups
                        else len(subscribe_topics)
                    )
                    dimensions.append(
                        ModelRuntimeHealthDimension(
                            name="topic_coverage",
                            status="HEALTHY",
                            detail=(
                                f"All {coverage_target_count} expected "
                                "consumer group(s) covered"
                            ),
                        )
                    )

            except ImportError:
                logger.debug(
                    "Runtime health: confluent-kafka not available — consumer checks skipped"
                )
                dimensions.append(
                    ModelRuntimeHealthDimension(
                        name="consumer_coverage",
                        status="HEALTHY",
                        detail="confluent-kafka not installed — consumer checks skipped",
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
        if self._started_at is None:
            self._started_at = time.monotonic()
        elapsed = time.monotonic() - self._started_at
        if elapsed < self._boot_grace_seconds:
            logger.debug(
                "ServiceRuntimeHealthMonitor: suppressing emit during boot grace "
                "(elapsed=%.1fs grace=%.1fs)",
                elapsed,
                self._boot_grace_seconds,
            )
            return
        if not self._boot_grace_complete_logged:
            logger.info("ServiceRuntimeHealthMonitor boot grace complete")
            self._boot_grace_complete_logged = True

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
