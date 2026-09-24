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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, NamedTuple

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.health.model_consumer_sync_status import (
    ModelConsumerSyncStatus,
)
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
from omnibase_infra.protocols.protocol_consumer_sync_source import (
    ProtocolConsumerSyncSource,
)
from omnibase_infra.runtime.health.projection_apply_flow import (
    FALLBACK_IMMUTABLE_GRAIN_PROJECTIONS,
    describe_projection_apply_divergence,
    describe_projection_delta_dropped,
    evaluate_projection_apply_flow,
    projection_apply_divergence_status,
    projection_delta_dropped_status,
)
from omnibase_infra.runtime.health.projection_liveness import (
    describe_dlq_saturation,
    describe_projection_attachment,
    describe_projection_write_path,
    dlq_saturation_status,
    evaluate_projection_liveness,
    select_kernel_nonwriting_projections,
    select_nonprojection_group_infixes,
    select_projection_contracts,
    select_projection_group_suffixes,
)
from omnibase_infra.runtime.health.runtime_lane_identity import (
    resolve_runtime_lane,
)
from omnibase_infra.runtime.observability import get_consumer_flow_counters
from omnibase_infra.runtime.observability.projection_apply_counters import (
    get_projection_apply_counters,
)
from omnibase_infra.runtime.projection_dispatch_ledger import (
    projections_with_no_live_dispatcher,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.utils import (
    apply_instance_discriminator,
    compute_consumer_group_id,
)
from omnibase_infra.utils.correlation import generate_correlation_id

if TYPE_CHECKING:
    from omnibase_infra.models.health.model_projection_contract_ref import (
        ModelProjectionContractRef,
    )
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
    from omnibase_infra.runtime.runtime_profile import resolve_runtime_profile_name

    manifest = discover_contracts()
    # OMN-17985: the ONE validated read of RUNTIME_PROFILE for an ownership
    # decision. The raw environment read this replaces spelled the same "main"
    # default but skipped the registry check, so an unregistered role name was
    # filtered against a list no contract declares -- every contract skipped,
    # the manifest empty -- and this monitor would report that as a finding
    # about the FLEET rather than as the misconfiguration it is.
    runtime_profile = resolve_runtime_profile_name()
    ownership_result = filter_manifest_for_runtime_profile(
        manifest=manifest,
        runtime_profile=runtime_profile,
    )
    # Why: Runtime validation guarantees the returned value matches the contract.
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
    from omnibase_infra.runtime.runtime_profile import resolve_runtime_profile_name

    if not isinstance(manifest, ModelAutoWiringManifest):
        return manifest

    # OMN-17985: same validated read as `_discover_contracts` above. Unset and
    # blank still resolve to "main" -- consolidating adds the registry check,
    # it does not move the ownership default.
    ownership_result = filter_manifest_for_runtime_profile(
        manifest=manifest,
        runtime_profile=resolve_runtime_profile_name(),
    )
    if ownership_result.skipped_contracts:
        logger.debug(
            "Runtime health monitor profile ownership: profile=%s owned=%d skipped=%d",
            ownership_result.runtime_profile,
            ownership_result.manifest.total_discovered,
            len(ownership_result.skipped_contracts),
        )
    return ownership_result.manifest


def _discover_contracts_for_runtime_profile() -> ProtocolAutoWiringManifestLike:
    """Discover and profile-filter, as one synchronous unit to hand to a thread.

    BOTH NAMES ARE RESOLVED THROUGH THIS MODULE'S GLOBALS at call time, which
    is what keeps the two existing patch points working: every test of this
    monitor patches ``_discover_contracts`` and most patch
    ``_filter_manifest_for_runtime_profile`` as well, and a version of this
    that closed over either one would silently run the real scan under them.
    """
    return _filter_manifest_for_runtime_profile(_discover_contracts())


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

    from omnibase_infra.event_bus.kafka_auth import (
        build_confluent_auth_config_from_env,
    )

    timeout_seconds = max(request_timeout_ms / 1000.0, 1.0)
    # OMN-18012: the aiokafka data plane in this same container authenticates
    # from KAFKA_SECURITY_PROTOCOL/KAFKA_SASL_* while this admin client opened
    # PLAINTEXT, so consumer_coverage failed every cycle on a SASL lane and the
    # container was marked unhealthy. Same resolver, confluent projection: the
    # spread is empty on a PLAINTEXT lane.
    admin = AdminClient(
        {
            "bootstrap.servers": bootstrap_servers,
            "socket.timeout.ms": request_timeout_ms,
            "request.timeout.ms": request_timeout_ms,
            **build_confluent_auth_config_from_env(),
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


# Cap on how many failing entry points are named in the discovery_errors detail.
# The detail string is served on every /health response and shipped in every
# health event; a runaway discovery failure must not turn it into a log dump.
_MAX_NAMED_DISCOVERY_ERRORS = 8


def _describe_discovery_errors(
    manifest: ProtocolAutoWiringManifestLike, error_count: int
) -> str:
    """Build the discovery_errors detail, naming the failing entry points.

    OMN-15217: the pre-existing detail was ``"4 contract(s) failed to load"`` —
    a count with no identities, which forced every investigation to go back to
    raw container logs to learn *which* contracts failed (and the boot-time
    ``Failed to load entry point`` lines roll out of the log buffer long before
    anyone looks). Naming them here makes the health surface itself
    self-diagnosing.

    ``ProtocolAutoWiringManifestLike`` only guarantees the counts, so the error
    tuple is read defensively: any manifest that does not expose it degrades to
    the original count-only detail rather than raising inside a health check.
    """
    errors = getattr(manifest, "errors", None)
    names: list[str] = []
    if isinstance(errors, Sequence) and not isinstance(errors, str | bytes):
        for error in errors[:_MAX_NAMED_DISCOVERY_ERRORS]:
            entry_point = getattr(error, "entry_point_name", None)
            if entry_point:
                names.append(str(entry_point))

    base = f"{error_count} contract(s) failed to load"
    if not names:
        return base
    listed = ", ".join(names)
    remaining = error_count - len(names)
    if remaining > 0:
        listed = f"{listed}, +{remaining} more"
    return f"{base}: {listed}"


def _worst(statuses: list[_HealthStatus]) -> _HealthStatus:
    """Return the worst status from a list."""
    if "CRITICAL" in statuses:
        return "CRITICAL"
    if "DEGRADED" in statuses:
        return "DEGRADED"
    return "HEALTHY"


# --- consumer_sync dimension (OMN-18640 AC1) -------------------------------

#: Groups named in the dimension detail before it is capped. The detail lands
#: in container logs and in ``docker inspect`` output on every probe interval;
#: a runtime with fifty wedged groups has one problem, and the first few name
#: it.
_MAX_NAMED_SYNC_GROUPS = 5

#: What the dimension reports when the transport cannot answer the question --
#: the in-memory bus, and any process that consumes nothing. Stated rather
#: than omitted: a dimension that disappears is indistinguishable from one
#: that was never added, and this whole ticket is about surfaces that were
#: silent when they should have been speaking.
CONSUMER_SYNC_UNAVAILABLE = (
    "consumer-group sync reporting is not available on this transport"
)


def describe_consumer_sync(statuses: Sequence[ModelConsumerSyncStatus]) -> str:
    """Render the evidence for the ``consumer_sync`` dimension.

    Always carries the measured numbers, green or red. A detail that is only
    informative when it is failing cannot distinguish "measured and fine" from
    "not measured", which is the distinction the 2026-09-19T04:51Z wedge came
    down to.
    """
    if not statuses:
        return "no consumer groups attached yet"

    out_of_sync = [status for status in statuses if not status.ready]
    if not out_of_sync:
        max_lag = max(status.backlog_records for status in statuses)
        newest_advance = min(status.seconds_since_last_record for status in statuses)
        return (
            f"{len(statuses)} consumer group(s) in sync; "
            f"max lag {max_lag} record(s), "
            f"last advance {newest_advance:.0f}s ago"
        )

    named = [
        _describe_one_group(status) for status in out_of_sync[:_MAX_NAMED_SYNC_GROUPS]
    ]
    remainder = len(out_of_sync) - len(named)
    if remainder > 0:
        named.append(f"+{remainder} more")
    return (
        f"{len(out_of_sync)} of {len(statuses)} consumer group(s) out of sync: "
        + "; ".join(named)
    )


def _describe_one_group(status: ModelConsumerSyncStatus) -> str:
    """One out-of-sync group, with the two facts that prove it."""
    parts = [
        status.consumer_group,
        f"lag={status.backlog_records}",
        f"stalled={status.stalled_seconds:.0f}s",
        f"last_advance={status.seconds_since_last_record:.0f}s",
        f"reason={status.reason.value}",
    ]
    if status.last_rejoin_failed:
        # Named separately from the stall: the group being behind and the
        # self-heal being unable to fix it are two findings, and only the
        # second one says the container needs outside help.
        parts.append(f"rejoin_failed(attempts={status.rejoin_count})")
    return " ".join(parts)


def evaluate_consumer_sync(
    event_bus: object | None,
) -> tuple[_HealthStatus, str]:
    """Grade the consumer-sync dimension from whatever bus the kernel wired.

    CRITICAL rather than DEGRADED when a group is out of sync, for two
    reasons. A runtime holding a group it is not draining is not partially
    doing that topic's work, it is doing none of it while every record
    published meanwhile waits on a client that will not return unaided. And
    CRITICAL is the only grade that fails the container healthcheck
    irrespective of that lane's ``--degraded-policy``, which is what makes
    this fail-closed rather than dependent on a flag.
    """
    if not isinstance(event_bus, ProtocolConsumerSyncSource):
        return "HEALTHY", CONSUMER_SYNC_UNAVAILABLE
    try:
        statuses = tuple(event_bus.consumer_sync_statuses())
    except Exception as read_error:  # noqa: BLE001 — boundary: a health service must not die reporting on something else
        logger.warning(
            "consumer_sync dimension could not read the event bus: %s",
            read_error,
            exc_info=True,
        )
        return "DEGRADED", f"consumer-group sync could not be read: {read_error}"
    detail = describe_consumer_sync(statuses)
    if any(not status.ready for status in statuses):
        return "CRITICAL", detail
    return "HEALTHY", detail


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
        # OMN-15217: retain the latest verdict so the HTTP health surface can
        # publish it. Before this the verdict existed only in logs and on the
        # Kafka health topic, so /health reported healthy while the runtime was
        # DEGRADED — see runtime_health_block for the full mask description.
        self._latest_event: ModelRuntimeHealthCheckEvent | None = None

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

    @property
    def latest_event(self) -> ModelRuntimeHealthCheckEvent | None:
        """Return the most recent health-check event, or ``None`` before the first cycle.

        ``None`` means "no verdict computed yet" — a distinct state from
        HEALTHY. Consumers that need proof of health must fail closed on
        ``None`` rather than reading absence as green (OMN-15217).
        """
        return self._latest_event

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
        # OMN-16994: the contract-declared projection set and the topics that
        # actually have a live subscription. Held outside the try so a discovery
        # failure leaves them at "unobservable" (empty) rather than at a
        # fabricated all-clear.
        projections: tuple[ModelProjectionContractRef, ...] = ()
        # OMN-17562. Bound alongside ``projections`` because discovery raising
        # leaves BOTH halves unresolved, and the liveness evaluation below still
        # runs so the remaining dimensions are reported. An empty tuple is the
        # honest value there: with no manifest, nothing is known to be
        # kernel-nonwriting, and the discovery dimension already carries the
        # failure.
        nonwriting_projections: tuple[ModelProjectionContractRef, ...] = ()
        attached_topics: frozenset[str] = frozenset()
        # OMN-16753. Group-suffix -> projection name, so a flow delta on a
        # topic several projections declare is attributed to the subscription
        # that produced it instead of to the alphabetically last declarer.
        # Empty is the honest value when discovery raised: with no manifest
        # nothing can be attributed, and the saturation half then reports only
        # what a sole-declarer topic proves.
        projection_group_suffixes: dict[str, str] = {}
        # OMN-16753 round 3. The complement: group infixes belonging to
        # consumers that are NOT projections (a reducer, an effect, a
        # forwarder). Their flow on a topic a projection also declares is
        # dropped from the arithmetic rather than counted as unattributable,
        # which is what failed the stability refresh at 915a10446. Empty is the
        # conservative value: nothing excluded, so nothing is silenced.
        nonprojection_group_infixes: dict[str, str] = {}
        try:
            # OFF THE EVENT LOOP (OMN-19373). Contract discovery is a fully
            # synchronous scan: it walks every ``onex.nodes`` entry point,
            # reads and parses the ``contract.yaml`` beside each one, and logs
            # a line per contract. On the .201 runtime that is ~1,000
            # contracts and it takes ABOUT ELEVEN SECONDS. Awaited inline it
            # did not merely make this check slow -- it froze the whole
            # process, because a synchronous call in a coroutine holds the
            # loop until it returns, and this loop is also the one answering
            # the gateway.
            #
            # What that cost, measured on 2026-09-24 against ``onex-dev`` on
            # i-06169517a92b45f86: ``onex-api`` publishes a heartbeat command
            # and waits 10.0s for ``gateway-session.v1``. Any heartbeat that
            # arrived inside a sweep could not be answered before the budget
            # expired, so the customer got a 503 while the work was merely
            # late -- in both captured cases runtime-effects published the
            # answer in the very second the sweep ended, after the caller had
            # already abandoned it. 2 of 14 heartbeats over one 3m42s run at
            # the designed 15s cadence; ~1s is the healthy round trip.
            #
            # This check runs every ``check_interval_seconds`` (300s), so the
            # refusal recurred about every five minutes and no client cadence
            # could avoid it. ``asyncio.to_thread`` is the same primitive this
            # very method already uses for the Kafka admin call below; this
            # line was the one blocking call in it that never got it.
            manifest = await asyncio.to_thread(_discover_contracts_for_runtime_profile)
            contract_count = manifest.total_discovered
            discovery_error_count = manifest.total_errors
            subscribe_topics = set(manifest.all_subscribe_topics())
            expected_groups = _expected_consumer_groups_from_manifest(
                manifest, os.environ.get("KAFKA_INSTANCE_ID")
            )
            # OMN-17562. The kernel withholds the Kafka subscription for a
            # contract EVERY handler entry of which wires a no-op dispatch, so
            # its topics correctly leave the live bus registry. This selector is
            # manifest-derived while ``attached_topics`` is that registry, so
            # keeping those contracts in the attachment scope would report a
            # fleet-wide false outage on ``projection_attachment`` in place of
            # the one this change removed from ``projection_write_path``. The
            # excluded half is resolved back to refs so the write-path leg can
            # still name it and can still detect one that is STILL attached.
            kernel_nonwriting = projections_with_no_live_dispatcher()
            projections = select_projection_contracts(
                manifest, kernel_nonwriting=kernel_nonwriting
            )
            nonwriting_projections = select_kernel_nonwriting_projections(
                manifest, kernel_nonwriting
            )
            projection_group_suffixes = select_projection_group_suffixes(
                manifest, projections
            )
            nonprojection_group_infixes = select_nonprojection_group_infixes(
                manifest, projection_group_suffixes
            )
            live_expected_groups = _expected_consumer_groups_from_event_bus(
                self._event_bus
            )
            # OMN-16994: capture the live registry BEFORE it overwrites
            # ``expected_groups`` below. That override is what hid nineteen
            # unattached projections for months — a contract that never
            # subscribed is absent from the registry, so replacing the
            # manifest-derived expectation set with it makes the missing
            # consumer stop being expected instead of being reported.
            attached_topics = frozenset(
                expected.topic for expected in live_expected_groups
            )
            if live_expected_groups:
                expected_groups = live_expected_groups
                subscribe_topics = {expected.topic for expected in live_expected_groups}

            if discovery_error_count > 0:
                dimensions.append(
                    ModelRuntimeHealthDimension(
                        name="discovery_errors",
                        status="DEGRADED",
                        detail=_describe_discovery_errors(
                            manifest, discovery_error_count
                        ),
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

        # --- Dimension 4 & 5: Projection liveness (OMN-16994) ----------------
        # OMN-16843's deferred AC6. Every dimension above measures
        # CONNECTEDNESS; none of them measures whether a projection persists
        # anything. A projection that never attached, and a projection that
        # attaches and quarantines 100% of what it takes, both read green on
        # consumer-group state, on lag, and on process health.
        liveness = evaluate_projection_liveness(
            projections=projections,
            attached_topics=attached_topics,
            flow_windows=get_consumer_flow_counters().retained_windows.snapshot(),
            kernel_nonwriting=nonwriting_projections,
            projection_group_suffixes=projection_group_suffixes,
            nonprojection_group_infixes=nonprojection_group_infixes,
        )
        dimensions.append(
            ModelRuntimeHealthDimension(
                name="projection_attachment",
                status="DEGRADED" if liveness.unattached_projections else "HEALTHY",
                detail=describe_projection_attachment(liveness),
            )
        )
        # OMN-16753. The status comes from ``dlq_saturation_status`` rather than
        # being recomputed here, so it cannot disagree with the prose beside it.
        # It is produced in this dimension's own vocabulary, so there is no
        # narrowing step between the two that could regrade a value silently.
        # It is DEGRADED on an unattributable topic as well as on a measured
        # saturation: flow this process cannot attribute is excluded from every
        # ratio, and publishing HEALTHY over that exclusion is a false all-clear
        # on the one dimension that exists to catch a silent total loss.
        dimensions.append(
            ModelRuntimeHealthDimension(
                name="projection_dlq_saturation",
                status=dlq_saturation_status(liveness),
                detail=describe_dlq_saturation(liveness),
            )
        )
        # --- Dimension 6: Projection write path (OMN-17448/OMN-17562) -------
        # The third way to persist nothing. Dimensions 4 and 5 both read green
        # through it by construction: the topic IS attached (only the dispatch
        # is a no-op), and nothing raises so nothing reaches a DLQ. Measured
        # live on the .201 dev lane 2026-09-01 -- this monitor logged
        # `status=HEALTHY ... projections=13 unattached_projections=0` while
        # node_projection_tenant_registry had no writer on ANY lane and
        # tenant_registry_mirror held 0 rows.
        #
        # OMN-17562 splits the fact the status is taken from. DEGRADED is
        # reserved for the half this process can actually observe AND is
        # actually doing wrong: still SUBSCRIBED while dispatching nothing, so
        # offsets commit over destroyed events. Merely having no in-process
        # dispatcher is a deployment fact -- whether a dedicated writer runs on
        # this lane is a corpus-level claim over the deployment manifests
        # (OMN-17448 AC5) that a kernel process cannot see, and asserting it
        # here would be a permanent DEGRADED no operator action could clear.
        # The names stay on the detail either way.
        dimensions.append(
            ModelRuntimeHealthDimension(
                name="projection_write_path",
                status=(
                    "DEGRADED"
                    if liveness.nonwriting_attached_projections
                    else "HEALTHY"
                ),
                detail=describe_projection_write_path(liveness),
            )
        )

        # --- Dimension 7: Consumer group sync (OMN-18640 AC1) ---------------
        # Every dimension above reads BROKER-side or PROCESS-side state. None
        # of them reads whether this runtime's own consumers are still
        # fetching. On 2026-09-19T04:51:48Z the .201 dev-lane broker was
        # recreated, the effects consumer wedged with its offsets frozen at
        # 7540, and it stayed that way for about fifty minutes until the whole
        # lane was recreated at 05:41:22Z -- while the group reported Stable
        # with its partition assigned (so `empty_consumer_groups` and
        # `consumer_coverage` were green), the contracts had all loaded, the
        # projections were attached, and the container reported Up with
        # RestartCount 0. The deploy agent's force-recreate backstop never
        # fired because the only thing it probes is that container's health.
        #
        # This is deliberately NOT computed from the profile-filtered manifest
        # like the coverage dimensions above. The supervisors exist one per
        # (topic, group) this PROCESS actually polls, so the set is scoped by
        # construction -- there is no ownership filter here to get wrong.
        consumer_sync_status, consumer_sync_detail = evaluate_consumer_sync(
            self._event_bus
        )
        dimensions.append(
            ModelRuntimeHealthDimension(
                name="consumer_sync",
                status=consumer_sync_status,
                detail=consumer_sync_detail,
            )
        )

        # --- Dimensions 8 and 9: Projection flow invariants (OMN-18910) -----
        # Every dimension above answers whether a consumer is attached and
        # MOVING. None answers whether it is moving and WRITING, and those
        # come apart: a consumer that refuses a message still commits its
        # offset, a projection that returns without writing still commits its
        # offset, and a cache that drops a delta still consumed it. Lag was
        # zero throughout OMN-18880 (nine hours of total refusal behind three
        # green surfaces), OMN-18905 (a snapshot cache frozen mid-replay while
        # readiness reported every topic bootstrapped) and OMN-18769 (a writer
        # that wrote nothing and raised nothing).
        #
        # The honest claim: this does not prevent any of the three. It
        # shortens detection, which in all three was set by a person happening
        # to look.
        apply_counters = get_projection_apply_counters()
        # This cycle is also the window closer, and deliberately the ONLY one.
        # The first revision closed the apply window on the heartbeat tick,
        # beside the throughput window, which coupled the dimension to
        # introspection being enabled AND to this node holding flow-window
        # carriage. On a lane where neither holds, no window would ever close
        # and both dimensions would sit permanently DEGRADED on their
        # unobserved branch -- a check that cannot pass, which Operating Rule
        # 24's correction is explicit is worse than one that cannot fail,
        # because it stops delivery rather than missing a defect. Closing here
        # means a window exists from the first cycle on any lane that runs
        # this monitor at all.
        try:
            apply_counters.close_window()
        except Exception:  # noqa: BLE001 -- the dimension fails closed below
            logger.warning(
                "Projection apply window could not be closed; the apply-flow "
                "dimensions will report their unobserved outcome rather than "
                "a clean one",
                exc_info=True,
            )
        # OMN-19081. The exemption set comes from the CONTRACT-declared grain
        # each projection registered with at wiring time, not from a literal
        # list in this repository. A copy of a declared fact in the consuming
        # repo fails quietly both ways: a new content-addressed exposure is
        # alarmed on until somebody edits a tuple here, and one wrongly listed
        # here is exempt forever with no evidence behind it.
        apply_flow = evaluate_projection_apply_flow(
            windows=apply_counters.retained_windows(),
            registered_projections=apply_counters.registered_projections(),
            immutable_grain_projections=apply_counters.immutable_grain_projections(),
            grain_unresolved_projections=apply_counters.grain_unresolved_projections(),
            # OMN-19081, operator ruling 2026-09-21. Consulted ONLY where the
            # contract resolved nothing, which is the window in which a
            # deployed runtime predates the key_grain declaration. Deleting
            # the literal outright was measured on dogfood-101 to turn the two
            # content-addressed exposures DEGRADED on healthy behaviour.
            fallback_immutable_projections=FALLBACK_IMMUTABLE_GRAIN_PROJECTIONS,
        )
        dimensions.append(
            ModelRuntimeHealthDimension(
                name="projection_apply_divergence",
                status=projection_apply_divergence_status(apply_flow),
                detail=describe_projection_apply_divergence(apply_flow),
            )
        )
        dimensions.append(
            ModelRuntimeHealthDimension(
                name="projection_delta_dropped",
                status=projection_delta_dropped_status(apply_flow),
                detail=describe_projection_delta_dropped(apply_flow),
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
            projection_count=liveness.projection_count,
            unattached_projection_count=len(liveness.unattached_projections),
            dlq_saturated_projection_count=len(liveness.dlq_saturated_projections),
            nonwriting_projection_count=len(liveness.nonwriting_projections),
            # OMN-18769: the lane this verdict is ABOUT. Absent when the
            # deployment does not name one -- see the field's own note on
            # why that is nullable rather than defaulted.
            lane=resolve_runtime_lane(),
        )

        logger.info(
            "Runtime health check: status=%s contracts=%d errors=%d "
            "consumer_groups=%d empty=%d uncovered_topics=%d "
            "projections=%d unattached_projections=%d dlq_saturated_projections=%d "
            "nonwriting_projections=%d",
            aggregate_status,
            contract_count,
            discovery_error_count,
            consumer_group_count,
            empty_consumer_group_count,
            uncovered_topic_count,
            liveness.projection_count,
            len(liveness.unattached_projections),
            len(liveness.dlq_saturated_projections),
            len(liveness.nonwriting_projections),
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

        self._latest_event = event

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
            tenant_id=None,
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
