# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka Topic Provisioner for automatic topic creation on startup.

Ensures that all ONEX topics (platform + domain plugins) exist before the
runtime begins consuming or producing events. Uses AIOKafkaAdminClient to
create topics that are missing, with best-effort semantics (warnings on
failure, never blocks startup).

Design:
    - Best-effort: Logs warnings but never blocks startup on failure
    - Idempotent: Safe to call multiple times (skips existing topics)
    - Compatible: Works with both Redpanda and Apache Kafka
    - Configurable: Supports custom topic configs via ModelSnapshotTopicConfig

Related Tickets:
    - OMN-1990: Kafka topic auto-creation gap
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.event_bus.enum_topic_readiness_failure_reason import (
    EnumTopicReadinessFailureReason,
)
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_from_env
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_readiness_failure import (
    ModelTopicReadinessFailure,
)
from omnibase_infra.event_bus.model_topic_set_readiness import (
    ModelTopicSetReadiness,
)
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.topics.broker_capacity_probe import (
    bind_policy_to_broker_capacity,
    is_invalid_replication_factor_error,
)
from omnibase_infra.topics.model_topic_provisioning_diff import (
    ModelTopicProvisioningDiff,
    build_provisioning_diff,
)
from omnibase_infra.topics.model_topic_provisioning_policy import (
    ModelTopicProvisioningPolicy,
    resolve_specs_for_creation,
)
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.models.projection.model_snapshot_topic_config import (
        ModelSnapshotTopicConfig,
    )

logger = logging.getLogger(__name__)

# OMN-8783: No default — KAFKA_BOOTSTRAP_SERVERS must be set via overlay.
ENV_BOOTSTRAP_SERVERS = "KAFKA_BOOTSTRAP_SERVERS"
ENV_TOPIC_PARTITION_CAP = "ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS"

# Default partition count for standard event topics.
#
# OMN-15395: the companion ``DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR = 1`` is
# GONE. It was the mechanism that overrode the MSK broker's own RF2 default down
# to RF1 on every topic whose contract declared nothing — 519 RF1 topics.
# Replication is now resolved by ``ModelTopicProvisioningPolicy`` on the
# creation path, which fails closed against a managed cluster.
DEFAULT_EVENT_TOPIC_PARTITIONS = 6


def topic_partition_cap_from_env() -> int | None:
    """Return the lane's partition cap, or ``None`` when uncapped.

    Public because the operator CLI (``scripts/create_kafka_topics.py``) must
    apply the SAME cap the runtime provisioner applies. A CLI that creates a
    topic at its contract-declared 6 partitions on a lane the runtime caps to 1
    manufactures permanent partition drift between the two live creation paths.
    """
    raw_value = os.environ.get(ENV_TOPIC_PARTITION_CAP)
    if raw_value is None or raw_value.strip() == "":
        return None

    try:
        cap = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; expected a positive integer",
            ENV_TOPIC_PARTITION_CAP,
            raw_value,
        )
        return None

    if cap in (-1, 0):
        return None

    if cap < 0:
        logger.warning(
            "Ignoring invalid %s=%r; expected -1, zero, or a positive integer",
            ENV_TOPIC_PARTITION_CAP,
            raw_value,
        )
        return None

    return cap


def _topic_provisioning_sort_key(spec: ModelTopicSpec) -> tuple[int, str]:
    """Sort topics using contract-declared provisioning priority."""
    return (spec.provisioning_priority, spec.suffix)


def unhostable_replication_error(
    *,
    topic: str,
    requested_replication_factor: int | None,
    policy: ModelTopicProvisioningPolicy,
    cause: BaseException,
) -> TopicReplicationPolicyError:
    """Build the typed, receipted error for a broker-rejected replica count."""
    context = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.KAFKA,
        operation="create_topic",
        target_name=topic,
    )
    measured = (
        f"the cluster measured {policy.broker_count} broker(s)"
        if policy.broker_count is not None
        else (
            "the cluster's broker count could NOT be measured, so no capacity "
            "ceiling was installed and the declared value was passed through "
            "unreduced"
        )
    )
    return TopicReplicationPolicyError(
        f"Broker refused to create topic {topic!r} with "
        f"replication_factor={requested_replication_factor}: "
        f"INVALID_REPLICATION_FACTOR. {measured}. This is a hard provisioning "
        "failure, not a best-effort miss — the topic does NOT exist. Fix the "
        f"owning contract's topic_config.replication_factor, or run against a "
        "cluster whose describe_cluster is reachable so the measured capacity "
        f"ceiling can reduce it (OMN-15395). Broker error: {cause!r}",
        context=context,
        topic=topic,
        declared_replication_factor=requested_replication_factor,
        profile=policy.profile.value,
        broker_count=policy.broker_count,
    )


class TopicProvisioner:
    """Provisions Kafka topics automatically on startup.

    Creates ONEX platform topics if they don't already exist, using
    AIOKafkaAdminClient. Topic creation is best-effort: failures log
    warnings but never block startup.

    The provisioner handles two categories of topics:
    1. **Standard event topics**: Created with default settings (delete cleanup)
    2. **Snapshot topics**: Created with compaction settings from ModelSnapshotTopicConfig

    Thread Safety:
        This class is coroutine-safe. All methods are async and use
        the AIOKafkaAdminClient which handles its own connection pooling.

    Example:
        >>> provisioner = TopicProvisioner(contracts_root=Path("src/.../nodes"))
        >>> await provisioner.ensure_provisioned_topics_exist()
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        request_timeout_ms: int = 30000,
        *,
        contracts_root: Path,
        skill_manifests_root: Path | None = None,
        skill_manifests_roots: list[Path] | None = None,
        policy: ModelTopicProvisioningPolicy | None = None,
    ) -> None:
        """Initialize the topic provisioner.

        Args:
            bootstrap_servers: Kafka broker addresses. If None, reads from
                KAFKA_BOOTSTRAP_SERVERS env var (raises KeyError if absent).
            request_timeout_ms: Timeout for admin operations in milliseconds.
            contracts_root: Path to contract.yaml root directory. Required.
                Topics are discovered from contracts via
                ContractTopicExtractor. The directory must exist; a
                ``FileNotFoundError`` is raised at construction time if it
                does not.
            skill_manifests_root: Optional single path to omniclaude skills
                root (plugins/onex/skills/). Kept for backwards compatibility.
            skill_manifests_roots: Optional list of paths to scan for
                topics.yaml manifests (supports multiple roots: skills,
                CLI relays, services). When both singular and plural are set,
                the singular root is prepended to the list.
            policy: Replication policy for the target broker (OMN-15395).
                Defaults to the policy derived from the live Kafka client
                configuration, so a managed (MSK) target rejects a declared RF1
                fail-closed and resolves an undeclared replication factor to the
                managed durability floor rather than to 1. Whatever is supplied
                here is UNMEASURED — it carries no capacity ceiling until the
                first admin connection binds a live ``describe_cluster`` broker
                count to it (see :meth:`_measured_policy`). The measurement may
                only install a ceiling and raise an undeclared default; it never
                weakens the durability floor.

        Raises:
            FileNotFoundError: If *contracts_root* does not point to an
                existing directory.

        Ticket: OMN-4594, OMN-4622, OMN-5132, OMN-15395
        """
        if not contracts_root.is_dir():
            raise FileNotFoundError(
                f"contracts_root does not exist or is not a directory: {contracts_root}"
            )
        # OMN-8783: Hard-fail if not provided and env var absent.
        self._bootstrap_servers = bootstrap_servers or os.environ[ENV_BOOTSTRAP_SERVERS]
        self._request_timeout_ms = request_timeout_ms
        self._contracts_root = contracts_root
        self._skill_manifests_root = skill_manifests_root
        self._skill_manifests_roots = skill_manifests_roots
        self._topic_partition_cap = topic_partition_cap_from_env()
        self._policy = policy or ModelTopicProvisioningPolicy.from_env()
        # OMN-15395 (D4): memoize the capacity probe ATTEMPT, not merely a
        # successful one. Keying the "already probed?" test on
        # ``policy.broker_count is None`` re-probed an UNMEASURABLE cluster on
        # every entrypoint — three entrypoints, three describe_cluster round
        # trips, none of which could ever succeed — which is the per-call
        # fan-out (d) exists to eliminate, reintroduced on the failure path. A
        # policy supplied already-measured counts as probed.
        self._capacity_probed = self._policy.broker_count is not None
        self._topic_specs = self._build_topic_specs()
        # OMN-15395 (c): the contract-derived spec registry every path resolves
        # against, so a caller that knows only a topic NAME still creates that
        # topic to its owning contract's declared partitions/replication/config
        # instead of falling back to bare module defaults.
        self._spec_by_name: dict[str, ModelTopicSpec] = {
            spec.suffix: spec for spec in self._topic_specs
        }
        # OMN-15395 (d): live broker snapshot, so a pass over an
        # already-provisioned cluster issues zero CreateTopics instead of
        # ~1,280 blind authorizations. Lifecycle: re-fetched at the top of every
        # ``ensure_provisioned_topics_exist`` pass, fetched lazily once for the
        # ``ensure_topic_exists`` path, and folded forward on each create. It is
        # deliberately not invalidated on a timer — the only staleness a
        # provisioner instance can observe is a topic deleted out-of-band, and
        # the cost there is a skipped create that the next full pass repairs.
        self._existing_topics: frozenset[str] | None = None
        # OMN-15395 (c): resolved specs of topics THIS provisioner created, used
        # as the readiness expectation so a freshly created topic is confirmed
        # against the spec it was created with.
        self._created_specs: dict[str, ModelTopicSpec] = {}

    @property
    def policy(self) -> ModelTopicProvisioningPolicy:
        """The replication policy this provisioner resolves specs against.

        Unmeasured until the first admin connection; thereafter bound to the
        cluster's live broker count (OMN-15395).
        """
        return self._policy

    async def _measured_policy(self, admin: object) -> ModelTopicProvisioningPolicy:
        """Bind the policy to the cluster's live broker count, once.

        The capacity ceiling that reduces a contract-declared replication
        factor MUST come from a measurement of the target broker, never from an
        inference off the SASL mechanism: ``ModelKafkaEventBusConfig`` accepts
        PLAIN / SCRAM / OAUTHBEARER as well as MSK IAM, so "not IAM" says
        nothing at all about node count. Measuring here — on the same admin
        client that is about to issue the ``CreateTopics`` — is the only place
        the ceiling can be honest.

        The ATTEMPT is memoized, not the success (OMN-15395 D4). A cluster whose
        ``describe_cluster`` is absent or failing leaves ``broker_count`` at
        ``None`` forever, so testing that field re-ran the probe on every
        entrypoint; the sentinel bounds it to one attempt per instance.
        """
        if self._capacity_probed:
            return self._policy
        self._capacity_probed = True
        self._policy = await bind_policy_to_broker_capacity(admin, self._policy)
        return self._policy

    def _creation_partitions(self, spec: ModelTopicSpec) -> int:
        if self._topic_partition_cap is None:
            return spec.partitions
        return min(spec.partitions, self._topic_partition_cap)

    def _creation_spec(self, spec: ModelTopicSpec) -> ModelTopicSpec:
        """Return the spec as this provisioner would actually CREATE it.

        Replication resolved through the policy, partitions clamped by the
        lane's env cap. This is the single "effective spec" every site must
        compare against — BOTH creation sites (the per-topic
        :meth:`ensure_topic_exists` and the batch
        :meth:`ensure_provisioned_topics_exist` loop), the (OMN-15395 D3) drift
        site, and — through what those creation sites hand
        :meth:`_note_topic_created` — the readiness site, which reads
        ``_created_specs``. "There is one resolver" only holds if every
        consumer of that record is fed by it: recording anything else here
        poisons readiness for the whole life of the process (OMN-16844).

        Raises:
            TopicReplicationPolicyError: The spec violates the policy.
        """
        resolved = self._resolve_spec(spec)
        partitions = self._creation_partitions(resolved)
        if partitions == resolved.partitions:
            return resolved
        return resolved.model_copy(update={"partitions": partitions})

    def _resolve_spec(self, spec: ModelTopicSpec) -> ModelTopicSpec:
        """Resolve one spec's replication factor through the environment policy.

        Raises:
            TopicReplicationPolicyError: RF below the environment floor, or
                undeclared RF where the policy has no default.
        """
        return self._policy.resolve_spec(spec)

    def _resolve_specs_for_creation(
        self,
        specs: Sequence[ModelTopicSpec],
        correlation_id: UUID,
    ) -> tuple[ModelTopicSpec, ...]:
        """Resolve every spec BEFORE any ``CreateTopics`` is issued (OMN-15395 a/b).

        Fail-closed and batch-scoped: a single spec that violates the
        environment replication policy — the RF1-on-MSK case — aborts the whole
        pass with ZERO creates issued. Not a warning, not a clamp-and-continue,
        and not a per-topic skip that lets the rest of the pass proceed while a
        durability defect sits unfixed in a contract we own. Every violation is
        collected first so one boot surfaces every offending contract instead of
        one per redeploy.

        The batch resolution itself lives in
        :func:`~omnibase_infra.topics.model_topic_provisioning_policy.resolve_specs_for_creation`
        so the operator CLI (``scripts/create_kafka_topics.py``) enforces the
        identical fail-closed rule; this wrapper only adds the
        correlation-scoped log line.

        Returns:
            The resolved specs, each carrying an explicit replication factor.

        Raises:
            TopicReplicationPolicyError: Any spec violates the policy.
        """
        try:
            return resolve_specs_for_creation(self._policy, specs)
        except TopicReplicationPolicyError:
            logger.exception(
                "Refusing to provision topic(s) under the %s replication "
                "policy; no CreateTopics issued (correlation_id=%s)",
                self._policy.profile.value,
                correlation_id,
            )
            raise

    async def _fetch_broker_topic_metadata(
        self,
        admin: object,
    ) -> tuple[dict[str, Mapping[str, object]], frozenset[str]]:
        """Snapshot the broker's live topics in a single metadata request.

        Returns ``(metadata_by_topic, existing_names)``. One metadata request
        replaces the previous "issue CreateTopics for every known topic and use
        ``TopicAlreadyExistsError`` as flow control" pattern.
        """
        describe = getattr(admin, "describe_topics", None)
        entries: object = []
        if describe is not None:
            entries = await describe()
        metadata: dict[str, Mapping[str, object]] = {}
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                name = entry.get("topic")
                if not isinstance(name, str):
                    continue
                error_code = entry.get("error_code")
                if isinstance(error_code, int) and error_code != 0:
                    # Broker knows the name but cannot serve it (e.g. unknown
                    # topic) — treat as absent so it is created, not skipped.
                    continue
                metadata[name] = entry
        return metadata, frozenset(metadata)

    async def _existing_topic_names(self, admin: object) -> frozenset[str]:
        """Return the cached live topic snapshot, fetching it once if needed."""
        if self._existing_topics is None:
            _, names = await self._fetch_broker_topic_metadata(admin)
            self._existing_topics = names
        return self._existing_topics

    def _note_topic_created(
        self,
        topic_name: str,
        spec: ModelTopicSpec | None = None,
    ) -> None:
        """Fold a freshly created topic into the cached snapshot + readiness specs."""
        if self._existing_topics is not None:
            # ``.union`` rather than ``|``: the repo's union-count ratchet parses
            # a bare ``|`` here as a type union and counts it against the budget.
            self._existing_topics = self._existing_topics.union({topic_name})
        if spec is not None:
            self._created_specs[topic_name] = spec

    def _report_spec_drift(
        self,
        present_topics: Sequence[str],
        metadata: Mapping[str, Mapping[str, object]],
        correlation_id: UUID,
    ) -> list[str]:
        """Report partition/replication drift on already-existing topics.

        OMN-15395 (d): drift on a live topic is REPORTED, never silently
        re-created or mutated — repairing the 519 pre-existing RF1 topics is the
        operator-gated WS-M reassignment lane, not this provisioner's call.
        Reuses ``evaluate_topic_readiness`` rather than re-implementing the
        comparison.

        The expectation is the spec this provisioner would actually CREATE —
        :meth:`_creation_spec`, i.e. replication resolved through the policy AND
        partitions clamped by the lane's env cap. "There is one resolver" only
        holds if every site uses what it returns, and that includes the site
        that decides what counts as drift:

        * comparing a broker against an unresolved RF2 on a cluster measured at
          one node reports every RF2 topic as drifted even though the
          provisioner deliberately and correctly created it at RF1 there; and
        * comparing against the UNCAPPED ``spec.partitions`` reports every
          contract-declared 6-partition topic as ``partition_mismatch`` on every
          dev/stability lane, where ``ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS=1``
          means the provisioner itself created them with one partition
          (OMN-15395 D3).

        Both seed the operator-gated WS-M reassignment queue that consumes this
        feed with targets the cluster cannot host — a provisioner reporting
        drift against topics it just created, correctly.

        Partition divergence the cap *explains* is not silently dropped either:
        a pre-existing topic carrying more partitions than the current cap
        allows (created before the cap was lowered) is reported under a distinct
        ``partition_cap_suppressed`` label. Kafka cannot reduce a partition
        count, so emitting that as ``partition_mismatch`` would feed the repair
        queue an impossible instruction; emitting nothing at all would hide a
        real difference between contract and broker.

        A spec the policy REFUSES (a contract declaring RF1 against managed) is
        reported here rather than raised: the fail-closed abort is scoped to
        topics being created, and a pre-existing topic is already on the broker.
        """
        expected: dict[str, ModelTopicSpec] = {}
        refusals: list[str] = []
        for name in present_topics:
            declared = self._spec_by_name.get(name)
            if declared is None:
                continue
            try:
                expected[name] = self._creation_spec(declared)
            except TopicReplicationPolicyError as exc:
                refusals.append(f"{name}: replication_policy_violation: {exc}")
        if refusals:
            logger.warning(
                "%d existing topic(s) have a contract spec the %s replication "
                "policy would refuse to create: %s (correlation_id=%s)",
                len(refusals),
                self._policy.profile.value,
                refusals,
                correlation_id,
            )
        if not expected:
            return refusals
        evaluation = evaluate_topic_readiness(
            tuple(expected),
            [metadata[name] for name in expected if name in metadata],
            expected_specs=expected,
        )
        drift: list[str] = []
        cap_suppressed: list[str] = []
        for failure in evaluation.failures:
            if failure.reason not in (
                EnumTopicReadinessFailureReason.PARTITION_MISMATCH,
                EnumTopicReadinessFailureReason.REPLICATION_MISMATCH,
            ):
                continue
            entry = f"{failure.topic}: {failure.reason.value}: {failure.detail}"
            if (
                failure.reason is EnumTopicReadinessFailureReason.PARTITION_MISMATCH
                and self._partition_gap_is_cap_explained(
                    failure.topic, metadata.get(failure.topic)
                )
            ):
                cap_suppressed.append(
                    f"{failure.topic}: partition_cap_suppressed: {failure.detail} "
                    f"(explained by {ENV_TOPIC_PARTITION_CAP}="
                    f"{self._topic_partition_cap}; partitions cannot be reduced, "
                    "so this is NOT a reassignment target)"
                )
                continue
            drift.append(entry)
        if cap_suppressed:
            logger.info(
                "Partition divergence on %d existing topic(s) is explained by "
                "the lane partition cap — NOT reported as drift (OMN-15395): "
                "%s (correlation_id=%s)",
                len(cap_suppressed),
                cap_suppressed,
                correlation_id,
            )
        if drift:
            logger.warning(
                "Topic spec drift on %d existing topic(s) — reported, NOT "
                "re-created or mutated (OMN-15395): %s (correlation_id=%s)",
                len(drift),
                drift,
                correlation_id,
            )
        return refusals + drift

    def _partition_gap_is_cap_explained(
        self,
        topic: str,
        entry: Mapping[str, object] | None,
    ) -> bool:
        """True when the broker's extra partitions are explained by the cap.

        The lane cap can only ever be *lowered* against topics that already
        exist — Kafka has no partition-reduction operation — so a topic whose
        broker partition count sits between the capped expectation
        (exclusive) and the contract-declared count (inclusive) is a topic
        created before the current cap, not a contract/broker disagreement the
        reassignment lane can act on.
        """
        if self._topic_partition_cap is None or entry is None:
            return False
        declared = self._spec_by_name.get(topic)
        if declared is None:
            return False
        raw = entry.get("partitions")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return False
        observed = len([p for p in raw if isinstance(p, Mapping)])
        return self._topic_partition_cap < observed <= declared.partitions

    def _build_topic_specs(self) -> tuple[ModelTopicSpec, ...]:
        """Build topic specs from contract YAML extraction.

        Topics are derived entirely from contract YAML extraction via
        ``ContractTopicExtractor.extract_all()``. There is no fallback to
        the Python constant registry (``ALL_PROVISIONED_TOPIC_SPECS``).

        Raises:
            ImportError: If ``ContractTopicExtractor`` is not importable.
            RuntimeError: If extraction fails unexpectedly.

        Ticket: OMN-4594, OMN-4622, OMN-5132
        """
        from omnibase_infra.tools.contract_topic_extractor import (
            ContractTopicExtractor,
        )

        extractor = ContractTopicExtractor(include_installed_packages=True)
        contract_entries = extractor.extract_all(
            contracts_root=self._contracts_root,
            skill_manifests_root=self._skill_manifests_root,
            skill_manifests_roots=self._skill_manifests_roots,
        )

        result_specs: list[ModelTopicSpec] = []
        for entry in contract_entries:
            # Per-topic config (OMN-13238): when a contract declares a
            # ``topic_config`` block the extractor carries partitions /
            # replication_factor / kafka_config.
            #
            # OMN-15395: an undeclared replication_factor is carried through as
            # None — "the contract declared nothing" — and is resolved (or
            # refused) by the environment policy on the creation path. It is NOT
            # silently coerced to 1 here any more.
            result_specs.append(
                ModelTopicSpec(
                    suffix=entry.topic,
                    provisioning_priority=entry.provisioning_priority,
                    partitions=(
                        entry.partitions
                        if entry.partitions is not None
                        else DEFAULT_EVENT_TOPIC_PARTITIONS
                    ),
                    replication_factor=entry.replication_factor,
                    kafka_config=(
                        dict(entry.kafka_config)
                        if entry.kafka_config is not None
                        else None
                    ),
                )
            )

        result = tuple(sorted(result_specs, key=_topic_provisioning_sort_key))

        skill_count = len([e for e in contract_entries if "omniclaude" in e.topic])
        logger.info(
            "topic provisioning (contract-first) — total: %d, "
            "skill-manifest topics: %d",
            len(result),
            skill_count,
        )

        return result

    async def ensure_provisioned_topics_exist(
        self,
        correlation_id: UUID | None = None,
    ) -> dict[str, list[str] | str]:
        """Ensure all ONEX provisioned topics exist.

        Lists the broker's live topics FIRST and creates only the genuinely
        missing ones (OMN-15395 d) — a pass over an already-provisioned cluster
        issues zero ``CreateTopics``. Every created topic's replication factor is
        resolved through the environment policy before any create is issued
        (OMN-15395 a/b), so an RF1 spec against managed staging aborts the whole
        pass instead of creating a topic that cannot survive a broker loss.

        Individual topic creation failures are best-effort: they log warnings and
        do not prevent other topics from being created. Unrecoverable failures
        (connection, authentication) also degrade to a warning and never block
        startup. A replication-policy violation is NOT best-effort — it
        propagates to the caller with nothing created.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Summary dict with:
                - created: List of newly created topic names
                - existing: List of topics that already existed
                - failed: List of topics that failed to create
                - drift: Partition/replication drift found on existing topics
                  (reported only — never re-created or mutated)
                - status: "success", "partial", or "unavailable"

        Raises:
            TopicReplicationPolicyError: A missing topic's spec violates the
                environment replication policy (raised before any
                ``CreateTopics`` — the pass creates nothing), or the broker
                refused a ``CreateTopics`` with ``INVALID_REPLICATION_FACTOR``
                (OMN-15395 D5 — an unhostable replica count leaves the topic
                absent, so it is never degraded to a warning + ``failed``).
        """
        correlation_id = correlation_id or uuid4()
        created: list[str] = []
        existing: list[str] = []
        failed: list[str] = []
        drift: list[str] = []

        try:
            from aiokafka.admin import AIOKafkaAdminClient, NewTopic
            from aiokafka.errors import (
                TopicAlreadyExistsError as _TopicAlreadyExistsError,
            )
        except ImportError:
            logger.warning(
                "aiokafka not available, skipping topic auto-creation. "
                "Install aiokafka to enable automatic topic management.",
                extra={"correlation_id": str(correlation_id)},
            )
            return {
                "created": created,
                "existing": existing,
                "failed": [s.suffix for s in self._topic_specs],
                "drift": drift,
                "status": "unavailable",
            }

        # Bind to local after successful import block
        TopicAlreadyExistsError = _TopicAlreadyExistsError

        admin: AIOKafkaAdminClient | None = None
        try:
            auth_kwargs = build_aiokafka_auth_kwargs_from_env()
            admin = AIOKafkaAdminClient(
                bootstrap_servers=self._bootstrap_servers,
                request_timeout_ms=self._request_timeout_ms,
                **auth_kwargs,
            )
            await admin.start()

            # Measure the cluster's node count BEFORE resolving anything: the
            # capacity ceiling that may reduce a contract-declared replication
            # factor has to be a measurement of this broker, not an inference
            # from its auth mechanism (OMN-15395).
            await self._measured_policy(admin)

            # (d) One metadata request replaces the blind create-everything
            # sweep. Only names the broker does not already have are candidates.
            metadata, existing_names = await self._fetch_broker_topic_metadata(admin)
            self._existing_topics = existing_names
            diff: ModelTopicProvisioningDiff = build_provisioning_diff(
                (spec.suffix for spec in self._topic_specs), existing_names
            )
            existing.extend(diff.present_topics)
            drift.extend(
                self._report_spec_drift(diff.present_topics, metadata, correlation_id)
            )

            missing = set(diff.missing_topics)
            missing_specs = [
                spec for spec in self._topic_specs if spec.suffix in missing
            ]
            logger.info(
                "Topic provisioning diff: desired=%d present=%d missing=%d "
                "(correlation_id=%s)",
                len(diff.desired_topics),
                len(diff.present_topics),
                len(diff.missing_topics),
                correlation_id,
            )

            # (a)/(b) Resolve every missing spec's replication factor BEFORE the
            # first CreateTopics. A floor violation raises out of this method —
            # it is deliberately outside the best-effort boundary below.
            resolved_specs = self._resolve_specs_for_creation(
                missing_specs, correlation_id
            )

            for spec in resolved_specs:
                try:
                    # One spec object drives BOTH the NewTopic and the record,
                    # exactly as the single-topic path does. Building the
                    # NewTopic from the clamped partition count while recording
                    # the uncapped `spec` made `_created_specs` describe a topic
                    # that was never created: the readiness gate then compared
                    # broker-actual (1) against the contract-declared count (6)
                    # and NEVER converged, so on any lane with
                    # ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS set, every node whose
                    # topics were created in this boot stayed NOT-READY for the
                    # life of the process and only attached after an unrelated
                    # restart made its topics pre-existing (OMN-16844).
                    creation_spec = self._creation_spec(spec)
                    new_topic = NewTopic(
                        name=creation_spec.suffix,
                        num_partitions=creation_spec.partitions,
                        replication_factor=creation_spec.replication_factor,
                        topic_configs=dict(creation_spec.kafka_config)
                        if creation_spec.kafka_config
                        else {},
                    )

                    await admin.create_topics([new_topic])
                    created.append(creation_spec.suffix)
                    self._note_topic_created(creation_spec.suffix, creation_spec)
                    logger.info(
                        "Created topic: %s (partitions=%d, replication_factor=%s)",
                        creation_spec.suffix,
                        creation_spec.partitions,
                        creation_spec.replication_factor,
                        extra={"correlation_id": str(correlation_id)},
                    )

                except TopicAlreadyExistsError:
                    existing.append(spec.suffix)
                    self._note_topic_created(spec.suffix)
                    logger.debug(
                        "Topic already exists: %s",
                        spec.suffix,
                        extra={"correlation_id": str(correlation_id)},
                    )

                except Exception as e:
                    # Boundary: an unhostable replica count is re-raised
                    # fail-closed; everything else degrades to a warning.
                    if is_invalid_replication_factor_error(e):
                        # (D5) A replica count the broker cannot host is a
                        # durability failure, not a best-effort miss. It leaves
                        # the topic ABSENT, so degrading it to a warning +
                        # status="partial" is the silent-uncreated-topic bug the
                        # capacity measurement was supposed to make impossible.
                        raise unhostable_replication_error(
                            topic=spec.suffix,
                            requested_replication_factor=spec.replication_factor,
                            policy=self._policy,
                            cause=e,
                        ) from e
                    failed.append(spec.suffix)
                    logger.warning(
                        "Failed to create topic %s: %s",
                        spec.suffix,
                        type(e).__name__,
                        extra={
                            "correlation_id": str(correlation_id),
                            "error": sanitize_error_message(e),
                        },
                    )

        except TopicReplicationPolicyError:
            # Durability violations are fail-closed: never degraded to a warning.
            raise

        except Exception as e:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "Topic auto-creation interrupted by %s. "
                "Topics may need to be created manually or via broker auto-create.",
                type(e).__name__,
                extra={
                    "bootstrap_servers": self._bootstrap_servers,
                    "correlation_id": str(correlation_id),
                    "error": sanitize_error_message(e),
                },
            )
            # Separate individually-failed topics from those never attempted
            already_resolved = set(created) | set(existing) | set(failed)
            all_suffixes = {spec.suffix for spec in self._topic_specs}
            not_attempted = [s for s in all_suffixes if s not in already_resolved]
            if not_attempted:
                logger.warning(
                    "Topics not attempted due to early termination: %d topics",
                    len(not_attempted),
                    extra={
                        "not_attempted_count": len(not_attempted),
                        "correlation_id": str(correlation_id),
                    },
                )
            # Use "partial" if any topics succeeded before the interruption;
            # "unavailable" only when nothing was resolved at all.
            interrupted_status = "partial" if (created or existing) else "unavailable"
            return {
                "created": created,
                "existing": existing,
                "failed": failed + not_attempted,
                "drift": drift,
                "status": interrupted_status,
            }

        finally:
            if admin is not None:
                try:
                    await admin.close()
                except Exception:  # noqa: BLE001 — boundary: catch-all for resilience
                    pass  # Best-effort cleanup

        status = (
            "success"
            if not failed
            else ("partial" if created or existing else "unavailable")
        )

        logger.info(
            "Topic auto-creation complete",
            extra={
                "created_count": len(created),
                "existing_count": len(existing),
                "failed_count": len(failed),
                "drift_count": len(drift),
                "status": status,
                "correlation_id": str(correlation_id),
            },
        )

        return {
            "created": created,
            "existing": existing,
            "failed": failed,
            "drift": drift,
            "status": status,
        }

    async def ensure_topic_exists(
        self,
        topic_name: str,
        config: ModelSnapshotTopicConfig | None = None,
        correlation_id: UUID | None = None,
        *,
        spec: ModelTopicSpec | None = None,
    ) -> bool:
        """Ensure a single topic exists with optional custom config.

        Creates a new AIOKafkaAdminClient connection per call. For creating
        multiple topics, prefer :meth:`ensure_provisioned_topics_exist` which
        reuses a single admin connection for all topics.

        Spec resolution (OMN-15395 c): when the caller supplies neither *config*
        nor *spec*, the topic's OWN contract-derived spec is looked up from this
        provisioner's registry. The per-contract boot interleave calls this
        method with a bare topic name, and it used to land on a hardcoded RF1 —
        that bare-default branch no longer exists.

        Args:
            topic_name: The topic name to create.
            config: Optional snapshot-topic configuration (compaction etc.). If
                None, falls back to *spec*, then to the contract-derived spec.
            correlation_id: Optional correlation ID for tracing.
            spec: Optional contract-derived ``ModelTopicSpec`` (partitions,
                replication, kafka_config). Used by the per-contract boot
                interleave (OMN-13237) so a topic is created to its
                contract-declared spec rather than bare defaults. Ignored when
                *config* is supplied.

        Returns:
            True only after broker metadata confirms the topic is ready; False
            when creation, an already-exists race, or readiness confirmation
            cannot establish that proof.

        Raises:
            TopicReplicationPolicyError: The resolved spec violates the
                environment replication policy (raised before
                ``CreateTopics``), or the broker refused the ``CreateTopics``
                with ``INVALID_REPLICATION_FACTOR`` (OMN-15395 D5).
        """
        correlation_id = correlation_id or uuid4()

        try:
            from aiokafka.admin import AIOKafkaAdminClient, NewTopic
            from aiokafka.errors import (
                TopicAlreadyExistsError as _TopicAlreadyExistsError,
            )
        except ImportError:
            logger.warning(
                "aiokafka not available, cannot create topic %s",
                topic_name,
                extra={"correlation_id": str(correlation_id)},
            )
            return False

        # Bind to local after successful import block
        TopicAlreadyExistsError = _TopicAlreadyExistsError

        admin: AIOKafkaAdminClient | None = None
        # Recorded so a broker INVALID_REPLICATION_FACTOR rejection can name the
        # value it refused (OMN-15395 D5).
        requested_replication_factor: int | None = None
        try:
            auth_kwargs = build_aiokafka_auth_kwargs_from_env()
            admin = AIOKafkaAdminClient(
                bootstrap_servers=self._bootstrap_servers,
                request_timeout_ms=self._request_timeout_ms,
                **auth_kwargs,
            )
            await admin.start()

            # Same measured-capacity binding as the full pass: the ceiling is
            # read off this cluster, never inferred (OMN-15395).
            await self._measured_policy(admin)

            # (d) Skip creation entirely when the broker already has the topic.
            # The snapshot is fetched once per provisioner instance, so the
            # per-contract boot interleave costs ONE metadata request instead of
            # one blind CreateTopics authorization per contract topic.
            if topic_name in await self._existing_topic_names(admin):
                logger.debug(
                    "Topic already exists (broker snapshot), skipping create: %s",
                    topic_name,
                    extra={"correlation_id": str(correlation_id)},
                )
                return True

            created_spec: ModelTopicSpec | None = None
            if config is not None:
                # Snapshot-topic config carries its own replication factor; it
                # is still resolved through the SAME policy as every other
                # creation site, and it is the RESOLVER'S OUTPUT that reaches
                # NewTopic — not the raw declared value. "There is one resolver"
                # only holds if every creation site uses what it returns.
                resolved_rf = self._policy.resolve_replication_factor(
                    topic=topic_name, declared=config.replication_factor
                )
                requested_replication_factor = resolved_rf
                new_topic = NewTopic(
                    name=topic_name,
                    num_partitions=config.partition_count,
                    replication_factor=resolved_rf,
                    topic_configs=config.to_kafka_config(),
                )
                created_spec = ModelTopicSpec(
                    suffix=topic_name,
                    partitions=config.partition_count,
                    replication_factor=resolved_rf,
                    kafka_config=config.to_kafka_config(),
                )
            else:
                # (c) Caller-supplied spec wins; otherwise use the topic's own
                # contract-derived spec. Only a topic this provisioner knows
                # nothing about falls back to a bare spec, whose undeclared
                # replication factor the policy resolves or refuses.
                effective_spec = (
                    spec
                    if spec is not None
                    else self._spec_by_name.get(
                        topic_name, ModelTopicSpec(suffix=topic_name)
                    )
                )
                # Record the exact effective spec handed to the broker. The
                # dev/stability partition cap is part of creation semantics;
                # retaining the uncapped spec here makes the subsequent
                # readiness check expect (for example) 6 partitions after we
                # deliberately created 1, so a cold-start consumer can never
                # attach until the process restarts and treats the topic as
                # pre-existing (OMN-15978 live finding).
                resolved = self._creation_spec(effective_spec)
                created_spec = resolved
                requested_replication_factor = resolved.replication_factor
                new_topic = NewTopic(
                    name=topic_name,
                    num_partitions=resolved.partitions,
                    replication_factor=resolved.replication_factor,
                    topic_configs=dict(resolved.kafka_config)
                    if resolved.kafka_config
                    else {},
                )

            await admin.create_topics([new_topic])
            readiness = await self.confirm_topics_ready(
                [topic_name],
                expected_specs={topic_name: created_spec}
                if created_spec is not None
                else None,
                correlation_id=correlation_id,
            )
            if readiness.is_ready:
                self._note_topic_created(topic_name, created_spec)
                logger.info(
                    "Created topic: %s",
                    topic_name,
                    extra={"correlation_id": str(correlation_id)},
                )
                return True
            logger.warning(
                "Topic create did not materialize in broker metadata: %s (%s)",
                topic_name,
                readiness.status.value,
                extra={"correlation_id": str(correlation_id)},
            )
            return False

        except TopicAlreadyExistsError:
            readiness = await self.confirm_topics_ready(
                [topic_name],
                expected_specs={topic_name: created_spec}
                if created_spec is not None
                else None,
                correlation_id=correlation_id,
            )
            if readiness.is_ready:
                self._note_topic_created(topic_name)
                logger.debug(
                    "Topic already exists: %s",
                    topic_name,
                    extra={"correlation_id": str(correlation_id)},
                )
                return True
            logger.warning(
                "Topic already-exists race did not materialize in broker metadata: %s (%s)",
                topic_name,
                readiness.status.value,
                extra={"correlation_id": str(correlation_id)},
            )
            return False

        except TopicReplicationPolicyError:
            # Durability violations are fail-closed: never degraded to a
            # warning-and-False, which the caller would read as "best effort".
            raise

        except Exception as e:
            # Boundary: an unhostable replica count is re-raised fail-closed;
            # everything else degrades to a warning-and-False.
            if is_invalid_replication_factor_error(e):
                # (D5) Same rule as the batch path: a broker-rejected replica
                # count is fail-closed, never a warning-and-False the caller
                # reads as "best effort".
                raise unhostable_replication_error(
                    topic=topic_name,
                    requested_replication_factor=requested_replication_factor,
                    policy=self._policy,
                    cause=e,
                ) from e
            logger.warning(
                "Failed to create topic %s: %s",
                topic_name,
                type(e).__name__,
                extra={
                    "correlation_id": str(correlation_id),
                    "error": sanitize_error_message(e),
                },
            )
            return False

        finally:
            if admin is not None:
                try:
                    await admin.close()
                except Exception:  # noqa: BLE001 — boundary: catch-all for resilience
                    pass

    async def confirm_topics_ready(
        self,
        topics: Sequence[str],
        *,
        expected_specs: Mapping[str, ModelTopicSpec] | None = None,
        config: ModelTopicReadinessConfig | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelTopicSetReadiness:
        """Confirm broker metadata for ``topics`` converged (§3.7, OMN-13237).

        A topic is READY when broker metadata returns it, its partition count
        matches the expected spec, every partition has a leader, the reported
        replication factor matches the spec (where inspectable), and required
        config keys are visible. The poll is bounded by *config*'s timeout /
        cadence / max-attempts; on exhaustion each unready topic carries a
        classified failure reason.

        Spec pass-through (OMN-15395 c): when the caller supplies no
        *expected_specs*, the RESOLVED specs of the topics this provisioner
        actually created in this process are used, so a freshly created topic is
        verified against the partitions/replication it was created with. Specs
        are deliberately NOT injected for pre-existing topics: on a cluster
        carrying the 519 legacy RF1 topics, asserting the contract's RF against
        them would flip healthy topics to NOT-READY and block consumer attach.
        Drift on pre-existing topics is reported by
        :meth:`ensure_provisioned_topics_exist` instead (OMN-15395 d) and
        repaired by the operator-gated WS-M reassignment lane.

        Args:
            topics: The topic names to confirm.
            expected_specs: Optional per-topic expected spec (partitions/RF/
                kafka_config). Topics without a spec use default expectations.
            config: Bounded readiness knobs. Defaults to env-resolved knobs.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            A ``ModelTopicSetReadiness`` describing per-topic outcomes.
        """
        correlation_id = correlation_id or uuid4()
        requested = tuple(dict.fromkeys(topics))
        if not requested:
            return ModelTopicSetReadiness(status=EnumTopicReadinessStatus.SKIPPED)
        knobs = config or ModelTopicReadinessConfig()
        specs = (
            dict(expected_specs)
            if expected_specs is not None
            else {
                name: spec
                for name in requested
                if (spec := self._created_specs.get(name)) is not None
            }
        )

        try:
            from aiokafka.admin import AIOKafkaAdminClient
        except ImportError:
            logger.warning(
                "aiokafka not available, cannot confirm topic readiness",
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelTopicSetReadiness(
                topics=requested,
                status=EnumTopicReadinessStatus.UNAVAILABLE,
            )

        deadline = time.monotonic() + knobs.readiness_timeout_seconds
        poll_seconds = knobs.readiness_poll_interval_ms / 1000.0
        admin: AIOKafkaAdminClient | None = None
        last_evaluation: ModelTopicSetReadiness | None = None
        attempts = 0
        try:
            auth_kwargs = build_aiokafka_auth_kwargs_from_env()
            admin = AIOKafkaAdminClient(
                bootstrap_servers=self._bootstrap_servers,
                request_timeout_ms=self._request_timeout_ms,
                **auth_kwargs,
            )
            await admin.start()

            while attempts < knobs.max_attempts:
                attempts += 1
                metadata = await admin.describe_topics(list(requested))
                last_evaluation = evaluate_topic_readiness(
                    requested,
                    metadata,
                    expected_specs=specs,
                    attempts=attempts,
                )
                if last_evaluation.is_ready:
                    return last_evaluation
                if time.monotonic() >= deadline:
                    break
                await asyncio.sleep(poll_seconds)

        except Exception as e:  # noqa: BLE001 — boundary: degrades to not-ready
            logger.warning(
                "Topic readiness confirm interrupted by %s",
                type(e).__name__,
                extra={
                    "correlation_id": str(correlation_id),
                    "error": sanitize_error_message(e),
                },
            )
            return ModelTopicSetReadiness(
                topics=requested,
                status=EnumTopicReadinessStatus.UNAVAILABLE,
                attempts=attempts,
            )
        finally:
            if admin is not None:
                try:
                    await admin.close()
                except Exception:  # noqa: BLE001 — boundary: catch-all for resilience
                    pass

        if last_evaluation is not None:
            return last_evaluation
        # max_attempts must be >=1, so a loop that ran at least once always sets
        # last_evaluation; this guards the (unreachable) zero-iteration case.
        return ModelTopicSetReadiness(
            topics=requested,
            status=EnumTopicReadinessStatus.NOT_READY,
            attempts=attempts,
        )


def evaluate_topic_readiness(
    topics: Sequence[str],
    metadata: Sequence[Mapping[str, object]],
    *,
    expected_specs: Mapping[str, ModelTopicSpec] | None = None,
    attempts: int = 1,
) -> ModelTopicSetReadiness:
    """Classify broker metadata into a per-topic readiness outcome (§3.7).

    Pure function over the metadata shape returned by
    ``AIOKafkaAdminClient.describe_topics`` so the readiness semantics are
    unit-testable without a live broker. Each metadata entry is a mapping with
    keys ``topic``, ``error_code``, and ``partitions`` (a sequence of mappings
    with ``partition``, ``leader``, and ``replicas``).
    """
    requested = tuple(dict.fromkeys(topics))
    specs = dict(expected_specs or {})
    by_topic: dict[str, Mapping[str, object]] = {}
    for entry in metadata:
        name = entry.get("topic")
        if isinstance(name, str):
            by_topic[name] = entry

    ready: list[str] = []
    failures: list[ModelTopicReadinessFailure] = []
    for topic in requested:
        spec = specs.get(topic)
        topic_entry: Mapping[str, object] | None = by_topic.get(topic)
        if topic_entry is None:
            failures.append(
                ModelTopicReadinessFailure(
                    topic=topic,
                    reason=EnumTopicReadinessFailureReason.TOPIC_ABSENT,
                    detail="broker metadata did not return the topic",
                )
            )
            continue
        error_code = topic_entry.get("error_code")
        if isinstance(error_code, int) and error_code != 0:
            failures.append(
                ModelTopicReadinessFailure(
                    topic=topic,
                    reason=EnumTopicReadinessFailureReason.TOPIC_ABSENT,
                    detail=f"broker reported error_code={error_code}",
                )
            )
            continue
        partitions_raw = topic_entry.get("partitions")
        partitions: list[Mapping[str, object]] = (
            [p for p in partitions_raw if isinstance(p, Mapping)]
            if isinstance(partitions_raw, Sequence)
            and not isinstance(partitions_raw, (str, bytes))
            else []
        )
        if not partitions:
            failures.append(
                ModelTopicReadinessFailure(
                    topic=topic,
                    reason=EnumTopicReadinessFailureReason.PARTITION_MISMATCH,
                    detail="topic metadata reported zero partitions",
                )
            )
            continue
        if spec is not None and len(partitions) != spec.partitions:
            failures.append(
                ModelTopicReadinessFailure(
                    topic=topic,
                    reason=EnumTopicReadinessFailureReason.PARTITION_MISMATCH,
                    detail=(
                        f"expected {spec.partitions} partitions, "
                        f"broker reports {len(partitions)}"
                    ),
                )
            )
            continue
        no_leader = any(_partition_leader(p) is None for p in partitions)
        if no_leader:
            failures.append(
                ModelTopicReadinessFailure(
                    topic=topic,
                    reason=EnumTopicReadinessFailureReason.NO_LEADER,
                    detail="at least one partition has no available leader",
                )
            )
            continue
        if spec is not None:
            rf_mismatch = _replication_mismatch(partitions, spec.replication_factor)
            if rf_mismatch is not None:
                failures.append(
                    ModelTopicReadinessFailure(
                        topic=topic,
                        reason=(EnumTopicReadinessFailureReason.REPLICATION_MISMATCH),
                        detail=rf_mismatch,
                    )
                )
                continue
        ready.append(topic)

    status = (
        EnumTopicReadinessStatus.READY
        if not failures
        else EnumTopicReadinessStatus.NOT_READY
    )
    return ModelTopicSetReadiness(
        topics=requested,
        status=status,
        ready_topics=tuple(ready),
        failures=tuple(failures),
        attempts=attempts,
    )


def _partition_leader(partition: Mapping[str, object]) -> int | None:
    """Return the partition leader id, or None when no valid leader exists."""
    leader = partition.get("leader")
    if isinstance(leader, int) and leader >= 0:
        return leader
    return None


def _replication_mismatch(
    partitions: Sequence[Mapping[str, object]],
    expected_rf: int | None,
) -> str | None:
    """Return a detail string when replica counts disagree with the spec.

    Skipped (returns None) where the broker does not expose a replica list, or
    where the owning contract declared no replication factor (OMN-15395: an
    undeclared RF is not an expectation to assert against).
    """
    if expected_rf is None:
        return None
    for partition in partitions:
        replicas = partition.get("replicas")
        if not (
            isinstance(replicas, Sequence) and not isinstance(replicas, (str, bytes))
        ):
            return None  # RF not inspectable from this metadata shape
        if len(replicas) != expected_rf:
            return (
                f"expected replication_factor={expected_rf}, "
                f"broker reports {len(replicas)} replicas"
            )
    return None


def _cli_main() -> None:
    """CLI entrypoint for manual topic provisioning without runtime.

    Usage:
        uv run python -m omnibase_infra.event_bus.service_topic_manager \\
            --contracts-root src/omnibase_infra/nodes

    Useful for provisioning topics when running just Redpanda for development
    without the full runtime stack.
    """
    import argparse
    import asyncio
    import json

    parser = argparse.ArgumentParser(
        description="Provision Kafka topics from contract YAML."
    )
    parser.add_argument(
        "--contracts-root",
        type=Path,
        default=Path(os.environ.get("ONEX_CONTRACTS_DIR", "./contracts")),
        help=(
            "Root directory containing contract.yaml files. "
            "Defaults to ONEX_CONTRACTS_DIR env var or ./contracts."
        ),
    )
    args = parser.parse_args()

    async def _run() -> None:
        provisioner = TopicProvisioner(contracts_root=args.contracts_root)
        result = await provisioner.ensure_provisioned_topics_exist()
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


if __name__ == "__main__":
    _cli_main()


__all__ = [
    "TopicProvisioner",
    "evaluate_topic_readiness",
    "topic_partition_cap_from_env",
    "unhostable_replication_error",
]
