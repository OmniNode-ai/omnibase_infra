# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result publisher via rpk topic produce."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError

from deploy_agent.events import (
    TOPIC_DEV_LANE_DRIFT,
    TOPIC_DEV_TAG_HEARTBEAT,
    TOPIC_REBUILD_COMPLETED,
    ModelComposeInvocation,
    ModelContainerResidue,
    ModelDepsConvergenceFinding,
    ModelHealthCheck,
    ModelLaneMirrorReceipt,
    ModelOneShotRerun,
    ModelOnexApiDelivery,
    ModelRebuildCompleted,
    ModelRecreateSupervision,
    ModelVerifyRecreate,
    Phase,
)
from deploy_agent.job_state import JobState
from deploy_agent.kafka_config import ModelDeployAgentKafkaConfig
from deploy_agent.tracking_ref import load_tracking_remote_ref_from_env
from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.topics.broker_capacity_probe import bind_policy_to_broker_count
from omnibase_infra.topics.model_topic_provisioning_policy import (
    ModelTopicProvisioningPolicy,
)
from omnibase_infra.topics.topic_namespace import apply_topic_namespace

logger = logging.getLogger(__name__)

# Retry backoff: 30s, 60s, 120s, cap 300s. Give up after 10 min total.
RETRY_DELAYS = [30, 60, 120, 300, 300]
MAX_RETRY_TOTAL_SECONDS = 600

# Circuit breaker defaults — configurable via constructor.
DEFAULT_CB_MAX_CONSECUTIVE_FAILURES = 10
DEFAULT_CB_MAX_AGE_SECONDS = 3600.0  # 1 hour


@dataclass
class PublishCircuitBreaker:
    """Trips after too many consecutive failures or too much elapsed time on a single pending job.

    Once tripped, the caller must handle the stuck message (log CRITICAL, record
    friction, remove from pending) rather than retrying indefinitely.
    """

    max_consecutive_failures: int = DEFAULT_CB_MAX_CONSECUTIVE_FAILURES
    max_age_seconds: float = DEFAULT_CB_MAX_AGE_SECONDS

    # Per-correlation-id tracking: {correlation_id: (failure_count, first_failure_ts)}
    _state: dict[str, tuple[int, float]] = field(default_factory=dict)

    def record_failure(self, correlation_id: str) -> None:
        now = time.monotonic()
        count, first_ts = self._state.get(correlation_id, (0, now))
        if count == 0:
            first_ts = now
        self._state[correlation_id] = (count + 1, first_ts)

    def record_success(self, correlation_id: str) -> None:
        self._state.pop(correlation_id, None)

    def is_tripped(self, correlation_id: str) -> bool:
        entry = self._state.get(correlation_id)
        if entry is None:
            return False
        count, first_ts = entry
        elapsed = time.monotonic() - first_ts
        return count >= self.max_consecutive_failures or elapsed >= self.max_age_seconds

    def clear(self, correlation_id: str) -> None:
        self._state.pop(correlation_id, None)


def build_completion_payload(
    job: JobState,
    git_sha: str,
    health_checks: list[ModelHealthCheck] | None = None,
    *,
    services_restarted: list[str] | None = None,
    container_residue: list[ModelContainerResidue] | None = None,
    sibling_refs: dict[str, str] | None = None,
    recreate_supervision: list[ModelRecreateSupervision] | None = None,
    deps_convergence: list[ModelDepsConvergenceFinding] | None = None,
    compose_invocations: list[ModelComposeInvocation] | None = None,
    verify_recreate: list[ModelVerifyRecreate] | None = None,
    oneshot_reruns: list[ModelOneShotRerun] | None = None,
    onex_api_delivery: ModelOnexApiDelivery | None = None,
    lane_mirror: ModelLaneMirrorReceipt | None = None,
) -> dict[str, Any]:
    """Build the completion event payload from job state.

    OMN-18057. Three things changed here, each closing a way the terminal event
    could disagree with what happened:

    * The payload is VALIDATED through ``ModelRebuildCompleted`` instead of
      being hand-assembled as a dict that merely resembled it. The model was
      never on the publish path, so nothing enforced its shape and its
      ``status`` verdict never reached the wire at all -- every consumer had to
      re-derive a verdict from phase strings.
    * ``Phase.PUBLISH`` is stripped. This event IS the publish; it was always
      recorded as ``in_progress`` at the moment the payload was built, so a
      ``status`` derived from ``phase_results`` would have read "failed" for
      every deploy the agent ever completed.
    * ``services_restarted`` comes from what ``rebuild_scope`` actually brought
      up. It used to echo ``command["services"]``, which is EMPTY for a
      scope-default deploy -- so the field read as "nothing was restarted" on
      exactly the deploys that restarted everything.

    OMN-17135 adds ``sibling_refs``: repo -> the commit SHA RT-1 resolved for
    each sibling. ``requested_git_ref`` pins omnibase_infra only, so a reader
    of this event could not previously say which omnibase_core or omnimarket
    commit the image carried without going to the host and opening
    ``/app/build-provenance.json``.

    OMN-18640 adds ``deps_convergence`` and ``compose_invocations``. The first
    says, per core dependency, whether the deps leg was about to replace that
    container and why; the second records the argv of every compose command
    the deploy issued. Both existed nowhere durable before: a replaced
    dependency left only a container timestamp, and a deploy's flags were
    readable only by sampling the host process table while the child ran.

    OMN-18692 adds ``recreate_supervision``: what the deps-phase ceiling did.
    A deferral taken before the lane was touched, and a wait held past the
    ceiling rather than cancelling a live recreate, are both DECISIONS this
    agent made about a production-shaped mutation, and until now neither left
    any durable artifact. The 2026-09-18 kill had to be reconstructed from the
    dockerd journal by a third lane, because the terminal event recorded only
    that the phase failed.

    OMN-18572 adds ``onex_api_delivery``: what the onex-api pin delivery did
    on this job's tail. ``None`` says the delivery was never reached, which is
    a different fact from one that ran and refused -- and the two were
    indistinguishable while the event carried neither. It never affects
    ``status``: the compose lane's verdict is settled before the delivery runs,
    and a lane that converged is not broken because an image failed to reach
    it.

    OMN-18640 adds ``verify_recreate``: the runtime containers this deploy
    force-recreated because their own post-deploy health probe failed, and
    whether that repaired them. An empty list is the normal reading and is a
    positive fact -- verification found nothing to repair -- which is what
    makes the field's absence from an older event distinguishable from a
    deploy that repaired nothing.

    OMN-19911 adds ``oneshot_reruns``: the control-plane one-shots a runtime
    rebuild re-ran because their config hash moved, each with its exit code.
    """
    started_at = job.accepted_at
    completed_at = job.completed_at or datetime.now(UTC)
    duration = (completed_at - started_at).total_seconds()

    phase_results = {
        phase: status
        for phase, status in job.phase_results.items()
        if phase != Phase.PUBLISH
    }

    completed = ModelRebuildCompleted(
        correlation_id=job.correlation_id,
        # OMN-16442: the completion event records the ref this agent DECLARES
        # it tracks when the command omitted one -- never a literal "main".
        # This field is read back as the deployed lineage, so a wrong default
        # here misreports what a lane is running.
        requested_git_ref=job.command.get("git_ref")
        or load_tracking_remote_ref_from_env(),
        git_sha=git_sha,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=round(duration, 1),
        scope=job.command.get("scope", "runtime"),
        runtime_lane=job.command["runtime_lane"],
        image_ref=job.command.get("image_ref"),
        image_digest=job.command.get("image_digest"),
        services_restarted=list(
            services_restarted
            if services_restarted is not None
            else job.command.get("services", [])
        ),
        sibling_refs=dict(sibling_refs or {}),
        phase_results=phase_results,
        errors=job.errors,
        health_checks=list(health_checks or []),
        container_residue=list(container_residue or []),
        recreate_supervision=list(recreate_supervision or []),
        deps_convergence=list(deps_convergence or []),
        compose_invocations=list(compose_invocations or []),
        verify_recreate=list(verify_recreate or []),
        oneshot_reruns=list(oneshot_reruns or []),
        onex_api_delivery=onex_api_delivery,
        lane_mirror=lane_mirror,
    )
    return completed.model_dump(mode="json")


# OMN-17427: topics the agent alone publishes. No contract's event_bus declares them; until
# src/omnibase_infra/services/topics.yaml listed them nothing created them, and a lab deploy agent
# logged "Topic onex.evt.deploy.dev-tag-heartbeat.v1 not found in cluster metadata" on every
# heartbeat. A broker the provisioner has not run against still lacks them, so the agent creates
# each once per process before its first publish to it. A topic another node declares (the
# rebuild topics) is left to that node's contract.
AGENT_OWNED_TOPICS = frozenset({TOPIC_DEV_LANE_DRIFT, TOPIC_DEV_TAG_HEARTBEAT})
AGENT_TOPIC_PARTITIONS = 6
_ENSURED_TOPICS: set[str] = set()


def _broker_count(admin: KafkaAdminClient) -> int | None:
    """Live broker count from ``describe_cluster``, or None when it cannot be read."""
    brokers = admin.describe_cluster().get("brokers")
    return len(brokers) if isinstance(brokers, list) and brokers else None


def _ensure_topic(topic: str, kafka_config: ModelDeployAgentKafkaConfig) -> None:
    """Create ``topic`` when the broker lacks it, once per process; a failure is retried next time."""
    if topic in _ENSURED_TOPICS:
        return
    try:
        admin = KafkaAdminClient(**kafka_config.consumer_kwargs())
        try:
            # OMN-15395: the replication factor comes from the provisioning policy bound to
            # the measured broker count, never a literal. The agent only reaches SASL
            # PLAIN/SCRAM brokers, so the self-hosted profile applies.
            policy = bind_policy_to_broker_count(
                ModelTopicProvisioningPolicy.self_hosted(), _broker_count(admin)
            )
            replication_factor = policy.resolve_replication_factor(
                topic=topic, declared=None
            )
            admin.create_topics(
                [
                    NewTopic(
                        name=topic,
                        num_partitions=AGENT_TOPIC_PARTITIONS,
                        replication_factor=replication_factor,
                    )
                ]
            )
            logger.info("Created topic %s", topic)
        except TopicAlreadyExistsError:
            pass
        finally:
            admin.close()
    except (KafkaError, OSError, TopicReplicationPolicyError) as e:
        logger.warning("Could not ensure topic %s: %s", topic, e)
        return
    _ENSURED_TOPICS.add(topic)


def _produce(
    topic: str,
    key: str,
    payload: dict[str, Any],
    kafka_config: ModelDeployAgentKafkaConfig,
) -> bool:
    """Produce one JSON record on the agent's control bus; False when it failed."""
    if topic in AGENT_OWNED_TOPICS:
        _ensure_topic(apply_topic_namespace(topic), kafka_config)
    try:
        producer = KafkaProducer(
            **kafka_config.producer_kwargs(),
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda v: str(v).encode("utf-8"),
        )
        producer.send(apply_topic_namespace(topic), key=key, value=payload)
        producer.flush(timeout=30)
        producer.close()
        logger.info("Published %s", key)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Publish of %s failed: %s", key, e)
        return False


def publish_result(
    payload: dict[str, Any], kafka_config: ModelDeployAgentKafkaConfig
) -> bool:
    """Publish completion event to the same control bus consumed by deploy-agent."""
    correlation_id = payload.get("correlation_id", "")
    return _produce(
        TOPIC_REBUILD_COMPLETED,
        f"deploy-result/{correlation_id}",
        payload,
        kafka_config,
    )


def publish_dev_lane_drift(
    payload: dict[str, Any], kafka_config: ModelDeployAgentKafkaConfig
) -> bool:
    """Publish a dev-lane drift event (OMN-21146), keyed by instance and repository.

    One attempt, no retry: the agent publishes the lane's drift again on its
    next check while the lane stays overdue, so a lost event is replaced by a
    newer one rather than replayed.
    """
    key = (
        f"dev-lane-drift/{payload.get('instance', '')}/{payload.get('repository', '')}"
    )
    return _produce(TOPIC_DEV_LANE_DRIFT, key, payload, kafka_config)


def publish_dev_tag_heartbeat(
    payload: dict[str, Any], kafka_config: ModelDeployAgentKafkaConfig
) -> bool:
    """Publish a dev-tag heartbeat (OMN-19998), keyed by instance and repository.

    One attempt, no retry: the next heartbeat replaces a lost one.
    """
    key = (
        f"dev-tag-heartbeat/{payload.get('instance', '')}/"
        f"{payload.get('repository', '')}"
    )
    return _produce(TOPIC_DEV_TAG_HEARTBEAT, key, payload, kafka_config)


def publish_with_retry(
    payload: dict[str, Any], kafka_config: ModelDeployAgentKafkaConfig
) -> bool:
    """Attempt to publish with exponential backoff."""
    total_waited = 0
    for delay in RETRY_DELAYS:
        if publish_result(payload, kafka_config):
            return True
        if total_waited + delay > MAX_RETRY_TOTAL_SECONDS:
            logger.error(
                "Giving up publishing for %s after %ds",
                payload.get("correlation_id"),
                total_waited,
            )
            return False
        time.sleep(delay)
        total_waited += delay

    return publish_result(payload, kafka_config)
