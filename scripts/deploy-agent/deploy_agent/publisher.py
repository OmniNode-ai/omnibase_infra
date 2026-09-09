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

from kafka import KafkaProducer

from deploy_agent.events import (
    TOPIC_REBUILD_COMPLETED,
    ModelContainerResidue,
    ModelHealthCheck,
    ModelRebuildCompleted,
    Phase,
)
from deploy_agent.job_state import JobState
from deploy_agent.kafka_config import ModelDeployAgentKafkaConfig
from deploy_agent.tracking_ref import load_tracking_remote_ref_from_env

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
        phase_results=phase_results,
        errors=job.errors,
        health_checks=list(health_checks or []),
        container_residue=list(container_residue or []),
    )
    return completed.model_dump(mode="json")


def publish_result(
    payload: dict[str, Any], kafka_config: ModelDeployAgentKafkaConfig
) -> bool:
    """Publish completion event to the same control bus consumed by deploy-agent."""
    try:
        producer = KafkaProducer(
            **kafka_config.producer_kwargs(),
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda v: str(v).encode("utf-8"),
        )
        correlation_id = payload.get("correlation_id", "")
        producer.send(
            TOPIC_REBUILD_COMPLETED,
            key=f"deploy-result/{correlation_id}",
            value=payload,
        )
        producer.flush(timeout=30)
        producer.close()
        logger.info("Published result for %s", correlation_id)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Publish failed: %s", e)
        return False


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
