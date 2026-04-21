# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result publisher via rpk topic produce."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from deploy_agent.events import (
    TOPIC_REBUILD_COMPLETED,
    ModelHealthCheck,
)
from deploy_agent.job_state import JobState

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
    job: JobState, git_sha: str, health_checks: list[ModelHealthCheck] | None = None
) -> dict:
    """Build the completion event payload from job state."""
    started_at = job.accepted_at
    completed_at = job.completed_at or datetime.now(UTC)
    duration = (completed_at - started_at).total_seconds()

    phase_results = {str(k): str(v) for k, v in job.phase_results.items()}
    checks = [c.model_dump() for c in (health_checks or [])]

    return {
        "correlation_id": str(job.correlation_id),
        "requested_git_ref": job.command.get("git_ref", "origin/main"),
        "git_sha": git_sha,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round(duration, 1),
        "scope": job.command.get("scope", "runtime"),
        "services_restarted": job.command.get("services", []),
        "phase_results": phase_results,
        "errors": job.errors,
        "health_checks": checks,
    }


def publish_result(payload: dict) -> bool:
    """Publish completion event via rpk inside the redpanda container. Returns True on success."""
    try:
        data = json.dumps(payload) + "\n"
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "omnibase-infra-redpanda",
                "rpk",
                "topic",
                "produce",
                TOPIC_REBUILD_COMPLETED,
            ],
            input=data,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Published result for %s", payload.get("correlation_id"))
            return True
        logger.warning("rpk produce failed: %s", result.stderr)
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning("Publish failed: %s", e)
        return False


def publish_with_retry(payload: dict) -> bool:
    """Attempt to publish with exponential backoff."""
    total_waited = 0
    for delay in RETRY_DELAYS:
        if publish_result(payload):
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

    return publish_result(payload)
