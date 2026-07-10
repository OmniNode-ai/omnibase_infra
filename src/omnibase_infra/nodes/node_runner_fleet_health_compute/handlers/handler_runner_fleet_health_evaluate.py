# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that classifies a runner-fleet snapshot into a health verdict (OMN-13942).

This is a COMPUTE handler -- pure, deterministic, no I/O. It performs NO
probing; every fact it classifies was gathered upstream by
``node_runner_health_snapshot_effect``.

Classification precedence (most severe first) ports:
  - ``collector_runner_health.py::_classify_runner`` (CRASH_LOOPING via
    RestartCount, OFFLINE_IDLE via GitHub status)
  - OMN-13915's ``_diag`` heartbeat-freshness rule (LISTENER_ZOMBIE)
  - ``runner-monitor.sh``'s SILENT-WEDGE (OMN-13109) and CRASHLOOP_RESTART_
    THRESHOLD heuristics (WEDGED)
  - NEW (OMN-13932): BUILDX_UNAVAILABLE, CODELOAD_THROTTLED
  - NEW: SATURATED (the 2026-07-04 zero-idle incident this ticket responds to)

OMN-14228 Slice A precondition fix: this handler used to classify every
runner as if the upstream GitHub/Docker sources always succeeded, silently
defaulting a failed source's facts (e.g. ``docker_restart_count=0`` when the
SSH probe failed entirely) into a confident classification -- fail OPEN. A
docker-source outage meant CRASH_LOOPING/LISTENER_ZOMBIE (the two
highest-confidence recommended actions) could never fire even when the
runner really was crash-looping, because the missing fact silently read as
"no restarts." This handler now threads ``github_source_ok``/
``docker_source_ok`` onto every assessment and the verdict, and preserves the
buildx probe's tri-state (unknown vs. confirmed-available) instead of
collapsing ``None`` into ``False``, so a downstream remediation gate can fail
CLOSED on indeterminate health instead of treating a source outage as
verified-healthy. No gate/executor logic is added here -- it is precondition
data only.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_recommended_action_type import (
    EnumRecommendedActionType,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_fleet_health_state import (
    EnumRunnerFleetHealthState,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_recommended_action import (
    ModelRecommendedAction,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_verdict import (
    ModelRunnerFleetHealthVerdict,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_health_assessment import (
    ModelRunnerHealthAssessment,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_runner_fact import (
    ModelRunnerFleetRunnerFact,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot import (
    ModelRunnerFleetSnapshot,
)

logger = logging.getLogger(__name__)

# Same env-overridable defaults as the EFFECT + the legacy bash surfaces
# (runner-monitor.sh, healthcheck.sh) so all three surfaces agree on
# thresholds during the trust-building period (OMN-13109/OMN-13912/OMN-13915).
_CRASHLOOP_RESTART_THRESHOLD = int(os.environ.get("CRASHLOOP_RESTART_THRESHOLD", "5"))
_RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS = int(
    os.environ.get("RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS", "900")
)
_WEDGE_QUEUE_AGE_SECONDS = int(os.environ.get("WEDGE_QUEUE_AGE_SECONDS", "600"))

_EnumState = EnumRunnerFleetHealthState


def _classify_runner(
    fact: ModelRunnerFleetRunnerFact,
    *,
    fleet_wedged: bool,
    fleet_saturated: bool,
    buildx_available: bool | None,
    codeload_throttled: bool,
) -> tuple[_EnumState, str]:
    """Classify a single runner. Returns (state, detail). Pure, no I/O."""
    if fact.docker_restart_count > _CRASHLOOP_RESTART_THRESHOLD:
        return (
            _EnumState.CRASH_LOOPING,
            f"RestartCount={fact.docker_restart_count} > threshold={_CRASHLOOP_RESTART_THRESHOLD}",
        )
    if (
        fact.diag_heartbeat_age_seconds is not None
        and fact.diag_heartbeat_age_seconds > _RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS
    ):
        return (
            _EnumState.LISTENER_ZOMBIE,
            (
                f"_diag heartbeat age={fact.diag_heartbeat_age_seconds:.0f}s > "
                f"threshold={_RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS}s"
            ),
        )
    if fact.github_status == "offline":
        return _EnumState.OFFLINE_IDLE, "GitHub API reports runner offline"
    if buildx_available is False:
        return _EnumState.BUILDX_UNAVAILABLE, "docker buildx unavailable on runner host"
    if codeload_throttled:
        return (
            _EnumState.CODELOAD_THROTTLED,
            "recent failed runs match codeload.github.com throttle signatures",
        )
    if fleet_wedged and not fact.github_busy:
        return (
            _EnumState.WEDGED,
            f"fleet-wide: queued job age >= {_WEDGE_QUEUE_AGE_SECONDS}s with zero busy runners",
        )
    if fleet_saturated and fact.github_busy:
        return (
            _EnumState.SATURATED,
            "fleet-wide: zero idle runners (saturation_ratio >= 1.0)",
        )
    return _EnumState.HEALTHY, ""


def _annotate_indeterminate(
    detail: str, *, github_source_ok: bool, docker_source_ok: bool
) -> str:
    """Append an honest indeterminacy note when a classification source failed.

    Pure string annotation -- does NOT change ``state``. A future remediation
    gate reads ``ModelRunnerHealthAssessment.is_determinate`` to decide
    ALLOW/SUPPRESS; this note keeps the human-readable ``detail`` from
    silently implying a source-outage classification was verified.
    """
    failed = []
    if not github_source_ok:
        failed.append("github_source_ok=False")
    if not docker_source_ok:
        failed.append("docker_source_ok=False")
    if not failed:
        return detail
    note = f"INDETERMINATE ({', '.join(failed)}): classification unreliable"
    return f"{detail}; {note}" if detail else note


def _recommend_for_assessment(
    assessment: ModelRunnerHealthAssessment,
) -> ModelRecommendedAction | None:
    """Map a per-runner assessment to a recommended (never-executed) action."""
    if assessment.state == _EnumState.CRASH_LOOPING:
        return ModelRecommendedAction(
            action_type=EnumRecommendedActionType.RESTART_RUNNER,
            target_id=assessment.name,
            reason=assessment.detail,
            confidence=0.9,
        )
    if assessment.state == _EnumState.LISTENER_ZOMBIE:
        return ModelRecommendedAction(
            action_type=EnumRecommendedActionType.RESTART_RUNNER,
            target_id=assessment.name,
            reason=assessment.detail,
            confidence=0.85,
        )
    if assessment.state == _EnumState.OFFLINE_IDLE:
        return ModelRecommendedAction(
            action_type=EnumRecommendedActionType.RESTART_RUNNER,
            target_id=assessment.name,
            reason=assessment.detail,
            confidence=0.6,
        )
    if assessment.state == _EnumState.WEDGED:
        return ModelRecommendedAction(
            action_type=EnumRecommendedActionType.RESTART_RUNNER,
            target_id=assessment.name,
            reason=assessment.detail,
            confidence=0.5,
        )
    if assessment.state == _EnumState.SATURATED:
        return ModelRecommendedAction(
            action_type=EnumRecommendedActionType.NONE,
            target_id=assessment.name,
            reason="fleet saturated (0 idle) -- requires capacity, not a per-runner action",
            confidence=0.0,
        )
    if assessment.state == _EnumState.BUILDX_UNAVAILABLE:
        return ModelRecommendedAction(
            action_type=EnumRecommendedActionType.NONE,
            target_id=assessment.name,
            reason="buildx unavailable on host -- needs host/image re-provisioning",
            confidence=0.0,
        )
    if assessment.state == _EnumState.CODELOAD_THROTTLED:
        return ModelRecommendedAction(
            action_type=EnumRecommendedActionType.NONE,
            target_id=assessment.name,
            reason="transient GitHub-side codeload throttling -- no fleet-side fix",
            confidence=0.0,
        )
    return None


class HandlerRunnerFleetHealthEvaluate:
    """Classifies a runner-fleet snapshot into a typed health verdict.

    Pure and deterministic: identical input always produces identical
    output, and no I/O happens anywhere in this class.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, correlation_id: UUID, snapshot: ModelRunnerFleetSnapshot
    ) -> ModelRunnerFleetHealthVerdict:
        """Classify ``snapshot`` into a ``ModelRunnerFleetHealthVerdict``.

        Args:
            correlation_id: Workflow correlation ID.
            snapshot: Facts-only snapshot gathered by the EFFECT node.

        Returns:
            ModelRunnerFleetHealthVerdict with per-runner states, fleet
            aggregates, and recorded (never executed) recommended actions.
        """
        online_count = sum(1 for r in snapshot.runners if r.github_status == "online")
        offline_count = sum(1 for r in snapshot.runners if r.github_status != "online")
        busy_count = sum(
            1 for r in snapshot.runners if r.github_status == "online" and r.github_busy
        )
        idle_count = online_count - busy_count
        saturation_ratio = (busy_count / online_count) if online_count else 0.0
        codeload_throttled = snapshot.codeload_throttle_signal_count > 0
        buildx_unavailable = snapshot.buildx_available is False
        # Preserve the tri-state instead of collapsing None -> False: None
        # means the probe could not determine availability, which must not
        # read the same as "confirmed available."
        buildx_determinate = snapshot.buildx_available is not None
        # Source failure is fleet-wide today (one `gh api` call, one `ssh`
        # call cover every runner) -- every assessment gets the same
        # determinacy value. The field is per-runner so Slice B/C can narrow
        # this once per-runner probes exist without another model change.
        is_determinate = snapshot.github_source_ok and snapshot.docker_source_ok

        fleet_wedged = (
            snapshot.oldest_queued_job_age_seconds is not None
            and snapshot.oldest_queued_job_age_seconds >= _WEDGE_QUEUE_AGE_SECONDS
            and busy_count == 0
            and online_count > 0
        )
        fleet_saturated = online_count > 0 and idle_count == 0

        assessments: list[ModelRunnerHealthAssessment] = []
        recommended_actions: list[ModelRecommendedAction] = []
        crash_looping_count = 0
        listener_zombie_count = 0
        wedged_count = 0

        for fact in snapshot.runners:
            state, detail = _classify_runner(
                fact,
                fleet_wedged=fleet_wedged,
                fleet_saturated=fleet_saturated,
                buildx_available=snapshot.buildx_available,
                codeload_throttled=codeload_throttled,
            )
            assessment = ModelRunnerHealthAssessment(
                name=fact.name,
                state=state,
                detail=_annotate_indeterminate(
                    detail,
                    github_source_ok=snapshot.github_source_ok,
                    docker_source_ok=snapshot.docker_source_ok,
                ),
                is_determinate=is_determinate,
                docker_restart_count=fact.docker_restart_count,
                diag_heartbeat_age_seconds=fact.diag_heartbeat_age_seconds,
            )
            assessments.append(assessment)
            if state == _EnumState.CRASH_LOOPING:
                crash_looping_count += 1
            elif state == _EnumState.LISTENER_ZOMBIE:
                listener_zombie_count += 1
            elif state == _EnumState.WEDGED:
                wedged_count += 1
            action = _recommend_for_assessment(assessment)
            if action is not None:
                recommended_actions.append(action)

        for candidate in snapshot.zombie_run_candidates:
            recommended_actions.append(
                ModelRecommendedAction(
                    action_type=EnumRecommendedActionType.CANCEL_RUN,
                    target_id=str(candidate.run_id),
                    reason=(
                        f"run {candidate.status} for {candidate.age_seconds:.0f}s in "
                        f"{candidate.repo}, exceeds wedge threshold "
                        f"({_WEDGE_QUEUE_AGE_SECONDS}s)"
                    ),
                    confidence=0.4,
                )
            )

        verdict = ModelRunnerFleetHealthVerdict(
            correlation_id=correlation_id,
            evaluated_at=datetime.now(tz=UTC),
            assessments=tuple(assessments),
            expected_count=snapshot.expected_count,
            observed_count=len(snapshot.runners),
            online_count=online_count,
            offline_count=offline_count,
            busy_count=busy_count,
            idle_count=idle_count,
            saturation_ratio=saturation_ratio,
            crash_looping_count=crash_looping_count,
            listener_zombie_count=listener_zombie_count,
            wedged_count=wedged_count,
            buildx_unavailable=buildx_unavailable,
            buildx_determinate=buildx_determinate,
            codeload_throttle_signal_count=snapshot.codeload_throttle_signal_count,
            recommended_actions=tuple(recommended_actions),
            source_errors=snapshot.source_errors,
            github_source_ok=snapshot.github_source_ok,
            docker_source_ok=snapshot.docker_source_ok,
        )
        logger.info(
            "Runner-fleet health verdict: %d/%d online, saturation=%.2f, %d recommended "
            "actions (correlation_id=%s)",
            online_count,
            snapshot.expected_count,
            saturation_ratio,
            len(recommended_actions),
            correlation_id,
        )
        return verdict


__all__ = ["HandlerRunnerFleetHealthEvaluate"]
