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

OMN-15255 (friction F-04) adds the composite readiness surface alongside the
precedence classification. Two things are true at once and must not be
conflated:

  - ``state`` answers "what is the single most severe thing wrong with this
    runner" by precedence. First match wins; later signals are not evaluated.
  - ``readiness`` answers "may this runner be routed governed work" by
    conjunction over six independently-probed signals. Every signal is
    evaluated every tick, including the passing ones.

They legitimately disagree. A GitHub-online runner with a fresh heartbeat, an
unhealthy container and two Runner.Listener processes is ``state=HEALTHY``
(neither container health nor listener topology is an input to the precedence
chain) and ``readiness=NOT_READY``. That gap is the F-04 finding: at one
closeout GitHub reported 64/64 online while 53 of 64 containers read
docker-unhealthy, and nothing in the system adjudicated.

The quarantine/bounce split fixes the other half -- the false-positive
restart storm. A stale-looking heartbeat alone can no longer produce a
restart recommendation: a bounce needs a determinate source, an idle runner,
and a failing signal a force-recreate can actually fix.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_readiness_signal_outcome import (
    EnumReadinessSignalOutcome,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_recommended_action_type import (
    EnumRecommendedActionType,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_fleet_health_state import (
    EnumRunnerFleetHealthState,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_readiness_signal import (
    EnumRunnerReadinessSignal,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_readiness_state import (
    EnumRunnerReadinessState,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_readiness_signal_rollup import (
    ModelReadinessSignalRollup,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_recommended_action import (
    ModelRecommendedAction,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_evaluate_command import (
    ModelRunnerFleetHealthEvaluateCommand,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_verdict import (
    ModelRunnerFleetHealthVerdict,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_health_assessment import (
    ModelRunnerHealthAssessment,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_readiness_signal import (
    ModelRunnerReadinessSignal,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_runner_fact import (
    ModelRunnerFleetRunnerFact,
)

logger = logging.getLogger(__name__)

# Same env-overridable defaults as the EFFECT + the legacy bash surfaces
# (runner-monitor.sh, healthcheck.sh) so all three surfaces agree on
# thresholds during the trust-building period (OMN-13109/OMN-13912/OMN-13915/OMN-15233).
_CRASHLOOP_RESTART_THRESHOLD = int(os.environ.get("CRASHLOOP_RESTART_THRESHOLD", "5"))
_RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS = int(
    os.environ.get("RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS", "4500")
)
_WEDGE_QUEUE_AGE_SECONDS = int(os.environ.get("WEDGE_QUEUE_AGE_SECONDS", "600"))
# OMN-15255: ceiling on runner-host disk usage. A host past this is out of
# space for checkouts/caches; every container on it is unfit for work even
# though each one still registers online and reports a fresh heartbeat.
_RUNNER_READINESS_MAX_DISK_USED_PERCENT = float(
    os.environ.get("RUNNER_READINESS_MAX_DISK_USED_PERCENT", "90")
)

_EnumState = EnumRunnerFleetHealthState
_EnumSignal = EnumRunnerReadinessSignal
_EnumOutcome = EnumReadinessSignalOutcome

# Docker health values that mean "the container itself is fit". ``none`` is a
# PASS, not a gap: it means the image declares no healthcheck, so this signal
# has nothing to assert and the other five carry the verdict.
_HEALTHY_DOCKER_HEALTH_VALUES = frozenset({"healthy", "none"})

# Signals a force-recreate can plausibly fix. DISK_CAPACITY is deliberately
# absent -- recreating a container frees no host disk, so bouncing on a full
# disk is pure churn. GITHUB_REGISTRATION is absent as a *standalone* trigger
# per OMN-14057 (raw GitHub offline over-reports under status lag); it only
# contributes when a local signal corroborates it.
_BOUNCE_REMEDIABLE_SIGNALS = frozenset(
    {
        _EnumSignal.DOCKER_HEALTH,
        _EnumSignal.DIAG_HEARTBEAT,
        _EnumSignal.LISTENER_TOPOLOGY,
        _EnumSignal.CONTAINER_STABILITY,
    }
)


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


def _signal(
    signal: EnumRunnerReadinessSignal,
    outcome: EnumReadinessSignalOutcome,
    detail: str = "",
) -> ModelRunnerReadinessSignal:
    return ModelRunnerReadinessSignal(signal=signal, outcome=outcome, detail=detail)


def _github_registration_signal(
    fact: ModelRunnerFleetRunnerFact, *, github_source_ok: bool
) -> ModelRunnerReadinessSignal:
    if not github_source_ok:
        return _signal(
            _EnumSignal.GITHUB_REGISTRATION,
            _EnumOutcome.UNKNOWN,
            "GitHub runners API source failed this tick",
        )
    if fact.github_status == "online":
        return _signal(_EnumSignal.GITHUB_REGISTRATION, _EnumOutcome.PASS)
    return _signal(
        _EnumSignal.GITHUB_REGISTRATION,
        _EnumOutcome.FAIL,
        f"GitHub reports status={fact.github_status!r}, not 'online'",
    )


def _docker_health_signal(
    fact: ModelRunnerFleetRunnerFact, *, docker_source_ok: bool
) -> ModelRunnerReadinessSignal:
    """The signal the legacy precedence classifier never evaluated at all."""
    if not docker_source_ok:
        return _signal(
            _EnumSignal.DOCKER_HEALTH,
            _EnumOutcome.UNKNOWN,
            "SSH/Docker source failed this tick",
        )
    if fact.docker_status and fact.docker_status != "running":
        return _signal(
            _EnumSignal.DOCKER_HEALTH,
            _EnumOutcome.FAIL,
            f"container state={fact.docker_status!r}, not 'running'",
        )
    if not fact.docker_health:
        return _signal(
            _EnumSignal.DOCKER_HEALTH,
            _EnumOutcome.UNKNOWN,
            "container health not reported by the probe",
        )
    if fact.docker_health in _HEALTHY_DOCKER_HEALTH_VALUES:
        return _signal(_EnumSignal.DOCKER_HEALTH, _EnumOutcome.PASS)
    if fact.docker_health == "starting":
        return _signal(
            _EnumSignal.DOCKER_HEALTH,
            _EnumOutcome.UNKNOWN,
            "healthcheck still in start_period (starting)",
        )
    return _signal(
        _EnumSignal.DOCKER_HEALTH,
        _EnumOutcome.FAIL,
        f"container health={fact.docker_health!r}",
    )


def _diag_heartbeat_signal(
    fact: ModelRunnerFleetRunnerFact, *, docker_source_ok: bool
) -> ModelRunnerReadinessSignal:
    """OMN-15233: the threshold must bracket a full token-refresh cycle.

    A live idle runner writes ``_diag`` on the listener's token-refresh
    cadence, measured at ~50-53 minutes on 2026-07-27. The retired 900s
    default therefore read a perfectly healthy idle runner as stale for ~35 of
    every 50 minutes -- the false-positive that drove the restart storm this
    signal exists to stop.
    """
    if fact.diag_heartbeat_age_seconds is None:
        return _signal(
            _EnumSignal.DIAG_HEARTBEAT,
            _EnumOutcome.UNKNOWN,
            (
                "SSH/Docker source failed this tick"
                if not docker_source_ok
                else "no _diag heartbeat age observed"
            ),
        )
    if fact.diag_heartbeat_age_seconds <= _RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS:
        return _signal(_EnumSignal.DIAG_HEARTBEAT, _EnumOutcome.PASS)
    return _signal(
        _EnumSignal.DIAG_HEARTBEAT,
        _EnumOutcome.FAIL,
        (
            f"_diag heartbeat age={fact.diag_heartbeat_age_seconds:.0f}s > "
            f"threshold={_RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS}s"
        ),
    )


def _listener_topology_signal(
    fact: ModelRunnerFleetRunnerFact,
) -> ModelRunnerReadinessSignal:
    if fact.listener_process_count is None or fact.orphaned_listener_count is None:
        return _signal(
            _EnumSignal.LISTENER_TOPOLOGY,
            _EnumOutcome.UNKNOWN,
            "listener process topology not observed",
        )
    if fact.orphaned_listener_count > 0:
        return _signal(
            _EnumSignal.LISTENER_TOPOLOGY,
            _EnumOutcome.FAIL,
            f"{fact.orphaned_listener_count} Runner.Listener process(es) at PPID 1",
        )
    if fact.listener_process_count != 1:
        return _signal(
            _EnumSignal.LISTENER_TOPOLOGY,
            _EnumOutcome.FAIL,
            f"{fact.listener_process_count} Runner.Listener process(es), expected exactly 1",
        )
    return _signal(_EnumSignal.LISTENER_TOPOLOGY, _EnumOutcome.PASS)


def _container_stability_signal(
    fact: ModelRunnerFleetRunnerFact, *, docker_source_ok: bool
) -> ModelRunnerReadinessSignal:
    if not docker_source_ok:
        return _signal(
            _EnumSignal.CONTAINER_STABILITY,
            _EnumOutcome.UNKNOWN,
            "SSH/Docker source failed this tick",
        )
    if fact.docker_restart_count > _CRASHLOOP_RESTART_THRESHOLD:
        return _signal(
            _EnumSignal.CONTAINER_STABILITY,
            _EnumOutcome.FAIL,
            (
                f"RestartCount={fact.docker_restart_count} > "
                f"threshold={_CRASHLOOP_RESTART_THRESHOLD}"
            ),
        )
    return _signal(_EnumSignal.CONTAINER_STABILITY, _EnumOutcome.PASS)


def _disk_capacity_signal(
    host_disk_used_percent: float | None,
) -> ModelRunnerReadinessSignal:
    if host_disk_used_percent is None:
        return _signal(
            _EnumSignal.DISK_CAPACITY,
            _EnumOutcome.UNKNOWN,
            "runner-host disk usage not observed",
        )
    if host_disk_used_percent < _RUNNER_READINESS_MAX_DISK_USED_PERCENT:
        return _signal(_EnumSignal.DISK_CAPACITY, _EnumOutcome.PASS)
    return _signal(
        _EnumSignal.DISK_CAPACITY,
        _EnumOutcome.FAIL,
        (
            f"runner-host disk used={host_disk_used_percent:.1f}% >= "
            f"ceiling={_RUNNER_READINESS_MAX_DISK_USED_PERCENT:.1f}%"
        ),
    )


def _evaluate_readiness_signals(
    fact: ModelRunnerFleetRunnerFact,
    *,
    github_source_ok: bool,
    docker_source_ok: bool,
    host_disk_used_percent: float | None,
) -> tuple[ModelRunnerReadinessSignal, ...]:
    """Evaluate every readiness signal for one runner. Pure, no short-circuit.

    Every signal is evaluated even when an earlier one already FAILed. That is
    the difference from ``_classify_runner``: a precedence chain stops at the
    first match and therefore cannot report that a runner failed three
    different ways.
    """
    return (
        _github_registration_signal(fact, github_source_ok=github_source_ok),
        _docker_health_signal(fact, docker_source_ok=docker_source_ok),
        _diag_heartbeat_signal(fact, docker_source_ok=docker_source_ok),
        _listener_topology_signal(fact),
        _container_stability_signal(fact, docker_source_ok=docker_source_ok),
        _disk_capacity_signal(host_disk_used_percent),
    )


def _readiness_state(
    signals: tuple[ModelRunnerReadinessSignal, ...],
) -> EnumRunnerReadinessState:
    """Conjunction: READY only when every signal PASSes."""
    if any(s.outcome == _EnumOutcome.FAIL for s in signals):
        return EnumRunnerReadinessState.NOT_READY
    if any(s.outcome == _EnumOutcome.UNKNOWN for s in signals):
        return EnumRunnerReadinessState.UNKNOWN
    return EnumRunnerReadinessState.READY


def _quarantine_reason(signals: tuple[ModelRunnerReadinessSignal, ...]) -> str:
    return "; ".join(
        f"{s.signal.value}: {s.detail}"
        for s in signals
        if s.outcome == _EnumOutcome.FAIL
    )


def _is_bounce_eligible(
    fact: ModelRunnerFleetRunnerFact,
    signals: tuple[ModelRunnerReadinessSignal, ...],
    *,
    readiness: EnumRunnerReadinessState,
    is_determinate: bool,
) -> bool:
    """Decide whether a force-recreate is a defensible remedy for this runner.

    Readiness fails CLOSED (UNKNOWN is not READY, so an unprobeable runner is
    not counted as capacity). This gate fails SAFE in the opposite direction:
    it mutates nothing on indeterminate evidence, never interrupts a job in
    flight, and never bounces for a cause a bounce cannot fix. The asymmetry
    is deliberate -- the cost of not routing to a good runner is one idle
    runner; the cost of recreating a busy or misread runner is a cancelled CI
    job plus the restart storm this ticket exists to stop.
    """
    if readiness != EnumRunnerReadinessState.NOT_READY:
        return False
    if not is_determinate:
        return False
    if fact.github_busy:
        return False
    failing = {s.signal for s in signals if s.outcome == _EnumOutcome.FAIL}
    return bool(failing & _BOUNCE_REMEDIABLE_SIGNALS)


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
    """Map a per-runner assessment to a recommended (never-executed) action.

    OMN-15255 replaced four state-keyed ``RESTART_RUNNER`` branches
    (``CRASH_LOOPING`` 0.9 / ``LISTENER_ZOMBIE`` 0.85 / ``OFFLINE_IDLE`` 0.6 /
    ``WEDGED`` 0.5) with one rule: a restart is recommended exactly when the
    runner is bounce-eligible. Four independently-tunable confidence
    heuristics over the same underlying facts is how a single misread
    threshold (the 900s heartbeat window) turned into a fleet-wide restart
    storm -- there was no second condition anywhere that could veto it.

    Quarantined-but-not-bounce-eligible still surfaces, as ``NONE`` with the
    failing signals as the reason: it is an operator item (host disk, GitHub
    status lag), not a restart.
    """
    if assessment.bounce_eligible:
        return ModelRecommendedAction(
            action_type=EnumRecommendedActionType.RESTART_RUNNER,
            target_id=assessment.name,
            reason=assessment.quarantine_reason or assessment.detail,
            confidence=0.9,
        )
    if assessment.quarantined:
        return ModelRecommendedAction(
            action_type=EnumRecommendedActionType.NONE,
            target_id=assessment.name,
            reason=(
                "quarantined; no failing signal a force-recreate can fix -- "
                f"{assessment.quarantine_reason}"
            ),
            confidence=0.0,
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
        self, request: ModelRunnerFleetHealthEvaluateCommand
    ) -> ModelRunnerFleetHealthVerdict:
        """Classify the command's snapshot into a ``ModelRunnerFleetHealthVerdict``.

        Canonical ONEX definition B (OMN-14355): a single typed-payload
        ``request`` the shared runtime adapter binds via the contract
        ``input_model``. The correlation id and facts-only snapshot are unpacked
        from the command; classification is byte-identical to the pre-flip
        two-argument entrypoint (OMN-14781 signature adaptation -- no behavior
        change).

        Args:
            request: Command carrying the workflow correlation id and the
                facts-only snapshot gathered upstream by the EFFECT node.

        Returns:
            ModelRunnerFleetHealthVerdict with per-runner states, fleet
            aggregates, and recorded (never executed) recommended actions.
        """
        correlation_id = request.correlation_id
        snapshot = request.snapshot
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
            signals = _evaluate_readiness_signals(
                fact,
                github_source_ok=snapshot.github_source_ok,
                docker_source_ok=snapshot.docker_source_ok,
                host_disk_used_percent=snapshot.host_disk_used_percent,
            )
            readiness = _readiness_state(signals)
            quarantined = readiness == EnumRunnerReadinessState.NOT_READY
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
                readiness=readiness,
                signals=signals,
                quarantined=quarantined,
                quarantine_reason=_quarantine_reason(signals) if quarantined else "",
                bounce_eligible=_is_bounce_eligible(
                    fact,
                    signals,
                    readiness=readiness,
                    is_determinate=is_determinate,
                ),
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

        ready_count = sum(
            1 for a in assessments if a.readiness == EnumRunnerReadinessState.READY
        )
        not_ready_count = sum(
            1 for a in assessments if a.readiness == EnumRunnerReadinessState.NOT_READY
        )
        readiness_unknown_count = sum(
            1 for a in assessments if a.readiness == EnumRunnerReadinessState.UNKNOWN
        )
        signal_rollups = tuple(
            ModelReadinessSignalRollup(
                signal=signal,
                fail_count=sum(
                    1
                    for a in assessments
                    for s in a.signals
                    if s.signal == signal and s.outcome == _EnumOutcome.FAIL
                ),
                unknown_count=sum(
                    1
                    for a in assessments
                    for s in a.signals
                    if s.signal == signal and s.outcome == _EnumOutcome.UNKNOWN
                ),
            )
            for signal in EnumRunnerReadinessSignal
        )

        verdict = ModelRunnerFleetHealthVerdict(
            correlation_id=correlation_id,
            evaluated_at=datetime.now(tz=UTC),
            assessments=tuple(assessments),
            ready_count=ready_count,
            not_ready_count=not_ready_count,
            readiness_unknown_count=readiness_unknown_count,
            fleet_ready=ready_count > 0 and not_ready_count == 0,
            quarantined_runners=tuple(a.name for a in assessments if a.quarantined),
            bounce_eligible_runners=tuple(
                a.name for a in assessments if a.bounce_eligible
            ),
            readiness_signal_rollups=signal_rollups,
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
            "Runner-fleet verdict: %d/%d online, %d READY / %d NOT_READY / %d UNKNOWN, "
            "%d quarantined (%d bounce-eligible), saturation=%.2f, %d recommended "
            "actions (correlation_id=%s)",
            online_count,
            snapshot.expected_count,
            ready_count,
            not_ready_count,
            readiness_unknown_count,
            len(verdict.quarantined_runners),
            len(verdict.bounce_eligible_runners),
            saturation_ratio,
            len(recommended_actions),
            correlation_id,
        )
        return verdict


__all__ = ["HandlerRunnerFleetHealthEvaluate"]
