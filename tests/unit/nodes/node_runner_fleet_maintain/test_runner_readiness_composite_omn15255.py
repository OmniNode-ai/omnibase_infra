# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Composite runner readiness + quarantine gate (OMN-15255, friction F-04).

These tests drive the real ``HandlerRunnerFleetHealthEvaluate`` -- the COMPUTE
handler the runtime dispatches -- against fixture snapshots. No I/O, no fleet
mutation, nothing here touches a live runner.

Two properties are under test and they are not the same property:

1. **Readiness is a conjunction.** Each signal alone is insufficient: a runner
   is READY only when all six PASS. The five/six single-signal-failure tests
   below each fail exactly one signal with the rest passing, which is the only
   way to prove a signal is load-bearing rather than decorative.
2. **Quarantine is not a restart.** The bounce gate is strictly narrower than
   the quarantine set. A stale-looking heartbeat, a GitHub status lag, or a
   full host disk must never on their own produce a RESTART_RUNNER -- that
   combination is the false-positive restart storm this ticket closes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_runner_fleet_health_compute.handlers import (
    handler_runner_fleet_health_evaluate as evaluate_module,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.handlers.handler_runner_fleet_health_evaluate import (
    HandlerRunnerFleetHealthEvaluate,
)
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
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_evaluate_command import (
    ModelRunnerFleetHealthEvaluateCommand,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_verdict import (
    ModelRunnerFleetHealthVerdict,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_runner_fact import (
    ModelRunnerFleetRunnerFact,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot import (
    ModelRunnerFleetSnapshot,
)

# A runner that passes every readiness signal. Each test below perturbs
# exactly one field, so any resulting NOT_READY is attributable to that one
# signal and nothing else.
_ALL_SIGNALS_PASSING: dict[str, object] = {
    "github_status": "online",
    "github_busy": False,
    "docker_status": "running",
    "docker_health": "healthy",
    "diag_heartbeat_age_seconds": 120.0,
    "listener_process_count": 1,
    "orphaned_listener_count": 0,
    "docker_restart_count": 0,
}

# Host disk well under the ceiling; the DISK_CAPACITY signal is snapshot-level.
_DISK_OK = 42.0


def _fact(
    name: str = "omninode-runner-1", **overrides: object
) -> ModelRunnerFleetRunnerFact:
    values: dict[str, object] = {"name": name, **_ALL_SIGNALS_PASSING}
    values.update(overrides)
    return ModelRunnerFleetRunnerFact(**values)  # type: ignore[arg-type]


def _snapshot(
    *facts: ModelRunnerFleetRunnerFact, **overrides: object
) -> ModelRunnerFleetSnapshot:
    values: dict[str, object] = {
        "correlation_id": uuid4(),
        "collected_at": datetime.now(tz=UTC),
        "host": "192.168.86.201",
        "expected_count": max(len(facts), 1),
        "runners": facts,
        "host_disk_used_percent": _DISK_OK,
    }
    values.update(overrides)
    return ModelRunnerFleetSnapshot(**values)  # type: ignore[arg-type]


async def _evaluate(
    snapshot: ModelRunnerFleetSnapshot,
) -> ModelRunnerFleetHealthVerdict:
    handler = HandlerRunnerFleetHealthEvaluate()
    return await handler.handle(
        ModelRunnerFleetHealthEvaluateCommand(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
    )


def _outcome(
    verdict: ModelRunnerFleetHealthVerdict,
    runner: str,
    signal: EnumRunnerReadinessSignal,
) -> EnumReadinessSignalOutcome:
    assessment = next(a for a in verdict.assessments if a.name == runner)
    return next(s.outcome for s in assessment.signals if s.signal == signal)


def _restart_targets(verdict: ModelRunnerFleetHealthVerdict) -> list[str]:
    return [
        a.target_id
        for a in verdict.recommended_actions
        if a.action_type == EnumRecommendedActionType.RESTART_RUNNER
    ]


@pytest.mark.unit
class TestReadinessIsAConjunction:
    """Each signal alone is insufficient -- the F-04 claim, tested one at a time."""

    @pytest.mark.asyncio
    async def test_all_signals_passing_is_ready(self):
        verdict = await _evaluate(_snapshot(_fact()))
        assessment = verdict.assessments[0]
        assert assessment.readiness == EnumRunnerReadinessState.READY
        assert assessment.quarantined is False
        assert assessment.bounce_eligible is False
        assert all(
            s.outcome == EnumReadinessSignalOutcome.PASS for s in assessment.signals
        )
        assert verdict.ready_count == 1
        assert verdict.fleet_ready is True
        assert verdict.recommended_actions == ()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("overrides", "failing_signal"),
        [
            pytest.param(
                {"github_status": "offline"},
                EnumRunnerReadinessSignal.GITHUB_REGISTRATION,
                id="github-offline",
            ),
            pytest.param(
                {"docker_health": "unhealthy"},
                EnumRunnerReadinessSignal.DOCKER_HEALTH,
                id="container-unhealthy",
            ),
            pytest.param(
                {"diag_heartbeat_age_seconds": 6000.0},
                EnumRunnerReadinessSignal.DIAG_HEARTBEAT,
                id="heartbeat-past-corrected-cadence",
            ),
            pytest.param(
                {"listener_process_count": 2},
                EnumRunnerReadinessSignal.LISTENER_TOPOLOGY,
                id="duplicate-listener",
            ),
            pytest.param(
                {"orphaned_listener_count": 1},
                EnumRunnerReadinessSignal.LISTENER_TOPOLOGY,
                id="ppid-1-orphan",
            ),
            pytest.param(
                {"docker_restart_count": 12},
                EnumRunnerReadinessSignal.CONTAINER_STABILITY,
                id="crash-looping",
            ),
        ],
    )
    async def test_single_failing_signal_makes_the_runner_not_ready(
        self, overrides: dict[str, object], failing_signal: EnumRunnerReadinessSignal
    ):
        verdict = await _evaluate(_snapshot(_fact(**overrides)))
        assessment = verdict.assessments[0]
        assert assessment.readiness == EnumRunnerReadinessState.NOT_READY
        assert assessment.quarantined is True
        assert (
            _outcome(verdict, assessment.name, failing_signal)
            == EnumReadinessSignalOutcome.FAIL
        )
        assert failing_signal.value in assessment.quarantine_reason
        # Every OTHER signal still passed -- the failure is attributable.
        others = [s for s in assessment.signals if s.signal != failing_signal]
        assert all(s.outcome == EnumReadinessSignalOutcome.PASS for s in others)
        assert verdict.ready_count == 0
        assert verdict.not_ready_count == 1
        assert verdict.fleet_ready is False
        assert verdict.quarantined_runners == (assessment.name,)

    @pytest.mark.asyncio
    async def test_host_disk_over_ceiling_fails_the_whole_fleet(self):
        """DISK_CAPACITY is host-scoped: one full disk, every runner unfit."""
        verdict = await _evaluate(
            _snapshot(
                _fact("omninode-runner-1"),
                _fact("omninode-runner-2"),
                host_disk_used_percent=97.5,
            )
        )
        assert verdict.ready_count == 0
        assert verdict.not_ready_count == 2
        assert set(verdict.quarantined_runners) == {
            "omninode-runner-1",
            "omninode-runner-2",
        }
        assert all(
            _outcome(verdict, a.name, EnumRunnerReadinessSignal.DISK_CAPACITY)
            == EnumReadinessSignalOutcome.FAIL
            for a in verdict.assessments
        )

    @pytest.mark.asyncio
    async def test_docker_health_none_is_a_pass_not_a_gap(self):
        """An image with no declared healthcheck must not be permanently UNKNOWN."""
        verdict = await _evaluate(_snapshot(_fact(docker_health="none")))
        assert verdict.assessments[0].readiness == EnumRunnerReadinessState.READY

    @pytest.mark.asyncio
    async def test_every_signal_is_evaluated_not_short_circuited(self):
        """The precedence chain reports one cause; the conjunction reports all of them.

        This is the operational difference: an operator triaging a runner needs
        to know it is failing three ways, not just the highest-precedence one.
        """
        verdict = await _evaluate(
            _snapshot(
                _fact(
                    github_status="offline",
                    docker_health="unhealthy",
                    listener_process_count=3,
                )
            )
        )
        assessment = verdict.assessments[0]
        failing = {
            s.signal
            for s in assessment.signals
            if s.outcome == EnumReadinessSignalOutcome.FAIL
        }
        assert failing == {
            EnumRunnerReadinessSignal.GITHUB_REGISTRATION,
            EnumRunnerReadinessSignal.DOCKER_HEALTH,
            EnumRunnerReadinessSignal.LISTENER_TOPOLOGY,
        }


@pytest.mark.unit
class TestCompositeCatchesWhatPrecedenceMisses:
    """The live 2026-07-27 divergence: 64/64 online, 53/64 containers unhealthy."""

    @pytest.mark.asyncio
    async def test_online_plus_unhealthy_container_plus_duplicate_listener(self):
        verdict = await _evaluate(
            _snapshot(
                _fact(
                    github_status="online",
                    docker_health="unhealthy",
                    listener_process_count=2,
                    diag_heartbeat_age_seconds=60.0,
                )
            )
        )
        assessment = verdict.assessments[0]
        # The legacy precedence classifier evaluates neither container health
        # nor listener topology, so it still calls this runner HEALTHY.
        assert assessment.state == EnumRunnerFleetHealthState.HEALTHY
        # The composite does not.
        assert assessment.readiness == EnumRunnerReadinessState.NOT_READY
        assert assessment.quarantined is True

    @pytest.mark.asyncio
    async def test_fleet_rollup_reconciles_the_two_surfaces(self):
        """`online_count` and `ready_count` are different numbers, on purpose."""
        verdict = await _evaluate(
            _snapshot(
                _fact("omninode-runner-1"),
                _fact("omninode-runner-2", docker_health="unhealthy"),
                _fact("omninode-runner-3", docker_health="unhealthy"),
            )
        )
        assert verdict.online_count == 3
        assert verdict.ready_count == 1
        assert verdict.not_ready_count == 2
        rollup = {r.signal: r for r in verdict.readiness_signal_rollups}
        assert rollup[EnumRunnerReadinessSignal.DOCKER_HEALTH].fail_count == 2
        assert rollup[EnumRunnerReadinessSignal.GITHUB_REGISTRATION].fail_count == 0
        # Every signal appears in the rollup, including the clean ones.
        assert set(rollup) == set(EnumRunnerReadinessSignal)


@pytest.mark.unit
class TestQuarantineStopsTheFalsePositiveRestartStorm:
    """OMN-15233's root cause, expressed as a regression test.

    The listener writes ``_diag`` on a token-refresh cadence measured at
    ~50-53 minutes. Under the retired 900s threshold a healthy idle fleet
    read stale for ~35 of every 50 minutes and the auto-bounce loop recreated
    it. The corrected 4500s default brackets a full refresh cycle.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("idle_age", [1000.0, 1200.0, 2739.0, 3000.0, 4400.0])
    async def test_idle_fleet_past_the_retired_900s_window_is_ready_and_unbounced(
        self, idle_age: float
    ):
        verdict = await _evaluate(
            _snapshot(
                *(
                    _fact(f"omninode-runner-{i}", diag_heartbeat_age_seconds=idle_age)
                    for i in range(1, 9)
                )
            )
        )
        assert verdict.ready_count == 8
        assert verdict.quarantined_runners == ()
        assert verdict.bounce_eligible_runners == ()
        assert _restart_targets(verdict) == []

    @pytest.mark.asyncio
    async def test_the_retired_900s_threshold_quarantines_without_bounce(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Pin the old threshold and the same healthy fleet is quarantined.

        OMN-15234 adds a second guard after OMN-15255: a lone stale heartbeat
        can make a runner NOT_READY, but it still cannot produce a bounce
        without independent corroboration.
        """
        monkeypatch.setattr(evaluate_module, "_RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS", 900)
        verdict = await _evaluate(
            _snapshot(
                *(
                    _fact(f"omninode-runner-{i}", diag_heartbeat_age_seconds=1200.0)
                    for i in range(1, 9)
                )
            )
        )
        assert verdict.ready_count == 0
        assert len(verdict.quarantined_runners) == 8
        assert verdict.bounce_eligible_runners == ()
        assert _restart_targets(verdict) == []

    @pytest.mark.asyncio
    async def test_a_genuinely_dead_listener_still_bounces(self):
        """The fix must not have disarmed the check it recalibrated."""
        verdict = await _evaluate(
            _snapshot(
                _fact(diag_heartbeat_age_seconds=9000.0, listener_process_count=0)
            )
        )
        assert verdict.assessments[0].readiness == EnumRunnerReadinessState.NOT_READY
        assert verdict.assessments[0].bounce_eligible is True
        assert _restart_targets(verdict) == ["omninode-runner-1"]


@pytest.mark.unit
class TestBounceGateIsNarrowerThanQuarantine:
    @pytest.mark.asyncio
    async def test_github_offline_alone_quarantines_but_never_bounces(self):
        """OMN-14057: raw GitHub offline over-reports under status lag.

        Local evidence is intact -- one non-orphaned listener, fresh
        heartbeat, healthy container. Recreating this runner would destroy a
        working listener to fix a reporting delay.
        """
        verdict = await _evaluate(_snapshot(_fact(github_status="offline")))
        assessment = verdict.assessments[0]
        assert assessment.quarantined is True
        assert assessment.bounce_eligible is False
        assert _restart_targets(verdict) == []
        action = next(
            a for a in verdict.recommended_actions if a.target_id == assessment.name
        )
        assert action.action_type == EnumRecommendedActionType.NONE

    @pytest.mark.asyncio
    async def test_github_offline_corroborated_by_local_failure_does_bounce(self):
        """Corroboration, not suppression: offline + a real local fault bounces."""
        verdict = await _evaluate(
            _snapshot(_fact(github_status="offline", orphaned_listener_count=1))
        )
        assert verdict.assessments[0].bounce_eligible is True
        assert _restart_targets(verdict) == ["omninode-runner-1"]

    @pytest.mark.asyncio
    async def test_full_host_disk_quarantines_but_never_bounces(self):
        """A force-recreate frees no host disk; bouncing on it is pure churn."""
        verdict = await _evaluate(_snapshot(_fact(), host_disk_used_percent=99.0))
        assert verdict.assessments[0].quarantined is True
        assert verdict.assessments[0].bounce_eligible is False
        assert _restart_targets(verdict) == []

    @pytest.mark.asyncio
    async def test_busy_runner_is_never_bounced(self):
        """A recreate mid-job cancels that job. Never on a busy runner."""
        verdict = await _evaluate(
            _snapshot(_fact(github_busy=True, docker_health="unhealthy"))
        )
        assert verdict.assessments[0].quarantined is True
        assert verdict.assessments[0].bounce_eligible is False
        assert _restart_targets(verdict) == []

    @pytest.mark.asyncio
    async def test_bounce_eligible_is_always_a_subset_of_quarantined(self):
        verdict = await _evaluate(
            _snapshot(
                _fact("omninode-runner-1"),
                _fact("omninode-runner-2", docker_health="unhealthy"),
                _fact("omninode-runner-3", github_status="offline"),
                _fact("omninode-runner-4", github_busy=True, listener_process_count=0),
            )
        )
        assert set(verdict.bounce_eligible_runners) <= set(verdict.quarantined_runners)
        assert verdict.bounce_eligible_runners == ("omninode-runner-2",)


@pytest.mark.unit
class TestIndeterminateSourcesNeverProduceReadyOrBounce:
    @pytest.mark.asyncio
    async def test_docker_source_outage_is_unknown_not_ready(self):
        verdict = await _evaluate(
            _snapshot(
                _fact(),
                docker_source_ok=False,
                source_errors=("SSH/Docker exit code 255: connection refused",),
            )
        )
        assessment = verdict.assessments[0]
        assert assessment.readiness == EnumRunnerReadinessState.UNKNOWN
        assert assessment.quarantined is False
        assert assessment.bounce_eligible is False
        assert verdict.ready_count == 0
        assert verdict.readiness_unknown_count == 1
        assert verdict.fleet_ready is False

    @pytest.mark.asyncio
    async def test_indeterminate_sources_never_bounce_even_when_a_signal_fails(self):
        """Fail-safe on mutation: a confirmed FAIL under a dead source is still
        not enough to recreate a container."""
        verdict = await _evaluate(
            _snapshot(
                _fact(github_status="offline", docker_health="unhealthy"),
                docker_source_ok=False,
            )
        )
        assessment = verdict.assessments[0]
        assert assessment.quarantined is True
        assert assessment.is_determinate is False
        assert assessment.bounce_eligible is False
        assert _restart_targets(verdict) == []

    @pytest.mark.asyncio
    async def test_unprobed_readiness_facts_read_unknown_not_pass(self):
        """A pre-rollout snapshot (no health/listener/disk facts) must not read READY.

        This is the fail-safe that makes shipping this ahead of the probe
        rollout inert: unknown everywhere means nothing is quarantined and
        nothing is bounced.
        """
        verdict = await _evaluate(
            ModelRunnerFleetSnapshot(
                correlation_id=uuid4(),
                collected_at=datetime.now(tz=UTC),
                host="192.168.86.201",
                expected_count=1,
                runners=(
                    ModelRunnerFleetRunnerFact(
                        name="omninode-runner-1",
                        github_status="online",
                        github_busy=False,
                        docker_status="running",
                        diag_heartbeat_age_seconds=60.0,
                    ),
                ),
            )
        )
        assessment = verdict.assessments[0]
        assert assessment.readiness == EnumRunnerReadinessState.UNKNOWN
        assert verdict.quarantined_runners == ()
        assert verdict.bounce_eligible_runners == ()
        unknown = {
            s.signal
            for s in assessment.signals
            if s.outcome == EnumReadinessSignalOutcome.UNKNOWN
        }
        assert unknown == {
            EnumRunnerReadinessSignal.DOCKER_HEALTH,
            EnumRunnerReadinessSignal.LISTENER_TOPOLOGY,
            EnumRunnerReadinessSignal.DISK_CAPACITY,
        }

    @pytest.mark.asyncio
    async def test_container_not_running_fails_docker_health(self):
        verdict = await _evaluate(
            _snapshot(_fact(docker_status="exited", docker_health=""))
        )
        assert (
            _outcome(
                verdict, "omninode-runner-1", EnumRunnerReadinessSignal.DOCKER_HEALTH
            )
            == EnumReadinessSignalOutcome.FAIL
        )
        assert verdict.assessments[0].bounce_eligible is True

    @pytest.mark.asyncio
    async def test_starting_healthcheck_is_unknown_not_fail(self):
        """A container inside its start_period has not failed anything yet."""
        verdict = await _evaluate(_snapshot(_fact(docker_health="starting")))
        assert verdict.assessments[0].readiness == EnumRunnerReadinessState.UNKNOWN
        assert verdict.assessments[0].quarantined is False


@pytest.mark.unit
class TestReadinessDeterminism:
    @pytest.mark.asyncio
    async def test_identical_snapshot_yields_identical_readiness(self):
        snapshot = _snapshot(
            _fact("omninode-runner-1", docker_health="unhealthy"),
            _fact("omninode-runner-2", github_status="offline"),
        )
        first = await _evaluate(snapshot)
        second = await _evaluate(snapshot)
        assert [a.readiness for a in first.assessments] == [
            a.readiness for a in second.assessments
        ]
        assert first.bounce_eligible_runners == second.bounce_eligible_runners
        assert [
            (s.signal, s.outcome) for a in first.assessments for s in a.signals
        ] == [(s.signal, s.outcome) for a in second.assessments for s in a.signals]
