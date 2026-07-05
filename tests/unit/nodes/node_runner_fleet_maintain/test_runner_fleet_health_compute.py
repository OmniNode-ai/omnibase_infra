# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerRunnerFleetHealthEvaluate (OMN-13942).

Pure classification logic against fixture fleet states. No test in this
module restarts, cancels, or mutates anything real -- there is no I/O
anywhere in the COMPUTE node under test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_runner_fleet_health_compute.handlers.handler_runner_fleet_health_evaluate import (
    HandlerRunnerFleetHealthEvaluate,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_recommended_action_type import (
    EnumRecommendedActionType,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_fleet_health_state import (
    EnumRunnerFleetHealthState,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_runner_fact import (
    ModelRunnerFleetRunnerFact,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot import (
    ModelRunnerFleetSnapshot,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_zombie_run_candidate import (
    ModelZombieRunCandidate,
)


def _snapshot(**overrides: object) -> ModelRunnerFleetSnapshot:
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "collected_at": datetime.now(tz=UTC),
        "host": "192.168.86.201",
        "expected_count": 3,
        "runners": (),
    }
    defaults.update(overrides)
    return ModelRunnerFleetSnapshot(**defaults)  # type: ignore[arg-type]


def _fact(name: str, **overrides: object) -> ModelRunnerFleetRunnerFact:
    defaults: dict[str, object] = {
        "name": name,
        "github_status": "online",
        "github_busy": False,
    }
    defaults.update(overrides)
    return ModelRunnerFleetRunnerFact(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestRunnerFleetHealthComputeHealthy:
    @pytest.mark.asyncio
    async def test_all_online_and_idle_is_healthy(self):
        snapshot = _snapshot(
            runners=(
                _fact("omninode-runner-1", github_busy=False),
                _fact("omninode-runner-2", github_busy=False),
            )
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        assert all(
            a.state == EnumRunnerFleetHealthState.HEALTHY for a in verdict.assessments
        )
        assert verdict.online_count == 2
        assert verdict.busy_count == 0
        assert verdict.idle_count == 2
        assert verdict.saturation_ratio == 0.0
        assert verdict.recommended_actions == ()


@pytest.mark.unit
class TestRunnerFleetHealthComputeSaturated:
    @pytest.mark.asyncio
    async def test_zero_idle_all_busy_is_saturated(self):
        """Mirrors the live 2026-07-04 incident: 43 online, 0 idle, all busy."""
        snapshot = _snapshot(
            expected_count=3,
            runners=(
                _fact("omninode-runner-1", github_busy=True),
                _fact("omninode-runner-2", github_busy=True),
                _fact("omninode-runner-3", github_busy=True),
            ),
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        assert all(
            a.state == EnumRunnerFleetHealthState.SATURATED for a in verdict.assessments
        )
        assert verdict.saturation_ratio == 1.0
        assert verdict.idle_count == 0
        # SATURATED recommends NONE -- there is no per-runner action for capacity exhaustion.
        assert all(
            a.action_type == EnumRecommendedActionType.NONE
            for a in verdict.recommended_actions
        )


@pytest.mark.unit
class TestRunnerFleetHealthComputeDegraded:
    @pytest.mark.asyncio
    async def test_crash_loop_and_zombie_present(self):
        snapshot = _snapshot(
            expected_count=3,
            runners=(
                _fact("omninode-runner-1", docker_restart_count=9),
                _fact("omninode-runner-2", diag_heartbeat_age_seconds=1200.0),
                _fact("omninode-runner-3", github_busy=False),
            ),
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        states = {a.name: a.state for a in verdict.assessments}
        assert states["omninode-runner-1"] == EnumRunnerFleetHealthState.CRASH_LOOPING
        assert states["omninode-runner-2"] == EnumRunnerFleetHealthState.LISTENER_ZOMBIE
        assert states["omninode-runner-3"] == EnumRunnerFleetHealthState.HEALTHY
        assert verdict.crash_looping_count == 1
        assert verdict.listener_zombie_count == 1

        restart_action = next(
            a for a in verdict.recommended_actions if a.target_id == "omninode-runner-1"
        )
        assert restart_action.action_type == EnumRecommendedActionType.RESTART_RUNNER
        assert restart_action.confidence > 0.0

    @pytest.mark.asyncio
    async def test_offline_runner_is_offline_idle(self):
        snapshot = _snapshot(
            expected_count=1,
            runners=(_fact("omninode-runner-1", github_status="offline"),),
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        assert verdict.assessments[0].state == EnumRunnerFleetHealthState.OFFLINE_IDLE
        assert verdict.offline_count == 1


@pytest.mark.unit
class TestRunnerFleetHealthComputeZombiePresent:
    @pytest.mark.asyncio
    async def test_zombie_run_candidate_recommends_cancel_run(self):
        snapshot = _snapshot(
            expected_count=1,
            runners=(_fact("omninode-runner-1"),),
            zombie_run_candidates=(
                ModelZombieRunCandidate(
                    repo="OmniNode-ai/omnibase_infra",
                    run_id=123456,
                    workflow_name="CI",
                    status="queued",
                    age_seconds=900.0,
                ),
            ),
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        cancel_actions = [
            a
            for a in verdict.recommended_actions
            if a.action_type == EnumRecommendedActionType.CANCEL_RUN
        ]
        assert len(cancel_actions) == 1
        assert cancel_actions[0].target_id == "123456"


@pytest.mark.unit
class TestRunnerFleetHealthComputeBuildxAndCodeload:
    """Covers the two NEW checks that close the OMN-13932 blind spot."""

    @pytest.mark.asyncio
    async def test_buildx_unavailable_classifies_runners(self):
        snapshot = _snapshot(
            expected_count=1,
            runners=(_fact("omninode-runner-1"),),
            buildx_available=False,
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        assert verdict.buildx_unavailable is True
        assert (
            verdict.assessments[0].state
            == EnumRunnerFleetHealthState.BUILDX_UNAVAILABLE
        )

    @pytest.mark.asyncio
    async def test_codeload_throttle_signals_classify_runners(self):
        snapshot = _snapshot(
            expected_count=1,
            runners=(_fact("omninode-runner-1"),),
            codeload_throttle_signal_count=3,
            codeload_throttle_examples=(
                "OmniNode-ai/omnibase_infra#42: matched 'GnuTLS'",
            ),
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        assert verdict.codeload_throttle_signal_count == 3
        assert (
            verdict.assessments[0].state
            == EnumRunnerFleetHealthState.CODELOAD_THROTTLED
        )

    @pytest.mark.asyncio
    async def test_buildx_takes_precedence_over_codeload(self):
        """CRASH_LOOPING/ZOMBIE/OFFLINE outrank both new checks; buildx outranks codeload."""
        snapshot = _snapshot(
            expected_count=1,
            runners=(_fact("omninode-runner-1"),),
            buildx_available=False,
            codeload_throttle_signal_count=1,
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        assert (
            verdict.assessments[0].state
            == EnumRunnerFleetHealthState.BUILDX_UNAVAILABLE
        )


@pytest.mark.unit
class TestRunnerFleetHealthComputeWedged:
    @pytest.mark.asyncio
    async def test_fleet_wide_wedge_with_zero_busy(self):
        """Mirrors runner-monitor.sh's SILENT-WEDGE (OMN-13109): online + idle fleet-wide,
        but a job has been queued past the wedge-age threshold."""
        snapshot = _snapshot(
            expected_count=2,
            runners=(
                _fact("omninode-runner-1", github_busy=False),
                _fact("omninode-runner-2", github_busy=False),
            ),
            oldest_queued_job_age_seconds=900.0,
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        assert all(
            a.state == EnumRunnerFleetHealthState.WEDGED for a in verdict.assessments
        )
        assert verdict.wedged_count == 2

    @pytest.mark.asyncio
    async def test_no_wedge_when_queue_age_below_threshold(self):
        snapshot = _snapshot(
            expected_count=1,
            runners=(_fact("omninode-runner-1", github_busy=False),),
            oldest_queued_job_age_seconds=10.0,
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        assert verdict.assessments[0].state == EnumRunnerFleetHealthState.HEALTHY


@pytest.mark.unit
class TestRunnerFleetHealthComputeDeterminism:
    @pytest.mark.asyncio
    async def test_identical_input_produces_identical_classification(self):
        """Pure/deterministic: same snapshot facts always yield the same per-runner states."""
        snapshot = _snapshot(
            expected_count=2,
            runners=(
                _fact("omninode-runner-1", docker_restart_count=9),
                _fact("omninode-runner-2", github_status="offline"),
            ),
        )
        handler = HandlerRunnerFleetHealthEvaluate()
        verdict_a = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        verdict_b = await handler.handle(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
        assert [a.state for a in verdict_a.assessments] == [
            a.state for a in verdict_b.assessments
        ]
