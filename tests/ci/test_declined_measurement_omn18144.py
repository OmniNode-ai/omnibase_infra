# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18144 AC5: a declined measurement is never reported as a lane fault.

THE INCIDENT THIS REPLAYS (2026-09-19, release-train run ``35439999118``).

``omnibase_infra`` could not be released. The train refused with
``omnibase_infra SKIP 0.38.33 306f7430a3e7 lab_receipt_fail``, reason "the
compose dev lane recorded FAIL for this sha; checks not passing:
deployed_revision". The lane had not failed. Receipt artifact ``10581858331``
carries exactly one non-ok check, ``deployed_revision``, outcome
``indeterminate``, whose own evidence ends "This asserts nothing about the
lane": the guard declined to start its wait because the agent had ONE command
ahead of this one and the derived bound, 3003s, exceeded the 1620s the job
could watch for.

Live readback taken while the train was refusing: the lane carried
``306f7430a3e7`` at ``0.38.33``, ``:8085/ready`` answered 200, and the
containers were created at 09:39:10Z -- 554 seconds after the 09:29:56Z probe
that declined to look. The lane converged well inside the window the guard
declared it could not afford. Nothing measured it, and the train read the
silence as a failure.

TWO DEFECTS, BOTH PINNED HERE.

1. THE REFUSAL PREDICATE USED A WORST CASE TO DECIDE WHETHER TO MEASURE AT ALL.
   ``derived_wait_bound_seconds`` is ``commands_ahead x mean service + the
   lane's own grant + a poll margin`` -- an honest upper bound, and correct as
   the receipt EVIDENCE that AC5 requires it to carry. It is the wrong
   predicate for "is this wait worth starting", because it charges the window
   for the lane's entire post-acceptance grant (1500s), which this window was
   never sized to cover: the job ceiling allocates the settle budget
   SEPARATELY, to the probe step that follows (``2700 - 1500 - 900 - 120``, see
   ``lane_settle_budget.STEP_OVERHEAD_SECONDS``). Removing that grant term was
   right and it stays removed.

   **SUPERSEDED IN PART, same day.** This file originally went on to argue that
   the predicate is therefore the time to REACH this command, and two tests
   here asserted it. That was wrong, and lane
   ``post-merge-lab-verify-reds-diag-1230`` measured why within the hour: the
   mean service time is ACCEPT-TO-COMPLETION, so reaching acceptance still
   leaves this command's own full service before the lane carries the sha. At
   ``commands_ahead = 1`` the reach predicate starts a watch that cannot
   converge and holds the single verify runner for the whole window to reach
   the same INDETERMINATE. The predicate is now the CONVERGENCE horizon, one
   measured service per place in line, and an empty queue never refuses --
   see ``tests/ci/test_convergence_horizon_omn18144.py``, which carries the
   measured table and owns this half. The two tests below were rewritten to
   assert the corrected behaviour rather than deleted, so the refuted claim
   stays visible next to what replaced it.

2. THE TRAIN COLLAPSED "THE LANE FAILED" INTO "NOTHING MEASURED THE LANE".
   AC5's fourth bullet requires that a queue the lane did not cause is a
   statement about the RUN, never a FAIL. The receipt delivers that at the
   CHECK level -- ``outcome: indeterminate`` -- and the train then read
   ``result`` alone and called it ``lab_receipt_fail``. The receipt model is
   not weakened here and its ``result`` stays two-valued on purpose (see
   ``lab_pass_receipt.EnumLabPassResult``): an indeterminate check keeps the
   receipt non-PASS, so the delivery gate of rule 24(b) stays shut exactly as
   before. What changes is only the NAME the train gives its refusal, which is
   the same distinction ``LAB_RECEIPT_PENDING`` was split from
   ``LAB_RECEIPT_ABSENT`` to make, for the same reason: collapsing them told a
   reader a healthy system was broken.

NEITHER DEFECT'S FIX CAN OPEN A GATE. Fix 1 can only turn a refusal into a real
measurement or into the same INDETERMINATE later; ``test_a_started_wait_can_
never_manufacture_a_fail`` pins that it can never produce a FAIL that the
refusal would not have. Fix 2 changes a reason string, and both reasons are
non-None, so the train skips the cut in both cases -- pinned by
``test_an_indeterminate_receipt_still_refuses_the_cut``.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci import release_train as rt
from scripts.ci.check_dev_lane_staleness import (
    Ancestry,
    EnumConvergenceOutcome,
    LaneRevision,
    ModelAcceptanceProbe,
    ModelAgentAcceptance,
    ModelQueueFacts,
    queue_exceeds_bound,
    run_convergence_wait,
)

pytestmark = pytest.mark.unit

# --------------------------------------------------------------------------- #
# The 2026-09-19 numbers, every one of them read off receipt artifact          #
# 10581858331 or off the workflow that produced it. None is chosen.            #
# --------------------------------------------------------------------------- #

#: The merge the release train could not cut.
MERGE_SHA = "306f7430a3e7a233f8e524d01c9f5907267c9b97"
#: What the lane was still carrying when the guard looked.
STALE_SHA = "e7f65bb8e50a1122334455667788990011223344"
CORRELATION_ID = "e8e9a169-d7c9-4327-b567-a606a0ad7b40"

#: ``started_at`` on the receipt.
T0 = datetime(2026, 9, 19, 9, 29, 55, tzinfo=UTC)

#: ``--wait-timeout 25m``, the lane's post-acceptance grant.
DECLARED = timedelta(minutes=25)
#: ``--poll-interval 60s``.
POLL = timedelta(seconds=60)
#: ``CONVERGE_WALL_CLOCK`` on the day: the 45-minute ceiling less the declared
#: settle budget, the reserved tail and the step overhead.
WALL_CLOCK = timedelta(seconds=45 * 60 - 900 - 120 - 60)
#: The agent's observed mean over its 10 completed jobs.
SERVICE_SECONDS = 1443.3
#: Containers were created 09:39:10Z against a 09:29:56Z probe.
ACTUAL_CONVERGENCE_SECONDS = 554


def _lane(revision: str) -> LaneRevision:
    return LaneRevision(
        revision=revision,
        compose_project="omnibase-infra",
        build_source="workspace",
        state="running",
    )


def _ancestry(observed: str) -> Ancestry:
    if observed == MERGE_SHA:
        return Ancestry(
            relation="identical", commits_ahead=0, observed_on_branch=True, branch="dev"
        )
    return Ancestry(
        relation="ancestor", commits_ahead=1, observed_on_branch=True, branch="dev"
    )


def _queue(commands_ahead: int) -> ModelQueueFacts:
    return ModelQueueFacts(
        commands_ahead=commands_ahead,
        mean_service_time_seconds=SERVICE_SECONDS,
        service_sample_size=10,
        in_flight_correlation_id="1f2e3d4c-5555-4b3a-8c1d-9e0f1a2b3c4d",
        unread_reason="",
        source="http://host.docker.internal:8098/queue",
    )


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _wait(
    *,
    clock: _Clock,
    queue: ModelQueueFacts,
    converges_after: int | None = None,
    accepted_after: int | None = None,
):
    """Run the guard against a lane that converges (or does not) on a schedule."""
    start = clock.now

    def read_lane() -> LaneRevision:
        if (
            converges_after is not None
            and (clock.now - start).total_seconds() >= converges_after
        ):
            return _lane(MERGE_SHA)
        return _lane(STALE_SHA)

    def resolve_acceptance() -> ModelAcceptanceProbe:
        if (
            accepted_after is not None
            and (clock.now - start).total_seconds() >= accepted_after
        ):
            return ModelAcceptanceProbe(
                acceptance=ModelAgentAcceptance(
                    correlation_id=CORRELATION_ID,
                    accepted_at=start + timedelta(seconds=accepted_after),
                    source="http://host.docker.internal:8098/job/x",
                ),
                reason="",
            )
        return ModelAcceptanceProbe(
            acceptance=None,
            reason=(
                "the deploy agent reports no job for correlation "
                f"{CORRELATION_ID} (HTTP 404). The command has not been handed "
                "to the agent yet, so the lane's budget has not started."
            ),
        )

    return run_convergence_wait(
        expected_revision=MERGE_SHA,
        read_lane=read_lane,
        resolve_ancestry=_ancestry,
        resolve_acceptance=resolve_acceptance,
        declared_budget=DECLARED,
        wall_clock=WALL_CLOCK,
        poll_interval=POLL,
        clock=clock,
        sleep=clock.sleep,
        resolve_queue=lambda: queue,
    )


# --------------------------------------------------------------------------- #
# Defect 1 -- the refusal predicate.                                           #
# --------------------------------------------------------------------------- #
class TestTheMeasurementThatWasDeclined:
    """Run 35439999118, replayed with the lane converging when it really did."""

    def test_the_incident_is_refused_and_is_not_this_predicates_to_fix(
        self,
    ) -> None:
        """REWRITTEN. This asserted the opposite until the correction.

        The incident converged 554 seconds after the probe at
        ``commands_ahead = 1`` -- far too fast for a queue drain plus a full
        service at a 1443s mean. Read with the supersession finding on the same
        lane, the in-flight command was building a DESCENDANT, so the lane
        picked up this sha when that job completed. A queue-depth predicate
        cannot see that and should not pretend to: the resolution is the
        agent's supersession signal reaching CI (OMN-18816), not a wider
        window. So this case is refused, and refused for the honest reason.
        """
        assert queue_exceeds_bound(
            _queue(1),
            lane_budget_seconds=int(DECLARED.total_seconds()),
            wall_clock_seconds=int(WALL_CLOCK.total_seconds()),
            margin_seconds=int(POLL.total_seconds()),
        )

    def test_the_predicate_is_the_convergence_horizon_not_the_reach(self) -> None:
        """REWRITTEN. 1443 + 60 fits in 1620; 1443 + 1443 + 60 does not.

        The reach fits and the horizon does not, so a predicate reading the
        reach would start this watch. It is the horizon that decides.
        """
        facts = _queue(1)
        margin = int(POLL.total_seconds())
        window = int(WALL_CLOCK.total_seconds())
        reach = facts.reach_bound_seconds(margin_seconds=margin)
        horizon = facts.convergence_horizon_seconds(margin_seconds=margin)
        assert reach is not None and horizon is not None
        assert reach <= window < horizon
        assert queue_exceeds_bound(
            facts,
            lane_budget_seconds=int(DECLARED.total_seconds()),
            wall_clock_seconds=window,
            margin_seconds=margin,
        )

    def test_a_queue_the_window_cannot_reach_is_still_refused(self) -> None:
        """AC4 survives. Two ahead is 2947s to reach, against 1620s."""
        reason = queue_exceeds_bound(
            _queue(2),
            lane_budget_seconds=int(DECLARED.total_seconds()),
            wall_clock_seconds=int(WALL_CLOCK.total_seconds()),
            margin_seconds=int(POLL.total_seconds()),
        )
        assert reason
        assert "2 command(s) ahead" in reason

    def test_a_refused_run_is_indeterminate_and_names_the_depth(self) -> None:
        clock = _Clock(T0)
        result = _wait(clock=clock, queue=_queue(2))
        assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
        assert "2 command(s) ahead" in result.reason

    def test_the_evidence_bound_is_unchanged_and_still_the_worst_case(self) -> None:
        """AC5 requires ``derived_wait_bound_s`` in the evidence. 3003s, as recorded."""
        bound = _queue(1).derived_wait_bound_seconds(
            lane_budget_seconds=int(DECLARED.total_seconds()),
            margin_seconds=int(POLL.total_seconds()),
        )
        assert bound == 3003

    @pytest.mark.parametrize("commands_ahead", [0, 1, 2, 5])
    def test_a_started_wait_can_never_manufacture_a_fail(
        self, commands_ahead: int
    ) -> None:
        """The safety property: this change cannot blame the lane.

        A FAIL needs acceptance established AND its full 1500s grant spent. The
        window is 1620s, so that is only reachable when the agent accepted at
        once -- which is not a queue case at all, and was always a FAIL. Every
        run that never converges stops on its wall clock, which is
        INDETERMINATE.
        """
        clock = _Clock(T0)
        result = _wait(clock=clock, queue=_queue(commands_ahead), converges_after=None)
        assert result.outcome is not EnumConvergenceOutcome.FAIL


# --------------------------------------------------------------------------- #
# Defect 2 -- what the train calls a declined measurement.                     #
# --------------------------------------------------------------------------- #
def _receipt(*, checks: tuple[Any, ...]) -> Any:
    lab = rt.lab_pass_receipt
    all_ok = all(c.ok for c in checks)
    return lab.ModelLabPassReceipt(
        sha=MERGE_SHA,
        lane=lab.EnumLabLane.COMPOSE_DEV,
        started_at=T0,
        finished_at=T0 + timedelta(seconds=1),
        result=lab.EnumLabPassResult.PASS if all_ok else lab.EnumLabPassResult.FAIL,
        checks=checks,
        agent_command_id=CORRELATION_ID,
    )


def _ok_check(name: str) -> Any:
    return rt.lab_pass_receipt.ModelLabPassCheck(
        name=name, ok=True, evidence=f"GET /{name} -> 200"
    )


def _indeterminate_check(name: str) -> Any:
    return rt.lab_pass_receipt.ModelLabPassCheck.indeterminate_check(
        name,
        "INDETERMINATE: the deploy agent has 1 command(s) ahead of this one. "
        "This asserts nothing about the lane; the receipt is still non-PASS.",
    )


def _failed_check(name: str) -> Any:
    return rt.lab_pass_receipt.ModelLabPassCheck(
        name=name, ok=False, evidence=f"GET /{name} -> 503"
    )


def _classify(receipt: Any) -> tuple[Any, str]:
    return rt.classify_lab_receipt(
        "omnibase_infra",
        MERGE_SHA,
        list_artifacts=lambda repo, name: [
            {"id": 1, "created_at": "2026-09-19T09:29:56Z"}
        ],
        download_receipt=lambda repo, artifact_id: receipt,
    )


class TestADeclinedMeasurementIsNotALaneFault:
    def test_an_all_indeterminate_receipt_is_not_lab_receipt_fail(self) -> None:
        """The exact shape of artifact 10581858331."""
        reason, detail = _classify(
            _receipt(
                checks=(
                    _indeterminate_check("deployed_revision"),
                    _ok_check("ready_main"),
                    _ok_check("ready_effects"),
                )
            )
        )
        assert reason is not rt.EnumTrainReason.LAB_RECEIPT_FAIL
        assert reason is rt.EnumTrainReason.LAB_RECEIPT_INDETERMINATE
        assert "deployed_revision" in detail

    def test_a_genuinely_failed_check_is_still_lab_receipt_fail(self) -> None:
        """The distinction has to cut both ways or it is just a rename."""
        reason, _ = _classify(
            _receipt(checks=(_failed_check("ready_main"), _ok_check("ready_effects")))
        )
        assert reason is rt.EnumTrainReason.LAB_RECEIPT_FAIL

    def test_one_real_failure_among_indeterminates_is_still_a_fail(self) -> None:
        """A lane fault is not laundered by an unmeasured check beside it."""
        reason, _ = _classify(
            _receipt(
                checks=(
                    _indeterminate_check("deployed_revision"),
                    _failed_check("ready_main"),
                )
            )
        )
        assert reason is rt.EnumTrainReason.LAB_RECEIPT_FAIL

    def test_an_indeterminate_receipt_still_refuses_the_cut(self) -> None:
        """Nothing opens. The premise is unproven either way."""
        reason, _ = _classify(
            _receipt(
                checks=(
                    _indeterminate_check("deployed_revision"),
                    _ok_check("ready_main"),
                )
            )
        )
        assert reason is not None

    def test_a_pass_receipt_is_still_the_only_thing_that_cuts(self) -> None:
        reason, detail = _classify(
            _receipt(checks=(_ok_check("deployed_revision"), _ok_check("ready_main")))
        )
        assert reason is None
        assert "PASS" in detail
