# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18144: the refusal predicate is the CONVERGENCE horizon, not the reach.

THIS CORRECTS A CHANGE THAT MERGED FOUR HOURS EARLIER THE SAME DAY.

``omnibase_infra#3823`` (squash ``d7530fb0093f``, merged 2026-09-19T12:33:02Z)
replaced the refusal predicate with :meth:`ModelQueueFacts.reach_bound_seconds`
-- the time for the agent to REACH this command -- on the argument that
reaching acceptance is what makes a measurement possible. That argument is
wrong, and lane ``post-merge-lab-verify-reds-diag-1230`` measured why within
the hour.

``mean_service_time_seconds`` is the agent's ACCEPT-TO-COMPLETION time. So
reaching acceptance at ``commands_ahead x mean`` leaves this command's OWN full
service before the lane carries the sha. The reach is not the horizon; it is
the horizon minus one service time. Against the five real cases on record:

| case                  | ahead | mean   | reach | reach + own | window |
|-----------------------|-------|--------|-------|-------------|--------|
| #3815 ``5425a634``    |     2 | 1360.5 |  2781 |        4142 |   1620 |
| #3816 ``d4b668c2``    |     2 | 1360.5 |  2781 |        4142 |   1620 |
| #3817 ``61b55f08``    |     1 | 1308.0 |  1368 |        2676 |   1620 |
| #3820 ``07d1d4ac``    |     1 | 1308.0 |  1368 |        2676 |   1620 |
| incident ``306f7430`` |     1 | 1443.3 |  1503 |        2947 |   1620 |

Not one of them can see convergence inside the window. Under the merged reach
predicate the three at ``commands_ahead = 1`` nonetheless START a watch, hold
the single host-201 verify runner for the full 1620s, and arrive at the same
INDETERMINATE the refusal would have written in about a second -- while the
next merge's guard waits behind them. Two of the four runs measured that
morning sat at exactly that queue position, so it is the common case, not an
edge. Freeing the runner rather than spending it on a foregone conclusion is
the trade AC4 asks for and the trade the original refusal made on purpose.

WHAT IS KEPT FROM THE MERGED CHANGE, AND WHY IT WAS STILL RIGHT
---------------------------------------------------------------
The term that came OUT was the lane's 1500s post-acceptance grant
(``--wait-timeout``), and it should stay out: it is an arbitrary declared
ceiling, it is funded separately out of the job ceiling for the probe step that
follows, and it is not an estimate of anything this command will actually
spend. What replaces it here is this command's own MEASURED service time, the
same number already used for every command ahead of it. So the horizon is
``queue_position_at_start x mean + margin`` -- this command is Nth in line, each
takes a mean service, convergence is at N means.

AN EMPTY QUEUE NEVER REFUSES, AND THAT IS LOAD-BEARING
-------------------------------------------------------
With ``commands_ahead = 0`` there is no queue to decline on, and the refusal's
own message is about a queue ahead of this run. It is also the case that
normally PASSES -- the 12:33Z run carried ``commands_ahead=0`` with a 1360s mean
and converged -- and a horizon that charged it a full mean would refuse it
whenever the agent's mean drifted above the window, silently turning the
healthy path into a non-PASS. ``test_an_empty_queue_never_refuses`` pins that
across mean service times far beyond the window.

WHAT THIS DOES NOT CLAIM TO FIX
--------------------------------
The 09:29Z incident converged 554 seconds after the probe at
``commands_ahead = 1``. That is far too fast for a queue drain plus a full
service, and read with the same lane's supersession finding the in-flight
command was almost certainly building a DESCENDANT, so the lane picked up the
sha when that job completed. The predicate is the wrong instrument for that
class: the resolution is the agent's supersession signal reaching CI, which is
OMN-18816 and is not touched here. This change makes the predicate honest; it
does not make that incident PASS, and it should not pretend to.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.check_dev_lane_staleness import (
    Ancestry,
    EnumConvergenceOutcome,
    LaneRevision,
    ModelAcceptanceProbe,
    ModelQueueFacts,
    queue_exceeds_bound,
    run_convergence_wait,
)

pytestmark = pytest.mark.unit

MERGE_SHA = "306f7430a3e7a233f8e524d01c9f5907267c9b97"
STALE_SHA = "e7f65bb8e50a1122334455667788990011223344"
CORRELATION_ID = "e8e9a169-d7c9-4327-b567-a606a0ad7b40"
T0 = datetime(2026, 9, 19, 9, 29, 55, tzinfo=UTC)

DECLARED = timedelta(minutes=25)
POLL = timedelta(seconds=60)
WALL_CLOCK = timedelta(seconds=45 * 60 - 900 - 120 - 60)
WINDOW = int(WALL_CLOCK.total_seconds())
LANE_BUDGET = int(DECLARED.total_seconds())
MARGIN = int(POLL.total_seconds())

#: The four runs lane ``post-merge-lab-verify-reds-diag-1230`` measured on
#: 2026-09-19, plus the incident, as (label, commands_ahead, mean service).
MEASURED_CASES = [
    ("#3815 5425a634", 2, 1360.5),
    ("#3816 d4b668c2", 2, 1360.5),
    ("#3817 61b55f08", 1, 1308.0),
    ("#3820 07d1d4ac", 1, 1308.0),
    ("incident 306f7430", 1, 1443.3),
]


def _queue(commands_ahead: int, mean: float = 1360.5) -> ModelQueueFacts:
    return ModelQueueFacts(
        commands_ahead=commands_ahead,
        mean_service_time_seconds=mean,
        service_sample_size=10,
        in_flight_correlation_id="1f2e3d4c-5555-4b3a-8c1d-9e0f1a2b3c4d",
        unread_reason="",
        source="http://host.docker.internal:8098/queue",
    )


def _refuses(facts: ModelQueueFacts) -> str:
    return queue_exceeds_bound(
        facts,
        lane_budget_seconds=LANE_BUDGET,
        wall_clock_seconds=WINDOW,
        margin_seconds=MARGIN,
    )


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class TestTheHorizonIsReachPlusThisCommandsOwnService:
    @pytest.mark.parametrize(
        ("label", "commands_ahead", "mean"),
        MEASURED_CASES,
        ids=[c[0] for c in MEASURED_CASES],
    )
    def test_every_measured_case_is_refused(
        self, label: str, commands_ahead: int, mean: float
    ) -> None:
        """None of the five can converge in the window, so none starts a watch."""
        assert _refuses(_queue(commands_ahead, mean))

    def test_the_horizon_charges_one_service_per_place_in_line(self) -> None:
        facts = _queue(2, 1360.5)
        assert facts.convergence_horizon_seconds(margin_seconds=MARGIN) == round(
            3 * 1360.5 + MARGIN
        )

    def test_the_horizon_is_the_reach_plus_one_more_service(self) -> None:
        """Stated as the relation, so the two cannot drift apart silently.

        Within a second: each total is rounded once, so a fractional mean can
        put the two roundings on opposite sides.
        """
        facts = _queue(2, 1360.5)
        reach = facts.reach_bound_seconds(margin_seconds=MARGIN)
        horizon = facts.convergence_horizon_seconds(margin_seconds=MARGIN)
        assert reach is not None and horizon is not None
        assert abs(horizon - (reach + 1360.5)) <= 1

    def test_the_lane_grant_is_not_a_term_in_the_predicate(self) -> None:
        """The 1500s wait-timeout stays out; #3823 was right to remove it.

        Same queue, same window, wildly different declared lane grant: the
        verdict must not move, because the grant is funded separately out of
        the job ceiling for the probe step that follows.
        """
        facts = _queue(1, 700.0)
        verdicts = {
            bool(
                queue_exceeds_bound(
                    facts,
                    lane_budget_seconds=grant,
                    wall_clock_seconds=WINDOW,
                    margin_seconds=MARGIN,
                )
            )
            for grant in (0, 900, 1500, 3600)
        }
        assert verdicts == {False}


class TestAnEmptyQueueNeverRefuses:
    """The healthy path. A refusal is about a QUEUE, and there is none."""

    @pytest.mark.parametrize("mean", [60.0, 1360.5, 1620.0, 5000.0])
    def test_an_empty_queue_never_refuses(self, mean: float) -> None:
        assert not _refuses(_queue(0, mean))

    def test_the_12_33z_shape_still_watches(self) -> None:
        """commands_ahead=0 at a 1360s mean -- the run that converged."""
        assert not _refuses(_queue(0, 1360.5))


class TestTheRefusalStillExplainsItself:
    def test_it_names_the_depth_the_means_and_the_window(self) -> None:
        reason = _refuses(_queue(2, 1360.5))
        assert "2 command(s) ahead" in reason
        assert "1360s mean service time" in reason
        assert str(WINDOW) in reason

    def test_a_refused_run_is_indeterminate_not_a_fail(self) -> None:
        """Unchanged, and the property that keeps this off the lane's record."""
        clock = _Clock(T0)

        def read_lane() -> LaneRevision:
            return LaneRevision(
                revision=STALE_SHA,
                compose_project="omnibase-infra",
                build_source="workspace",
                state="running",
            )

        result = run_convergence_wait(
            expected_revision=MERGE_SHA,
            read_lane=read_lane,
            resolve_ancestry=lambda observed: Ancestry(
                relation="ancestor",
                commits_ahead=1,
                observed_on_branch=True,
                branch="dev",
            ),
            resolve_acceptance=lambda: ModelAcceptanceProbe(
                acceptance=None,
                reason=(
                    "the deploy agent reports no job for correlation "
                    f"{CORRELATION_ID} (HTTP 404). The command has not been "
                    "handed to the agent yet, so the lane's budget has not "
                    "started."
                ),
            ),
            declared_budget=DECLARED,
            wall_clock=WALL_CLOCK,
            poll_interval=POLL,
            clock=clock,
            sleep=clock.sleep,
            resolve_queue=lambda: _queue(2, 1360.5),
        )
        assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
        assert "2 command(s) ahead" in result.reason
        # And it cost the runner nothing: the refusal is answered up front.
        assert result.waited.total_seconds() == 0


class TestTheEvidenceBoundIsUntouched:
    def test_the_worst_case_in_the_receipt_still_carries_the_lane_grant(self) -> None:
        """AC5 requires ``derived_wait_bound_s``; this change does not move it."""
        assert (
            _queue(1, 1443.3).derived_wait_bound_seconds(
                lane_budget_seconds=LANE_BUDGET, margin_seconds=MARGIN
            )
            == 3003
        )

    def test_an_unreadable_queue_still_falls_back_rather_than_refusing(self) -> None:
        assert not _refuses(
            ModelQueueFacts.unread("the queue endpoint refused the connection")
        )
