# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18144 -- the verify job's wait is bounded by queue position, not a clock.

WHAT THESE PIN
--------------
OMN-18573 anchored the LANE's convergence budget to the deploy agent's
acceptance of the command. That fixed everything after acceptance and left the
wait BEFORE it bounded by nothing but the CI job's own ceiling.

Measured 2026-09-18 (report ``dev-lane-agent-dispatch-diag-1700``; evidence on
OMN-18143 and OMN-18436): four runtime merges landed inside 33 minutes against
a deploy agent servicing roughly 32 minutes per command. The fourth merge's
command sat at control-topic offset 293 with a consumer lag of 2 -- one job in
flight and two unconsumed, so third in line -- its ``/job/<correlation_id>``
answered 404 for the whole verify window, and the compose-dev receipt for
``11e8951f`` was minted FAIL with ``deployed_revision`` INDETERMINATE. The lane
was healthy and strictly monotone throughout, and a manual re-run after the
agent caught up passed first time.

THE SHAPES
----------
* two commands ahead at ~24 min each against 1680s of affordable clock ->
  INDETERMINATE **naming the depth**, taken in seconds rather than after the
  whole 28-minute watch, and never a FAIL;
* nothing ahead -> byte-for-byte today's behaviour;
* an unreadable queue -> today's behaviour, with the evidence saying in words
  that the queue could not be read;
* an ALREADY-ACCEPTED command -> the queue is spent and the branch is
  unreachable, so every run whose command the agent has taken behaves exactly
  as it did before this change.

The four AC5 fields ride on every verdict, PASS included: a receipt that does
not record what the queue looked like cannot be used to check that a later
non-PASS was really the queue's doing.
"""

from __future__ import annotations

import json
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
    ModelAgentAcceptance,
    ModelQueueFacts,
    convergence_evidence,
    queue_exceeds_bound,
    read_agent_queue,
    run_convergence_wait,
)

pytestmark = pytest.mark.unit

MERGE_SHA = "11e8951f0c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f"
STALE_SHA = "843fe808cc7891a1b2c3d4e5f60718293a4b5c6d"
CORRELATION_ID = "0a3d0f1e-1111-4c2a-9f3b-2a6c8d4e5f60"
T0 = datetime(2026, 9, 18, 17, 0, 0, tzinfo=UTC)

DECLARED = timedelta(minutes=25)
#: The job's 45-minute ceiling less the declared settle budget and the reserved
#: tail. Not a number this change may move.
WALL_CLOCK = timedelta(seconds=45 * 60 - 900 - 120)
POLL = timedelta(seconds=60)

#: The agent's measured service time on 2026-09-18.
SERVICE_SECONDS = 24 * 60


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
        relation="ancestor", commits_ahead=0, observed_on_branch=True, branch="dev"
    )


def _queue(
    commands_ahead: int, *, service_seconds: float | None = SERVICE_SECONDS
) -> ModelQueueFacts:
    return ModelQueueFacts(
        commands_ahead=commands_ahead,
        mean_service_time_seconds=service_seconds,
        service_sample_size=6 if service_seconds else 0,
        in_flight_correlation_id="7f6e5d4c-2222-4b3a-8c1d-9e0f1a2b3c4d",
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
    converges_at: datetime | None = None,
    acceptance_at: datetime | None = None,
    wall_clock: timedelta = WALL_CLOCK,
):
    def read_lane() -> LaneRevision:
        if converges_at is not None and clock.now >= converges_at:
            return _lane(MERGE_SHA)
        return _lane(STALE_SHA)

    def resolve_acceptance() -> ModelAcceptanceProbe:
        if acceptance_at is not None and clock.now >= acceptance_at:
            return ModelAcceptanceProbe(
                acceptance=ModelAgentAcceptance(
                    correlation_id=CORRELATION_ID,
                    accepted_at=acceptance_at,
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
        wall_clock=wall_clock,
        poll_interval=POLL,
        clock=clock,
        sleep=clock.sleep,
        resolve_queue=lambda: queue,
    )


class TestTheMeasuredTimeline:
    """The 2026-09-18 run, replayed."""

    def test_a_merge_third_in_line_is_indeterminate_naming_the_depth(self) -> None:
        clock = _Clock(T0)
        result = _wait(clock=clock, queue=_queue(2))

        assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
        assert "2 command(s) ahead" in result.reason
        assert "number 3 in line" in result.reason

    def test_it_is_not_a_fail_because_the_lane_is_not_what_went_wrong(self) -> None:
        """The receipt for 11e8951f read FAIL against a healthy, monotone lane."""
        result = _wait(clock=_Clock(T0), queue=_queue(2))
        assert result.outcome is not EnumConvergenceOutcome.FAIL

    def test_the_refusal_is_taken_in_seconds_not_after_the_whole_ceiling(self) -> None:
        """AC4: the single physical verify runner is not held for 28 minutes to
        reach a verdict that was knowable at the start.

        Measured at the parent commit on this same loop and this same
        timeline: 0h28m watched, INDETERMINATE, no depth named.
        """
        clock = _Clock(T0)
        result = _wait(clock=clock, queue=_queue(2))

        assert result.waited == timedelta(0)
        assert clock.now == T0, (
            "the wait must not be entered at all; holding the runner open "
            "serializes the next merge's guard behind this one"
        )

    def test_the_queue_facts_survive_onto_the_result(self) -> None:
        result = _wait(clock=_Clock(T0), queue=_queue(2))
        assert result.queue is not None
        assert result.queue.commands_ahead == 2
        assert result.queue.queue_position_at_start == 3


class TestNoBehaviourChangeWhereTheQueueIsEmpty:
    def test_nothing_ahead_still_waits_and_still_converges(self) -> None:
        clock = _Clock(T0)
        result = _wait(
            clock=clock,
            queue=_queue(0),
            acceptance_at=T0,
            converges_at=T0 + timedelta(minutes=8),
        )
        assert result.outcome is EnumConvergenceOutcome.OK

    def test_a_lane_that_had_its_whole_budget_and_missed_is_still_a_fail(self) -> None:
        """The FAIL direction is not widened away. A statement about the lane
        stays a statement about the lane."""
        clock = _Clock(T0)
        result = _wait(clock=clock, queue=_queue(0), acceptance_at=T0)
        assert result.outcome is EnumConvergenceOutcome.FAIL

    def test_an_already_accepted_command_ignores_the_queue_entirely(self) -> None:
        """Once the agent has taken the command the queue ahead is spent.

        The refusal branch is scoped to an unestablished acceptance precisely
        so that every run whose command was already accepted behaves as it did
        before this change -- even one whose queue reading is nonsense.
        """
        clock = _Clock(T0)
        result = _wait(
            clock=clock,
            queue=_queue(99),
            acceptance_at=T0,
            converges_at=T0 + timedelta(minutes=5),
        )
        assert result.outcome is EnumConvergenceOutcome.OK

    def test_an_already_converged_lane_is_never_refused_over_a_queue(self) -> None:
        clock = _Clock(T0)
        result = _wait(clock=clock, queue=_queue(9), converges_at=T0)
        assert result.outcome is EnumConvergenceOutcome.OK


class TestUnreadableQueueFallsBackByName:
    def test_an_unread_queue_does_not_refuse_the_wait(self) -> None:
        clock = _Clock(T0)
        result = _wait(
            clock=clock,
            queue=ModelQueueFacts.unread("this agent predates the queue endpoint"),
            acceptance_at=T0,
            converges_at=T0 + timedelta(minutes=6),
        )
        assert result.outcome is EnumConvergenceOutcome.OK, (
            "the change may only ever make a verdict better informed, never "
            "harder to obtain"
        )

    def test_the_evidence_says_the_queue_was_unread_rather_than_empty(self) -> None:
        facts = ModelQueueFacts.unread("HTTP 404 from the agent")
        clause = facts.evidence_clause(lane_budget_seconds=1500, margin_seconds=60)
        assert "commands_ahead=UNREAD" in clause
        assert "HTTP 404" in clause
        assert "commands_ahead=0" not in clause

    def test_a_queue_with_no_service_time_yet_cannot_derive_a_bound(self) -> None:
        """A bound derived from an absent service time is a guess in a number's
        clothes, so there is none and the wait proceeds."""
        facts = _queue(5, service_seconds=None)
        assert (
            facts.derived_wait_bound_seconds(
                lane_budget_seconds=1500, margin_seconds=60
            )
            is None
        )
        assert not queue_exceeds_bound(
            facts, lane_budget_seconds=1500, wall_clock_seconds=100, margin_seconds=60
        )

    def test_facts_carry_exactly_one_of_a_count_and_a_reason(self) -> None:
        with pytest.raises(ValueError, match="EXACTLY one"):
            ModelQueueFacts(
                commands_ahead=2,
                mean_service_time_seconds=10.0,
                service_sample_size=1,
                in_flight_correlation_id=None,
                unread_reason="and a reason",
                source="",
            )


class TestTheDerivedBound:
    def test_it_is_the_queue_wait_plus_the_lane_budget_plus_the_poll_margin(
        self,
    ) -> None:
        """Every term declared or measured; none invented."""
        bound = _queue(2).derived_wait_bound_seconds(
            lane_budget_seconds=1500, margin_seconds=60
        )
        assert bound == 2 * SERVICE_SECONDS + 1500 + 60

    def test_a_bound_inside_the_clock_does_not_refuse(self) -> None:
        assert not queue_exceeds_bound(
            _queue(1, service_seconds=60),
            lane_budget_seconds=1500,
            wall_clock_seconds=2520,
            margin_seconds=60,
        )

    def test_the_refusal_names_every_number_it_used(self) -> None:
        reason = queue_exceeds_bound(
            _queue(2),
            lane_budget_seconds=1500,
            wall_clock_seconds=int(WALL_CLOCK.total_seconds()),
            margin_seconds=60,
        )
        assert "2 command(s) ahead" in reason
        assert "1440s mean service time" in reason
        assert str(int(WALL_CLOCK.total_seconds())) in reason


class TestReceiptEvidenceFields:
    """AC5: four named fields, on every verdict."""

    @pytest.mark.parametrize("converged", [True, False])
    def test_the_four_fields_are_present_on_both_verdicts(
        self, converged: bool
    ) -> None:
        clause = _queue(2).evidence_clause(lane_budget_seconds=1500, margin_seconds=60)
        evidence = convergence_evidence(
            lane=_lane(MERGE_SHA if converged else STALE_SHA),
            expected_revision=MERGE_SHA,
            ancestry=_ancestry(MERGE_SHA if converged else STALE_SHA),
            waited=timedelta(minutes=3),
            converged=converged,
            queue_clause=clause,
        )
        for field in (
            "queue_position_at_start=3",
            "commands_ahead=2",
            "derived_wait_bound_s=",
            "mean_service_time_s=1440.0",
        ):
            assert field in evidence, f"{field} missing from: {evidence}"

    def test_the_fields_are_present_on_an_indeterminate_verdict(self) -> None:
        clause = _queue(2).evidence_clause(lane_budget_seconds=1500, margin_seconds=60)
        evidence = convergence_evidence(
            lane=_lane(STALE_SHA),
            expected_revision=MERGE_SHA,
            ancestry=_ancestry(STALE_SHA),
            waited=timedelta(0),
            converged=False,
            indeterminate_reason="2 command(s) ahead of this one",
            queue_clause=clause,
        )
        assert "INDETERMINATE" in evidence
        assert "commands_ahead=2" in evidence

    def test_the_evidence_stays_one_shell_safe_line(self) -> None:
        """It is written to GITHUB_OUTPUT and re-read as one receipt field."""
        clause = _queue(2).evidence_clause(lane_budget_seconds=1500, margin_seconds=60)
        evidence = convergence_evidence(
            lane=_lane(STALE_SHA),
            expected_revision=MERGE_SHA,
            ancestry=_ancestry(STALE_SHA),
            waited=timedelta(0),
            converged=False,
            indeterminate_reason="queued behind two commands",
            queue_clause=clause,
        )
        assert "\n" not in evidence
        assert not set(evidence) & {"`", "$", '"'}


class TestReadingTheAgentSurface:
    def _opener(self, status: int, body: str):  # type: ignore[no-untyped-def]
        def _fetch(url: str, timeout: float) -> tuple[int, str]:
            return status, body

        return _fetch

    def test_a_well_formed_payload_reads(self) -> None:
        facts = read_agent_queue(
            "http://agent:8098",
            opener=self._opener(
                200,
                json.dumps(
                    {
                        "commands_ahead": 3,
                        "store_depth": 1,
                        "control_topic_lag": 2,
                        "mean_service_time_seconds": 1920.0,
                        "service_sample_size": 4,
                        "in_flight_correlation_id": CORRELATION_ID,
                    }
                ),
            ),
        )
        assert facts.commands_ahead == 3
        assert facts.queue_position_at_start == 4
        assert facts.mean_service_time_seconds == 1920.0

    def test_a_404_is_an_older_agent_and_says_so(self) -> None:
        """The agent self-updates from dev, so this is the normal state between
        a merge and its next re-exec -- not an outage."""
        facts = read_agent_queue(
            "http://agent:8098", opener=self._opener(404, "not found")
        )
        assert facts.commands_ahead is None
        assert "predates the queue endpoint" in facts.unread_reason

    def test_a_depth_the_agent_reports_unknown_stays_unknown(self) -> None:
        facts = read_agent_queue(
            "http://agent:8098",
            opener=self._opener(
                200,
                json.dumps(
                    {
                        "commands_ahead": None,
                        "control_topic_lag_reason": "no partition assignment yet",
                    }
                ),
            ),
        )
        assert facts.commands_ahead is None
        assert "no partition assignment" in facts.unread_reason

    @pytest.mark.parametrize(
        "body", ["not json", json.dumps([1, 2]), json.dumps({"commands_ahead": "two"})]
    )
    def test_an_unreadable_body_is_unread_never_zero(self, body: str) -> None:
        facts = read_agent_queue("http://agent:8098", opener=self._opener(200, body))
        assert facts.commands_ahead is None
        assert facts.unread_reason

    def test_a_transport_failure_does_not_raise(self) -> None:
        """This read is additive. A guard that died because the queue surface
        was unreachable would be strictly worse than the one it replaces."""

        def _boom(url: str, timeout: float) -> tuple[int, str]:
            raise OSError("connection refused")

        facts = read_agent_queue("http://agent:8098", opener=_boom)
        assert facts.commands_ahead is None
        assert "connection refused" in facts.unread_reason

    def test_a_negative_count_is_refused_rather_than_clamped(self) -> None:
        facts = read_agent_queue(
            "http://agent:8098",
            opener=self._opener(200, json.dumps({"commands_ahead": -1})),
        )
        assert facts.commands_ahead is None
