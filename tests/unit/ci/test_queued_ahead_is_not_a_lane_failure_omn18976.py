# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18976 — a merge queued behind another deploy is not a failed lab pass.

WHAT WAS BROKEN
---------------
When the deploy agent already holds commands ahead of this merge, the verify job
correctly declines to hold the single lab verify runner open for a queue it
cannot outlast (OMN-18144). It then recorded that decision as
``EnumConvergenceOutcome.INDETERMINATE``, which maps to a lab-pass check with
``ok: false``, and the receipt verdict rule is "PASS iff every check passed" --
so the receipt was emitted **FAIL** and rule 24(b) refused a sha whose only
fault was being second in line.

The job's own evidence said so in as many words: *"This asserts nothing about
the lane; the receipt is still non-PASS."* A check that asserts nothing about
the lane and a check the lane failed are not the same claim, and only one of
them should refuse a delivery.

Measured on the .201 dev lane, two consecutive receipts, each probed while the
lane still carried the PREVIOUS sha:

    artifact 10622202672, sha 22a0ca18, lane at 08db8946 -> INDETERMINATE -> FAIL
    artifact 10623436027, sha 4277d6e9, lane at 22a0ca18 -> INDETERMINATE -> FAIL

Under a steady merge rate the queue never drains inside one verify window, so
this is the normal outcome for a busy period rather than an edge case.

WHY THE FIX IS A DISTINCT OUTCOME AND NO RECEIPT
-------------------------------------------------
``EnumLabPassResult`` is deliberately two-valued and its own docstring states
the contract: *"An in-flight or indeterminate lab pass emits NO receipt at all
rather than a PENDING one, so the gate's 'absent' branch and its 'not yet
passing' branch are the same branch, and both fail closed."* A merge whose turn
in the queue has not come is in flight. Emitting a terminal FAIL for it is the
one behaviour that contract rules out.

So the queued case becomes its own convergence outcome, and it emits nothing.
**Nothing opens**: the gate's absent branch already fails closed, so the sha is
still refused -- what changes is that the refusal is recoverable by a later
convergence instead of being contradicted forever by a FAIL artifact.

The three existing outcomes keep their meanings exactly:

* ``OK`` -- the lane contains the merge sha.
* ``FAIL`` -- the lane had its whole budget and does not. About the LANE.
* ``INDETERMINATE`` -- the budget could not be established, or this run ran out
  of its own clock. About the RUN.
* ``QUEUED`` (new) -- the wait never started, because the queue ahead of this
  command is longer than this window can outlast. About the QUEUE.

Every assertion here pins one of those distinctions, because collapsing any two
of them is how this defect was introduced in the first place.
"""

from __future__ import annotations

import pytest

from scripts.ci.check_dev_lane_staleness import (
    EnumConvergenceOutcome,
    ModelQueueFacts,
    convergence_check_outcome,
    queue_exceeds_bound,
)
from scripts.ci.lab_pass_receipt import EnumLabPassCheckOutcome

pytestmark = pytest.mark.unit


class TestQueuedIsItsOwnOutcome:
    """AC-1: a queued-ahead result is a distinct non-terminal state."""

    def test_the_outcome_exists_and_is_distinct(self) -> None:
        assert hasattr(EnumConvergenceOutcome, "QUEUED"), (
            "a merge waiting its turn behind another deploy has no outcome of its "
            "own, so it is recorded as INDETERMINATE and the receipt reads FAIL "
            "for a sha nothing is wrong with"
        )
        values = {
            EnumConvergenceOutcome.OK,
            EnumConvergenceOutcome.FAIL,
            EnumConvergenceOutcome.INDETERMINATE,
            EnumConvergenceOutcome.QUEUED,
        }
        assert len(values) == 4, "QUEUED collapsed onto an existing outcome"

    def test_the_three_existing_outcomes_keep_their_meanings(self) -> None:
        # Positive control on the assertion above: this change must be additive.
        # If it silently re-pointed FAIL or INDETERMINATE the test above would
        # still pass while the guard had been weakened.
        assert EnumConvergenceOutcome.OK.value == "ok"
        assert EnumConvergenceOutcome.FAIL.value == "fail"
        assert EnumConvergenceOutcome.INDETERMINATE.value == "indeterminate"


class TestQueuedEmitsNoTerminalReceipt:
    """AC-1 continued: it must not become a check with ``ok: false``."""

    def test_queued_maps_to_no_lab_pass_check(self) -> None:
        mapped = convergence_check_outcome(EnumConvergenceOutcome.QUEUED)
        assert mapped is None, (
            "a queued merge still maps onto a lab-pass check, so a receipt is "
            "written for it. EnumLabPassResult's own contract is that an "
            "in-flight pass emits NO receipt rather than a terminal one; a FAIL "
            "here is what rule 24(b) then refuses forever"
        )

    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [
            (EnumConvergenceOutcome.OK, EnumLabPassCheckOutcome.PASS),
            (EnumConvergenceOutcome.FAIL, EnumLabPassCheckOutcome.FAIL),
            (
                EnumConvergenceOutcome.INDETERMINATE,
                EnumLabPassCheckOutcome.INDETERMINATE,
            ),
        ],
        ids=["ok", "fail", "indeterminate"],
    )
    def test_every_other_outcome_still_maps_exactly_as_before(
        self, outcome: EnumConvergenceOutcome, expected: EnumLabPassCheckOutcome
    ) -> None:
        # AC-5. The three pre-existing mappings are the guard; this change may
        # only add a fourth case, never re-point one of them.
        assert convergence_check_outcome(outcome) is expected


class TestTheQueueRefusalStillFiresForTheRightReason:
    """The OMN-18144 decision is unchanged; only its recording moves."""

    def test_an_empty_queue_never_refuses(self) -> None:
        # Guards the common healthy path: a horizon applied to an empty queue
        # would refuse every merge whenever the agent's mean drifted up.
        facts = ModelQueueFacts(
            commands_ahead=0,
            mean_service_time_seconds=1052.1,
            service_sample_size=10,
            in_flight_correlation_id=None,
            unread_reason="",
            source="test",
        )
        assert (
            queue_exceeds_bound(
                facts,
                lane_budget_seconds=1500,
                wall_clock_seconds=1620,
                margin_seconds=60,
            )
            == ""
        )

    def test_a_queue_this_window_cannot_outlast_refuses_and_names_the_depth(
        self,
    ) -> None:
        facts = ModelQueueFacts(
            commands_ahead=1,
            mean_service_time_seconds=1052.1,
            service_sample_size=10,
            in_flight_correlation_id="c0ffee",
            unread_reason="",
            source="test",
        )
        refusal = queue_exceeds_bound(
            facts,
            lane_budget_seconds=1500,
            wall_clock_seconds=1620,
            margin_seconds=60,
        )
        assert refusal, "the queue refusal stopped firing; OMN-18144 regressed"
        assert "command(s) ahead" in refusal
        assert "number 2 in line" in refusal


class TestTheGateReadsTheNewestReceipt:
    """AC-3, pinned rather than left as a docstring claim.

    ``evaluate_gate`` sorts an exact-name artifact query newest-first and reads
    only the newest. That is load-bearing for any later convergence being able
    to supersede an earlier non-terminal answer, and until now it was asserted
    only in prose.
    """

    def test_the_gate_sorts_artifacts_newest_first(self) -> None:
        import inspect

        from scripts.ci import lab_pass_receipt

        source = inspect.getsource(lab_pass_receipt.evaluate_gate)
        assert "created_at" in source, (
            "evaluate_gate no longer orders candidate artifacts by creation "
            "time, so a later receipt for a sha cannot supersede an earlier one"
        )
        assert "reverse=True" in source or "[-1]" in source or "max(" in source, (
            "evaluate_gate reads artifacts in an order that is not newest-first"
        )
