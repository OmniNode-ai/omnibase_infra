# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19869: a cancellation whose replacement run is still running is PENDING.

The shape replayed here was measured on omnibase_infra#4209 (head 153a007e9) on
2026-09-27. The OCC bot's body edit fired ``edited``; the Hostile Reviewer run
36320397834 was cancelled at 12:52:32Z by its replacement 36320434712 (same
workflow, same head). CI Summary failed closed at 13:03:21Z -- 649s after the
cancellation, past the 600s grace -- and the replacement's ``Hostile Review
Gate`` row concluded ``success`` at 13:03:27Z. The same shape repeated on #4210
(10m42s) and #4207 (10m21s).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "ci"))

from ci_summary_gate import (
    CANCELLED_SUPERSESSION_GRACE_S,
    EXIT_FAILURE,
    EXIT_PENDING,
    EXIT_SUCCESS,
    cancelled_rows_with_replacement_in_flight,
    evaluate,
    evaluate_external_contexts,
    replacement_run_in_flight,
)

pytestmark = pytest.mark.unit

HEAD = "153a007e9b32e62580e505c94c30c63a0393784b"
REPO = "https://github.com/OmniNode-ai/omnibase_infra"
HOSTILE_WORKFLOW_ID = 111
REJECT_WORKFLOW_ID = 222
CANCELLED_RUN = 36320397834
REPLACEMENT_RUN = 36320434712
CANCELLED_AT = datetime(2026, 9, 27, 12, 52, 32, tzinfo=UTC)
# The poll that reddened #4209: 649s after the cancellation.
MEASURED_POLL = CANCELLED_AT + timedelta(seconds=649)
GATE = "Hostile Review Gate"
REJECT = "call-reject-skip-token / scan / reject-skip-gate-token"


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(
    name: str,
    run_id: int,
    *,
    conclusion: str | None,
    status: str = "completed",
    at: datetime = CANCELLED_AT,
    row_id: int = 1,
) -> dict[str, object]:
    return {
        "id": row_id,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started_at": _iso(at),
        "completed_at": _iso(at) if status == "completed" else None,
        "html_url": f"{REPO}/actions/runs/{run_id}/job/{row_id}",
        "details_url": f"{REPO}/actions/runs/{run_id}/job/{row_id}",
    }


def _run(
    run_id: int,
    *,
    status: str,
    workflow_id: int = HOSTILE_WORKFLOW_ID,
    head_sha: str = HEAD,
) -> dict[str, object]:
    return {
        "id": run_id,
        "workflow_id": workflow_id,
        "head_sha": head_sha,
        "status": status,
        "event": "pull_request",
    }


def _cancelled_gate() -> dict[str, object]:
    return _row(GATE, CANCELLED_RUN, conclusion="cancelled")


def _summary_jobs() -> list[dict[str, object]]:
    # A CI run whose own jobs are all green, so only layer 5 decides.
    return [
        {
            "name": "CI Summary",
            "status": "in_progress",
            "conclusion": None,
            "run_attempt": 1,
        }
    ]


def _evaluate(
    check_runs: list[dict[str, object]],
    workflow_runs: list[dict[str, object]] | None,
    now: datetime,
) -> tuple[int, str]:
    return evaluate(
        _summary_jobs(),
        run_attempt=1,
        strict_gates=(),
        skippable_gates=(),
        check_runs=check_runs,
        external_contexts=(),
        now=now,
        sweep_exclusions={},
        conditional_sweep_exclusions={},
        workflow_runs=workflow_runs,
    )


class TestInFlight:
    def test_in_flight_newer_run_of_the_same_workflow_is_a_replacement(self) -> None:
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="in_progress"),
        ]
        assert replacement_run_in_flight(_cancelled_gate(), runs) is True

    def test_in_flight_queued_replacement_counts(self) -> None:
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="queued"),
        ]
        assert replacement_run_in_flight(_cancelled_gate(), runs) is True

    def test_in_flight_rerun_of_the_cancelled_run_itself_counts(self) -> None:
        runs = [_run(CANCELLED_RUN, status="in_progress")]
        assert replacement_run_in_flight(_cancelled_gate(), runs) is True

    def test_in_flight_external_context_is_pending_past_the_grace(self) -> None:
        rows = [_row(REJECT, CANCELLED_RUN, conclusion="cancelled")]
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="in_progress"),
        ]
        in_flight = cancelled_rows_with_replacement_in_flight(rows, runs)
        failures, unresolved = evaluate_external_contexts(
            rows,
            (REJECT,),
            now=MEASURED_POLL,
            replacements_in_flight=in_flight,
        )
        assert failures == []
        assert unresolved == [REJECT]


class TestMeasured:
    def test_measured_4209_shape_is_pending_not_failure(self) -> None:
        assert (MEASURED_POLL - CANCELLED_AT).total_seconds() > (
            CANCELLED_SUPERSESSION_GRACE_S
        )
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="in_progress"),
        ]
        code, report = _evaluate([_cancelled_gate()], runs, MEASURED_POLL)
        assert code == EXIT_PENDING, report

    def test_measured_4209_shape_without_the_runs_payload_still_fails(self) -> None:
        # The positive control: the shipped behaviour this ticket measured.
        code, report = _evaluate([_cancelled_gate()], None, MEASURED_POLL)
        assert code == EXIT_FAILURE, report
        assert "cancelled_without_replacement" in report

    def test_measured_4209_replacement_row_resolves_it(self) -> None:
        rows = [
            _cancelled_gate(),
            _row(
                GATE,
                REPLACEMENT_RUN,
                conclusion="success",
                at=CANCELLED_AT + timedelta(seconds=655),
                row_id=2,
            ),
        ]
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="completed"),
        ]
        code, report = _evaluate(rows, runs, MEASURED_POLL + timedelta(seconds=10))
        assert code == EXIT_SUCCESS, report


class TestFailsClosed:
    def test_fails_closed_when_the_newer_run_completed_without_a_row(self) -> None:
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="completed"),
        ]
        assert replacement_run_in_flight(_cancelled_gate(), runs) is False
        code, report = _evaluate([_cancelled_gate()], runs, MEASURED_POLL)
        assert code == EXIT_FAILURE, report

    def test_fails_closed_on_another_workflow_in_flight(self) -> None:
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="in_progress", workflow_id=REJECT_WORKFLOW_ID),
        ]
        assert replacement_run_in_flight(_cancelled_gate(), runs) is False

    def test_fails_closed_on_another_head_in_flight(self) -> None:
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="in_progress", head_sha="0" * 40),
        ]
        assert replacement_run_in_flight(_cancelled_gate(), runs) is False

    def test_fails_closed_on_an_older_run_in_flight(self) -> None:
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(CANCELLED_RUN - 5, status="in_progress"),
        ]
        assert replacement_run_in_flight(_cancelled_gate(), runs) is False

    def test_fails_closed_when_the_cancelled_run_is_not_in_the_payload(self) -> None:
        runs = [_run(REPLACEMENT_RUN, status="in_progress")]
        assert replacement_run_in_flight(_cancelled_gate(), runs) is False

    def test_fails_closed_on_a_row_without_a_run_url(self) -> None:
        row = _cancelled_gate()
        row["html_url"] = ""
        row["details_url"] = ""
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="in_progress"),
        ]
        assert replacement_run_in_flight(row, runs) is False

    def test_fails_closed_on_a_run_without_a_workflow_id(self) -> None:
        origin = _run(CANCELLED_RUN, status="completed")
        origin["workflow_id"] = None
        runs = [origin, _run(REPLACEMENT_RUN, status="in_progress")]
        assert replacement_run_in_flight(_cancelled_gate(), runs) is False

    def test_fails_closed_with_no_runs_payload(self) -> None:
        assert replacement_run_in_flight(_cancelled_gate(), None) is False
        assert replacement_run_in_flight(_cancelled_gate(), []) is False


class TestNeverGreen:
    @pytest.mark.parametrize("conclusion", ["failure", "timed_out", "action_required"])
    def test_never_green_a_real_red_is_not_held_by_an_in_flight_run(
        self, conclusion: str
    ) -> None:
        row = _row(GATE, CANCELLED_RUN, conclusion=conclusion)
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="in_progress"),
        ]
        assert replacement_run_in_flight(row, runs) is False

    def test_never_green_an_in_flight_replacement_holds_pending_only(self) -> None:
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN, status="in_progress"),
        ]
        # However long the replacement runs, the verdict is PENDING, never
        # SUCCESS: the poller's own deadline is what ends the wait.
        code, _ = _evaluate(
            [_cancelled_gate()], runs, CANCELLED_AT + timedelta(hours=2)
        )
        assert code == EXIT_PENDING
        assert code != EXIT_SUCCESS

    def test_never_green_the_winner_must_be_the_cancellation(self) -> None:
        # A later real failure on the same name wins latest-wins, so the set
        # does not name it and the failure reds.
        rows = [
            _cancelled_gate(),
            _row(
                GATE,
                REPLACEMENT_RUN,
                conclusion="failure",
                at=CANCELLED_AT + timedelta(seconds=30),
                row_id=2,
            ),
        ]
        runs = [
            _run(CANCELLED_RUN, status="completed"),
            _run(REPLACEMENT_RUN + 1, status="in_progress"),
        ]
        assert cancelled_rows_with_replacement_in_flight(rows, runs) == frozenset()
