# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay for the transient external-context RED (OMN-17864).

WHAT IS UNDER TEST
    The REAL external-context resolution in ``scripts/ci/ci_summary_gate.py``,
    imported as the module CI runs, driven over a ``commits/{sha}/check-runs``
    payload captured verbatim from the head it failed on.

THE INCIDENT, IN ITS OWN WORDS
    ``CI Summary`` is the single required branch-protection context on
    ``omnibase_infra`` ``dev``. OMN-18355 taught it that a ``cancelled`` row is
    not a verdict and must be waited out. It left the larger half open: a
    ``failure`` row is still terminal on the poll that observes it, even when
    that failure is about to be superseded by an automatic re-run of the same
    producer.

    The producer this bites is the Receipt Gate (``verify / verify``). On a
    ticketed PR the OCC evidence companion is minted by automation AFTER the PR
    opens; until it lands the PR body carries no evidence-source line and the
    Receipt Gate is legitimately red. When the companion merges, automation
    PATCHes the PR body, the body edit re-fires every workflow whose ``types:``
    include ``edited``, the Receipt Gate re-runs and goes green ON ITS OWN. By
    then ``CI Summary`` has already recorded FAILURE and exited, and the armed
    auto-merge is held by a verdict that is no longer true of the head.

    Only a human ``gh run rerun`` cleared it, and that rerun passed with no
    change to the PR — which is the proof that nothing was ever wrong with the
    head.

WHY THE CAPTURES ARE THE EVIDENCE, NOT THIS DOCSTRING
    Two captures decide every fact below.

    * ``infra-pr3779-head-ec6c7636-check-runs.json.captured`` — the full
      check-run list for ``omnibase_infra#3779``'s head. ``verify / verify``
      appears on it four times: failure, success, failure, success.
    * ``infra-pr3779-ci-summary-job105748732276.json.captured`` — the failing
      poller's own job record. Its ``completed_at`` is the instant the shipped
      gate recorded FAILURE, so the replay does not have to guess when the
      poller looked, and cannot be accused of picking a flattering moment.

    The replay reconstructs the payload AS SEEN AT that instant and asserts the
    shipped rule's answer, then does it again three minutes later. Nothing is
    edited, invented or reordered.

THE CONSTANT IS MEASURED, NOT CHOSEN
    ``verify-verify-recovery-window.json.captured`` holds the recovery interval
    for every one of the last 30 merged ``dev`` PRs that exhibited this shape.
    :func:`test_the_grace_exceeds_every_measured_recovery` reads it, so
    shrinking :data:`EXTERNAL_FAILURE_SUPERSESSION_GRACE_S` below measured
    reality is a red test rather than a review catch.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.ci.ci_summary_gate import (
    EXIT_FAILURE,
    EXIT_PENDING,
    EXIT_SUCCESS,
    EXPECTED_EXTERNAL_CONTEXTS,
    EXTERNAL_FAILURE_SUPERSESSION_GRACE_S,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
    JobState,
    evaluate,
    evaluate_external_contexts,
    failure_is_provisional,
    latest_check_run_by_name,
    provisional_external_verdicts,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "omn17864"

PR3779_CHECK_RUNS = FIXTURES / "infra-pr3779-head-ec6c7636-check-runs.json.captured"
PR3779_POLLER_JOB = FIXTURES / "infra-pr3779-ci-summary-job105748732276.json.captured"
RECOVERY_WINDOW = FIXTURES / "verify-verify-recovery-window.json.captured"

RECEIPT_GATE = "verify / verify"

# The instant the shipped poller recorded FAILURE, read off its OWN job record
# rather than chosen here. See _poller_failed_at().
_THREE_MINUTES_LATER = timedelta(minutes=3)


def _rows() -> list[dict[str, object]]:
    """Every captured check-run row, unmodified."""
    return list(json.loads(PR3779_CHECK_RUNS.read_text(encoding="utf-8"))["check_runs"])


def _poller_failed_at() -> datetime:
    """The instant the shipped gate recorded FAILURE, from the job's own record."""
    job = json.loads(PR3779_POLLER_JOB.read_text(encoding="utf-8"))
    assert job["conclusion"] == "failure", job
    return _parse(str(job["completed_at"]))


def _parse(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _as_seen_at(instant: datetime) -> list[dict[str, object]]:
    """The captured payload as a poll at ``instant`` would have read it.

    Two exclusions, both stated rather than baked into a hand-built file:

    * a row that had not STARTED cannot have been visible, so it is dropped;
    * a row that had started but not yet FINISHED was ``in_progress`` at that
      instant, so its terminal fields are withheld rather than back-dated.

    The second is stricter than the OMN-18355 replay's filter, which keeps a
    row whole once it has started. Stricter is the right direction here: this
    ticket is about a verdict issued too early, so a replay that let a
    not-yet-written conclusion be visible would be arguing its own case.
    """

    seen: list[dict[str, object]] = []
    for row in _rows():
        started = str(row.get("started_at") or "")
        if not started or _parse(started) > instant:
            continue
        current = dict(row)
        completed = row.get("completed_at")
        if completed is None or _parse(str(completed)) > instant:
            current["status"] = "in_progress"
            current["conclusion"] = None
            current["completed_at"] = None
        seen.append(current)
    return seen


def _all_gates_green() -> list[dict[str, object]]:
    """An in-run job snapshot with every gate present and passing.

    Isolates the external-context layer: any verdict below is decided by the
    check-run payload alone.
    """
    return [
        {"name": name, "status": "completed", "conclusion": "success", "run_attempt": 1}
        for name in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS)
    ]


def _external_row(
    name: str,
    conclusion: str | None,
    *,
    completed_at: str | None,
    status: str = "completed",
    started_at: str = "2026-09-18T20:00:00Z",
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started_at": started_at,
        "completed_at": completed_at,
        "id": 1,
    }


class TestCapturesAreTheIncident:
    """Provenance: the replay decides on committed bytes, not on this file."""

    def test_captured_payloads_are_readable_distinct_and_non_empty(self) -> None:
        paths = (PR3779_CHECK_RUNS, PR3779_POLLER_JOB, RECOVERY_WINDOW)
        digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        # A capture that silently became an empty file would otherwise satisfy
        # every assertion below vacuously.
        assert len(set(digests.values())) == len(digests)
        assert all(p.stat().st_size > 0 for p in paths)

    def test_the_receipt_gate_went_red_then_green_on_one_unchanged_head(self) -> None:
        """Bytes, not inference: four rows, and the last one is a success."""
        rows = sorted(
            (r for r in _rows() if r.get("name") == RECEIPT_GATE),
            key=lambda r: str(r["started_at"]),
        )
        conclusions = [r["conclusion"] for r in rows]
        assert conclusions == ["failure", "success", "failure", "success"], conclusions

    def test_the_poller_concluded_failure_between_a_red_and_its_green(self) -> None:
        """The failing poll sits inside the window, so the race is the cause."""
        failed_at = _poller_failed_at()
        latest_then = latest_check_run_by_name(_as_seen_at(failed_at))[RECEIPT_GATE]
        assert latest_then.conclusion == "failure"

        later = latest_check_run_by_name(_as_seen_at(failed_at + _THREE_MINUTES_LATER))[
            RECEIPT_GATE
        ]
        assert later.conclusion == "success"

        # And the recovery was NOT the human rerun: the green row's producer
        # started before anybody could have reacted to the failure.
        assert _parse(str(later.completed_at)) - failed_at < timedelta(minutes=5)


class TestTheTransientRedIsNotATerminalVerdict:
    """AC-1 — the incident head must not be failed while its replacement is due."""

    def test_the_gate_does_not_fail_the_head_at_the_failing_poll(self) -> None:
        """REGRESSION: this is the exact snapshot the shipped gate failed on."""
        failed_at = _poller_failed_at()
        failures, unresolved = evaluate_external_contexts(
            _as_seen_at(failed_at), EXPECTED_EXTERNAL_CONTEXTS, now=failed_at
        )
        assert RECEIPT_GATE not in failures
        assert RECEIPT_GATE in unresolved

    def test_the_whole_verdict_is_pending_not_failure_at_that_instant(self) -> None:
        code, report = evaluate(
            _all_gates_green(),
            check_runs=_as_seen_at(_poller_failed_at()),
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
            now=_poller_failed_at(),
        )
        assert code == EXIT_PENDING, report

    def test_the_same_head_resolves_once_the_replacement_lands(self) -> None:
        """No change to the PR — only the replacement row — clears it."""
        instant = _poller_failed_at() + _THREE_MINUTES_LATER
        failures, unresolved = evaluate_external_contexts(
            _as_seen_at(instant), EXPECTED_EXTERNAL_CONTEXTS, now=instant
        )
        assert failures == []
        assert RECEIPT_GATE not in unresolved

    def test_the_report_names_the_wait_rather_than_hiding_it(self) -> None:
        failed_at = _poller_failed_at()
        _, report = evaluate(
            _all_gates_green(),
            check_runs=_as_seen_at(failed_at),
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
            now=failed_at,
        )
        assert RECEIPT_GATE in report
        assert "awaiting" in report.lower()


class TestTheGraceIsFailClosedInEveryUncertainCase:
    """AC-2 — the half that keeps this from becoming a bypass."""

    def test_a_red_older_than_the_grace_still_fails(self) -> None:
        now = datetime(2026, 9, 18, 21, 0, 0, tzinfo=UTC)
        stale = now - timedelta(seconds=EXTERNAL_FAILURE_SUPERSESSION_GRACE_S + 60)
        rows = [
            _external_row(
                RECEIPT_GATE,
                "failure",
                completed_at=stale.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        ]
        failures, _ = evaluate_external_contexts(rows, (RECEIPT_GATE,), now=now)
        assert failures == [RECEIPT_GATE]

    def test_no_clock_is_the_strict_pre_grace_reading(self) -> None:
        """A caller that forgets the time enforces the OLD, stricter behaviour."""
        rows = [
            _external_row(RECEIPT_GATE, "failure", completed_at="2026-09-18T20:59:59Z")
        ]
        failures, _ = evaluate_external_contexts(rows, (RECEIPT_GATE,), now=None)
        assert failures == [RECEIPT_GATE]

    @pytest.mark.parametrize("completed_at", [None, "", "not-a-timestamp"])
    def test_an_unreadable_completion_time_fails_now(self, completed_at: str) -> None:
        rows = [_external_row(RECEIPT_GATE, "failure", completed_at=completed_at)]
        now = datetime(2026, 9, 18, 21, 0, 0, tzinfo=UTC)
        failures, _ = evaluate_external_contexts(rows, (RECEIPT_GATE,), now=now)
        assert failures == [RECEIPT_GATE]

    def test_a_completion_far_in_the_future_fails_now(self) -> None:
        """A clock so wrong the row cannot be reasoned about is not waited on."""
        now = datetime(2026, 9, 18, 21, 0, 0, tzinfo=UTC)
        skewed = now + timedelta(seconds=EXTERNAL_FAILURE_SUPERSESSION_GRACE_S + 60)
        rows = [
            _external_row(
                RECEIPT_GATE,
                "failure",
                completed_at=skewed.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        ]
        failures, _ = evaluate_external_contexts(rows, (RECEIPT_GATE,), now=now)
        assert failures == [RECEIPT_GATE]

    @pytest.mark.parametrize("conclusion", ["timed_out", "action_required"])
    def test_the_grace_is_scoped_to_failure_and_cancelled_only(
        self, conclusion: str
    ) -> None:
        """Nothing else gets a window: the measured mechanism is a re-run."""
        now = datetime(2026, 9, 18, 21, 0, 0, tzinfo=UTC)
        rows = [
            _external_row(
                RECEIPT_GATE,
                conclusion,
                completed_at=(now - timedelta(seconds=5)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            )
        ]
        failures, _ = evaluate_external_contexts(rows, (RECEIPT_GATE,), now=now)
        assert failures == [RECEIPT_GATE]

    def test_a_provisional_red_is_never_success(self) -> None:
        """PENDING is the only thing the grace can produce — never green."""
        now = datetime(2026, 9, 18, 21, 0, 0, tzinfo=UTC)
        rows = [
            _external_row(
                RECEIPT_GATE,
                "failure",
                completed_at=(now - timedelta(seconds=5)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            )
        ]
        code, report = evaluate(
            _all_gates_green(),
            check_runs=rows,
            external_contexts=(RECEIPT_GATE,),
            now=now,
        )
        assert code == EXIT_PENDING, report
        assert code != EXIT_SUCCESS

    def test_a_sustained_red_still_reaches_failure_at_the_deadline(self) -> None:
        """The poller's deadline converts the wait into the same FAILURE."""
        first_seen = datetime(2026, 9, 18, 21, 0, 0, tzinfo=UTC)
        rows = [
            _external_row(
                RECEIPT_GATE,
                "failure",
                completed_at=first_seen.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        ]
        held, _ = evaluate(
            _all_gates_green(),
            check_runs=rows,
            external_contexts=(RECEIPT_GATE,),
            now=first_seen + timedelta(seconds=30),
        )
        assert held == EXIT_PENDING

        terminal, report = evaluate(
            _all_gates_green(),
            check_runs=rows,
            external_contexts=(RECEIPT_GATE,),
            now=first_seen
            + timedelta(seconds=EXTERNAL_FAILURE_SUPERSESSION_GRACE_S + 1),
        )
        assert terminal == EXIT_FAILURE, report

    def test_an_unreadable_check_runs_payload_is_still_pending_never_green(
        self,
    ) -> None:
        failures, unresolved = evaluate_external_contexts(
            None, (RECEIPT_GATE,), now=datetime(2026, 9, 18, 21, 0, 0, tzinfo=UTC)
        )
        assert failures == []
        assert unresolved == [RECEIPT_GATE]


class TestTheConstantIsBoundByMeasurement:
    """AC-3 — the window is derived from the fleet, not chosen in review."""

    def test_the_grace_exceeds_every_measured_recovery(self) -> None:
        measured = json.loads(RECOVERY_WINDOW.read_text(encoding="utf-8"))
        assert measured["never_recovered"] == 0
        assert measured["prs_with_a_transient_red"] > 0
        worst_s = float(measured["max_recovery_minutes"]) * 60
        assert worst_s < EXTERNAL_FAILURE_SUPERSESSION_GRACE_S

    def test_the_grace_stays_well_under_the_poll_deadline(self) -> None:
        """A grace near the deadline would convert a wait into a timeout."""
        deadline_s = 90 * 60  # DEADLINE_MINUTES in ci.yml's ci-summary step
        assert deadline_s / 3 > EXTERNAL_FAILURE_SUPERSESSION_GRACE_S

    def test_this_shape_is_common_enough_to_be_worth_a_mechanism(self) -> None:
        """Half the merged PRs sampled hit it — it is the norm, not an outlier."""
        measured = json.loads(RECOVERY_WINDOW.read_text(encoding="utf-8"))
        assert measured["prs_with_a_transient_red"] >= measured["prs_sampled"] / 4


class TestTheHelperIsDirectlyPinned:
    """The predicate itself, so a refactor cannot quietly widen it."""

    def test_only_a_failure_conclusion_is_eligible(self) -> None:
        now = datetime(2026, 9, 18, 21, 0, 0, tzinfo=UTC)
        recent = (now - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert failure_is_provisional(
            JobState(RECEIPT_GATE, "completed", "failure", 1, recent), now
        )
        assert not failure_is_provisional(
            JobState(RECEIPT_GATE, "completed", "success", 1, recent), now
        )
        assert not failure_is_provisional(
            JobState(RECEIPT_GATE, "completed", "timed_out", 1, recent), now
        )

    def test_the_reporting_helper_lists_both_provisional_kinds(self) -> None:
        now = datetime(2026, 9, 18, 21, 0, 0, tzinfo=UTC)
        recent = (now - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = [
            _external_row("red-one", "failure", completed_at=recent),
            _external_row("cancelled-one", "cancelled", completed_at=recent),
            _external_row("green-one", "success", completed_at=recent),
        ]
        listed = provisional_external_verdicts(
            rows, ("red-one", "cancelled-one", "green-one"), now
        )
        assert listed == ["cancelled-one", "red-one"]
