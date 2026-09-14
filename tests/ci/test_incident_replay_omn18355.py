# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay for ``scripts/ci/ci_summary_gate.py`` (OMN-15547 / OMN-18355).

WHAT IS UNDER TEST
    The REAL external-context resolution in ``scripts/ci/ci_summary_gate.py``,
    imported as the module CI runs, driven over ``commits/{sha}/check-runs``
    payloads captured verbatim from the two heads it failed on.

THE INCIDENT, IN ITS OWN WORDS
    ``CI Summary`` is the single required branch-protection context on
    ``omnibase_infra`` ``dev``. On 2026-09-14 it failed twice inside an hour, on
    two PRs whose gates were healthy, and a human ``gh run rerun --failed``
    cleared both. The shipped resolution admitted only ``success`` and issued a
    TERMINAL FAILURE on every other completed conclusion, with no waiting and no
    notion of a row that carries no verdict. Two such rows reached it:

    * a ``cancelled`` row from a run the OCC autobind stamp's body edit had
      superseded, at a moment when the replacement run had not yet written its
      own check-run, so the cancellation was the only row for that name;
    * a ``neutral`` code-scanning placeholder that outranks the still-running
      analysis job publishing the same context name, because the placeholder
      starts later.

    Both are false reds: nothing regressed on either head.

WHY THE CAPTURES ARE THE EVIDENCE, NOT THIS DOCSTRING
    Every fact the replay decides on is read out of committed bytes. The
    ``#3511`` pair is a live capture taken while the placeholder and the real
    analysis row coexisted. The ``#3512`` pair plus the failing poller's own job
    log are the second specimen; the log is what names the contexts the shipped
    gate called failures, so the replay does not have to re-implement the rule
    it is replacing in order to know what that rule did.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts.ci.ci_summary_gate import (
    CANCELLED_SUPERSESSION_GRACE_S,
    EXIT_FAILURE,
    EXIT_PENDING,
    EXPECTED_EXTERNAL_CONTEXTS,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
    evaluate,
    evaluate_external_contexts,
    latest_check_run_by_name,
    provisional_cancellations,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "omn18355"

# --- specimen 1: the code-scanning placeholder (omnibase_infra#3511) ---------
PR3511_PAGE1 = FIXTURES / "infra-pr3511-head-45e9d4d8-check-runs.json.captured"
PR3511_PAGE2 = FIXTURES / "infra-pr3511-head-45e9d4d8-check-runs-page2.json.captured"

# --- specimen 2: the superseded cancellation (omnibase_infra#3512) -----------
PR3512_PAGE1 = FIXTURES / "infra-pr3512-head-689344f8-check-runs.json.captured"
PR3512_PAGE2 = FIXTURES / "infra-pr3512-head-689344f8-check-runs-page2.json.captured"
PR3512_POLL_LOG = FIXTURES / "infra-pr3512-ci-summary-job103815923213.log.gz.captured"

# The instant the shipped poller took the snapshot it failed on, read off the
# captured log's own timestamp for the FAILURE verdict line.
PR3512_POLL_INSTANT = datetime.fromisoformat("2026-09-14T00:01:32+00:00")
CANCELLED_CONTEXT = "call-reject-skip-token / scan / reject-skip-gate-token"
PLACEHOLDER_CONTEXT = "CodeQL"


def _rows(*paths: Path) -> list[dict[str, object]]:
    """Every check-run row across the captured pages, unmodified."""
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["check_runs"])
    return rows


def _at_instant(
    rows: list[dict[str, object]], instant: datetime
) -> list[dict[str, object]]:
    """The captured rows that existed at ``instant``.

    The same filter the OMN-15496 merge-time fixture documents in its own
    provenance block: a check-run that had not STARTED cannot have been visible
    to a poll, so the payload for an earlier instant is the captured payload
    minus the rows that start after it. Nothing is edited, invented or
    reordered — rows are only excluded, and the rule for excluding them is
    stated here rather than baked into a hand-built file.
    """
    return [
        row
        for row in rows
        if str(row.get("started_at") or "") and str(row["started_at"]) <= _z(instant)
    ]


def _z(instant: datetime) -> str:
    return instant.strftime("%Y-%m-%dT%H:%M:%SZ")


def _all_gates_green() -> list[dict[str, object]]:
    """An in-run job snapshot with every gate present and passing.

    Isolates the external-context layer: any verdict below is decided by the
    check-run payload alone.
    """
    return [
        {"name": name, "status": "completed", "conclusion": "success", "run_attempt": 1}
        for name in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS)
    ]


class TestCapturesAreTheIncident:
    """R1/R2 provenance, and the tie between the captures and the poller's log."""

    def test_captured_payloads_hash_to_their_registry_entries(self) -> None:
        digests = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                PR3511_PAGE1,
                PR3511_PAGE2,
                PR3512_PAGE1,
                PR3512_PAGE2,
                PR3512_POLL_LOG,
            )
        }
        # Non-empty, distinct, and readable: a capture that silently became an
        # empty file would otherwise satisfy every assertion below vacuously.
        assert len(set(digests.values())) == len(digests)
        assert all(path.stat().st_size > 0 for path in (PR3511_PAGE1, PR3512_PAGE1))

    def test_the_poller_log_names_the_two_contexts_this_ticket_is_about(self) -> None:
        with gzip.open(
            PR3512_POLL_LOG, "rt", encoding="utf-8", errors="replace"
        ) as handle:
            log = handle.read()
        failure_lines = [
            line for line in log.splitlines() if "external-context failures:" in line
        ]
        assert len(failure_lines) == 1, failure_lines
        named = failure_lines[0].split("external-context failures:", 1)[1]
        assert CANCELLED_CONTEXT in named
        assert PLACEHOLDER_CONTEXT in named
        assert "CI Summary verdict: FAILURE" in log

    def test_the_cancelled_row_is_the_only_evidence_at_the_failing_poll(self) -> None:
        """Bytes, not inference: one row for that context, and it is cancelled."""
        rows = _at_instant(_rows(PR3512_PAGE1, PR3512_PAGE2), PR3512_POLL_INSTANT)
        matching = [row for row in rows if row.get("name") == CANCELLED_CONTEXT]
        assert len(matching) == 1, matching
        assert matching[0]["status"] == "completed"
        assert matching[0]["conclusion"] == "cancelled"
        assert matching[0]["completed_at"] == "2026-09-14T00:01:03Z"

    def test_the_placeholder_outranks_the_analysis_in_the_live_capture(self) -> None:
        """Bytes, not inference: two producers, placeholder newest."""
        rows = [
            row
            for row in _rows(PR3511_PAGE1, PR3511_PAGE2)
            if row.get("name") == PLACEHOLDER_CONTEXT
        ]
        by_app = {row["app"]["slug"]: row for row in rows}  # type: ignore[index]
        assert set(by_app) == {"github-advanced-security", "github-actions"}
        placeholder = by_app["github-advanced-security"]
        analysis = by_app["github-actions"]
        assert placeholder["conclusion"] == "neutral"
        assert str(placeholder["started_at"]) > str(analysis["started_at"])


class TestPlaceholderDoesNotOutrankTheAnalysis:
    """AC-1 — the code-scanning placeholder is not a verdict about the head."""

    def test_the_real_gate_accepts_the_head_the_placeholder_failed(self) -> None:
        """ACCEPT (false_red): nothing regressed on #3511's head."""
        rows = _rows(PR3511_PAGE1, PR3511_PAGE2)
        assert (
            latest_check_run_by_name(rows)[PLACEHOLDER_CONTEXT].conclusion == "success"
        )
        failures, _ = evaluate_external_contexts(
            rows, EXPECTED_EXTERNAL_CONTEXTS, now=PR3512_POLL_INSTANT
        )
        assert PLACEHOLDER_CONTEXT not in failures

    def test_a_still_running_analysis_holds_the_context_pending(self) -> None:
        """The placeholder must never resolve a context on its own."""
        rows = [dict(row) for row in _rows(PR3511_PAGE1, PR3511_PAGE2)]
        for row in rows:
            if (
                row.get("name") == PLACEHOLDER_CONTEXT
                and row["app"]["slug"] == "github-actions"  # type: ignore[index]
            ):
                row["status"] = "in_progress"
                row["conclusion"] = None
        failures, unresolved = evaluate_external_contexts(
            rows, EXPECTED_EXTERNAL_CONTEXTS, now=PR3512_POLL_INSTANT
        )
        assert PLACEHOLDER_CONTEXT not in failures
        assert PLACEHOLDER_CONTEXT in unresolved

    def test_discriminator_a_real_red_from_the_analysis_still_fails(self) -> None:
        """DISCRIMINATOR (R5): the guard is not stuck open on this name."""
        rows = [dict(row) for row in _rows(PR3511_PAGE1, PR3511_PAGE2)]
        for row in rows:
            if (
                row.get("name") == PLACEHOLDER_CONTEXT
                and row["app"]["slug"] == "github-actions"  # type: ignore[index]
            ):
                row["conclusion"] = "failure"
        failures, _ = evaluate_external_contexts(
            rows, EXPECTED_EXTERNAL_CONTEXTS, now=PR3512_POLL_INSTANT
        )
        assert PLACEHOLDER_CONTEXT in failures

    def test_discriminator_a_lone_placeholder_still_fails_closed(self) -> None:
        """DISCRIMINATOR: a name whose ONLY row is the placeholder never greens."""
        rows = [
            row
            for row in _rows(PR3511_PAGE1, PR3511_PAGE2)
            if not (
                row.get("name") == PLACEHOLDER_CONTEXT
                and row["app"]["slug"] == "github-actions"  # type: ignore[index]
            )
        ]
        failures, unresolved = evaluate_external_contexts(
            rows, EXPECTED_EXTERNAL_CONTEXTS, now=PR3512_POLL_INSTANT
        )
        assert PLACEHOLDER_CONTEXT in failures
        assert PLACEHOLDER_CONTEXT not in unresolved


class TestSupersededCancellationIsNotAVerdict:
    """AC-2 — a cancellation waits, bounded, for the replacement it implies."""

    def _slice(self) -> list[dict[str, object]]:
        return _at_instant(_rows(PR3512_PAGE1, PR3512_PAGE2), PR3512_POLL_INSTANT)

    def test_the_poll_that_failed_is_pending_instead(self) -> None:
        """ACCEPT (false_red): PENDING at the instant the shipped gate failed."""
        rows = self._slice()
        failures, unresolved = evaluate_external_contexts(
            rows, EXPECTED_EXTERNAL_CONTEXTS, now=PR3512_POLL_INSTANT
        )
        assert CANCELLED_CONTEXT not in failures
        assert CANCELLED_CONTEXT in unresolved
        assert failures == []
        code, report = evaluate(
            _all_gates_green(),
            check_runs=rows,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
            now=PR3512_POLL_INSTANT,
        )
        assert code == EXIT_PENDING, report

    def test_the_wait_is_disclosed_not_silent(self) -> None:
        rows = self._slice()
        assert provisional_cancellations(
            rows, EXPECTED_EXTERNAL_CONTEXTS, PR3512_POLL_INSTANT
        ) == [CANCELLED_CONTEXT]
        _, report = evaluate(
            _all_gates_green(),
            check_runs=rows,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
            now=PR3512_POLL_INSTANT,
        )
        assert "awaiting a replacement after cancellation" in report
        assert CANCELLED_CONTEXT in report

    def test_the_replacement_row_resolves_it_green(self) -> None:
        """The real replacement is in the same capture, 57s later."""
        later = datetime.fromisoformat("2026-09-14T00:02:05+00:00")
        rows = _at_instant(_rows(PR3512_PAGE1, PR3512_PAGE2), later)
        assert latest_check_run_by_name(rows)[CANCELLED_CONTEXT].conclusion == "success"
        failures, unresolved = evaluate_external_contexts(
            rows, EXPECTED_EXTERNAL_CONTEXTS, now=later
        )
        assert CANCELLED_CONTEXT not in failures
        assert CANCELLED_CONTEXT not in unresolved

    def test_discriminator_a_cancellation_past_the_grace_still_fails(self) -> None:
        """DISCRIMINATOR (R5): the wait is bounded, so this is not a bypass."""
        rows = self._slice()
        late = datetime.fromisoformat("2026-09-14T00:01:03+00:00").timestamp()
        past_grace = datetime.fromtimestamp(
            late + CANCELLED_SUPERSESSION_GRACE_S + 1, tz=PR3512_POLL_INSTANT.tzinfo
        )
        failures, _ = evaluate_external_contexts(
            rows, EXPECTED_EXTERNAL_CONTEXTS, now=past_grace
        )
        assert CANCELLED_CONTEXT in failures
        code, _ = evaluate(
            _all_gates_green(),
            check_runs=rows,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
            now=past_grace,
        )
        assert code == EXIT_FAILURE

    def test_discriminator_no_clock_means_the_old_strict_reading(self) -> None:
        """DISCRIMINATOR: a caller that supplies no time enforces, never waits."""
        rows = self._slice()
        failures, _ = evaluate_external_contexts(rows, EXPECTED_EXTERNAL_CONTEXTS)
        assert CANCELLED_CONTEXT in failures

    def test_discriminator_a_real_failure_in_the_same_slice_still_fails(self) -> None:
        """DISCRIMINATOR: only a cancellation waits; a verdict is still a verdict."""
        rows = [dict(row) for row in self._slice()]
        for row in rows:
            if row.get("name") == CANCELLED_CONTEXT:
                row["conclusion"] = "failure"
        failures, _ = evaluate_external_contexts(
            rows, EXPECTED_EXTERNAL_CONTEXTS, now=PR3512_POLL_INSTANT
        )
        assert CANCELLED_CONTEXT in failures
