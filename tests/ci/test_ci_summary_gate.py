# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed verdict tests for the ``CI Summary`` poller (OMN-14127).

The ``CI Summary`` required context is posted by a NO-``needs`` poller that
calls ``scripts/ci/ci_summary_gate.py``. These tests pin the fail-closed,
default-deny verdict so the required gate can never silently rubber-stamp, and
they assert the infra-specific strict/skippable gate split faithfully mirrors
the old needs-based ci-summary pass/fail condition.
"""

from __future__ import annotations

import pytest

from scripts.ci.ci_summary_gate import (
    EXIT_FAILURE,
    EXIT_PENDING,
    EXIT_SUCCESS,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
    evaluate,
)

pytestmark = pytest.mark.unit


def _job(
    name: str, conclusion: str | None, *, status: str = "completed", attempt: int = 1
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "run_attempt": attempt,
    }


def _all_gates(conclusion: str = "success") -> list[dict[str, object]]:
    """A full, passing snapshot: every strict + skippable gate present+good."""
    return [_job(g, conclusion) for g in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS)]


class TestCiSummaryGate:
    def test_all_gates_success_is_success(self) -> None:
        code, _ = evaluate(_all_gates("success") + [_job("Detect Changes", "success")])
        assert code == EXIT_SUCCESS

    def test_skippable_gate_skipped_counts_as_pass(self) -> None:
        # A skippable gate (e.g. migration-integration on docs-only) may skip.
        jobs = [_job(g, "success") for g in STRICT_GATE_JOBS]
        jobs += [_job(g, "skipped") for g in SKIPPABLE_GATE_JOBS]
        code, _ = evaluate(jobs)
        assert code == EXIT_SUCCESS

    def test_strict_gate_skipped_is_failure(self) -> None:
        # A STRICT gate is unconditional in ci.yml; a skip must fail closed
        # (mirrors the old ``== "success"`` condition), never rubber-stamp.
        jobs = _all_gates("success")
        jobs[0] = _job(STRICT_GATE_JOBS[0], "skipped")
        code, report = evaluate(jobs)
        assert code == EXIT_FAILURE
        assert STRICT_GATE_JOBS[0] in report

    def test_strict_gate_failure_is_failure(self) -> None:
        jobs = _all_gates("success")
        jobs[1] = _job(STRICT_GATE_JOBS[1], "failure")
        code, report = evaluate(jobs)
        assert code == EXIT_FAILURE
        assert STRICT_GATE_JOBS[1] in report

    def test_strict_gate_cancelled_is_failure(self) -> None:
        jobs = _all_gates("success")
        jobs[2] = _job(STRICT_GATE_JOBS[2], "cancelled")
        code, _ = evaluate(jobs)
        assert code == EXIT_FAILURE

    def test_skippable_gate_failure_is_failure(self) -> None:
        jobs = _all_gates("success")
        # Replace a skippable gate with a hard failure.
        target = SKIPPABLE_GATE_JOBS[0]
        jobs = [j for j in jobs if j["name"] != target] + [_job(target, "failure")]
        code, report = evaluate(jobs)
        assert code == EXIT_FAILURE
        assert target in report

    def test_missing_gate_is_pending(self) -> None:
        # One aggregate gate absent entirely → not yet provable → PENDING.
        code, _ = evaluate(_all_gates("success")[:-1])
        assert code == EXIT_PENDING

    def test_gate_still_running_is_pending(self) -> None:
        jobs = _all_gates("success")
        jobs[0] = _job(STRICT_GATE_JOBS[0], None, status="in_progress")
        code, _ = evaluate(jobs)
        assert code == EXIT_PENDING

    def test_empty_run_is_pending_not_vacuous_success(self) -> None:
        # No jobs at all must never be a vacuous green.
        code, _ = evaluate([])
        assert code == EXIT_PENDING

    def test_leaf_failure_fails_even_before_gates_exist(self) -> None:
        # Default-deny sweep: a non-allowlisted leaf failure fails fast, even if
        # the aggregate gates have not been created yet. This is the class the
        # old ci-summary missed (detect-changes failure skips test-parallel →
        # tests-gate greens on ``skipped``).
        jobs = [_job("Detect Changes", "failure")]
        code, report = evaluate(jobs)
        assert code == EXIT_FAILURE
        assert "Detect Changes" in report

    def test_test_split_failure_is_caught_by_sweep(self) -> None:
        # A single templated matrix leg failing must fail the gate.
        jobs = _all_gates("success") + [_job("Tests (Split 3/15)", "failure")]
        code, report = evaluate(jobs)
        assert code == EXIT_FAILURE
        assert "Tests (Split 3/15)" in report

    def test_allowlisted_advisory_failure_is_ignored(self) -> None:
        # A failing advisory job (Test-Failure Ratchet Gate) must NOT block.
        jobs = _all_gates("success") + [_job("Test-Failure Ratchet Gate", "failure")]
        code, _ = evaluate(jobs)
        assert code == EXIT_SUCCESS

    def test_allowlisted_runtime_boot_smoke_failure_is_ignored(self) -> None:
        # runtime-boot-smoke is advisory (OMN-9120); its reusable inner jobs
        # surface prefixed and must be covered by prefix-aware allowlisting.
        jobs = _all_gates("success") + [
            _job("Runtime Boot Smoke (compose) / boot", "failure")
        ]
        code, _ = evaluate(jobs)
        assert code == EXIT_SUCCESS

    def test_allowlisted_zone_filter_inner_failure_is_ignored(self) -> None:
        jobs = _all_gates("success") + [_job("zone-filter / detect-zone", "failure")]
        code, _ = evaluate(jobs)
        assert code == EXIT_SUCCESS

    def test_migration_conflict_failure_is_ignored(self) -> None:
        # Cross-Repo Migration Conflicts is not required and not in ci-summary.
        jobs = _all_gates("success") + [
            _job("Cross-Repo Migration Conflicts", "failure")
        ]
        code, _ = evaluate(jobs)
        assert code == EXIT_SUCCESS

    def test_self_job_is_excluded(self) -> None:
        # The poller's own in-progress/failed record must not affect the verdict.
        jobs = _all_gates("success") + [_job("CI Summary", None, status="in_progress")]
        code, _ = evaluate(jobs)
        assert code == EXIT_SUCCESS

    def test_partial_rerun_uses_latest_attempt(self) -> None:
        # Attempt 1 failed; attempt 2 re-ran the same gate and passed → SUCCESS.
        jobs = _all_gates("success")
        jobs[0] = _job(STRICT_GATE_JOBS[0], "failure", attempt=1)
        jobs.append(_job(STRICT_GATE_JOBS[0], "success", attempt=2))
        code, _ = evaluate(jobs)
        assert code == EXIT_SUCCESS

    def test_run_attempt_filters_stale_failure_from_previous_attempt(self) -> None:
        jobs = [_job(j["name"], "failure", attempt=1) for j in _all_gates()]
        jobs.extend(_job(j["name"], "success", attempt=2) for j in _all_gates())
        code, _ = evaluate(jobs, run_attempt=2)
        assert code == EXIT_SUCCESS

    def test_current_attempt_missing_gate_is_pending_not_stale_failure(self) -> None:
        jobs = [_job(j["name"], "failure", attempt=1) for j in _all_gates()]
        current_attempt = _all_gates()[:-1]
        jobs.extend(_job(j["name"], "success", attempt=2) for j in current_attempt)
        code, report = evaluate(jobs, run_attempt=2)
        assert code == EXIT_PENDING
        assert _all_gates()[-1]["name"] in report

    def test_same_attempt_duplicate_job_names_keep_failure(self) -> None:
        jobs = _all_gates("success")
        jobs += [
            _job("Tests (Split)", "failure", attempt=1),
            _job("Tests (Split)", "success", attempt=1),
        ]
        code, report = evaluate(jobs)
        assert code == EXIT_FAILURE
        assert "Tests (Split)" in report

    def test_docs_only_snapshot_is_success(self) -> None:
        # Docs-only: skippable gates skip, strict gates still succeed.
        jobs = [_job(g, "success") for g in STRICT_GATE_JOBS]
        jobs += [_job(g, "skipped") for g in SKIPPABLE_GATE_JOBS]
        code, _ = evaluate(jobs)
        assert code == EXIT_SUCCESS
