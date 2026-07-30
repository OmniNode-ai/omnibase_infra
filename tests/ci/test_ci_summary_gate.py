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

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXIT_FAILURE,
    EXIT_PENDING,
    EXIT_SUCCESS,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
    evaluate,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DEPLOY_AGENT_GATE = "Deploy Agent Tests (OMN-15378) / deploy-agent-tests"


def _load_workflow(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _inner_job_names(workflow: dict[str, Any]) -> set[str]:
    """Job ids and display names declared by a (called) workflow."""
    jobs: dict[str, Any] = workflow["jobs"]
    names = {str(job_id) for job_id in jobs}
    names |= {
        str((body or {}).get("name"))
        for body in jobs.values()
        if (body or {}).get("name")
    }
    return names


def _observable_job_names(workflow: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Names the Actions jobs API can report for ``workflow``'s own run.

    Returns ``(exact_names, remote_caller_prefixes)``. Verified against live run
    30506617326: a plain job reports its display name (falling back to its job
    id); a reusable-workflow caller that EXECUTES reports only
    ``"<caller display name> / <inner job name>"`` rows and never a row under
    its own job id (``occ-preflight / eligibility``, not ``occ-preflight``) — a
    bare caller row appears only when the caller itself skipped (``zone-filter``,
    ``Runtime Boot Smoke (compose)``). Inner jobs of a REMOTE reusable cannot be
    resolved from this repo, so those callers are returned as prefixes.
    """
    exact: set[str] = set()
    remote_prefixes: set[str] = set()
    for job_id, raw in workflow["jobs"].items():
        body: dict[str, Any] = raw or {}
        display = str(body.get("name") or job_id)
        uses = str(body.get("uses") or "")
        if not uses:
            exact.add(display)
            continue
        if uses.startswith("./"):
            called_path = REPO_ROOT / uses[2:]
            assert called_path.is_file(), (
                f"ci.yml job {job_id!r} calls {uses!r}, which does not exist"
            )
            exact |= {
                f"{display} / {inner}"
                for inner in _inner_job_names(_load_workflow(called_path))
            }
        else:
            remote_prefixes.add(display)
    return exact, remote_prefixes


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

    def test_effect_assertion_gate_is_strict_and_fails_closed(self) -> None:
        # RT-5 (OMN-14467): the deploy-trigger effect-assertion gate must be a
        # STRICT CI Summary gate so a reintroduced silent-skip cannot merge to
        # dev green. Assert it is registered STRICT AND that a red result fails
        # the required "CI Summary" context (fail-closed, in-PR).
        assert "Effect-Assertion Gate (RT-5)" in STRICT_GATE_JOBS
        jobs = [
            j
            for j in _all_gates("success")
            if j["name"] != "Effect-Assertion Gate (RT-5)"
        ]
        jobs.append(_job("Effect-Assertion Gate (RT-5)", "failure"))
        code, report = evaluate(jobs)
        assert code == EXIT_FAILURE
        assert "Effect-Assertion Gate (RT-5)" in report

    def test_occ_companion_merged_gate_is_strict_and_fails_closed(self) -> None:
        # OMN-15214: the companion-merged gate makes the 2026-07-26 hygiene-sweep
        # trigger state (OPEN companion + MERGED product PR) unreachable via the
        # merge path. It must be STRICT so a red/absent result fails the required
        # "CI Summary" context — folding into the umbrella instead of adding a
        # new top-level required context avoids the never-reports wedge.
        gate = "OCC Companion Merged Gate (OMN-15214)"
        assert gate in STRICT_GATE_JOBS
        jobs = [j for j in _all_gates("success") if j["name"] != gate]
        jobs.append(_job(gate, "failure"))
        code, report = evaluate(jobs)
        assert code == EXIT_FAILURE
        assert gate in report
        # A skip must also fail closed — the job is unconditional in ci.yml.
        jobs = [j for j in _all_gates("success") if j["name"] != gate]
        jobs.append(_job(gate, "skipped"))
        code, _ = evaluate(jobs)
        assert code == EXIT_FAILURE

    def test_deploy_agent_tests_gate_is_strict_and_fails_closed(self) -> None:
        # OMN-15378 AC3: scripts/deploy-agent/tests/ (201 tests) was wired to RUN
        # on PRs but into no aggregator — absent from dev's required set (only
        # "CI Summary") and absent from both gate lists here, so a RED run left
        # the required context green: the guard was advisory, code not mechanism.
        # It must be STRICT so absent/red/skipped all fail the required context.
        assert DEPLOY_AGENT_GATE in STRICT_GATE_JOBS

        # (a) red → FAILURE
        jobs = [j for j in _all_gates("success") if j["name"] != DEPLOY_AGENT_GATE]
        jobs.append(_job(DEPLOY_AGENT_GATE, "failure"))
        code, report = evaluate(jobs)
        assert code == EXIT_FAILURE
        assert DEPLOY_AGENT_GATE in report

        # (b) skipped → FAILURE (the caller job is unconditional in ci.yml, so a
        # skip means someone re-added an `if:`/path filter, not a legitimate
        # no-op).
        jobs = [j for j in _all_gates("success") if j["name"] != DEPLOY_AGENT_GATE]
        jobs.append(_job(DEPLOY_AGENT_GATE, "skipped"))
        code, _ = evaluate(jobs)
        assert code == EXIT_FAILURE

        # (c) absent from the run entirely → PENDING, never a vacuous SUCCESS.
        # This is the exact pre-fix state (the tests ran in their own workflow,
        # so they never appeared in ci.yml's job list); the poller converts
        # PENDING to FAILURE at its deadline.
        jobs = [j for j in _all_gates("success") if j["name"] != DEPLOY_AGENT_GATE]
        code, report = evaluate(jobs)
        assert code == EXIT_PENDING
        assert DEPLOY_AGENT_GATE in report


class TestGateNamesResolveToRealJobs:
    """Every gate name must be a job the poller can actually observe.

    The poller reads ``actions/runs/${RUN_ID}/jobs`` for ci.yml's OWN run, so a
    gate naming a job that ci.yml never produces (e.g. a job that lives in a
    separately-triggered workflow) is never present → PENDING forever → the
    required context fails closed at the deadline on EVERY PR. That is the
    failure mode of "just add the standalone workflow's job name to
    STRICT_GATE_JOBS", and it is what this test makes unshippable.
    """

    def test_every_gate_name_resolves_to_a_ci_yml_job(self) -> None:
        observable, remote_caller_prefixes = _observable_job_names(
            _load_workflow(CI_WORKFLOW)
        )

        for gate in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS):
            if gate in observable:
                continue
            caller = gate.split(" / ", 1)[0]
            assert caller in remote_caller_prefixes, (
                f"gate {gate!r} is not a name the jobs API can report for "
                "ci.yml's own run. A plain job reports its display name; a "
                "reusable caller reports '<caller display name> / <inner job>' "
                "and NEVER its own job id. A gate the poller cannot observe is "
                "absent forever → PENDING → the required 'CI Summary' context "
                "fails closed at its deadline on every PR. Observable names: "
                f"{sorted(observable)}"
            )

    def test_deploy_agent_gate_caller_is_unconditional(self) -> None:
        # A STRICT gate may never legitimately skip, so the caller job must
        # carry no `if:` and no `needs:` (OMN-15378 AC3).
        job = _load_workflow(CI_WORKFLOW)["jobs"]["deploy-agent-tests"]
        assert job["name"] == "Deploy Agent Tests (OMN-15378)"
        assert job["uses"] == "./.github/workflows/deploy-agent-tests.yml"
        assert "if" not in job
        assert "needs" not in job

    def test_called_deploy_agent_workflow_does_not_self_trigger(self) -> None:
        # Self-triggering would double-run the suite on every PR (duplicate
        # producer) and re-open the path-filter blind spot the caller closes.
        called = _load_workflow(
            REPO_ROOT / ".github" / "workflows" / "deploy-agent-tests.yml"
        )
        # PyYAML parses the `on:` key as the boolean True (YAML 1.1).
        triggers = called.get(True, called.get("on"))
        assert isinstance(triggers, dict)
        assert "workflow_call" in triggers
        assert "pull_request" not in triggers
        assert "push" not in triggers
        assert "merge_group" not in triggers
