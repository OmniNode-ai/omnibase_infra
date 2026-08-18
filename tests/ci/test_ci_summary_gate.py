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

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    ACTOR_CONDITIONAL_CONTEXTS,
    EXIT_FAILURE,
    EXIT_PENDING,
    EXIT_SUCCESS,
    EXPECTED_EXTERNAL_CONTEXTS,
    MEASURED_NOT_ENFORCED_CONTEXTS,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
    applicable_external_contexts,
    evaluate,
    evaluate_external_contexts,
    latest_check_run_by_name,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DEPLOY_AGENT_GATE = "Deploy Agent Tests (OMN-15378) / deploy-agent-tests"
APPLICATION_DB_GATE = "Application Database Domain Enforcement (OMN-15361)"

# Real, unedited `commits/{sha}/check-runs` rows captured from the 16 dev PRs
# merged 2026-07-29T23:04Z → 2026-07-30T14:54Z, filtered to merge-time state.
# See the file's `_provenance` block for the exact capture command.
EXTERNAL_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "ci"
    / "fixtures"
    / "omn15496_merge_time_external_check_runs.json"
)


def _load_external_fixture() -> dict[str, Any]:
    loaded = json.loads(EXTERNAL_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _external_fixture(pr: str) -> list[dict[str, object]]:
    """Merge-time external check-runs for one real merged dev PR."""
    entry = _load_external_fixture()["pull_requests"][pr]
    runs = entry["check_runs"]
    assert isinstance(runs, list) and runs
    return [dict(row) for row in runs]


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

    def test_application_database_gate_is_strict_and_unconditional(self) -> None:
        """OMN-15361 source and rebuilt-Docker controls must gate CI Summary."""
        assert APPLICATION_DB_GATE in STRICT_GATE_JOBS
        jobs = [
            job for job in _all_gates("success") if job["name"] != APPLICATION_DB_GATE
        ]
        jobs.append(_job(APPLICATION_DB_GATE, "failure"))
        code, report = evaluate(jobs)
        assert code == EXIT_FAILURE
        assert APPLICATION_DB_GATE in report

        workflow_job = _load_workflow(CI_WORKFLOW)["jobs"][
            "application-database-domain-enforcement"
        ]
        assert workflow_job["name"] == APPLICATION_DB_GATE
        assert "if" not in workflow_job
        assert workflow_job["needs"] == "occ-preflight"


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


class TestExternalContextAssertion:
    """OMN-15496 — cross-workflow contexts must gate the required rollup.

    Checks 1-3 above read ``actions/runs/${RUN_ID}/jobs``: ci.yml's OWN run. A
    check produced by any other workflow file is invisible to them, and
    ``omnibase_infra``'s ``dev`` requires exactly one context (``CI Summary``,
    ``strict=false``) — so those checks were enforced by neither layer.

    Every fixture here is an UNEDITED ``commits/{sha}/check-runs`` payload from a
    real merged ``dev`` PR, filtered to merge-time state
    (``started_at <= mergedAt``). No hand-built dict stands in for the API shape.
    """

    def test_red_before_the_real_incident_greened(self) -> None:
        """The exact false green this gate exists to prevent.

        PR #2555 merged 2026-07-30T04:25:09Z with ``CI Summary`` = success while
        ``deploy-gate / deploy-gate`` = failure on the same head SHA. All 53
        in-run jobs really were green, so the run-scoped checks CANNOT catch it —
        this is "exists but wrong", not a missing import.
        """
        jobs = _all_gates("success")
        # Pre-condition — the run-scoped verdict alone (i.e. this module's
        # behaviour before OMN-15496) is SUCCESS on that very head SHA.
        assert evaluate(jobs)[0] == EXIT_SUCCESS

        # ...and the payload really does carry the red, so the RED below is not
        # passing for some unrelated reason.
        deploy_gate = [
            row
            for row in _external_fixture("2555")
            if row["name"] == "deploy-gate / deploy-gate"
        ]
        assert [row["conclusion"] for row in deploy_gate] == ["failure"]

        code, report = evaluate(
            jobs,
            check_runs=_external_fixture("2555"),
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_FAILURE
        assert "deploy-gate / deploy-gate" in report

    def test_falsification_control_tuple_entry_is_load_bearing(self) -> None:
        """Drop the context from the tuple and the SAME payload greens.

        Without this the RED above could be passing for an unrelated reason.
        """
        remaining = tuple(
            c for c in EXPECTED_EXTERNAL_CONTEXTS if c != "deploy-gate / deploy-gate"
        )
        assert len(remaining) == len(EXPECTED_EXTERNAL_CONTEXTS) - 1
        code, _ = evaluate(
            _all_gates("success"),
            check_runs=_external_fixture("2555"),
            external_contexts=remaining,
        )
        assert code == EXIT_SUCCESS

    def test_absent_context_is_pending_never_success(self) -> None:
        """A context that never reports must not read as passing.

        Absence and success are indistinguishable to branch protection; that is
        the OMN-14456 AC4 hole. PENDING is converted to FAILURE by the caller's
        deadline.
        """
        payload = [
            row
            for row in _external_fixture("2567")
            if row["name"] != "URL Authority Gate"
        ]
        code, report = evaluate(
            _all_gates("success"),
            check_runs=payload,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_PENDING
        assert "URL Authority Gate" in report

    def test_missing_payload_is_pending_never_success(self) -> None:
        """A failed check-runs fetch must not green the gate."""
        code, _ = evaluate(
            _all_gates("success"),
            check_runs=None,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_PENDING

    def test_still_running_context_is_pending(self) -> None:
        payload = [dict(row) for row in _external_fixture("2567")]
        for row in payload:
            if row["name"] == "CodeQL":
                row["status"] = "in_progress"
                row["conclusion"] = None
        code, _ = evaluate(
            _all_gates("success"),
            check_runs=payload,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_PENDING

    def test_skipped_external_context_fails_closed(self) -> None:
        """`skipped` is not a pass for an external context (OMN-15057 vector)."""
        payload = [dict(row) for row in _external_fixture("2567")]
        for row in payload:
            if row["name"] == "verify / verify":
                row["conclusion"] = "skipped"
        code, report = evaluate(
            _all_gates("success"),
            check_runs=payload,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_FAILURE
        assert "verify / verify" in report

    def test_no_external_contexts_means_no_assertion(self) -> None:
        """merge_group / workflow_dispatch have no PR-scoped context set."""
        code, _ = evaluate(_all_gates("success"), check_runs=None, external_contexts=())
        assert code == EXIT_SUCCESS

    def test_latest_wins_resolution_matches_github(self) -> None:
        """A rerun's green supersedes the earlier red for the same name.

        Check-runs accumulate on a SHA forever, so "any red fails" would make
        every transient red permanent and delete rerun as a recovery path.
        Measured: that rule blocks 6 of the 16 sampled merged PRs, 5 of them on
        already-reruns-green contexts.
        """
        red_then_green = [
            {
                "name": "CodeQL",
                "status": "completed",
                "conclusion": "failure",
                "started_at": "2026-07-30T01:00:00Z",
                "id": 1,
            },
            {
                "name": "CodeQL",
                "status": "completed",
                "conclusion": "success",
                "started_at": "2026-07-30T02:00:00Z",
                "id": 2,
            },
        ]
        resolved = latest_check_run_by_name(red_then_green)
        assert resolved["CodeQL"].conclusion == "success"

        # ...and the reverse order still resolves to the LATEST, not the best.
        green_then_red = [dict(row) for row in red_then_green]
        green_then_red[0]["conclusion"] = "success"
        green_then_red[1]["conclusion"] = "failure"
        assert (
            latest_check_run_by_name(green_then_red)["CodeQL"].conclusion == "failure"
        )

    def test_external_contexts_disjoint_from_in_run_gates(self) -> None:
        """No context may be asserted on both surfaces.

        ``occ-preflight / eligibility`` is the one name observed both inside and
        outside ci.yml's check suite; asserting it twice would double-count an
        ambiguous name (OMN-15112).
        """
        overlap = set(EXPECTED_EXTERNAL_CONTEXTS) & {
            *STRICT_GATE_JOBS,
            *SKIPPABLE_GATE_JOBS,
        }
        assert overlap == set(), f"asserted on both surfaces: {sorted(overlap)}"

    def test_excluded_contexts_are_recorded_with_a_reason(self) -> None:
        """Exclusions are data, not silence — each carries its measurement."""
        assert MEASURED_NOT_ENFORCED_CONTEXTS
        for context, reason in MEASURED_NOT_ENFORCED_CONTEXTS.items():
            assert context not in EXPECTED_EXTERNAL_CONTEXTS
            assert len(reason) > 40, f"{context} needs a substantive reason"

    def test_no_wedge_replay_over_sixteen_merged_dev_prs(self) -> None:
        """The admitted tuple must not block PRs that legitimately merged.

        A required gate that reds on healthy PRs is worse than the hole it
        closes. Replaying the merge-time payload of every dev PR merged
        2026-07-29T23:04Z → 2026-07-30T14:54Z must yield exactly ONE block, and
        it must be the real defect (#2555, deploy-gate red at merge).
        """
        fixture = _load_external_fixture()
        blocked: dict[str, list[str]] = {}
        for pr, entry in fixture["pull_requests"].items():
            failures, unresolved = evaluate_external_contexts(
                entry["check_runs"], EXPECTED_EXTERNAL_CONTEXTS
            )
            if failures or unresolved:
                blocked[pr] = failures + unresolved

        assert len(fixture["pull_requests"]) == 16
        assert blocked == {"2555": ["deploy-gate / deploy-gate"]}, (
            "seed membership changed the no-wedge profile. Re-measure per-context "
            "merge-time report rate over recent merged dev PRs before admitting a "
            f"context; got {blocked}"
        )

    def test_every_admitted_context_reported_on_every_sampled_pr(self) -> None:
        """Admission rule, enforced: 100% merge-time presence or it can wedge dev."""
        fixture = _load_external_fixture()
        missing: dict[str, list[str]] = {}
        for pr, entry in fixture["pull_requests"].items():
            observed = latest_check_run_by_name(entry["check_runs"])
            absent = [c for c in EXPECTED_EXTERNAL_CONTEXTS if c not in observed]
            if absent:
                missing[pr] = absent
        assert missing == {}, (
            "a context absent from any sampled PR does not report on every PR "
            f"shape; requiring it burns the poll deadline and wedges dev: {missing}"
        )


class TestExternalAssertionIsWiredIntoCiYml:
    """A rule is not a mechanism — the gate must be load-bearing, not merely defined."""

    def test_poll_step_passes_check_runs_and_event_name(self) -> None:
        job = _load_workflow(CI_WORKFLOW)["jobs"]["ci-summary"]
        poll = next(
            s for s in job["steps"] if "Poll run jobs" in str(s.get("name", ""))
        )
        run = str(poll["run"])
        assert "--check-runs-file check_runs.json" in run
        assert '--event-name "${EVENT_NAME}"' in run
        assert "commits/${HEAD_SHA}/check-runs" in run
        # The verdict step owns pass/fail: it must NOT be continue-on-error, or
        # the assertion is decorative exactly like the report-only cascade step.
        assert poll.get("continue-on-error") is not True
        assert job["permissions"]["checks"] == "read"
        assert "HEAD_SHA" in poll["env"] and "EVENT_NAME" in poll["env"]

    def test_failed_check_runs_fetch_removes_the_file(self) -> None:
        """A stale check_runs.json from a previous iteration must never be reused."""
        job = _load_workflow(CI_WORKFLOW)["jobs"]["ci-summary"]
        poll = next(
            s for s in job["steps"] if "Poll run jobs" in str(s.get("name", ""))
        )
        run = str(poll["run"])
        assert run.count("rm -f check_runs.json") >= 2


# --------------------------------------------------------------------------
# OMN-15532 — actor-conditional external contexts.
#
# OMN-15496 admitted `gate / CodeRabbit Thread Check` as a fail-closed external
# context after measuring it 16/16 present over #2546…#2567. That window held no
# Dependabot PR. cr-thread-gate-caller.yml skips the caller job when
# `github.actor == 'dependabot[bot]'`, and because the context is the
# `caller-job / reusable-job` form the inner job never materialises — the
# check-run is ABSENT, not `skipped`. Absent burns the 90 min deadline and then
# fails closed against the SOLE required context on infra dev.
# --------------------------------------------------------------------------

DEPENDABOT_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "ci"
    / "fixtures"
    / "omn15532_dependabot_pr2522_check_runs.json"
)
CR_THREAD_CONTEXT = "gate / CodeRabbit Thread Check"


def _dependabot_check_runs() -> list[dict[str, Any]]:
    """Real, unedited check-runs from Dependabot PR #2522 (head 2cdf352d)."""
    return list(json.loads(DEPENDABOT_FIXTURE.read_text())["check_runs"])


class TestActorConditionalExternalContexts:
    def test_fixture_is_the_real_absent_case(self) -> None:
        """Guard the premise: the fixture must genuinely lack ONLY this context."""
        rows = _dependabot_check_runs()
        names = {str(r["name"]) for r in rows}
        absent = [c for c in EXPECTED_EXTERNAL_CONTEXTS if c not in names]
        # If this ever changes, the exemption below is no longer justified.
        assert absent == [CR_THREAD_CONTEXT], (
            "fixture no longer isolates the CR-thread-gate absence; re-measure "
            "before trusting the exemption"
        )

    def test_dependabot_pr_is_not_wedged(self) -> None:
        """AC1 — the real absent-context payload resolves for dependabot[bot]."""
        code, report = evaluate(
            _all_gates("success"),
            check_runs=_dependabot_check_runs(),
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
            pr_author="dependabot[bot]",
        )
        assert code == EXIT_SUCCESS, report

    def test_same_payload_still_blocks_a_human_author(self) -> None:
        """AC2 — the control that matters: the exemption must not leak."""
        code, report = evaluate(
            _all_gates("success"),
            check_runs=_dependabot_check_runs(),
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
            pr_author="jonahgabriel",
        )
        assert code == EXIT_PENDING, report
        assert CR_THREAD_CONTEXT in report

    def test_absent_pr_author_enforces_everything(self) -> None:
        """A forgotten --pr-author must enforce, never exempt."""
        code, _ = evaluate(
            _all_gates("success"),
            check_runs=_dependabot_check_runs(),
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
            pr_author=None,
        )
        assert code == EXIT_PENDING

    def test_exemption_entry_is_load_bearing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC3 — falsification control.

        Genuinely REMOVE the registry entry and confirm the very same fixture
        and author go back to blocking. Without this, the AC1 green could come
        from the fixture rather than from the exemption.
        """
        import scripts.ci.ci_summary_gate as gate_mod

        monkeypatch.setattr(gate_mod, "ACTOR_CONDITIONAL_CONTEXTS", {})
        code, report = evaluate(
            _all_gates("success"),
            check_runs=_dependabot_check_runs(),
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
            pr_author="dependabot[bot]",
        )
        assert code == EXIT_PENDING, report
        assert CR_THREAD_CONTEXT in report

    def test_exemption_drops_only_that_one_context(self) -> None:
        """The dependabot exemption must not quietly widen."""
        pruned = applicable_external_contexts(
            EXPECTED_EXTERNAL_CONTEXTS, "dependabot[bot]"
        )
        assert set(EXPECTED_EXTERNAL_CONTEXTS) - set(pruned) == {CR_THREAD_CONTEXT}

    def test_registry_keys_are_asserted_contexts(self) -> None:
        """AC4 — cannot exempt a context that was never asserted."""
        for context in ACTOR_CONDITIONAL_CONTEXTS:
            assert context in EXPECTED_EXTERNAL_CONTEXTS, context

    def test_registry_actors_are_concrete_logins(self) -> None:
        """AC4 — no wildcard/empty actor may blanket-disable a context."""
        for context, actors in ACTOR_CONDITIONAL_CONTEXTS.items():
            assert actors, f"{context} declares an empty actor tuple"
            for actor in actors:
                assert actor.strip(), f"{context} declares a blank actor"
                assert actor not in {"*", "all"}, f"{context} declares wildcard {actor}"

    def test_a_failing_context_still_fails_for_dependabot(self) -> None:
        """The exemption is applicability, not a bypass: reds still block."""
        rows = _dependabot_check_runs()
        for row in rows:
            if row["name"] == "deploy-gate / deploy-gate":
                row["conclusion"] = "failure"
        code, report = evaluate(
            _all_gates("success"),
            check_runs=rows,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
            pr_author="dependabot[bot]",
        )
        assert code == EXIT_FAILURE, report

    def test_ci_yml_passes_pr_author(self) -> None:
        """AC5 — a rule is not a mechanism: the wiring must exist in ci.yml."""
        job = _load_workflow(CI_WORKFLOW)["jobs"]["ci-summary"]
        poll = next(
            s for s in job["steps"] if "Poll run jobs" in str(s.get("name", ""))
        )
        assert '--pr-author "${PR_AUTHOR:-}"' in str(poll["run"])
        assert "PR_AUTHOR" in poll["env"]


# --------------------------------------------------------------------------
# OMN-15979 — "Integration Test Removal Gate" (OMN-8732) admitted as an
# external context.
#
# Same residual class as OMN-15496: the job's own header says "No override
# mechanism", but before this admission it was reachable by neither branch
# protection (dev requires only `CI Summary`) nor this poller — grep for
# "integration test removal|test-removal" over the pre-OMN-15979 module
# returned nothing. PR #2720 merged 2026-08-11T04:47:58Z with this context
# `failure`.
# --------------------------------------------------------------------------

ITRG_CONTEXT = "Integration Test Removal Gate"

OMN15979_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "ci"
    / "fixtures"
    / "omn15979_merge_time_external_check_runs.json"
)


def _load_omn15979_fixture() -> dict[str, Any]:
    loaded = json.loads(OMN15979_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _omn15979_check_runs(pr: str) -> list[dict[str, object]]:
    """Merge-time external check-runs for one real merged dev PR (OMN-15979 window)."""
    entry = _load_omn15979_fixture()["pull_requests"][pr]
    runs = entry["check_runs"]
    assert isinstance(runs, list) and runs
    return [dict(row) for row in runs]


class TestIntegrationTestRemovalGateExternalContext:
    """OMN-15979 — admission proof + regression fixture for ITRG."""

    def test_context_is_admitted(self) -> None:
        assert ITRG_CONTEXT in EXPECTED_EXTERNAL_CONTEXTS

    def test_every_sampled_pr_reports_the_full_expected_set(self) -> None:
        """Admission rule, enforced: 100% merge-time presence over this window too.

        Same check as OMN-15496's ``test_every_admitted_context_reported_on_
        every_sampled_pr``, replayed against the OMN-15979 window (#2705…#2720)
        rather than the OMN-15496 window (#2546…#2567) — two independent
        16-PR samples, not one restated.
        """
        fixture = _load_omn15979_fixture()
        missing: dict[str, list[str]] = {}
        for pr, entry in fixture["pull_requests"].items():
            observed = latest_check_run_by_name(entry["check_runs"])
            absent = [c for c in EXPECTED_EXTERNAL_CONTEXTS if c not in observed]
            if absent:
                missing[pr] = absent
        assert missing == {}, (
            "a context absent from any PR in the OMN-15979 sample window does "
            f"not report on every PR shape: {missing}"
        )

    def test_no_wedge_replay_over_this_window(self) -> None:
        """Replaying the current window must block exactly the real red.

        16 dev PRs merged 2026-08-09T23:11:14Z → 2026-08-11T04:47:58Z
        (#2705…#2720): every context in :data:`EXPECTED_EXTERNAL_CONTEXTS` must
        resolve green on 15 of them and the block must be exactly #2720 on
        exactly ``Integration Test Removal Gate`` — not deploy-gate, not verify,
        not a second, unrelated context riding along.
        """
        fixture = _load_omn15979_fixture()
        blocked: dict[str, list[str]] = {}
        for pr, entry in fixture["pull_requests"].items():
            failures, unresolved = evaluate_external_contexts(
                entry["check_runs"], EXPECTED_EXTERNAL_CONTEXTS
            )
            if failures or unresolved:
                blocked[pr] = failures + unresolved

        assert len(fixture["pull_requests"]) == 16
        assert blocked == {"2720": [ITRG_CONTEXT]}, (
            "seed membership changed the no-wedge profile for the OMN-15979 "
            f"window; got {blocked}"
        )

    def test_red_at_merge_blocks_the_real_pr(self) -> None:
        """Live proof: #2720's real payload must FAIL the required rollup.

        This is the exact PR the ticket's evidence cites as merging past a red
        ``Integration Test Removal Gate`` — replayed here to prove that, with
        this entry wired, the identical payload would have been blocked
        end-to-end rather than merged.
        """
        jobs = _all_gates("success")
        # Pre-condition: the run-scoped verdict alone is SUCCESS on this head —
        # the in-run jobs are unaffected, so only the external assertion can
        # catch this.
        assert evaluate(jobs)[0] == EXIT_SUCCESS

        payload = _omn15979_check_runs("2720")
        itrg_rows = [row for row in payload if row["name"] == ITRG_CONTEXT]
        assert [row["conclusion"] for row in itrg_rows] == ["failure"]

        code, report = evaluate(
            jobs,
            check_runs=payload,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_FAILURE
        assert ITRG_CONTEXT in report

    def test_falsification_control_tuple_entry_is_load_bearing(self) -> None:
        """Drop the context from the tuple and the SAME real-red payload greens.

        Without this, the FAILURE above could come from an unrelated context
        (e.g. ``verify / verify``'s early reruns) rather than from ITRG itself.
        """
        remaining = tuple(c for c in EXPECTED_EXTERNAL_CONTEXTS if c != ITRG_CONTEXT)
        assert len(remaining) == len(EXPECTED_EXTERNAL_CONTEXTS) - 1
        code, report = evaluate(
            _all_gates("success"),
            check_runs=_omn15979_check_runs("2720"),
            external_contexts=remaining,
        )
        assert code == EXIT_SUCCESS, report

    def test_synthetic_red_flips_a_clean_pr_to_failure(self) -> None:
        """AC — a synthetic red on an otherwise-clean real PR flips the verdict.

        Starts from PR #2719's genuinely all-green merge-time payload, flips
        only ``Integration Test Removal Gate`` to ``failure`` in place (the
        rest of the real payload is untouched), and asserts CI Summary's
        verdict flips from SUCCESS to FAILURE — and back to SUCCESS once the
        synthetic row is reverted. This is the fixture the ticket's AC calls
        for: a synthetic red proven to flip the verdict once wired.
        """
        clean_payload = _omn15979_check_runs("2719")
        code, _ = evaluate(
            _all_gates("success"),
            check_runs=clean_payload,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_SUCCESS

        reddened = [dict(row) for row in clean_payload]
        flipped = False
        for row in reddened:
            if row["name"] == ITRG_CONTEXT:
                row["conclusion"] = "failure"
                flipped = True
        assert flipped, "fixture no longer carries an ITRG row to synthetically redden"

        code, report = evaluate(
            _all_gates("success"),
            check_runs=reddened,
            external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == EXIT_FAILURE
        assert ITRG_CONTEXT in report

    def test_measurement_windows_are_disjoint_from_the_original_seed(self) -> None:
        """Guard the premise: this is a second independent sample, not a restatement.

        If the two fixtures ever collapsed onto the same PR set, the "two
        independent windows" claim in the module docstring would be false.
        """
        omn15496_prs = set(_load_external_fixture()["pull_requests"])
        omn15979_prs = set(_load_omn15979_fixture()["pull_requests"])
        assert omn15496_prs.isdisjoint(omn15979_prs)
        assert len(omn15979_prs) == 16


# --------------------------------------------------------------------------
# OMN-16216 — draft-state CI admission gate (mirrors onex_change_control PR
# #6686 / OMN-15731). omnibase_infra's own `ci:ready` label-gate pilot
# (OMN-15731, infra#2693) was still OPEN/unmerged when this ticket was built
# -- there was nothing merged to "migrate from" the way OCC's #6216 -> #6686
# revision had. This class instead ships BOTH admission arms in one PR:
# native PR draft state as the PRIMARY signal, `ci:ready` retained as a
# transition-window dual-accept fallback for forward compatibility with
# #2693 (or any future equivalent), even though no producer of that label
# currently exists on this repo's live dev.
#
# `test-parallel` (the heavy split-matrix run, the actual expensive workload)
# is the load-bearing target. The trap this pins (AC(b) on OMN-16216):
# `tests-gate` ("CI Tests Gate", a STRICT_GATE_JOBS member) already treats a
# `skipped` test-parallel result as pass for the pre-existing docs-only
# exemption. Without the fix below, a draft, non-docs-only PR would hit that
# exact same skip-is-pass path and read CI Summary = SUCCESS with zero tests
# run. `deploy-gate`, `codeql`, and `hostile-review` are also gated -- each
# verified against its own fail-closed (or non-gating) path per the module
# docstring's OMN-15496 external-context mechanism, not asserted blind.
# --------------------------------------------------------------------------


DEPLOY_GATE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-gate.yml"
SECURITY_SCAN_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "security-scan.yml"
HOSTILE_REVIEWER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "hostile-reviewer.yml"


class TestDraftStateAdmissionGateOmn16216:
    def test_ci_yml_pull_request_trigger_includes_ready_for_review(self) -> None:
        """A draft->ready flip must re-evaluate the gate on the current head
        without requiring a new push (root brief invariant)."""
        workflow = _load_workflow(CI_WORKFLOW)
        # PyYAML's default (YAML 1.1) resolver parses the bare `on:` key as
        # the boolean True, not the string "on" -- this is not a typo.
        pr_trigger = workflow[True]["pull_request"]
        for event_type in ("labeled", "unlabeled", "ready_for_review"):
            assert event_type in pr_trigger["types"], (
                f"ci.yml pull_request.types is missing {event_type!r}; a "
                "label change or draft->ready flip would not retrigger the "
                "workflow"
            )

    def test_test_parallel_if_has_both_admission_arms_dual_accept(self) -> None:
        job = _load_workflow(CI_WORKFLOW)["jobs"]["test-parallel"]
        condition = str(job["if"])
        assert "!github.event.pull_request.draft" in condition
        assert "contains(github.event.pull_request.labels.*.name, 'ci:ready')" in (
            condition
        )
        # Non-pull_request events (push, merge_group, workflow_dispatch) carry
        # no PR draft/label state and must remain unaffected.
        assert "github.event_name != 'pull_request'" in condition
        # Main-boundary carve-out (CLAUDE.md rule 4): the dev->main promotion
        # boundary is never narrowed, so any PR targeting main must run the
        # full fleet unconditionally regardless of draft/ci:ready state.
        assert "github.base_ref != 'dev'" in condition
        # Both arms sit inside the same top-level `||` group, not nested
        # under an `&&` that would make one a precondition for the other.
        or_clause_start = condition.index("(github.event_name")
        or_clause = condition[or_clause_start:]
        assert or_clause.count("||") >= 3, (
            "expected a flat OR chain (event_name / base_ref / !draft / "
            f"ci:ready) -- got: {or_clause}"
        )

    def test_tests_gate_distinguishes_docs_only_skip_from_admission_skip(
        self,
    ) -> None:
        """The mechanism that closes the AC(b) trap must exist in ci.yml, not
        just in this poller's already-generic STRICT-gate failure path."""
        job = _load_workflow(CI_WORKFLOW)["jobs"]["tests-gate"]
        step = next(
            s
            for s in job["steps"]
            if "Check test matrix results" in str(s.get("name", ""))
        )
        run = str(step["run"])
        env = step["env"]
        assert "DOCS_ONLY" in env
        assert "ADMITTED" in env
        assert "!github.event.pull_request.draft" in str(env["ADMITTED"])
        assert "github.base_ref != 'dev'" in str(env["ADMITTED"])
        assert '[ "$DOCS_ONLY" = "true" ]' in run
        assert '[ "$ADMITTED" != "true" ]' in run
        # The admission-skip branch must exit non-zero (fail closed), distinct
        # from the docs-only branch which must not.
        assert "exit 1" in run
        assert "zone-filter" in job["needs"]

    def test_unadmitted_skip_reported_as_ci_tests_gate_failure_fails_closed(
        self,
    ) -> None:
        """Simulates the exact live outcome once tests-gate's bash exits 1 for
        an admission-skip: "CI Tests Gate" reports `failure`, and the
        poller's pre-existing (unmodified) STRICT-gate logic must fail the
        whole summary closed -- proving no new poller mechanism was needed."""
        gates = _all_gates("success")
        for gate in gates:
            if gate["name"] == "CI Tests Gate":
                gate["conclusion"] = "failure"
        code, report = evaluate(gates + [_job("Detect Changes", "success")])
        assert code == EXIT_FAILURE, report
        assert "CI Tests Gate" in report

    def test_docs_only_skip_is_still_success(self) -> None:
        """Control: the pre-existing docs-only exemption must be unaffected --
        "CI Tests Gate" reporting `success` (as it does when the bash
        script's docs-only branch completes without exiting 1) still
        passes."""
        code, report = evaluate(
            _all_gates("success") + [_job("Detect Changes", "success")]
        )
        assert code == EXIT_SUCCESS, report

    def test_deploy_gate_job_if_has_both_admission_arms(self) -> None:
        workflow = _load_workflow(DEPLOY_GATE_WORKFLOW)
        job = workflow["jobs"]["deploy-gate"]
        condition = str(job["if"])
        assert "!github.event.pull_request.draft" in condition
        assert "contains(github.event.pull_request.labels.*.name, 'ci:ready')" in (
            condition
        )
        assert "github.event_name != 'pull_request'" in condition
        assert "github.base_ref != 'dev'" in condition
        for event_type in ("labeled", "unlabeled", "ready_for_review"):
            assert event_type in workflow[True]["pull_request"]["types"]
        # "deploy-gate / deploy-gate" must stay a registered external context
        # so a skip fails closed via the EXPECTED_EXTERNAL_CONTEXTS path.
        assert "deploy-gate / deploy-gate" in EXPECTED_EXTERNAL_CONTEXTS

    def test_codeql_job_if_has_both_admission_arms(self) -> None:
        workflow = _load_workflow(SECURITY_SCAN_WORKFLOW)
        job = workflow["jobs"]["codeql"]
        condition = str(job["if"])
        assert "!github.event.pull_request.draft" in condition
        assert "contains(github.event.pull_request.labels.*.name, 'ci:ready')" in (
            condition
        )
        assert "github.event_name != 'pull_request'" in condition
        assert "github.base_ref != 'dev'" in condition
        for event_type in ("labeled", "unlabeled", "ready_for_review"):
            assert event_type in workflow[True]["pull_request"]["types"]
        assert "CodeQL" in EXPECTED_EXTERNAL_CONTEXTS

    def test_hostile_review_job_if_has_both_admission_arms(self) -> None:
        workflow = _load_workflow(HOSTILE_REVIEWER_WORKFLOW)
        job = workflow["jobs"]["hostile-review"]
        condition = str(job["if"])
        assert "!github.event.pull_request.draft" in condition
        assert "contains(github.event.pull_request.labels.*.name, 'ci:ready')" in (
            condition
        )
        assert "github.event_name != 'pull_request'" in condition
        assert "github.base_ref != 'dev'" in condition
        for event_type in ("labeled", "unlabeled", "ready_for_review"):
            assert event_type in workflow[True]["pull_request"]["types"]
        # Hostile Review Gate is deliberately absent from every gating
        # surface -- pure runner-cost win, not a fail-closed claim.
        assert "Hostile Review Gate" not in STRICT_GATE_JOBS
        assert "Hostile Review Gate" not in SKIPPABLE_GATE_JOBS
        assert "Hostile Review Gate" not in EXPECTED_EXTERNAL_CONTEXTS

    def test_red_control_pre_migration_test_parallel_shape_had_no_draft_arm(
        self,
    ) -> None:
        """RED-before control: pins that the pre-OMN-16216 `test-parallel`
        condition string had no draft-state clause at all -- a draft PR was
        previously ADMITTED (main/non-dev events aside) purely because no
        admission gate existed. This is the state this ticket replaces; it is
        not a live assertion against ci.yml (which now has the arm) -- it
        pins the pre-migration string so a future revert is caught by drift,
        not by eyeballing history."""
        pre_migration_condition = (
            "needs.occ-preflight.result == 'success' && "
            "(needs.zone-filter.outputs.docs_only != 'true')"
        )
        assert "!github.event.pull_request.draft" not in pre_migration_condition
        live_condition = str(_load_workflow(CI_WORKFLOW)["jobs"]["test-parallel"]["if"])
        assert live_condition != pre_migration_condition, (
            "live ci.yml test-parallel condition must have moved past the "
            "pre-migration shape captured above"
        )
