# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure-Python logic for audit-branch-protection.sh (OMN-9034).

Extracted from the shell script's inline python3 HEREDOCs so the logic can
be unit-tested by importing the functions directly. The shell wrapper calls
this module for Check A / Check B evaluation and fix-payload construction;
the tests exercise the same functions with synthetic gh-api JSON inputs.

See scripts/audit-branch-protection.sh for the orchestration layer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

GhCaller = Callable[[list[str]], tuple[int, str]]
"""A callable shaped like `(argv_without_gh) -> (returncode, stdout)`.

The production shell script passes a real `gh api`-invoking callable; tests
pass a fake that returns canned JSON. This is the injection seam that makes
the audit logic testable without spawning bash/gh as child processes.
"""

PAGE_SIZE = 100
"""GitHub REST pagination page size; 100 is the API maximum (OMN-9034 thread 7)."""

PR_ONLY_CONTEXTS: frozenset[str] = frozenset({"main-target-guard"})
"""Required contexts whose check-runs bind to the PR head SHA, never to a
default-branch commit, so Check B can never observe them on `main`.

`main-target-guard` (.github/workflows/main-target-guard.yml, OMN-12243)
triggers ONLY on `pull_request` events targeting `main` and binds its
check-run to the PR HEAD SHA (a feature-branch tip). It therefore can never
appear in `main`'s commit check-runs and was permanently mis-flagged as an
orphan across every repo that requires it. Excluded from orphan detection —
verified live, not config drift (OMN-13517). Genuinely-stale contexts on a
repo (e.g. a renamed CI job) are still flagged because they are not in this
allowlist."""


def parse_required_approving_review_count(protection_json: str) -> int:
    """Extract required_approving_review_count from a branch-protection API response.

    Returns 0 when the field is absent or the payload is malformed.
    """
    try:
        d = json.loads(protection_json)
    except json.JSONDecodeError:
        return 0
    prr = d.get("required_pull_request_reviews") or {}
    rac = prr.get("required_approving_review_count", 0)
    return int(rac) if isinstance(rac, int) else 0


def parse_required_contexts(protection_json: str) -> list[str]:
    """Extract required_status_checks.contexts from a branch-protection API response."""
    try:
        d = json.loads(protection_json)
    except json.JSONDecodeError:
        return []
    rsc = d.get("required_status_checks") or {}
    ctxs = rsc.get("contexts", [])
    return [str(c) for c in ctxs] if isinstance(ctxs, list) else []


def build_fix_payload(protection_json: str) -> dict[str, Any]:
    """Build the PUT payload used by --fix to drop required_pull_request_reviews.

    Preserves required_status_checks (contexts + strict) and enforce_admins,
    sets required_pull_request_reviews and restrictions to None. This mirrors
    the inline python3 block in audit-branch-protection.sh.
    """
    d = json.loads(protection_json)
    payload: dict[str, Any] = {}
    rsc = d.get("required_status_checks")
    if rsc:
        payload["required_status_checks"] = {
            "strict": rsc.get("strict", False),
            "contexts": rsc.get("contexts", []),
        }
    payload["enforce_admins"] = bool(
        (d.get("enforce_admins") or {}).get("enabled", False)
    )
    payload["required_pull_request_reviews"] = None
    payload["restrictions"] = None
    return payload


def collect_seen_check_run_names(
    owner: str,
    repo: str,
    commits: int,
    gh: GhCaller,
) -> set[str]:
    """Fetch check-run names across the last N commits on the default branch.

    Paginates check-runs per commit using PAGE_SIZE (thread 7 fix — was
    previously hardcoded to 50, missing contexts on busy repos).
    Returns the union of all check-run names seen.
    """
    rc, commits_stdout = gh(
        ["api", f"repos/{owner}/{repo}/commits?per_page={commits}", "--jq", ".[].sha"]
    )
    if rc != 0:
        return set()
    shas = [s for s in commits_stdout.splitlines() if s.strip()]

    seen: set[str] = set()
    for sha in shas:
        page = 1
        while True:
            rc, stdout = gh(
                [
                    "api",
                    f"repos/{owner}/{repo}/commits/{sha}/check-runs"
                    f"?per_page={PAGE_SIZE}&page={page}",
                ]
            )
            if rc != 0 or not stdout.strip():
                break
            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                break
            runs = data.get("check_runs", [])
            if not runs:
                break
            for run in runs:
                name = run.get("name")
                if isinstance(name, str):
                    seen.add(name)
            if len(runs) < PAGE_SIZE:
                break
            page += 1
    return seen


def find_orphan_contexts(required: list[str], seen: set[str]) -> list[str]:
    """Return contexts in `required` but absent from `seen` check-run names.

    Contexts in `PR_ONLY_CONTEXTS` are excluded: their check-runs bind to the
    PR head SHA (pull_request events) and never appear on default-branch
    commits, so Check B can never observe them — flagging them is a false
    positive (OMN-13517). All other unseen contexts are genuine orphans.
    """
    return [c for c in required if c not in seen and c not in PR_ONLY_CONTEXTS]


def audit_repo(
    owner: str,
    repo: str,
    gh: GhCaller,
    commits_to_scan: int = 5,
) -> dict[str, Any]:
    """Run Check A + Check B against a single repo.

    Returns a dict with:
      - status: "ok" | "skip" | "violation"
      - rac: int (required_approving_review_count, 0 if skipped)
      - required_contexts: list[str]
      - orphan_contexts: list[str]
      - message: human-readable summary
      - protection_json: raw protection payload (empty if skipped)
    """
    rc, protection = gh(["api", f"repos/{owner}/{repo}/branches/main/protection"])
    if rc != 0:
        return {
            "status": "skip",
            "rac": 0,
            "required_contexts": [],
            "orphan_contexts": [],
            "protection_json": "",
            "message": f"branch protection not enabled or inaccessible for {repo}",
        }

    rac = parse_required_approving_review_count(protection)
    required = parse_required_contexts(protection)
    seen = (
        collect_seen_check_run_names(owner, repo, commits_to_scan, gh)
        if required
        else set()
    )
    orphans = find_orphan_contexts(required, seen)

    violations: list[str] = []
    if rac > 0:
        violations.append(f"Check A: required_approving_review_count={rac} (must be 0)")
    for ctx in orphans:
        violations.append(
            f"Check B: context '{ctx}' not found in last-{commits_to_scan}-commit check-runs"
        )

    if violations:
        return {
            "status": "violation",
            "rac": rac,
            "required_contexts": required,
            "orphan_contexts": orphans,
            "protection_json": protection,
            "message": "; ".join(violations),
        }
    return {
        "status": "ok",
        "rac": rac,
        "required_contexts": required,
        "orphan_contexts": [],
        "protection_json": protection,
        "message": f"{repo}: clean",
    }


# ---------------------------------------------------------------------------
# Required-context parity ratchet (OMN-14288) — REPORT-ONLY logic
#
# The functions below are PURE: they take already-fetched live state (required
# contexts, parsed workflow-job graphs) and a declarative manifest, and return
# structured findings. All GitHub I/O (fetching protection + workflow contents)
# lives in audit_required_context_parity_cli.py so this module stays free of
# yaml / network dependencies and its existing Check A/B tests keep running
# under the shell script's bare `python3`.
#
# The audit that motivated this (docs/analyses enforcement-parity spec) proved
# the existing auditor only checks the ORPHANED direction (required -> seen) and
# never the MISSING direction (manifest -> required) nor `needs:`-closure
# coverage. These two additions close that hole. REPORT-ONLY: the caller never
# mutates branch protection and never fails a build on findings.
# ---------------------------------------------------------------------------

# Finding classes emitted by the parity evaluator.
PARITY_MISSING = "MISSING"  # declared direct gate absent from required_status_checks
PARITY_NEEDS_CLOSURE = (
    "NEEDS_CLOSURE"  # gate not covered by its aggregator's needs-closure
)
PARITY_UNPROTECTED = "UNPROTECTED"  # branch has no protection object at all
PARITY_INDETERMINATE = "INDETERMINATE"  # live state could not be resolved
PARITY_INVALID_MANIFEST = (
    "INVALID_MANIFEST"  # unknown coverage / policy value in manifest
)
# Merge-policy dimension (OMN-14288 scope extension): the manifest records the
# DECIDED per-repo/branch merge policy (queue on/off + strict "require branches
# up to date"), and the ratchet flags live drift from it. Same config-as-data
# principle as required contexts — one manifest asserts enforcement AND merge
# policy. REPORT-ONLY: findings never mutate branch protection.
PARITY_QUEUE_DRIFT = "QUEUE_DRIFT"  # live merge-queue state != declared policy
PARITY_STRICT_DRIFT = (
    "STRICT_DRIFT"  # live strict (branch-up-to-date) != declared policy
)


def normalize_context_forms(ctx: str) -> set[str]:
    """Return the set of equivalent forms a required/declared context can take.

    GitHub fuzzy-matches a reusable-workflow required context of the form
    ``"caller-job / reusable-job / leaf-job"`` against the *leaf* check-run name
    it actually reports (e.g. a required ``"deploy-gate / deploy-gate"`` is
    satisfied by a reported leaf ``"deploy-gate"``). To keep the parity check
    from false-positiving on the repos that DID wire the gate correctly, a
    context and the trailing ``" / "``-separated segment of any required context
    are treated as equivalent.
    """
    forms = {ctx.strip()}
    if " / " in ctx:
        forms.add(ctx.split(" / ")[-1].strip())
    return forms


def is_gate_directly_required(gate_context: str, required: list[str]) -> bool:
    """True iff ``gate_context`` is present in ``required`` under fuzzy-leaf
    normalization (mirrors GitHub's reusable-context matching)."""
    gate_forms = normalize_context_forms(gate_context)
    return any(gate_forms & normalize_context_forms(req) for req in required)


def compute_needs_closure(jobs: dict[str, Any], root_job_id: str) -> set[str]:
    """Transitive intra-workflow ``needs:`` closure of ``root_job_id``.

    ``jobs`` is the parsed ``.jobs`` mapping of a SINGLE workflow YAML file.
    Returns the set of job ids reachable from ``root_job_id`` via ``needs:``
    edges (the root itself is excluded). GitHub ``needs:`` is intra-workflow
    only, so a job living in a *different* workflow file can never appear in
    this closure — which is exactly the structural fact that makes a
    cross-workflow gate impossible to enforce through an aggregator (spec §6).
    """
    closure: set[str] = set()
    stack = [root_job_id]
    while stack:
        jid = stack.pop()
        block = jobs.get(jid) or {}
        needs = block.get("needs")
        if isinstance(needs, str):
            needs = [needs]
        for dep in needs or []:
            if dep not in closure:
                closure.add(dep)
                stack.append(dep)
    return closure


def evaluate_gate_parity(
    repo: str,
    branch: str,
    gate: dict[str, Any],
    required: list[str] | None,
    aggregator_jobs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Evaluate ONE manifest gate against live state. Pure.

    Returns a finding dict on violation, or ``None`` when the gate is covered.

    Args:
        required: live ``required_status_checks.contexts`` list, or ``None`` when
            the branch has no protection object at all (→ UNPROTECTED for every
            declared gate).
        aggregator_jobs: parsed ``.jobs`` mapping of the aggregator's workflow,
            needed only for ``coverage: needs_child`` gates.
    """
    context = str(gate["context"])
    coverage = str(gate.get("coverage", "direct"))
    rule = str(gate.get("rule", ""))

    def finding(cls: str, detail: str) -> dict[str, Any]:
        return {
            "repo": repo,
            "branch": branch,
            "gate": context,
            "coverage": coverage,
            "class": cls,
            "rule": rule,
            "detail": detail,
        }

    if required is None:
        return finding(
            PARITY_UNPROTECTED,
            f"branch '{branch}' has no branch-protection object; "
            f"declared gate '{context}' cannot be required",
        )

    if coverage == "direct":
        if is_gate_directly_required(context, required):
            return None
        return finding(
            PARITY_MISSING,
            f"declared direct gate '{context}' absent from "
            f"required_status_checks {sorted(required)}",
        )

    if coverage == "needs_child":
        aggregator = str(gate.get("aggregator", ""))
        aggregator_job_id = str(gate.get("aggregator_job_id", ""))
        gate_job_id = str(gate.get("gate_job_id", ""))
        # (i) the aggregator context must itself be a required context.
        if not is_gate_directly_required(aggregator, required):
            return finding(
                PARITY_NEEDS_CLOSURE,
                f"aggregator '{aggregator}' for gate '{context}' is not itself a "
                f"required context (contexts={sorted(required)})",
            )
        # (ii) the gate's emitting job must be in the aggregator's needs-closure.
        if aggregator_jobs is None:
            return finding(
                PARITY_INDETERMINATE,
                f"could not resolve workflow jobs for aggregator '{aggregator}' "
                f"to verify needs-closure of gate '{context}'",
            )
        closure = compute_needs_closure(aggregator_jobs, aggregator_job_id)
        if gate_job_id in closure:
            return None
        return finding(
            PARITY_NEEDS_CLOSURE,
            f"gate job '{gate_job_id}' is not in the transitive needs-closure of "
            f"aggregator job '{aggregator_job_id}' (closure size={len(closure)}) — "
            f"cross-workflow or orphaned, cannot be enforced via '{aggregator}'",
        )

    return finding(PARITY_INVALID_MANIFEST, f"unknown coverage mode '{coverage}'")


def evaluate_merge_policy_parity(
    repo: str,
    branch: str,
    policy: dict[str, Any],
    live_queue_enabled: bool | None,
    live_strict: bool | None,
) -> list[dict[str, Any]]:
    """Evaluate ONE branch's declared merge policy against live state. Pure.

    ``policy`` is the manifest's ``merge_policy`` block: ``{queue: enabled|disabled,
    strict: bool}``. Either key may be omitted (then that dimension is not asserted).
    Returns a list of findings (0, 1, or 2). A merge queue's only unique value is the
    ``merge_group`` re-test-against-latest-base; ``strict`` is the lighter
    "require-branches-up-to-date" combine-breakage guard. This records the DECIDED
    policy and flags live drift from it. REPORT-ONLY.
    """
    findings: list[dict[str, Any]] = []

    def finding(cls: str, detail: str, declared: Any, observed: Any) -> dict[str, Any]:
        return {
            "repo": repo,
            "branch": branch,
            "gate": "merge_policy",
            "coverage": "merge_policy",
            "class": cls,
            "rule": str(policy.get("rule", "")),
            "detail": detail,
            "declared": declared,
            "observed": observed,
        }

    if "queue" in policy:
        declared_queue = str(policy["queue"])
        if declared_queue not in {"enabled", "disabled"}:
            findings.append(
                finding(
                    PARITY_INVALID_MANIFEST,
                    f"merge_policy.queue must be 'enabled' or 'disabled', got '{declared_queue}'",
                    declared_queue,
                    None,
                )
            )
        elif live_queue_enabled is None:
            findings.append(
                finding(
                    PARITY_INDETERMINATE,
                    "could not resolve live merge-queue state",
                    declared_queue,
                    None,
                )
            )
        else:
            declared_enabled = declared_queue == "enabled"
            if live_queue_enabled != declared_enabled:
                observed = "enabled" if live_queue_enabled else "disabled"
                findings.append(
                    finding(
                        PARITY_QUEUE_DRIFT,
                        f"declared merge queue '{declared_queue}' but live queue is '{observed}'",
                        declared_queue,
                        observed,
                    )
                )

    if "strict" in policy:
        declared_strict = policy["strict"]
        if not isinstance(declared_strict, bool):
            findings.append(
                finding(
                    PARITY_INVALID_MANIFEST,
                    f"merge_policy.strict must be a bool, got {declared_strict!r}",
                    declared_strict,
                    None,
                )
            )
        elif live_strict is None:
            findings.append(
                finding(
                    PARITY_INDETERMINATE,
                    "could not resolve live strict (require-branches-up-to-date) state",
                    declared_strict,
                    None,
                )
            )
        elif live_strict != declared_strict:
            findings.append(
                finding(
                    PARITY_STRICT_DRIFT,
                    f"declared strict={declared_strict} but live strict={live_strict}",
                    declared_strict,
                    live_strict,
                )
            )

    return findings


def evaluate_manifest_parity(
    manifest: dict[str, Any],
    live: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate the whole parity manifest against pre-fetched live state. Pure.

    Args:
        manifest: parsed ``enforcement_parity_manifest.yaml`` (``{"repos": {...}}``).
        live: maps ``f"{repo}:{branch}"`` → {
            "required": list[str] | None,   # required_status_checks.contexts, None if unprotected
            "aggregator_jobs": {aggregator_context: jobs_dict, ...},  # for needs_child gates
            "queue_enabled": bool | None,   # live merge-queue state (for merge_policy)
            "strict": bool | None,          # live require-branches-up-to-date (for merge_policy)
        }

    Returns findings in manifest declaration order (repo, branch; gates then merge_policy).
    """
    findings: list[dict[str, Any]] = []
    repos = manifest.get("repos", {}) or {}
    for repo, branches in repos.items():
        for branch, spec in (branches or {}).items():
            live_entry = live.get(f"{repo}:{branch}", {})
            required = live_entry.get("required")
            agg_jobs_by_ctx = live_entry.get("aggregator_jobs", {}) or {}
            for gate in (spec or {}).get("load_bearing_gates", []) or []:
                aggregator_jobs = None
                if str(gate.get("coverage", "direct")) == "needs_child":
                    aggregator_jobs = agg_jobs_by_ctx.get(
                        str(gate.get("aggregator", ""))
                    )
                result = evaluate_gate_parity(
                    repo, branch, gate, required, aggregator_jobs
                )
                if result is not None:
                    findings.append(result)
            policy = (spec or {}).get("merge_policy")
            if policy:
                findings.extend(
                    evaluate_merge_policy_parity(
                        repo,
                        branch,
                        policy,
                        live_entry.get("queue_enabled"),
                        live_entry.get("strict"),
                    )
                )
    return findings
