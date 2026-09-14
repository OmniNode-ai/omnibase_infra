# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure assertion logic for the required-context parity ratchet (OMN-14288).

The declared policy lives in ``scripts/enforcement_parity_manifest.yaml``
(config-as-data: ``{repo -> branch -> {load_bearing_gates[], merge_policy}}``);
all GitHub I/O lives in ``scripts/audit_required_context_parity_cli.py``. This
module stays pure so the assertions can be unit-tested against synthetic live
state with no network and no yaml dependency.

This file was ``audit_branch_protection_lib.py`` until OMN-18346. It also held a
second, diverged copy of the branch-protection Check A / Check B logic, which
duplicated ``onex_change_control/scripts/audit_branch_protection.sh`` without its
review-gated carve-out and reported healthy repos as violations for seven days.
That half is gone: the shared checks are the canonical auditor's, and the one
check it lacks (orphaned required contexts) is
``scripts/audit_orphan_required_contexts.py``. The parity ratchet below is a
different audit with no canonical counterpart -- it checks the MISSING direction
(a declared load-bearing gate absent from ``required_status_checks``) and
``needs:``-closure coverage -- so it stays here, renamed to say what it is.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Required-context parity ratchet (OMN-14288) — REPORT-ONLY logic
#
# The functions below are PURE: they take already-fetched live state (required
# contexts, parsed workflow-job graphs) and a declarative manifest, and return
# structured findings. All GitHub I/O (fetching protection + workflow contents)
# lives in audit_required_context_parity_cli.py so this module stays free of
# yaml / network dependencies and its tests run with no network.
#
# The audit that motivated this (docs/analyses enforcement-parity spec) proved
# the branch-protection auditor only checks the ORPHANED direction (required ->
# seen, now scripts/audit_orphan_required_contexts.py) and never the MISSING
# direction (manifest -> required) nor `needs:`-closure coverage. These two
# additions close that hole. REPORT-ONLY: the caller never
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
