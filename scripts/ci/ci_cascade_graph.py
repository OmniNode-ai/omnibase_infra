# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-workflow CI cascade aggregator (OMN-14909 / CI-01, plan §3.C5).

Root cause this module addresses
--------------------------------
When one prerequisite fails (a missing ``Evidence-Source:`` makes
``occ-preflight / eligibility`` red), the failure *explodes*. On
``omnibase_infra#2370`` a single defect produced ~36 red check-runs — one red
``occ-preflight / eligibility`` per *calling workflow* (the C2 fan-out) plus a
red ``CI Summary`` — and ~74 skipped ``needs: occ-preflight`` dependents.
Roughly **37x amplification of a single defect**.

That wall of red is a *cost* driver, not only a diagnosis-quality one: because
each of the 35+ red ``occ-preflight`` checks belongs to its own workflow run,
each is **independently rerunnable**, which provokes the *rerun reflex* — a human
(or bot) re-runs the wall, and every re-run re-spends the full 40-way matrix.

The fix (count ROOTS, not checks)
---------------------------------
This module consumes the **full commit check-run list** for a head SHA (the
cross-workflow view from ``GET /repos/{o}/{r}/commits/{sha}/check-runs``) and
collapses it into:

- exactly **one** typed root cause — elected by the deterministic, unit-tested
  :func:`scripts.ci.product_reason_graph.build_reason_graph` classifier, so the
  OCC-independence property (a real product defect roots as ``PRODUCT_FAILED``,
  not ``EVIDENCE_MISSING``) is preserved byte-for-byte;
- every *other* red or skipped check-run marked ``BLOCKED_UPSTREAM`` and
  content-addressed back to the single root's ``root_receipt_id``;
- **exactly one rerunnable unit** — the root. Every dependent carries
  ``rerunnable == False``. That is the anti-cosmetic guarantee: if a dependent
  were still presented as independently rerunnable, the rerun reflex would
  survive and the fix would be cosmetic.

This is a **report-only projection**. It never changes any gate verdict: the
authoritative ``CI Summary`` poller (``scripts/ci/ci_summary_gate.py``) still
fails closed on the ``occ-preflight`` red. This module only renders *why* the
run is blocked as ONE root instead of N independent failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Reuse the WS1/OMN-14705 single-rooted classifier for root election. Works both
# as a bare script (script dir on sys.path[0]) and as a package import (pytest).
try:  # pragma: no cover - import shim, both branches exercised across contexts
    from scripts.ci.product_reason_graph import (
        _OUTCOME_TO_CHECK,
        DEPLOY_TRIGGER_FAILED,
        EVIDENCE_MISSING,
        GITHUB_API_OUTAGE,
        POLICY_HELD,
        PRODUCT_FAILED,
        PRODUCT_GREEN,
        RUNNER_INFRA,
        STATUS_BLOCKED_UPSTREAM,
        STATUS_PASS,
        build_reason_graph,
    )
except ImportError:  # pragma: no cover - script-run fallback
    from product_reason_graph import (  # type: ignore[no-redef]
        _OUTCOME_TO_CHECK,
        DEPLOY_TRIGGER_FAILED,
        EVIDENCE_MISSING,
        GITHUB_API_OUTAGE,
        POLICY_HELD,
        PRODUCT_FAILED,
        PRODUCT_GREEN,
        RUNNER_INFRA,
        STATUS_BLOCKED_UPSTREAM,
        STATUS_PASS,
        build_reason_graph,
    )

SCHEMA_VERSION = "ci-cascade-graph/v1"

STATUS_ROOT = "ROOT"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"

# A *skipped* check under a red root IS a dependent (an occ-gated job that never
# ran). ``neutral``/``success`` are genuinely passing and never counted.
_PASSING_CONCLUSIONS = frozenset({"success", "neutral"})
# Conclusions that represent an affirmative failure of the check itself.
_FAILING_CONCLUSIONS = frozenset(
    {"failure", "action_required", "cancelled", "timed_out", "startup_failure"}
)

# The occ-preflight eligibility check name (its reds are the EVIDENCE_MISSING
# signal and the canonical root reporter for that root kind).
_OCC_CHECK_NEEDLE = "occ-preflight"
_OCC_RED_CONCLUSIONS = frozenset({"failure", "action_required"})

# Product shadow subcheck display names -> the classifier subcheck key(s). Present
# so a genuine product defect roots as PRODUCT_FAILED (OCC-independence), rather
# than being mis-attributed to EVIDENCE_MISSING. ``tests+coverage (shadow)`` is a
# single job whose conclusion the shadow workflow passes as BOTH tests and
# coverage (product-readiness-shadow.yml), so it populates both keys here.
_SHADOW_TO_SUBCHECK: dict[str, tuple[str, ...]] = {
    "lint (shadow)": ("lint",),
    "typecheck (shadow)": ("typecheck",),
    "tests+coverage (shadow)": ("tests", "coverage"),
    "change detection (shadow)": ("change_detection",),
}

# All five product subcheck keys the classifier understands. A subcheck that is
# not present as a check-run defaults to ``success``: the aggregator collapses
# check-runs that ACTUALLY exist as red/skipped, and absence of a shadow check
# must never fabricate a PRODUCT_INFRA/RUNNER_INFRA root out of thin air.
_ALL_SUBCHECKS: tuple[str, ...] = (
    "change_detection",
    "lint",
    "typecheck",
    "tests",
    "coverage",
)

# The check name (substring, lowercased) that reports each non-product root, so
# the aggregator can elect ONE canonical rerunnable reporter check-run.
_ROOT_REPORTER_NEEDLE: dict[str, str] = {
    EVIDENCE_MISSING: _OCC_CHECK_NEEDLE,
    RUNNER_INFRA: "runner",
    GITHUB_API_OUTAGE: "github-api",
    POLICY_HELD: "policy",
    DEPLOY_TRIGGER_FAILED: "deploy",
}


def _norm(value: Any) -> str:
    return (str(value) if value is not None else "").strip().lower()


def _normalize_runs(check_runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Reduce raw check-run rows to the fields this aggregator needs."""

    out: list[dict[str, str]] = []
    for raw in check_runs:
        name = str(raw.get("name") or "")
        if not name:
            continue
        run_id = str(raw.get("run_id") or "")
        if not run_id:
            # Derive a run id from html_url when the reduced field is absent.
            url = str(raw.get("html_url") or "")
            if "/actions/runs/" in url:
                run_id = url.split("/actions/runs/")[1].split("/")[0]
        out.append(
            {
                "name": name,
                "status": str(raw.get("status") or ""),
                "conclusion": _norm(raw.get("conclusion")),
                "run_id": run_id,
            }
        )
    # Deterministic order so replay yields an identical projection.
    out.sort(key=lambda r: (r["name"], r["run_id"], r["conclusion"]))
    return out


def _derive_facts(runs: list[dict[str, str]], head_sha: str) -> dict[str, Any]:
    """Map the check-run list onto the ``build_reason_graph`` facts dict."""

    occ_red = any(
        _OCC_CHECK_NEEDLE in r["name"].lower()
        and r["conclusion"] in _OCC_RED_CONCLUSIONS
        for r in runs
    )

    # Product subchecks from the shadow checks when present; any subcheck with no
    # corresponding check-run defaults to ``success`` (see _ALL_SUBCHECKS). This
    # keeps a genuine product RED rooting as PRODUCT_FAILED while never inventing
    # a product-infra root from a merely-absent shadow check.
    subchecks: dict[str, str] = dict.fromkeys(_ALL_SUBCHECKS, "success")
    for r in runs:
        keys = _SHADOW_TO_SUBCHECK.get(r["name"].lower())
        if keys and r["conclusion"]:
            for key in keys:
                subchecks[key] = r["conclusion"]

    facts: dict[str, Any] = {"head_sha": head_sha, "subchecks": subchecks}
    if occ_red:
        facts["occ_eligibility"] = "failure"
    return facts


def _is_dependent(conclusion: str, status: str) -> bool:
    """A red or skipped *completed* check is a cascade dependent.

    ``skipped`` counts (an occ-gated job that never ran because the root was
    red). ``success``/``neutral`` never count. An incomplete/queued check is not
    yet a terminal dependent.
    """

    if conclusion in _PASSING_CONCLUSIONS:
        return False
    if conclusion in _FAILING_CONCLUSIONS or conclusion == "skipped":
        return True
    return False


def classify_check_runs(
    check_runs: list[dict[str, Any]], head_sha: str
) -> dict[str, Any]:
    """Collapse a head SHA's full check-run list into one typed root + dependents.

    Returns a projection dict whose ``root_count`` is the number of DISTINCT root
    causes (typically 1) and whose dependents are every other red/skipped
    check-run, each ``BLOCKED_UPSTREAM`` and NOT independently rerunnable.
    """

    runs = _normalize_runs(check_runs)
    facts = _derive_facts(runs, head_sha)
    graph = build_reason_graph(facts)
    root = graph["root"]
    root_kind = root["kind"] if root else None
    root_receipt_id = root["root_receipt_id"] if root else None

    # Elect ONE canonical rerunnable root reporter check-run. For a red root we
    # pick the first (deterministically ordered) matching red check; the other
    # same-cause reds collapse into BLOCKED_UPSTREAM duplicates so they are never
    # independently rerunnable.
    reporter_needle = _ROOT_REPORTER_NEEDLE.get(root_kind or "", "")
    root_run_index: int | None = None
    if root_kind is not None:
        if root_kind == PRODUCT_FAILED:
            # Root reporter is the failing product shadow subcheck. Invert the
            # (shadow name -> subcheck keys) map to (subcheck key -> shadow name).
            failing_subcheck = graph["product_outcome"]
            want_check = _OUTCOME_TO_CHECK.get(failing_subcheck)
            key_to_shadow = {
                key: shadow
                for shadow, keys in _SHADOW_TO_SUBCHECK.items()
                for key in keys
            }
            want_shadow = key_to_shadow.get(want_check or "")
            for i, r in enumerate(runs):
                if (
                    want_shadow
                    and r["name"].lower() == want_shadow
                    and (r["conclusion"] in _FAILING_CONCLUSIONS)
                ):
                    root_run_index = i
                    break
        else:
            for i, r in enumerate(runs):
                if (
                    reporter_needle
                    and reporter_needle in r["name"].lower()
                    and _is_dependent(r["conclusion"], r["status"])
                ):
                    root_run_index = i
                    break

    nodes: list[dict[str, Any]] = []
    blocked_failure = 0
    blocked_skipped = 0
    independent_failure = 0
    for i, r in enumerate(runs):
        node: dict[str, Any] = {
            "name": r["name"],
            "run_id": r["run_id"],
            "original_conclusion": r["conclusion"],
        }
        member = _is_dependent(r["conclusion"], r["status"])
        if i == root_run_index:
            node["status"] = STATUS_ROOT
            node["is_root"] = True
            node["rerunnable"] = True
            node["root_receipt_id"] = root_receipt_id
        elif root_kind is not None and member:
            # A red/skipped check under an elected root collapses onto it.
            node["status"] = STATUS_BLOCKED_UPSTREAM
            node["is_root"] = False
            node["rerunnable"] = False
            node["root_receipt_id"] = root_receipt_id
            if r["conclusion"] == "skipped":
                blocked_skipped += 1
            else:
                blocked_failure += 1
        elif member:
            # No single root explains this red/skip (e.g. an independent
            # deploy-gate failure). Report it HONESTLY rather than collapsing or
            # dropping it: an unexplained red stays its own rerunnable failure; a
            # skip with no root is genuinely not-applicable, not a dependent.
            if r["conclusion"] == "skipped":
                node["status"] = STATUS_SKIPPED
                node["rerunnable"] = False
            else:
                node["status"] = STATUS_FAILED
                node["rerunnable"] = True
                independent_failure += 1
            node["is_root"] = False
            node["root_receipt_id"] = None
        else:
            # Passing / neutral / still-running: not part of the cascade.
            node["status"] = STATUS_PASS
            node["is_root"] = False
            node["rerunnable"] = False
            node["root_receipt_id"] = None
        nodes.append(node)

    # If a root was elected but no reporter check-run matched (e.g. the occ
    # signal came from a check whose name did not contain the needle), synthesize
    # a single rerunnable root node so there is always exactly one rerunnable
    # unit — never zero, never many.
    if root_kind is not None and root_run_index is None:
        nodes.insert(
            0,
            {
                "name": f"<root:{root_kind}>",
                "run_id": "",
                "original_conclusion": "",
                "status": STATUS_ROOT,
                "is_root": True,
                "rerunnable": True,
                "root_receipt_id": root_receipt_id,
            },
        )

    rerunnable_units = sorted(
        {n["root_receipt_id"] for n in nodes if n.get("rerunnable")}
    )
    dependent_count = blocked_failure + blocked_skipped
    ready = root_kind is None and graph["product_outcome"] == PRODUCT_GREEN

    return {
        "schema_version": SCHEMA_VERSION,
        "head_sha": head_sha,
        "root": root,
        "root_count": 1 if root_kind is not None else 0,
        "dependent_count": dependent_count,
        "blocked_failure_count": blocked_failure,
        "blocked_skipped_count": blocked_skipped,
        # Reds that no single root explained — reported as their own failures
        # rather than force-collapsed onto a fabricated root.
        "independent_failure_count": independent_failure,
        # Exactly one rerunnable unit when a root is elected (the anti-cosmetic
        # invariant). A consumer counts ROOTS, not the N red/skipped checks.
        "rerunnable_unit_count": len(rerunnable_units),
        "rerunnable_units": rerunnable_units,
        "total_check_runs": len(runs),
        "nodes": nodes,
        "ready": ready,
    }


def render_summary(projection: dict[str, Any]) -> str:
    root = projection["root"]
    lines = ["## CI Cascade Reason Graph (report-only)", ""]
    if root is None:
        lines.append(
            f"> READY — no blocking root over {projection['total_check_runs']} "
            "check-runs. Single-node projection, no BLOCKED_UPSTREAM dependents."
        )
        return "\n".join(lines)
    lines += [
        f"> **1 root cause** — `{root['kind']}` — signal "
        f"`{root['primary_signal']}` — receipt `{root['root_receipt_id']}`",
        "",
        f"- root_count = **{projection['root_count']}** "
        f"(rerunnable units: **{projection['rerunnable_unit_count']}**)",
        f"- dependents classified `BLOCKED_UPSTREAM` = **{projection['dependent_count']}** "
        f"({projection['blocked_failure_count']} red + "
        f"{projection['blocked_skipped_count']} skipped)",
        f"- total check-runs collapsed = {projection['total_check_runs']}",
        "",
        "A single defect is presented as ONE rerunnable root, not "
        f"{projection['dependent_count']} independently-rerunnable failures — "
        "the rerun reflex has no wall of red to re-spend the matrix on.",
    ]
    return "\n".join(lines)


def _load_check_runs(path: str | None) -> tuple[list[dict[str, Any]], str]:
    """Load check-runs and a head SHA from a file (or stdin).

    Accepts: the reduced fixture object ``{head_sha, check_runs:[...]}``; the raw
    GitHub endpoint object ``{total_count, check_runs:[...]}``; a bare JSON array;
    or NDJSON (one check-run object per line, as ``gh api --jq '.check_runs[]'``
    emits). The head SHA is taken from the object when present, else "".
    """

    if path is None or path == "-":
        raw = sys.stdin.read()
    else:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
    head = ""
    if not raw.strip():
        return [], head
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # NDJSON fallback (multi-line, one check-run object per line).
        runs = [json.loads(line) for line in raw.splitlines() if line.strip()]
        return runs, head
    if isinstance(data, list):
        return data, head
    if isinstance(data, dict):
        head = str(data.get("head_sha") or "")
        if "check_runs" in data:
            runs = data["check_runs"]
            if not isinstance(runs, list):
                raise ValueError("'check_runs' must be a list")
            return runs, head
        # A single check-run object (one-line NDJSON parsed as one dict).
        if "name" in data:
            return [data], head
        return [], head
    raise ValueError("check-runs payload must be a list or an object with check_runs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collapse a head SHA's full check-run cascade into one typed root."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("classify", help="Emit the cascade projection as JSON")
    p.add_argument(
        "--check-runs-file",
        default="-",
        help="Path to the check-runs JSON/NDJSON (default: stdin).",
    )
    p.add_argument(
        "--head-sha",
        default="",
        help="Head SHA the projection is content-addressed to (overrides file).",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Also print a Markdown summary block to stderr.",
    )
    args = parser.parse_args(argv)

    if args.command == "classify":
        runs, file_head = _load_check_runs(args.check_runs_file)
        head_sha = args.head_sha or file_head
        projection = classify_check_runs(runs, head_sha)
        print(json.dumps(projection, sort_keys=True))
        if args.summary:
            print(render_summary(projection), file=sys.stderr)
        # ALWAYS exit 0. This is a report-only projection; it must never change
        # a gate verdict. The authoritative CI Summary poller owns pass/fail.
        return 0

    parser.error(f"unknown command: {args.command}")  # NoReturn


if __name__ == "__main__":
    raise SystemExit(main())
