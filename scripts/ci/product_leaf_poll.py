#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Map the head's LEAF product check-runs to Product Readiness facts (OMN-14707).

Why this exists (infra extends the ``ci-summary`` no-needs poller posture)
--------------------------------------------------------------------------
omnibase_infra is the worst OCC cascade and its product jobs are heavy and
runtime-critical (they clone three sibling repos, ``uv sync``, run a 15-shard
pytest matrix and a compose boot). Re-running them inside a shadow workflow —
the way the omnimarket canary does — would be wasteful and delicate. So the
infra Product Readiness shadow instead follows the ``ci-summary`` NO-``needs``
poller posture (``scripts/ci/ci_summary_gate.py``): the shadow workflow
self-instantiates its check-run immediately and **polls the head's already-
running LEAF product check-runs via the Checks API**, mapping their conclusions
onto the five canonical product subchecks the classifier consumes.

This module is the pure, stdlib-only mapper. The network call
(``gh api /repos/{owner}/{repo}/commits/{sha}/check-runs``) happens in the
workflow; this module only *maps* the returned check-run rows to a facts dict
for ``scripts/ci/product_reason_graph.py``. It performs no I/O of its own.

Invariants
----------
- **No occ-preflight.** The mapped LEAF set is deliberately the OCC-independent
  ci.yml product surface (``Lint``, ``ONEX Validators``, ``Detect Changes``,
  ``CI Tests Gate``). ``occ-preflight`` is NEVER polled and NEVER mapped, so the
  shadow reports the product dimension independent of any OCC evidence.
- **Self-exclusion.** Only the fixed :data:`LEAF_CHECK_MAP` names are consulted,
  so the shadow's own ``product-readiness-shadow`` check-run can never be polled
  (no self-deadlock — the omniclaude/omnibase self-poll trap).
- **Fail closed.** A missing or still-running leaf yields ``""`` (ABSENT) for its
  dimension, which the classifier maps to ``product_infra`` — never a silent
  pass. ``map`` exits 2 (PENDING) until every mapped leaf is ``completed`` so the
  workflow keeps polling to a bounded deadline, then emits the graph regardless.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:  # pragma: no cover - import shim, both branches exercised across contexts
    from scripts.ci.product_readiness import (
        EnumSubcheckOutcome,
        categorize_conclusion,
    )
except ImportError:  # pragma: no cover - script-run fallback
    from product_readiness import (  # type: ignore[no-redef]
        EnumSubcheckOutcome,
        categorize_conclusion,
    )

# Canonical product dimension -> the OCC-independent ci.yml LEAF check-run job
# names that report it. A dimension is the fail-closed aggregate of its leaves.
# NOTE: `occ-preflight` is intentionally absent — the shadow is OCC-independent.
#   - typecheck folds into `Lint` (infra runs mypy inside the Lint job).
#   - standards/validators (`ONEX Validators`) fold into the `lint` static
#     dimension (they are the standards/contract product gates).
#   - coverage folds into `CI Tests Gate` (the 15-shard tests aggregate; infra
#     enforces coverage inside pytest, not as a separate check-run).
LEAF_CHECK_MAP: dict[str, tuple[str, ...]] = {
    "change_detection": ("Detect Changes",),
    "lint": ("Lint", "ONEX Validators"),
    "typecheck": ("Lint",),
    "tests": ("CI Tests Gate",),
    "coverage": ("CI Tests Gate",),
}

# All distinct leaf names we ever poll (used for the terminal/pending decision).
POLLED_LEAVES: frozenset[str] = frozenset(
    name for names in LEAF_CHECK_MAP.values() for name in names
)

# Aggregation precedence for a multi-leaf dimension (fail-closed, worst wins):
# an affirmative failure dominates; an unconfirmed (absent/pending) leaf outranks
# an infra skip; only all-green is green.
_AGG_ORDER: tuple[EnumSubcheckOutcome, ...] = (
    EnumSubcheckOutcome.FAIL,
    EnumSubcheckOutcome.ABSENT,
    EnumSubcheckOutcome.INFRA,
    EnumSubcheckOutcome.PASS,
)
_CATEGORY_TO_RAW: dict[EnumSubcheckOutcome, str] = {
    EnumSubcheckOutcome.FAIL: "failure",
    EnumSubcheckOutcome.ABSENT: "",
    EnumSubcheckOutcome.INFRA: "skipped",
    EnumSubcheckOutcome.PASS: "success",
}

EXIT_TERMINAL = 0
EXIT_PENDING = 2


def _latest_by_name(check_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep the most recent check-run per name (by started_at, then id)."""
    latest: dict[str, dict[str, Any]] = {}
    for run in check_runs:
        name = str(run.get("name") or "")
        if not name:
            continue
        prev = latest.get(name)
        if prev is None or _sort_key(run) >= _sort_key(prev):
            latest[name] = run
    return latest


def _sort_key(run: dict[str, Any]) -> tuple[str, int]:
    started = str(run.get("started_at") or "")
    try:
        run_id = int(run.get("id") or 0)
    except (TypeError, ValueError):
        run_id = 0
    return (started, run_id)


def _leaf_category(run: dict[str, Any] | None) -> EnumSubcheckOutcome:
    """Coarse category for a single leaf, fail-closed on absent/incomplete."""
    if run is None:
        return EnumSubcheckOutcome.ABSENT
    if str(run.get("status") or "") != "completed":
        # queued / in_progress / waiting -> not yet a confirmable result.
        return EnumSubcheckOutcome.ABSENT
    return categorize_conclusion(run.get("conclusion"))


def _aggregate(categories: list[EnumSubcheckOutcome]) -> str:
    """Reduce a dimension's leaf categories to one raw conclusion string."""
    if not categories:
        return ""
    for cat in _AGG_ORDER:
        if cat in categories:
            return _CATEGORY_TO_RAW[cat]
    return ""  # pragma: no cover - _AGG_ORDER is exhaustive


def resolve_subchecks(check_runs: list[dict[str, Any]]) -> dict[str, str]:
    """Map the head's LEAF check-runs to the five product subcheck conclusions."""
    latest = _latest_by_name(check_runs)
    resolved: dict[str, str] = {}
    for dimension, leaves in LEAF_CHECK_MAP.items():
        resolved[dimension] = _aggregate(
            [_leaf_category(latest.get(n)) for n in leaves]
        )
    return resolved


def all_leaves_terminal(check_runs: list[dict[str, Any]]) -> bool:
    """True only when every polled LEAF is present and ``completed``."""
    latest = _latest_by_name(check_runs)
    for name in POLLED_LEAVES:
        run = latest.get(name)
        if run is None or str(run.get("status") or "") != "completed":
            return False
    return True


def build_facts(head_sha: str, check_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The facts dict consumed by ``product_reason_graph.build_reason_graph``."""
    return {"head_sha": head_sha, "subchecks": resolve_subchecks(check_runs)}


def _load_check_runs(path: str | None) -> list[dict[str, Any]]:
    if path is None or path == "-":
        raw = sys.stdin.read()
    else:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
    raw = raw.strip()
    if not raw:
        return []
    data = json.loads(raw)
    # Accept the raw endpoint object ({"check_runs": [...]}), a bare array of
    # check-run objects, or (from `gh api --paginate --slurp`) an array of page
    # objects each carrying a "check_runs" array.
    if isinstance(data, dict):
        runs = data.get("check_runs", [])
    elif isinstance(data, list):
        if data and all(isinstance(x, dict) and "check_runs" in x for x in data):
            runs = [r for page in data for r in page.get("check_runs", [])]
        else:
            runs = data
    else:
        runs = []
    if not isinstance(runs, list):
        raise ValueError(
            "check-runs payload must be a list or an object with a 'check_runs' array"
        )
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map head LEAF product check-runs to Product Readiness facts"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("map", help="Emit the facts JSON for the reason-graph")
    p.add_argument("--head-sha", required=True, help="Exact head SHA of the PR")
    p.add_argument(
        "--check-runs-file",
        default="-",
        help="Path to the Checks-API check-runs JSON (default: stdin). Accepts "
        "the raw endpoint object or a bare array of check-run objects.",
    )

    args = parser.parse_args(argv)

    if args.command == "map":
        check_runs = _load_check_runs(args.check_runs_file)
        facts = build_facts(args.head_sha, check_runs)
        print(json.dumps(facts, sort_keys=True))
        return EXIT_TERMINAL if all_leaves_terminal(check_runs) else EXIT_PENDING

    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover - argparse exits first


if __name__ == "__main__":
    raise SystemExit(main())
