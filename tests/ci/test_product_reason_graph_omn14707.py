# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Phase 3 Product Readiness reason-graph tests (OMN-14707, infra).

This is the omnibase_infra slice of the design fixture matrix
(``docs/plans/2026-07-17-product-first-ci-decouple-design.md`` §4). It proves:

* A seeded product/standards failure surfaces in Product Readiness as a typed
  ``PRODUCT_FAILED`` root — independent of OCC (the #1450/#1451 collapse fix).
* The reason-graph is single-rooted and replay-deterministic.
* Structurally, the shadow surface RUNS its OWN occ-independent product subchecks
  (ruff / mypy / a hermetic pytest slice) with NO ``occ-preflight`` anywhere in
  its needs-chain, and mints NO OCC request (RUN-OWN-SUBCHECKS, not a poller —
  the omniclaude#1906 / omnibase_infra#2325 pattern correction).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.product_readiness import EnumSubcheckOutcome, categorize_conclusion
from scripts.ci.product_reason_graph import (
    DEPLOY_TRIGGER_FAILED,
    EVIDENCE_MISSING,
    GITHUB_API_OUTAGE,
    POLICY_HELD,
    PRODUCT_FAILED,
    RUNNER_INFRA,
    STATUS_BLOCKED_UPSTREAM,
    STATUS_FAILED,
    build_reason_graph,
    main,
    root_receipt_id,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHADOW_WF = _REPO_ROOT / ".github" / "workflows" / "product-readiness-shadow.yml"

_HEAD = "a" * 40


def _green_subchecks() -> dict[str, str]:
    return dict.fromkeys(
        ("change_detection", "lint", "typecheck", "tests", "coverage"), "success"
    )


# --------------------------------------------------------------------------
# Reason-graph: seeded product failures — PRODUCT_FAILED root, OCC-independent.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failing_check", "expected_signal"),
    [
        ("lint", "lint=failure"),
        ("typecheck", "typecheck=failure"),
        ("tests", "tests=failure"),
        ("coverage", "coverage=failure"),
    ],
)
def test_seeded_product_failure_roots_as_product_failed(
    failing_check: str, expected_signal: str
) -> None:
    subchecks = _green_subchecks()
    subchecks[failing_check] = "failure"
    graph = build_reason_graph({"head_sha": _HEAD, "subchecks": subchecks})

    assert graph["root"] is not None
    assert graph["root"]["kind"] == PRODUCT_FAILED
    assert graph["root"]["primary_signal"] == expected_signal
    assert graph["blocked_candidate_count"] == 1
    reporter = next(n for n in graph["nodes"] if n["name"] == failing_check)
    assert reporter["status"] == STATUS_FAILED
    assert reporter["is_root"] is True
    assert reporter["root_receipt_id"] == graph["root"]["root_receipt_id"]


def test_product_failed_is_occ_independent() -> None:
    # A real product defect surfaces as PRODUCT_FAILED even when OCC eligibility
    # is red — the two dimensions no longer collapse (the #1450/#1451 fix).
    subchecks = _green_subchecks()
    subchecks["tests"] = "failure"
    graph = build_reason_graph(
        {"head_sha": _HEAD, "subchecks": subchecks, "occ_eligibility": "failure"}
    )
    assert graph["root"]["kind"] == PRODUCT_FAILED
    assert graph["root"]["primary_signal"] == "tests=failure"


# --------------------------------------------------------------------------
# Green — single-node graph, freeze-eligible.
# --------------------------------------------------------------------------


def test_green_all_pass_is_ready_without_blocked_dependents() -> None:
    graph = build_reason_graph({"head_sha": _HEAD, "subchecks": _green_subchecks()})
    assert graph["root"] is None
    assert graph["ready"] is True
    assert graph["freeze_eligible"] is True
    assert graph["blocked_candidate_count"] == 0
    assert graph["blocked_upstream_count"] == 0
    assert len(graph["nodes"]) == 5
    assert all(n["status"] != STATUS_BLOCKED_UPSTREAM for n in graph["nodes"])


def test_neutral_and_non_string_conclusions_fail_closed() -> None:
    assert categorize_conclusion("neutral") is EnumSubcheckOutcome.INFRA
    assert categorize_conclusion({"bad": "shape"}) is EnumSubcheckOutcome.INFRA

    subchecks = _green_subchecks()
    subchecks["lint"] = "neutral"
    graph = build_reason_graph({"head_sha": _HEAD, "subchecks": subchecks})
    assert graph["root"]["kind"] == RUNNER_INFRA
    assert graph["root"]["primary_signal"] == "product_infra:lint"


def test_malformed_subchecks_fail_closed_without_crashing() -> None:
    graph = build_reason_graph({"head_sha": _HEAD, "subchecks": []})
    assert graph["root"]["kind"] == RUNNER_INFRA
    assert graph["root"]["primary_signal"] == (
        "product_infra:change_detection,lint,typecheck,tests,coverage"
    )


def test_head_sha_is_required_for_content_addressed_receipts() -> None:
    with pytest.raises(ValueError, match="head_sha is required"):
        build_reason_graph({"head_sha": "", "subchecks": _green_subchecks()})


# --------------------------------------------------------------------------
# EVIDENCE_MISSING cascade collapse — the CI-01 projection contract.
# --------------------------------------------------------------------------


def test_evidence_missing_collapses_cascade_to_one_root() -> None:
    # occ-preflight red while all product checks are SKIPPED (the needs:
    # occ-preflight jobs that never ran). Exactly one EVIDENCE_MISSING root; the
    # M skipped checks are all BLOCKED_UPSTREAM under the SAME receipt id.
    subchecks = dict.fromkeys(
        ("change_detection", "lint", "typecheck", "tests", "coverage"), "skipped"
    )
    graph = build_reason_graph(
        {"head_sha": _HEAD, "subchecks": subchecks, "occ_eligibility": "failure"}
    )
    assert graph["root"]["kind"] == EVIDENCE_MISSING
    assert graph["blocked_candidate_count"] == 1  # not M
    receipt = graph["root"]["root_receipt_id"]
    dependents = [n for n in graph["nodes"] if n["status"] == STATUS_BLOCKED_UPSTREAM]
    assert len(dependents) == 5  # the five skipped product subchecks
    assert all(n["root_receipt_id"] == receipt for n in dependents)


def test_evidence_missing_infra_slice_product_green_still_ready() -> None:
    # Design fixture `evidence-missing-cascade-collapse-infra`: occ-preflight red
    # while the product checks are GREEN. One EVIDENCE_MISSING root (the OCC
    # dimension), yet the product dimension independently reports green — the two
    # dimensions no longer collapse.
    graph = build_reason_graph(
        {
            "head_sha": _HEAD,
            "subchecks": _green_subchecks(),
            "occ_eligibility": "failure",
        }
    )
    assert graph["root"]["kind"] == EVIDENCE_MISSING
    assert graph["blocked_candidate_count"] == 1
    assert graph["product_outcome"] == "product_green"
    assert graph["freeze_eligible"] is True
    # Green product nodes are PASS (not counted as blocked); only the synthetic
    # occ-preflight reporter is the root.
    assert graph["blocked_upstream_count"] == 0


def test_absent_occ_input_does_not_fire_evidence_missing() -> None:
    graph = build_reason_graph(
        {"head_sha": _HEAD, "subchecks": _green_subchecks(), "occ_eligibility": ""}
    )
    assert graph["root"] is None
    assert graph["ready"] is True


# --------------------------------------------------------------------------
# Single-rooting precedence — deterministic arbitration.
# --------------------------------------------------------------------------


def test_precedence_infra_and_api_outrank_product_and_evidence() -> None:
    subchecks = _green_subchecks()
    subchecks["lint"] = "failure"
    facts = {
        "head_sha": _HEAD,
        "subchecks": subchecks,
        "occ_eligibility": "failure",
        "policy": "prod-hold",
        "runner_signal": "disk-preflight",
        "gh_api": "5xx",
        "deploy_trigger": "failure",
    }
    assert build_reason_graph(facts)["root"]["kind"] == GITHUB_API_OUTAGE

    del facts["gh_api"]
    assert build_reason_graph(facts)["root"]["kind"] == RUNNER_INFRA

    del facts["runner_signal"]
    assert build_reason_graph(facts)["root"]["kind"] == POLICY_HELD

    del facts["policy"]
    assert build_reason_graph(facts)["root"]["kind"] == PRODUCT_FAILED

    facts["subchecks"]["lint"] = "success"
    assert build_reason_graph(facts)["root"]["kind"] == EVIDENCE_MISSING


def test_deploy_trigger_failed_is_lowest_precedence() -> None:
    graph = build_reason_graph(
        {
            "head_sha": _HEAD,
            "subchecks": _green_subchecks(),
            "deploy_trigger": "failure",
        }
    )
    assert graph["root"]["kind"] == DEPLOY_TRIGGER_FAILED


# --------------------------------------------------------------------------
# Replay determinism — identical head + facts => identical receipt id.
# --------------------------------------------------------------------------


def test_replay_is_byte_identical() -> None:
    subchecks = _green_subchecks()
    subchecks["lint"] = "failure"
    facts = {"head_sha": _HEAD, "subchecks": subchecks}
    first = build_reason_graph(dict(facts))
    second = build_reason_graph(dict(facts))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["root"]["root_receipt_id"] == second["root"]["root_receipt_id"]


def test_receipt_id_is_head_and_kind_sensitive() -> None:
    a = root_receipt_id(_HEAD, PRODUCT_FAILED, "lint=failure")
    assert len(a) == 16
    assert a != root_receipt_id("b" * 40, PRODUCT_FAILED, "lint=failure")
    assert a != root_receipt_id(_HEAD, PRODUCT_FAILED, "tests=failure")
    assert a != root_receipt_id(_HEAD, EVIDENCE_MISSING, "lint=failure")


def test_synchronize_new_head_supersedes_receipt() -> None:
    subchecks = _green_subchecks()
    subchecks["lint"] = "failure"
    old = build_reason_graph({"head_sha": "a" * 40, "subchecks": subchecks})
    new = build_reason_graph({"head_sha": "b" * 40, "subchecks": subchecks})
    assert old["root"]["root_receipt_id"] != new["root"]["root_receipt_id"]


# --------------------------------------------------------------------------
# Reason-graph CLI — report-only, always exit 0.
# --------------------------------------------------------------------------


def test_cli_graph_is_report_only_exit_zero_on_red() -> None:
    facts = {"head_sha": _HEAD, "subchecks": {"lint": "failure"}}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ci.product_reason_graph",
            "graph",
            "--facts-json",
            json.dumps(facts),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["root"]["kind"] == PRODUCT_FAILED


# --------------------------------------------------------------------------
# ENFORCING mode (OMN-14709) — the load-bearing exit-code matrix.
#
# The shadow was report-only (always exit 0), so its check conclusion was
# meaninglessly green. `--enforce` makes the reason-graph exit NON-ZERO ONLY on a
# genuine PRODUCT_FAILED root while every non-product root and a green head stay
# exit 0 — generating the red-side parity signal for a NON-required context.
# --------------------------------------------------------------------------


def _run_enforce(facts: dict) -> int:
    """Invoke the CLI `graph --enforce` path in-process and return the exit code."""
    return main(["graph", "--facts-json", json.dumps(facts), "--enforce"])


@pytest.mark.parametrize(
    "failing_check",
    ["lint", "typecheck", "tests", "coverage"],
)
def test_enforce_exits_nonzero_on_product_failed_root(failing_check: str) -> None:
    # A seeded PRODUCT_FAILED facts input must fail the enforcing shadow.
    subchecks = _green_subchecks()
    subchecks[failing_check] = "failure"
    facts = {"head_sha": _HEAD, "subchecks": subchecks}
    # Precondition: the elected root is genuinely PRODUCT_FAILED.
    assert build_reason_graph(facts)["root"]["kind"] == PRODUCT_FAILED
    assert _run_enforce(facts) != 0


def test_enforce_exits_zero_on_green() -> None:
    # A fully green facts input stays exit 0 under enforcement.
    facts = {"head_sha": _HEAD, "subchecks": _green_subchecks()}
    assert build_reason_graph(facts)["root"] is None
    assert _run_enforce(facts) == 0


@pytest.mark.parametrize(
    ("facts", "expected_root"),
    [
        # RUNNER_INFRA: an unconfirmable (skipped) product subcheck with no OCC-red
        # explanation — infra, not a product defect. Must stay non-fatal.
        (
            {
                "head_sha": _HEAD,
                "subchecks": {
                    **dict.fromkeys(
                        ("change_detection", "lint", "typecheck", "tests", "coverage"),
                        "success",
                    ),
                    "lint": "skipped",
                },
            },
            RUNNER_INFRA,
        ),
        # RUNNER_INFRA via explicit runner signal.
        (
            {
                "head_sha": _HEAD,
                "subchecks": dict.fromkeys(
                    ("change_detection", "lint", "typecheck", "tests", "coverage"),
                    "success",
                ),
                "runner_signal": "disk-preflight",
            },
            RUNNER_INFRA,
        ),
        # EVIDENCE_MISSING: OCC red while product checks are skipped — the OCC
        # dimension, not a product defect. Must stay non-fatal.
        (
            {
                "head_sha": _HEAD,
                "subchecks": dict.fromkeys(
                    (
                        "change_detection",
                        "lint",
                        "typecheck",
                        "tests",
                        "coverage",
                    ),
                    "skipped",
                ),
                "occ_eligibility": "failure",
            },
            EVIDENCE_MISSING,
        ),
        # GITHUB_API_OUTAGE: infra/API root. Must stay non-fatal even with a
        # co-present product failure (higher precedence non-product root wins).
        (
            {
                "head_sha": _HEAD,
                "subchecks": {
                    **dict.fromkeys(
                        ("change_detection", "lint", "typecheck", "tests", "coverage"),
                        "success",
                    ),
                    "lint": "failure",
                },
                "gh_api": "5xx",
            },
            GITHUB_API_OUTAGE,
        ),
        # POLICY_HELD: an intentional hold. Must stay non-fatal.
        (
            {
                "head_sha": _HEAD,
                "subchecks": dict.fromkeys(
                    ("change_detection", "lint", "typecheck", "tests", "coverage"),
                    "success",
                ),
                "policy": "prod-hold",
            },
            POLICY_HELD,
        ),
    ],
)
def test_enforce_exits_zero_on_non_product_roots(
    facts: dict, expected_root: str
) -> None:
    # Non-product roots report but never fail the enforcing shadow (exit 0).
    assert build_reason_graph(facts)["root"]["kind"] == expected_root
    assert _run_enforce(facts) == 0


def test_default_graph_stays_report_only_exit_zero_on_product_failed() -> None:
    # Without --enforce the surface is unchanged: report-only, exit 0 even on a
    # PRODUCT_FAILED root (preserves the historical default for other callers).
    subchecks = _green_subchecks()
    subchecks["tests"] = "failure"
    facts = {"head_sha": _HEAD, "subchecks": subchecks}
    assert main(["graph", "--facts-json", json.dumps(facts)]) == 0


def test_cli_subprocess_enforce_exits_nonzero_on_product_failed() -> None:
    # End-to-end proof through the real `python -m ... --enforce` entrypoint the
    # workflow invokes: a PRODUCT_FAILED head exits non-zero AND still prints the
    # graph JSON on stdout (so the workflow can render its summary).
    facts = {"head_sha": _HEAD, "subchecks": {"lint": "failure"}}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ci.product_reason_graph",
            "graph",
            "--facts-json",
            json.dumps(facts),
            "--enforce",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["root"]["kind"] == PRODUCT_FAILED


def test_cli_subprocess_enforce_exits_zero_on_non_product_root() -> None:
    # End-to-end: a RUNNER_INFRA (skipped-lint) head reports but exits 0.
    subchecks = _green_subchecks()
    subchecks["lint"] = "skipped"
    facts = {"head_sha": _HEAD, "subchecks": subchecks}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ci.product_reason_graph",
            "graph",
            "--facts-json",
            json.dumps(facts),
            "--enforce",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["root"]["kind"] == RUNNER_INFRA


# --------------------------------------------------------------------------
# STRUCTURAL — the shadow RUNS its own occ-independent subchecks, never OCC.
# --------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_shadow_workflow_has_no_occ_preflight_in_needs_chain() -> None:
    wf = _load_yaml(_SHADOW_WF)
    jobs = wf["jobs"]
    for name, job in jobs.items():
        needs = job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "occ-preflight" not in needs, f"job {name} must not need occ-preflight"
        assert not any("occ" in str(n).lower() for n in needs), (
            f"job {name} must not need any occ job"
        )


def test_shadow_subchecks_have_no_needs_and_graph_needs_only_subchecks() -> None:
    # RUN-OWN-SUBCHECKS shape: the three product subcheck jobs run FIRST with NO
    # `needs:` (so they execute regardless of occ-preflight), and the reason-graph
    # depends ONLY on those three subchecks.
    wf = _load_yaml(_SHADOW_WF)
    jobs = wf["jobs"]
    subcheck_jobs = {"lint-shadow", "typecheck-shadow", "tests-shadow"}
    assert subcheck_jobs <= set(jobs), "expected the three occ-independent subchecks"
    for name in subcheck_jobs:
        assert not jobs[name].get("needs"), (
            f"subcheck {name} must have NO needs (runs before OCC)"
        )
    graph_needs = jobs["reason-graph"].get("needs", [])
    if isinstance(graph_needs, str):
        graph_needs = [graph_needs]
    assert set(graph_needs) == subcheck_jobs


def test_shadow_workflow_never_triggers_occ_request() -> None:
    text = _SHADOW_WF.read_text(encoding="utf-8")
    executable = [
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert not any("call-occ-preflight" in ln for ln in executable)
    assert not any("occ-preflight" in ln for ln in executable)


def test_shadow_workflow_runs_own_subchecks_not_poller() -> None:
    text = _SHADOW_WF.read_text(encoding="utf-8")
    # It RUNS the real product toolchain (ruff / mypy / pytest) as occ-independent
    # subchecks — it does NOT poll the head's occ-gated leaf check-runs.
    assert "ruff check" in text
    assert "mypy" in text
    assert "pytest" in text
    assert "python -m scripts.ci.product_reason_graph" in text
    assert "--cov=scripts.ci.product_readiness" in text
    assert "--cov=scripts.ci.product_reason_graph" in text
    assert "--cov-fail-under=60" in text
    # The deprecated poller module must NOT be referenced in any executable line.
    executable = [
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert not any("product_leaf_poll" in ln for ln in executable)
    assert not any("check-runs" in ln for ln in executable)


def test_shadow_workflow_has_no_paths_filter() -> None:
    # The omniclaude#1906 self-exclusion lesson: a `paths:` filter that excludes
    # the PR's own changed files silently prevents the workflow from firing.
    wf = _load_yaml(_SHADOW_WF)
    on = wf.get(True, wf.get("on", {}))  # PyYAML parses `on:` as the bool True
    pr = on.get("pull_request", {}) or {}
    assert "paths" not in pr
    assert "paths-ignore" not in pr
