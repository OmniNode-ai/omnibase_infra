# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Phase 3 Product Readiness reason-graph + leaf-poller tests (OMN-14707, infra).

This is the omnibase_infra slice of the design fixture matrix
(``docs/plans/2026-07-17-product-first-ci-decouple-design.md`` §4). It proves:

* A seeded product/standards failure surfaces in Product Readiness as a typed
  ``PRODUCT_FAILED`` root — independent of OCC (the #1450/#1451 collapse fix).
* The reason-graph is single-rooted and replay-deterministic.
* The infra LEAF poller maps the head's OCC-independent ci.yml product check-runs
  onto the five product subchecks, self-excludes the shadow's own check-run, and
  NEVER polls or maps ``occ-preflight``.
* Structurally, the shadow surface is a NO-``needs`` poller with NO
  ``occ-preflight`` in its needs-chain and mints NO OCC request.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.product_leaf_poll import (
    LEAF_CHECK_MAP,
    POLLED_LEAVES,
    all_leaves_terminal,
    build_facts,
)
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
    root_receipt_id,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "product_reason_graph.py"
_POLL_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "product_leaf_poll.py"
_SHADOW_WF = _REPO_ROOT / ".github" / "workflows" / "product-readiness-shadow.yml"

_HEAD = "a" * 40


def _green_subchecks() -> dict[str, str]:
    return dict.fromkeys(
        ("change_detection", "lint", "typecheck", "tests", "coverage"), "success"
    )


def _check_run(name: str, conclusion: str, status: str = "completed") -> dict:
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started_at": "2026-07-17T00:00:00Z",
        "id": abs(hash(name)) % 1_000_000,
    }


def _all_leaves(conclusion: str) -> list[dict]:
    """One completed check-run per polled LEAF, all with ``conclusion``."""
    return [_check_run(name, conclusion) for name in sorted(POLLED_LEAVES)]


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


def test_green_all_pass_is_ready_single_node() -> None:
    graph = build_reason_graph({"head_sha": _HEAD, "subchecks": _green_subchecks()})
    assert graph["root"] is None
    assert graph["ready"] is True
    assert graph["freeze_eligible"] is True
    assert graph["blocked_candidate_count"] == 0
    assert graph["blocked_upstream_count"] == 0
    assert all(n["status"] != STATUS_BLOCKED_UPSTREAM for n in graph["nodes"])


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
# Infra LEAF poller — maps OCC-independent ci.yml check-runs to product facts.
# --------------------------------------------------------------------------


def test_leaf_map_never_includes_occ_preflight() -> None:
    all_leaves = {n for names in LEAF_CHECK_MAP.values() for n in names}
    assert "occ-preflight" not in all_leaves
    assert not any("occ" in n.lower() for n in all_leaves)
    assert frozenset(all_leaves) == POLLED_LEAVES


def test_leaf_poll_green_maps_to_ready_graph() -> None:
    check_runs = _all_leaves("success")
    facts = build_facts(_HEAD, check_runs)
    assert facts["subchecks"] == _green_subchecks()
    assert all_leaves_terminal(check_runs) is True
    assert build_reason_graph(facts)["ready"] is True


def test_seeded_standards_fail_infra_maps_to_product_failed() -> None:
    # `seeded-standards-fail-infra`: the infra standards/validator LEAF
    # ("ONEX Validators") concludes failure. It folds into the `lint` static
    # dimension and surfaces as PRODUCT_FAILED — even under occ_eligibility=red.
    check_runs = _all_leaves("success")
    for run in check_runs:
        if run["name"] == "ONEX Validators":
            run["conclusion"] = "failure"
    facts = build_facts(_HEAD, check_runs)
    assert facts["subchecks"]["lint"] == "failure"
    facts["occ_eligibility"] = "failure"
    graph = build_reason_graph(facts)
    assert graph["root"]["kind"] == PRODUCT_FAILED
    assert graph["root"]["primary_signal"] == "lint=failure"


def test_leaf_poll_pending_leaf_is_absent_and_nonterminal() -> None:
    check_runs = _all_leaves("success")
    # Make the tests gate still in-progress.
    for run in check_runs:
        if run["name"] == "CI Tests Gate":
            run["status"] = "in_progress"
            run["conclusion"] = None
    facts = build_facts(_HEAD, check_runs)
    # tests + coverage both fold from `CI Tests Gate` -> absent (fail-closed).
    assert facts["subchecks"]["tests"] == ""
    assert facts["subchecks"]["coverage"] == ""
    assert all_leaves_terminal(check_runs) is False


def test_leaf_poll_missing_leaf_is_absent() -> None:
    # Only a subset of leaves present -> the rest are ABSENT, non-terminal.
    check_runs = [_check_run("Lint", "success")]
    facts = build_facts(_HEAD, check_runs)
    assert facts["subchecks"]["change_detection"] == ""
    assert facts["subchecks"]["tests"] == ""
    assert all_leaves_terminal(check_runs) is False


def test_leaf_poll_self_excludes_shadow_and_ignores_occ_check_run() -> None:
    # The shadow's own check-run and a red occ-preflight check-run are present on
    # the head but neither is polled/mapped (no self-deadlock, OCC-independent).
    check_runs = _all_leaves("success") + [
        _check_run("product-readiness (shadow)", None, status="in_progress"),
        _check_run("occ-preflight / eligibility", "failure"),
    ]
    facts = build_facts(_HEAD, check_runs)
    assert facts["subchecks"] == _green_subchecks()
    assert all_leaves_terminal(check_runs) is True
    # OCC-independent: a red occ-preflight does not appear in the product facts.
    assert build_reason_graph(facts)["ready"] is True


def test_leaf_poll_accepts_paginated_slurp_page_array() -> None:
    # `gh api --paginate --slurp` yields an array of page objects; the CLI must
    # flatten them.
    pages = [
        {"check_runs": [_check_run("Lint", "success")]},
        {"check_runs": [_check_run("ONEX Validators", "success")]},
        {
            "check_runs": [
                _check_run("Detect Changes", "success"),
                _check_run("CI Tests Gate", "success"),
            ]
        },
    ]
    proc = subprocess.run(
        [sys.executable, str(_POLL_SCRIPT), "map", "--head-sha", _HEAD],
        input=json.dumps(pages),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0  # all leaves terminal
    facts = json.loads(proc.stdout)
    assert facts["subchecks"] == _green_subchecks()


def test_leaf_poll_cli_pending_exit_code_two() -> None:
    check_runs = _all_leaves("success")
    for run in check_runs:
        if run["name"] == "Detect Changes":
            run["status"] = "queued"
            run["conclusion"] = None
    proc = subprocess.run(
        [sys.executable, str(_POLL_SCRIPT), "map", "--head-sha", _HEAD],
        input=json.dumps({"check_runs": check_runs}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2  # PENDING — workflow keeps polling


# --------------------------------------------------------------------------
# Reason-graph CLI — report-only, always exit 0.
# --------------------------------------------------------------------------


def test_cli_graph_is_report_only_exit_zero_on_red() -> None:
    facts = {"head_sha": _HEAD, "subchecks": {"lint": "failure"}}
    proc = subprocess.run(
        [
            sys.executable,
            str(_GRAPH_SCRIPT),
            "graph",
            "--facts-json",
            json.dumps(facts),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["root"]["kind"] == PRODUCT_FAILED


# --------------------------------------------------------------------------
# STRUCTURAL — the shadow surface is a NO-needs poller, never couples to OCC.
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
        # Infra shadow is a strict NO-needs poller (reports FIRST on the head).
        assert not needs, f"job {name} must be a NO-needs poller, got needs={needs}"


def test_shadow_workflow_never_triggers_occ_request() -> None:
    text = _SHADOW_WF.read_text(encoding="utf-8")
    executable = [
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert not any("call-occ-preflight" in ln for ln in executable)
    assert not any("occ-preflight" in ln for ln in executable)


def test_shadow_workflow_is_leaf_poller_not_job_rerun() -> None:
    text = _SHADOW_WF.read_text(encoding="utf-8")
    # Extends the poller posture: it maps the head's LEAF check-runs and emits
    # the reason graph — it does NOT re-run the heavy infra product jobs.
    assert "scripts/ci/product_leaf_poll.py" in text
    assert "scripts/ci/product_reason_graph.py" in text
    assert "commits/${REPO}" not in text  # sanity: templated path uses envs
    assert "check-runs" in text
    # No heavy re-run of the product surface inside the shadow.
    assert "uv sync" not in text
    assert "pytest" not in text


def test_shadow_workflow_has_no_paths_filter() -> None:
    # The omniclaude#1906 self-exclusion lesson: a `paths:` filter that excludes
    # the PR's own changed files silently prevents the workflow from firing.
    wf = _load_yaml(_SHADOW_WF)
    on = wf.get(True, wf.get("on", {}))  # PyYAML parses `on:` as the bool True
    pr = on.get("pull_request", {}) or {}
    assert "paths" not in pr
    assert "paths-ignore" not in pr
