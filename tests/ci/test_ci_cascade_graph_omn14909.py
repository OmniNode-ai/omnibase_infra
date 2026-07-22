# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI-01 cascade aggregator replay proof (OMN-14909, plan §3.C5).

The load-bearing test (:func:`test_replay_2370_collapses_to_one_root`) replays
``omnibase_infra#2370``'s real blocked-head check-run state — captured verbatim
in ``tests/ci/fixtures/omn14909_2370_cascade_checkruns.json`` — through the new
aggregator and proves the plan §3.C5 acceptance criteria:

* the projection reports **exactly one** typed root cause (``EVIDENCE_MISSING``);
* every other red or skipped check-run is classified ``BLOCKED_UPSTREAM`` and
  content-addressed to that single root; and
* **no dependent is independently rerunnable** — there is exactly ONE rerunnable
  unit (the root). The last clause is the anti-cosmetic guarantee: a surviving
  independently-rerunnable dependent would keep the rerun reflex alive.

The remaining tests pin the OCC-independence property, replay determinism, the
green-head base case, and the report-only CLI contract.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from scripts.ci.ci_cascade_graph import (
    STATUS_BLOCKED_UPSTREAM,
    STATUS_ROOT,
    classify_check_runs,
    main,
    render_summary,
)
from scripts.ci.product_reason_graph import EVIDENCE_MISSING, PRODUCT_FAILED

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "omn14909_2370_cascade_checkruns.json"
)

# Pinned snapshot of the captured #2370 blocked head (head e5404e36…). These are
# the recorded numbers from the real cascade; a change here means the fixture was
# re-captured and the proof must be re-read, not silently updated.
_PINNED_TOTAL = 122
_PINNED_RED = 36  # 35 occ-preflight/eligibility + 1 CI Summary
_PINNED_SKIPPED = 74
_PINNED_OCC_RED = 35


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _passing(conclusion: str) -> bool:
    return conclusion in ("success", "neutral")


def _is_cascade_member(conclusion: str) -> bool:
    return conclusion == "skipped" or conclusion in (
        "failure",
        "action_required",
        "cancelled",
        "timed_out",
        "startup_failure",
    )


# ---------------------------------------------------------------------------
# THE PROOF (plan §3.C5): replay #2370 -> 1 root, all else BLOCKED_UPSTREAM,
# no dependent independently rerunnable.
# ---------------------------------------------------------------------------


def test_fixture_matches_recorded_2370_snapshot() -> None:
    """Guard the fixture itself so the proof cannot drift silently."""
    fixture = _load_fixture()
    runs = fixture["check_runs"]
    conc = collections.Counter(r["conclusion"] for r in runs)
    assert len(runs) == _PINNED_TOTAL
    assert conc["failure"] == _PINNED_RED
    assert conc["skipped"] == _PINNED_SKIPPED
    occ_red = sum(
        1
        for r in runs
        if r["name"] == "occ-preflight / eligibility" and r["conclusion"] == "failure"
    )
    assert occ_red == _PINNED_OCC_RED
    # The occ reds live in distinct workflow runs -> each independently
    # rerunnable TODAY. That is exactly the amplification the aggregator kills.
    distinct_failed_runs = {
        r["run_id"] for r in runs if r["conclusion"] == "failure" and r["run_id"]
    }
    assert len(distinct_failed_runs) == _PINNED_OCC_RED


def test_replay_2370_collapses_to_one_root() -> None:
    fixture = _load_fixture()
    proj = classify_check_runs(fixture["check_runs"], fixture["head_sha"])

    # (1) EXACTLY one typed root cause.
    assert proj["root_count"] == 1
    assert proj["root"] is not None
    assert proj["root"]["kind"] == EVIDENCE_MISSING
    root_receipt = proj["root"]["root_receipt_id"]

    # Exactly one node is the ROOT, and it is the only rerunnable unit.
    root_nodes = [n for n in proj["nodes"] if n["status"] == STATUS_ROOT]
    assert len(root_nodes) == 1
    assert root_nodes[0]["rerunnable"] is True
    assert root_nodes[0]["root_receipt_id"] == root_receipt

    # (2) Every red/skipped non-root check-run is BLOCKED_UPSTREAM, bound to the
    #     single root, and NOT a pass.
    deps = [n for n in proj["nodes"] if n["status"] == STATUS_BLOCKED_UPSTREAM]
    for node in deps:
        assert node["root_receipt_id"] == root_receipt
        assert node["is_root"] is False
        assert not _passing(node["original_conclusion"])

    # (3) NO dependent is independently rerunnable — exactly one rerunnable unit.
    assert proj["rerunnable_unit_count"] == 1
    assert proj["rerunnable_units"] == [root_receipt]
    assert all(n["rerunnable"] is False for n in deps)

    # Completeness: every cascade member (red or skipped) is classified, and the
    # root reporter is one of the reds (so total classified == root + dependents).
    fixture_members = [
        r for r in fixture["check_runs"] if _is_cascade_member(r["conclusion"])
    ]
    assert proj["dependent_count"] + 1 == len(fixture_members)
    assert proj["blocked_failure_count"] == _PINNED_RED - 1  # 1 red is the root
    assert proj["blocked_skipped_count"] == _PINNED_SKIPPED

    # The 35 occ-preflight reds collapse: 1 ROOT + 34 BLOCKED_UPSTREAM, so none of
    # the 34 duplicate reds is independently rerunnable.
    occ_nodes = [n for n in proj["nodes"] if n["name"] == "occ-preflight / eligibility"]
    occ_roots = [n for n in occ_nodes if n["status"] == STATUS_ROOT]
    occ_blocked = [n for n in occ_nodes if n["status"] == STATUS_BLOCKED_UPSTREAM]
    assert len(occ_roots) == 1
    assert len(occ_blocked) == _PINNED_OCC_RED - 1
    assert all(n["rerunnable"] is False for n in occ_blocked)

    # The red CI Summary is a dependent, not an independent failure.
    ci_summary = [n for n in proj["nodes"] if n["name"] == "CI Summary"]
    assert len(ci_summary) == 1
    assert ci_summary[0]["status"] == STATUS_BLOCKED_UPSTREAM
    assert ci_summary[0]["rerunnable"] is False


def test_replay_is_deterministic() -> None:
    fixture = _load_fixture()
    a = classify_check_runs(fixture["check_runs"], fixture["head_sha"])
    b = classify_check_runs(list(reversed(fixture["check_runs"])), fixture["head_sha"])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# OCC-independence: a genuine product defect roots as PRODUCT_FAILED even while
# occ-preflight is red (the #1450/#1451 collapse fix, preserved by reuse).
# ---------------------------------------------------------------------------


def test_product_defect_roots_as_product_failed_even_with_occ_red() -> None:
    head = "b" * 40
    runs = [
        {
            "name": "occ-preflight / eligibility",
            "status": "completed",
            "conclusion": "failure",
            "run_id": "1",
        },
        {
            "name": "lint (shadow)",
            "status": "completed",
            "conclusion": "failure",
            "run_id": "2",
        },
        {
            "name": "typecheck (shadow)",
            "status": "completed",
            "conclusion": "success",
            "run_id": "2",
        },
        {
            "name": "tests+coverage (shadow)",
            "status": "completed",
            "conclusion": "success",
            "run_id": "2",
        },
    ]
    proj = classify_check_runs(runs, head)
    assert proj["root"]["kind"] == PRODUCT_FAILED
    # Still exactly one rerunnable root, and the occ red becomes a dependent.
    assert proj["rerunnable_unit_count"] == 1
    root_nodes = [n for n in proj["nodes"] if n["status"] == STATUS_ROOT]
    assert len(root_nodes) == 1
    assert root_nodes[0]["name"] == "lint (shadow)"


def test_green_head_has_no_root_and_no_dependents() -> None:
    head = "c" * 40
    runs = [
        {
            "name": "lint (shadow)",
            "status": "completed",
            "conclusion": "success",
            "run_id": "9",
        },
        {
            "name": "typecheck (shadow)",
            "status": "completed",
            "conclusion": "success",
            "run_id": "9",
        },
        {
            "name": "tests+coverage (shadow)",
            "status": "completed",
            "conclusion": "success",
            "run_id": "9",
        },
        {
            "name": "change detection (shadow)",
            "status": "completed",
            "conclusion": "success",
            "run_id": "9",
        },
    ]
    proj = classify_check_runs(runs, head)
    assert proj["root"] is None
    assert proj["root_count"] == 0
    assert proj["dependent_count"] == 0
    assert proj["rerunnable_unit_count"] == 0
    assert proj["ready"] is True


# ---------------------------------------------------------------------------
# Report-only CLI contract: always exit 0 (never a gate verdict).
# ---------------------------------------------------------------------------


def test_cli_is_report_only_exit_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["classify", "--check-runs-file", str(_FIXTURE), "--summary"])
    assert code == 0
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert payload["root"]["kind"] == EVIDENCE_MISSING
    assert payload["rerunnable_unit_count"] == 1
    assert "1 root cause" in out.err


def test_render_summary_reports_one_root() -> None:
    fixture = _load_fixture()
    proj = classify_check_runs(fixture["check_runs"], fixture["head_sha"])
    text = render_summary(proj)
    assert "1 root cause" in text
    assert "EVIDENCE_MISSING" in text
