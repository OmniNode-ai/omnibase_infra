# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay for the orphaned-required-context audit (OMN-18346, OMN-15547 R5).

The regression is a false_red, and it ran at full scale: `Branch Protection
Audit` failed 47 of its 47 scheduled runs in seven days, every one of them on a
fleet whose branch protection was healthy. The auditor compared `main`'s
`required_status_checks.contexts` -- PR-time gates that bind their check-runs to
a PR head SHA -- against check-runs on the last five commits of the DEFAULT
branch, which are post-merge pushes and carry only the push-triggered subset.
The two sets essentially cannot intersect, so the audit reported omniclaude's
`main` as 54 contexts orphaned and named, among others, `CI Summary`,
`verify / verify` and `Security Gate`: three gates that were running green on
every PR in the repository at that moment.

A red gate that is red on everything is worse than no gate. Nobody can tell the
day it starts being right, which is the second half of this incident: two of
omniclaude's required contexts really ARE orphaned. No workflow in that repo
emits `CodeRabbit Thread Check` any more, so those two contexts can never report
and block every PR targeting `main`. That real finding was invisible inside 54
false ones for the whole seven days.

These tests drive the REAL audit module over REAL captured bytes:

  * the check-runs of omniclaude PR #2151's head SHA (both pages, 121 distinct
    names) -- the PR-time evidence source the old implementation never read;
  * the check-runs of omniclaude dev push commit 28ababda (9 names) -- the
    push-only evidence source it read instead;
  * the live `main` branch-protection payload carrying all 56 required contexts.

Under the push-only source the real old guard returned `violation` with 54
orphans; the replay pins that the corrected module returns `ok` over the same
repo's PR-time evidence.

THE DISCRIMINATOR IS LOAD-BEARING, not a formality. A check that reported
nothing would satisfy the accept case and enforce exactly zero, and it would be
indistinguishable from a working one from outside -- a silent audit and a clean
fleet look the same in the run log. So the same function, over the same evidence
set, is required to still name `gate / CodeRabbit Thread Check`: a context that
is genuinely orphaned in this very capture.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "audit_orphan_required_contexts.py"
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "omn18346"

# The registered artifact: page 1 of the check-runs on the head SHA of omniclaude
# PR #2151 (merged 2026-09-13T21:39:27Z). This is the surface the corrected audit
# reads and the old one never did.
_PR_HEAD_PAGE1 = _FIXTURES / "omniclaude-pr2151-head-32f33409-check-runs.json.captured"
_PR_HEAD_PAGE1_SHA256 = (
    "791267d3c29931812ab8f7a8c32d46f949705b8750fea826cab9d2ec462b0181"
)
# Page 2 of the same resource. The PR head carries 154 check-runs, so a reader
# that does not paginate sees a truncated evidence set and manufactures orphans
# of its own -- the replay drives the real paginating fetcher over both pages.
_PR_HEAD_PAGE2 = (
    _FIXTURES / "omniclaude-pr2151-head-32f33409-check-runs-page2.json.captured"
)
# The push-only evidence source the buggy implementation used.
_PUSH_CAPTURE = _FIXTURES / "omniclaude-dev-push-28ababda-check-runs.json.captured"
# The live required-context list that was reported as orphaned.
_PROTECTION_CAPTURE = _FIXTURES / "omniclaude-main-protection.json.captured"

_PR_HEAD_SHA = "32f3340904ff7bffbb2e826695fc96b32b074340"
_PUSH_SHA = "28ababdac534d3e0e6df0ca2ef2770dc03f48008"

# The two contexts that are genuinely orphaned in this capture: omniclaude has no
# workflow emitting "CodeRabbit Thread Check", confirmed against origin/dev.
_GENUINE_ORPHANS = (
    "cr-thread-gate / CodeRabbit Thread Check",
    "gate / CodeRabbit Thread Check",
)
# Three of the 54 the buggy auditor named, each running green on every PR.
_FALSELY_FLAGGED = ("CI Summary", "verify / verify", "Security Gate")

_EMPTY_PAGE = json.dumps({"total_count": 0, "check_runs": []})


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_orphan_required_contexts", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    return _load_module()


def _gh_over_captures(*, pr_evidence: bool) -> Any:
    """A gh caller serving the captured bytes for omniclaude.

    ``pr_evidence=False`` reproduces the buggy evidence source exactly: the
    default-branch push commit and nothing else.
    """
    protection = _PROTECTION_CAPTURE.read_text()
    push = _PUSH_CAPTURE.read_text()
    pr_page1 = _PR_HEAD_PAGE1.read_text()
    pr_page2 = _PR_HEAD_PAGE2.read_text()

    def gh(argv: list[str]) -> tuple[int, str]:
        path = argv[1]
        if path.endswith("/branches/main/protection"):
            return 0, protection
        if "/commits?per_page=" in path:
            return 0, json.dumps([{"sha": _PUSH_SHA}])
        if "/pulls?" in path:
            if not pr_evidence:
                return 0, "[]"
            return 0, json.dumps(
                [{"merged_at": "2026-09-13T21:39:27Z", "head": {"sha": _PR_HEAD_SHA}}]
            )
        if "/check-runs" in path:
            if f"/commits/{_PUSH_SHA}/check-runs" in path:
                return (0, push) if path.endswith("&page=1") else (0, _EMPTY_PAGE)
            if f"/commits/{_PR_HEAD_SHA}/check-runs" in path:
                if path.endswith("&page=1"):
                    return 0, pr_page1
                if path.endswith("&page=2"):
                    return 0, pr_page2
                return 0, _EMPTY_PAGE
        return 1, ""

    return gh


# ---------------------------------------------------------------------------
# R1 -- the captured bytes are the bytes, unmodified.
# ---------------------------------------------------------------------------
def test_registered_artifact_matches_its_recorded_sha256() -> None:
    digest = hashlib.sha256(_PR_HEAD_PAGE1.read_bytes()).hexdigest()
    assert digest == _PR_HEAD_PAGE1_SHA256, (
        "the registered artifact has been edited; a reformatted capture is no "
        "longer the capture that exhibits the incident"
    )


# ---------------------------------------------------------------------------
# The incident itself: the evidence source the old auditor used cannot carry
# the contexts it was asked about. This pins WHY the audit was red, from the
# captured bytes rather than from the report.
# ---------------------------------------------------------------------------
def test_push_only_evidence_cannot_observe_the_pr_time_contexts(
    audit: ModuleType,
) -> None:
    required = audit.parse_required_contexts(_PROTECTION_CAPTURE.read_text())
    assert len(required) == 56

    push_names = {
        run["name"] for run in json.loads(_PUSH_CAPTURE.read_text())["check_runs"]
    }
    push_orphans = audit.find_orphan_contexts(required, push_names)

    # 55 of 56, on a repo whose gates were all green. The real old guard
    # reported 54 of these: `main-target-guard` was the single entry in its
    # PR_ONLY_CONTEXTS allowlist, the workaround that hid one symptom of this
    # very defect and that OMN-18346 removed along with its cause.
    assert len(push_orphans) == 55
    for falsely_flagged in _FALSELY_FLAGGED:
        assert falsely_flagged in push_orphans


# ---------------------------------------------------------------------------
# R5 false_red -- the corrected guard ACCEPTS the real healthy input.
# ---------------------------------------------------------------------------
def test_the_real_audit_accepts_the_repo_it_reported_54_orphans_on(
    audit: ModuleType,
) -> None:
    result = audit.audit_repo_main(
        "OmniNode-ai", "omniclaude", _gh_over_captures(pr_evidence=True)
    )

    assert len(result["required_contexts"]) == 56
    for falsely_flagged in _FALSELY_FLAGGED:
        assert falsely_flagged not in result["orphan_contexts"], (
            f"'{falsely_flagged}' reports green on every PR in this capture; "
            "calling it orphaned is the 47-of-47 false red"
        )
    # Only the genuinely-orphaned pair survives.
    assert sorted(result["orphan_contexts"]) == sorted(_GENUINE_ORPHANS)


def test_the_falsely_flagged_contexts_are_present_in_the_pr_time_capture(
    audit: ModuleType,
) -> None:
    """The accept above is earned by evidence, not by a widened matcher."""
    names: set[str] = set()
    for page in (_PR_HEAD_PAGE1, _PR_HEAD_PAGE2):
        names |= {run["name"] for run in json.loads(page.read_text())["check_runs"]}

    observed: set[str] = set()
    for name in names:
        observed |= audit.normalize_forms(name)
    for falsely_flagged in _FALSELY_FLAGGED:
        assert audit.normalize_forms(falsely_flagged) & observed


# ---------------------------------------------------------------------------
# R5 discriminator -- the same guard, same evidence, still REJECTS a real orphan.
# ---------------------------------------------------------------------------
def test_the_same_audit_still_names_the_context_nothing_emits(
    audit: ModuleType,
) -> None:
    result = audit.audit_repo_main(
        "OmniNode-ai", "omniclaude", _gh_over_captures(pr_evidence=True)
    )

    assert result["status"] == "violation"
    for orphan in _GENUINE_ORPHANS:
        assert orphan in result["orphan_contexts"], (
            f"'{orphan}' is required on omniclaude main and no workflow in that "
            "repo emits it; an audit that stops reporting it enforces nothing"
        )
        assert orphan in result["message"]


def test_an_audit_that_reported_nothing_would_fail_the_discriminator(
    audit: ModuleType,
) -> None:
    """State the stuck-open failure mode the discriminator exists to catch."""
    required = audit.parse_required_contexts(_PROTECTION_CAPTURE.read_text())
    stuck_open = audit.find_orphan_contexts(required, set(required))
    assert stuck_open == []
    # ...which is exactly why the reject assertion above is required alongside
    # the accept: an empty result satisfies "no false positives" trivially.
