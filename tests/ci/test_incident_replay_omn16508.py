# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for OMN-16508 -- the contract-compliance vacuous pass.

The guard under replay is :mod:`scripts.ci.resolve_contract_compliance_pr`, and
the incident it exists to catch is a **false green**: ``ci.yml``'s
``contract-compliance`` step met an empty ``PR_NUMBER`` with ``exit 0``, so on
``push`` and ``merge_group`` the required gate reported success having evaluated
zero DoD ``check_values``. ``"Contract Compliance Check"`` is a ``GATE_JOBS``
entry in :mod:`scripts.ci.ci_summary_gate`, and ``CI Summary`` is this repo's
sole required branch-protection context, so the umbrella poller counted that
run as a *provable* pass.

Both artifacts here are verbatim GitHub API responses, captured with ``gh api``
and committed unmodified. Neither is retyped or trimmed -- the whole point of
OMN-15547's rule is that a plausible-looking reconstruction can only exhibit
failure modes its author already imagined.

* **Reject direction (the incident).**
  ``tests/fixtures/omn16508/commit-6d7090da-pulls.gh-api.json.captured`` is what
  ``repos/OmniNode-ai/omnibase_infra/commits/{sha}/pulls`` really returns for a
  commit no merged PR ever produced -- ``6d7090da``, the head of PR #2890, which
  was closed without merging. The answer is an empty list, and that emptiness is
  precisely the state the pre-fix step turned into a green gate. The guard must
  say NO here.
* **Accept control.**
  ``tests/fixtures/omn16508/commit-5b904d88-pulls.gh-api.json.captured`` is the
  21KB response for ``5b904d88``, a real commit on ``main`` whose source PR
  (#2378) merged and whose ``merge_commit_sha`` is that same SHA. Without this
  half, a guard hard-wired to ``return None`` would replay the incident
  perfectly and still be useless: it would fail every push-to-``main`` run
  closed instead of open. R5 only demands the reject half for a ``false_green``
  case; the control is here because a reject-only proof cannot tell a working
  guard from one that is stuck shut.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.resolve_contract_compliance_pr import resolve_pr_number

pytestmark = pytest.mark.unit

REPO = "OmniNode-ai/omnibase_infra"
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "omn16508"

# The commit no merged PR produced (head of closed-unmerged PR #2890).
UNMERGED_COMMIT = "6d7090daa3c410d9ee51ddc3a0b61a82a8a25662"
UNMERGED_CAPTURE = FIXTURES / "commit-6d7090da-pulls.gh-api.json.captured"

# A real `main` commit whose source PR merged.
MERGED_COMMIT = "5b904d881ba51a697e5b3d50b28460abbb2fd5aa"
MERGED_CAPTURE = FIXTURES / "commit-5b904d88-pulls.gh-api.json.captured"


def _replay(capture: Path) -> Any:
    """Serve the captured bytes where the guard would call the live API."""
    payload = json.loads(capture.read_text(encoding="utf-8"))

    def _fetch(repo: str, sha: str) -> list[dict[str, Any]]:
        return list(payload)

    return _fetch


def test_the_real_no_merged_pr_response_is_rejected_not_vacuously_passed() -> None:
    """The incident input. Pre-fix this was `exit 0`; it must now resolve to nothing."""
    resolved = resolve_pr_number(
        repo=REPO,
        pr_number="",
        merge_group_ref="",
        sha=UNMERGED_COMMIT,
        fetch_commit_pulls=_replay(UNMERGED_CAPTURE),
    )
    assert resolved is None, (
        "GitHub reports no merged PR for this commit, so there are no PR-scoped "
        "DoD check_values to evaluate. Resolving anything here reinstates the "
        "OMN-16508 vacuous pass: a required gate reporting green on zero checks."
    )


def test_the_real_merged_pr_response_still_resolves_its_source_pr() -> None:
    """The control: fail-closed must not mean fail-always."""
    resolved = resolve_pr_number(
        repo=REPO,
        pr_number="",
        merge_group_ref="",
        sha=MERGED_COMMIT,
        fetch_commit_pulls=_replay(MERGED_CAPTURE),
    )
    assert resolved == 2378, (
        "a guard that rejected this too would replay the incident and still be "
        "useless -- every push-to-main run would fail closed on a commit whose "
        "check_values were in fact evaluated by PR #2378's own run"
    )


def test_the_captured_responses_are_the_shapes_this_replay_claims() -> None:
    """Guards the replay itself: a re-capture that drifted must not pass silently."""
    unmerged = json.loads(UNMERGED_CAPTURE.read_text(encoding="utf-8"))
    assert [entry for entry in unmerged if entry.get("merged_at")] == [], (
        f"{UNMERGED_CAPTURE.name} no longer represents a commit without a merged "
        f"source PR, so it can no longer exhibit the OMN-16508 failure"
    )

    merged = json.loads(MERGED_CAPTURE.read_text(encoding="utf-8"))
    winners = [
        entry
        for entry in merged
        if entry.get("merged_at") and entry.get("merge_commit_sha") == MERGED_COMMIT
    ]
    assert [entry["number"] for entry in winners] == [2378]
