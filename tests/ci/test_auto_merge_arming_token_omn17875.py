# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for the auto-merge arming-token repair (OMN-17875).

Root cause (measured live 2026-09-04T10:55-11:05Z): ``auto-merge.yml`` armed
``gh pr merge --squash --auto`` with ``secrets.GITHUB_TOKEN``. GitHub completes
an armed auto-merge as the identity that armed it, and it fires no ``push``
event for ``GITHUB_TOKEN``-authored commits (documented Actions-recursion
prevention). Every push-on-``dev`` workflow was therefore skipped on every PR
that landed through this job::

    9d273373f (#3171, github-actions[bot]) -> 0 push runs
    92c397643 (#3173, jonahgabriel)        -> 4 push runs

(``gh api "repos/OmniNode-ai/omnibase_infra/actions/runs?head_sha=<sha>&event=push"
--jq .total_count``.) Over the sampled window 97 of 101 bot-armed dev merges had
zero push runs; the OMN-16906 dev-candidate-delivery liveness guard already
reported ``NOT_FIRED`` for ``9d273373f`` (run 33859940620) without the cause
being found.

Fix: arm with ``secrets.CROSS_REPO_PAT`` — an existing org secret
(``visibility: all``) this repo already consumes in four other workflows, and
the same credential ``omninode_infra``'s auto-merge.yml deliberately retained
for this exact property (OMN-15769, re-affirmed by OMN-16373).

These tests are the mechanical guard (CLAUDE.md rule 5: enforcement, not
detection) so the swap cannot be silently reverted to ``GITHUB_TOKEN`` — or
"modernised" to an ``onexbot-occ-writer`` App installation token, which the
OMN-16373 controlled probe proved suppresses push events identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTO_MERGE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "auto-merge.yml"

# The steps that cause a commit to be attributed to the token: arming the
# squash merge, and update-branch/enqueue. Both must carry the non-suppressing
# identity.
MUTATING_STEP_NAMES: tuple[str, ...] = (
    "Enable auto-merge",
    "Enqueue armed PR and verify it entered the queue",
)

# The step that only reads PR/repo metadata. Safe under the default token
# because it creates no commit.
READ_ONLY_STEP_NAME = "Resolve PR and author"

REQUIRED_TOKEN_EXPR = "${{ secrets.CROSS_REPO_PAT }}"

# Substrings that would reintroduce the suppression on a different credential.
APP_TOKEN_MARKERS: tuple[str, ...] = (
    "create-github-app-token",
    "ONEXBOT_OCC_APP_ID",
    "ONEXBOT_OCC_PRIVATE_KEY",
)


def _load_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(AUTO_MERGE_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _steps() -> list[dict[str, Any]]:
    jobs = _load_workflow()["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["auto-merge"]
    steps = job["steps"]
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict)]


def _step_by_name(name: str) -> dict[str, Any]:
    matches = [step for step in _steps() if step.get("name") == name]
    assert len(matches) == 1, (
        f"expected exactly one step named {name!r}, got {len(matches)}"
    )
    return matches[0]


@pytest.mark.parametrize("step_name", MUTATING_STEP_NAMES)
def test_merge_state_mutating_steps_arm_with_cross_repo_pat(step_name: str) -> None:
    """The arming/enqueue steps must authenticate as CROSS_REPO_PAT.

    A GITHUB_TOKEN- or App-token-authored merge commit fires no push event, so
    this assertion is the difference between a staging candidate being built on
    every dev merge and being built on none.
    """
    step = _step_by_name(step_name)
    token = step.get("env", {}).get("GH_TOKEN")
    assert token == REQUIRED_TOKEN_EXPR, (
        f"{step_name!r} must arm with {REQUIRED_TOKEN_EXPR}; found {token!r}. "
        "GITHUB_TOKEN- and GitHub-App-token-authored merges suppress push-triggered "
        "workflow runs on dev (OMN-17875 / OMN-16373)."
    )


@pytest.mark.parametrize("step_name", MUTATING_STEP_NAMES)
def test_mutating_steps_have_no_github_token_fallback(step_name: str) -> None:
    """No ``|| secrets.GITHUB_TOKEN`` fallback on the mutating steps.

    A fallback would restore the defect silently on any run where the PAT is
    unavailable, which is exactly the invisible-failure shape this ticket exists
    to remove: the job would report success while starving every downstream
    push workflow.
    """
    step = _step_by_name(step_name)
    token = str(step.get("env", {}).get("GH_TOKEN", ""))
    assert "GITHUB_TOKEN" not in token, (
        f"{step_name!r} must not fall back to GITHUB_TOKEN; found {token!r}"
    )


def test_read_only_resolve_step_stays_on_default_token() -> None:
    """The read-only step keeps GITHUB_TOKEN — it creates no commit.

    Narrow blast radius is deliberate: the PAT is only granted to the steps that
    actually need the non-suppressing identity.
    """
    step = _step_by_name(READ_ONLY_STEP_NAME)
    assert step.get("env", {}).get("GH_TOKEN") == "${{ secrets.GITHUB_TOKEN }}"
    run = step.get("run", "")
    for mutating_verb in (
        "gh pr merge",
        "gh pr update-branch",
        "enqueuePullRequest",
        "git push",
    ):
        assert mutating_verb not in run, (
            f"{READ_ONLY_STEP_NAME!r} performs {mutating_verb!r} but runs under "
            "GITHUB_TOKEN; move it to a CROSS_REPO_PAT step or the suppression returns."
        )


def test_no_app_token_mint_is_introduced() -> None:
    """An onexbot-occ-writer App token is not a valid substitute here.

    The OMN-16373 controlled probe pushed commit ``38ffe1f4`` under the App
    identity and the push-triggered marker workflow did not fire, while the
    ``jonahgabriel`` control push (``fd534b2a``) fired run 32562988366.
    """
    raw = AUTO_MERGE_WORKFLOW.read_text(encoding="utf-8")
    body = raw.split("name: Auto-Merge", 1)[1]
    for marker in APP_TOKEN_MARKERS:
        assert marker not in body, (
            f"auto-merge.yml must not mint a GitHub App token ({marker!r}); App-token "
            "pushes suppress push-triggered runs identically to GITHUB_TOKEN (OMN-16373)."
        )


def test_header_documents_the_defect_and_cites_the_evidence_tickets() -> None:
    """The in-file rationale must survive, so the next reader does not revert it."""
    header = AUTO_MERGE_WORKFLOW.read_text(encoding="utf-8").split(
        "name: Auto-Merge", 1
    )[0]
    for citation in ("OMN-17875", "OMN-16373", "OMN-16906", "CROSS_REPO_PAT"):
        assert citation in header, f"auto-merge.yml header must cite {citation}"
