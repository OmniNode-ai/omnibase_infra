# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Hostile Reviewer mandatory CI gate coverage (OMN-8468).

Proves ``hostile-reviewer.yml`` exists, is wired to run on every PR into
``dev``/``main``, and that its terminal ``hostile-review-gate`` job fails
closed (non-zero exit) whenever the adversarial review job reports a
blocking (``failure``) result or the OCC preflight predecessor did not
succeed. This is the static-config half of the DoD; the runtime half
(the merge actually being blocked) is proven by registering
"Hostile Review Gate" as a required status check in live branch
protection -- see PR body for the ``gh api`` readback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
HOSTILE_REVIEWER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "hostile-reviewer.yml"


def _load_yaml(path: Path) -> dict[Any, Any]:
    # dict[Any, Any] (not dict[str, Any]): PyYAML 1.1 parses the bare `on:`
    # top-level key as the boolean True, not the string "on" -- callers must
    # be able to look it up either way.
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_hostile_reviewer_workflow_exists() -> None:
    """OMN-8468: the adversarial review gate must be a real, committed workflow."""
    assert HOSTILE_REVIEWER_WORKFLOW.exists(), (
        "hostile-reviewer.yml is missing -- hostile_reviewer is not wired as a CI gate (OMN-8468)"
    )


def test_hostile_reviewer_triggers_on_dev_and_main_prs() -> None:
    workflow = _load_yaml(HOSTILE_REVIEWER_WORKFLOW)
    # PyYAML parses the bare `on:` key as boolean True.
    on_block = workflow.get("on") or workflow.get(True)
    assert on_block is not None, "workflow has no trigger block"
    pr_trigger = on_block["pull_request"]
    assert "dev" in pr_trigger["branches"]
    assert "main" in pr_trigger["branches"]
    assert "opened" in pr_trigger["types"]
    assert "synchronize" in pr_trigger["types"]


def test_hostile_reviewer_gate_job_requires_both_predecessors() -> None:
    """The terminal gate job must depend on occ-preflight AND hostile-review.

    A PR that never runs (or fails) either predecessor must not be able to
    report a PASSED "Hostile Review Gate" -- this is the mechanism the
    required-status-check registration in branch protection relies on.
    """
    workflow = _load_yaml(HOSTILE_REVIEWER_WORKFLOW)
    jobs = workflow["jobs"]
    assert "occ-preflight" in jobs
    assert "hostile-review" in jobs
    assert "hostile-review-gate" in jobs

    gate_job = jobs["hostile-review-gate"]
    assert gate_job["name"] == "Hostile Review Gate"
    needs = gate_job["needs"]
    assert set(needs) == {"occ-preflight", "hostile-review"}
    # `if: always()` is required -- otherwise a failed predecessor would skip
    # this job entirely, and a *skipped* required check does not block merge
    # the way a *failed* one does.
    assert gate_job["if"] == "always()"


def test_hostile_reviewer_gate_blocks_on_failed_review_or_missing_preflight() -> None:
    """RED-to-GREEN target: the gate's own evaluation step must exit non-zero
    when hostile-review reported 'failure' or occ-preflight did not succeed.

    Before this PR, no hostile-reviewer.yml existed in this repo at all, so
    this assertion was unreachable (RED: FileNotFoundError via the fixture
    above). After this PR, the workflow exists and its gate step contains the
    fail-closed branches asserted here (GREEN).
    """
    workflow = _load_yaml(HOSTILE_REVIEWER_WORKFLOW)
    gate_job = workflow["jobs"]["hostile-review-gate"]
    steps = gate_job["steps"]
    evaluate_step = next(s for s in steps if s.get("name") == "Evaluate gate")
    script = evaluate_step["run"]

    assert 'PREFLIGHT="${{ needs.occ-preflight.result }}"' in script
    assert 'RESULT="${{ needs.hostile-review.result }}"' in script
    # Preflight not succeeding is a hard fail.
    assert '[ "$PREFLIGHT" != "success" ]' in script
    # A blocked (failure) adversarial review is a hard fail.
    assert '[ "$RESULT" = "failure" ]' in script
    # Both failure branches must actually exit non-zero.
    assert script.count("exit 1") >= 2


def test_hostile_review_job_uses_live_local_models() -> None:
    """deepseek-r1 no longer serves on the local fleet (OMN-14176); the gate
    must target the currently-live qwen3-review / qwen3-review-b pair so it
    can actually reach >=1 model instead of permanently degrading."""
    workflow = _load_yaml(HOSTILE_REVIEWER_WORKFLOW)
    review_job = workflow["jobs"]["hostile-review"]
    steps = review_job["steps"]
    review_step = next(s for s in steps if s.get("name") == "Run adversarial review")
    script = review_step["run"]
    assert "--model qwen3-review" in script
    assert "--model qwen3-review-b" in script


def test_hostile_review_wires_shared_secret_from_ci_secrets() -> None:
    """The omniintelligence LLM HTTP transport HMAC-signs every call with
    LOCAL_LLM_SHARED_SECRET and fail-closes if absent (OMN-14176). The
    workflow must source it from GitHub Actions secrets, not a hardcoded
    value or an unset env var."""
    workflow = _load_yaml(HOSTILE_REVIEWER_WORKFLOW)
    review_job = workflow["jobs"]["hostile-review"]
    steps = review_job["steps"]
    review_step = next(s for s in steps if s.get("name") == "Run adversarial review")
    assert (
        review_step["env"]["LOCAL_LLM_SHARED_SECRET"]
        == "${{ secrets.LOCAL_LLM_SHARED_SECRET }}"
    )
