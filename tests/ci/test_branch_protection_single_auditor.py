# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One auditor of record for branch protection (OMN-18346).

Until 2026-09-13 this repo ran a SECOND, diverged copy of the branch-protection
audit. The canonical one is `onex_change_control/scripts/audit_branch_protection.sh`
-- the script omni_home's `branch-protection-guard.yml` and `scheduled-gap-detect.yml`
both run. OMN-18287 added a code-owner review carve-out to that copy and proved it
green (0 failures across 102 checks); the in-repo copy never received it, so it kept
reporting `onex_change_control` as a review violation for a requirement that is
deliberate, on both `main` and `dev`, indefinitely.

Two copies of the same audit that can disagree is worse than one copy that is
wrong, because the disagreement is silent: each is green or red on its own
schedule and nothing compares them. These tests keep the dedupe from silently
regrowing -- they are cheap, and a re-added local copy fails them by name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "branch-protection-audit.yml"

# The diverged local copies OMN-18346 deleted. A file reappearing at any of these
# paths is a second auditor, whatever it is called internally.
_DELETED_DUPLICATES = (
    "scripts/audit-branch-protection.sh",
    "scripts/audit_branch_protection_lib_cli.py",
    "scripts/audit-branch-protection.py",
    "tests/ci/test_branch_protection_audit.py",
)

_CANONICAL_SCRIPT = "scripts/audit_branch_protection.sh"
_PINNED_REF_RE = re.compile(r"^[0-9a-f]{40}$")


@pytest.mark.parametrize("relative", _DELETED_DUPLICATES)
def test_the_diverged_local_copy_is_gone(relative: str) -> None:
    assert not (_REPO_ROOT / relative).exists(), (
        f"{relative} is a second branch-protection auditor. The canonical one is "
        f"onex_change_control/{_CANONICAL_SCRIPT}; extend that instead."
    )


def test_the_workflow_runs_the_canonical_auditor() -> None:
    body = _WORKFLOW.read_text()
    assert _CANONICAL_SCRIPT in body, (
        "branch-protection-audit.yml must invoke the canonical onex_change_control "
        "auditor for the shared checks"
    )


def test_the_canonical_auditor_is_pinned_to_a_full_commit_sha() -> None:
    """A floating ref would let a sibling repo's merge change this gate silently."""
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    ref = workflow.get("env", {}).get("ONEX_CHANGE_CONTROL_REF")
    assert isinstance(ref, str) and _PINNED_REF_RE.match(ref), (
        f"ONEX_CHANGE_CONTROL_REF must be a 40-hex commit sha, got {ref!r}"
    )


def test_the_orphan_check_this_repo_keeps_is_wired() -> None:
    """Check B has no counterpart in the canonical auditor, so it stays here."""
    body = _WORKFLOW.read_text()
    assert "scripts/audit_orphan_required_contexts.py" in body
    assert (_REPO_ROOT / "scripts" / "audit_orphan_required_contexts.py").is_file()


def test_no_audit_step_is_softened_into_a_report() -> None:
    """The audit steps aggregate into a hard fail; none of them is advisory.

    `continue-on-error` on an audit step is the shape that turns a red gate green
    without the findings going away, which is the outcome OMN-18346 forbids. The
    required-context parity ratchet (OMN-14288) is deliberately report-only and
    is the single exception, named here rather than matched loosely.
    """
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    steps = workflow["jobs"]["audit"]["steps"]
    softened = [
        step.get("name", "<unnamed>")
        for step in steps
        if step.get("continue-on-error") is True
    ]
    assert softened == ["Required-context parity report (REPORT-ONLY, non-blocking)"], (
        f"unexpected advisory audit step(s): {softened}"
    )
