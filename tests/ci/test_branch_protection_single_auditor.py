# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One auditor of record for branch protection (OMN-18346, OMN-20955, OMN-20074).

Until 2026-09-13 this repo ran a SECOND, diverged copy of the branch-protection
audit. The shared audit then lived in `onex_change_control` and this workflow
fetched it at a pinned sha; OMN-20955 moved it to the repository that now owns it, whose own
`branch-protection-audit.yml` audits this repository through its overlay, so the
fetch here is gone (OMN-20074). OMN-18287 had added a code-owner review carve-out
to the shared copy; the in-repo copy never received it, so it kept reporting
`onex_change_control` as a review violation for a requirement that is deliberate.

Two copies of the same audit that can disagree is worse than one copy that is
wrong, because the disagreement is silent: each is green or red on its own
schedule and nothing compares them. These tests keep the dedupe from silently
regrowing -- they are cheap, and a re-added local copy fails them by name.
"""

from __future__ import annotations

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

_SHARED_AUDIT_SCRIPT = "audit_branch_protection.sh"


@pytest.mark.parametrize("relative", _DELETED_DUPLICATES)
def test_the_diverged_local_copy_is_gone(relative: str) -> None:
    assert not (_REPO_ROOT / relative).exists(), (
        f"{relative} is a second branch-protection auditor. The shared one runs "
        "from the branch-protection-audit.yml of the repository that owns it (OMN-20955); extend that instead."
    )


def test_the_workflow_fetches_and_runs_no_shared_auditor() -> None:
    """The shared checks run from the repository that owns them, never from a fetch here."""
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    assert "ONEX_CHANGE_CONTROL_REF" not in (workflow.get("env") or {})
    runs = "\n".join(
        str(step.get("run", "")) for step in workflow["jobs"]["audit"]["steps"]
    )
    assert _SHARED_AUDIT_SCRIPT not in runs
    assert "git fetch" not in runs


def test_the_orphan_check_this_repo_keeps_is_wired() -> None:
    """Check B has no counterpart in the canonical auditor, so it stays here."""
    body = _WORKFLOW.read_text()
    assert "scripts/audit_orphan_required_contexts.py" in body
    assert (_REPO_ROOT / "scripts" / "audit_orphan_required_contexts.py").is_file()


def test_no_audit_step_is_softened_into_a_report() -> None:
    """The audit steps aggregate into a hard fail; none of them is advisory.

    `continue-on-error` on an audit step is the shape that turns a red gate green
    without the findings going away, which is the outcome OMN-18346 forbids. The
    required-context parity ratchet (OMN-14288) is report-only through its own
    exit code (``report`` exits 0 with findings), so it needs no exception here.
    """
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    steps = workflow["jobs"]["audit"]["steps"]
    softened = [
        step.get("name", "<unnamed>")
        for step in steps
        if step.get("continue-on-error") is True
    ]
    assert softened == [], f"unexpected advisory audit step(s): {softened}"
