# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One auditor of record for branch protection (OMN-18346, OMN-20955).

Until 2026-09-13 this repo ran a SECOND, diverged copy of the branch-protection
audit next to the one in `onex_change_control`. OMN-18287 added a code-owner review
carve-out to that other copy and proved it green (0 failures across 102 checks); the
in-repo copy never received it, so it kept reporting `onex_change_control` as a
review violation for a requirement that is deliberate, on both `main` and `dev`,
indefinitely.

Two copies of the same audit that can disagree is worse than one copy that is
wrong, because the disagreement is silent: each is green or red on its own
schedule and nothing compares them. OMN-18346 deleted the diverged local copy and
kept the `onex_change_control` one. OMN-20955 (OCC retirement) then moved that one
to the private tooling repository, where it runs on a schedule against the
organisation (`handler_audit_branch_protection.sh`). This repo therefore
runs NO shared branch-protection audit and fetches nothing from
`onex_change_control`: it keeps only the two checks that auditor does not carry
(orphaned required contexts, required-context parity). These tests keep a second
auditor, or the old fetch, from silently regrowing -- they are cheap, and a
re-added copy fails them by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "branch-protection-audit.yml"

# The local copies OMN-18346 and OMN-20955 deleted. A file reappearing at any of
# these paths is a second auditor, whatever it is called internally.
_DELETED_DUPLICATES = (
    "scripts/audit_branch_protection.sh",
    "scripts/audit-branch-protection.sh",
    "scripts/audit_branch_protection_lib_cli.py",
    "scripts/audit-branch-protection.py",
    "tests/ci/test_branch_protection_audit.py",
)

# Anything that points this workflow at the retired source of the shared auditor.
_RETIRED_SOURCE_MARKERS = (
    "onex_change_control",
    "ONEX_CHANGE_CONTROL_REF",
    "audit_branch_protection.sh",
)


def _executable_workflow_content() -> str:
    """The workflow as parsed YAML dumped back to text: comments are gone.

    The header comments keep the history (what the retirement replaced), so a
    raw-text scan would flag prose. The parsed content is what the runner executes.
    """
    return yaml.safe_dump(yaml.safe_load(_WORKFLOW.read_text()))


@pytest.mark.parametrize("relative", _DELETED_DUPLICATES)
def test_the_diverged_local_copy_is_gone(relative: str) -> None:
    assert not (_REPO_ROOT / relative).exists(), (
        f"{relative} is a second branch-protection auditor. The auditor of record is "
        "handler_audit_branch_protection.sh in the private tooling repository; extend that instead."
    )


@pytest.mark.parametrize("marker", _RETIRED_SOURCE_MARKERS)
def test_the_workflow_fetches_and_runs_no_shared_auditor(marker: str) -> None:
    """No fetch, pin, env var or invocation of the retired auditor source."""
    assert marker not in _executable_workflow_content(), (
        f"branch-protection-audit.yml still names {marker!r}; the shared auditor runs "
        "from the private tooling repository and this workflow keeps only the orphan and parity "
        "checks (OMN-20955)"
    )


def test_the_orphan_check_this_repo_keeps_is_wired() -> None:
    """Check B has no counterpart in the shared auditor, so it stays here."""
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


@pytest.mark.live_contact(
    "tests/ci/fixtures/shared_auditor_replacement_schedule_omn20955.json"
)
def test_recorded_replacement_runs_the_shared_auditor_on_a_schedule(
    recorded_response: dict[str, object],
) -> None:
    """The replacement home runs the shared auditor on a schedule, as a hard failure.

    This workflow stopped running the shared audit because the recorded workflow
    does: scheduled every four hours (the same cron this workflow keeps), no
    pull_request trigger, and an audit step with no continue-on-error. If the
    recording stopped showing that, dropping the shared audit here would lose the
    observation.
    """
    response = cast("dict[str, Any]", recorded_response["response"])
    assert response["schedule"] == [{"cron": "23 */4 * * *"}]
    assert response["has_pull_request_trigger"] is False
    (step,) = response["audit_steps"]
    assert step["runs_script"] == "handler_audit_branch_protection.sh"
    assert step["continue_on_error"] is None
    local = yaml.safe_load(_WORKFLOW.read_text())
    local_on = local.get(True) or local.get("on")
    assert local_on["schedule"] == response["schedule"], (
        "this workflow's cron should stay the same as the recorded replacement's"
    )
