# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shape invariants for the sibling rebuild trigger (OMN-18268).

Each assertion here is a property that, if it silently regressed, would turn the
workflow into the false green it exists to replace. They are asserted on the
parsed workflow, never by matching prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger-reusable.yml"
GUARD = REPO_ROOT / "scripts" / "ci" / "check_lane_sibling_revision.py"


def _load() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_is_callable_only_and_declares_the_facts_the_caller_must_supply() -> None:
    workflow = _load()
    triggers = workflow[True]
    assert set(triggers) == {"workflow_call"}, (
        "a sibling trigger is called by the sibling's CI; it must not carry an "
        "event trigger of its own, and a bare workflow_dispatch of a deploy is "
        "never the path"
    )
    inputs = triggers["workflow_call"]["inputs"]
    for required in (
        "source_repo",
        "repo_slug",
        "source_sha",
        "base_branch",
        "changed_files",
        "pr_number",
    ):
        assert inputs[required]["required"] is True, required
    assert set(triggers["workflow_call"]["secrets"]) == {
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
    }, (
        "the lane declares SASL and the publisher refuses to downgrade it; no "
        "HMAC secret is a precondition of this producer"
    )


def test_the_publisher_passes_both_the_sibling_sha_and_a_primary_ref() -> None:
    """Conflating the two publishes a ref the deploy clone cannot resolve."""
    workflow = _load()
    steps = workflow["jobs"]["trigger-rebuild"]["steps"]
    publish = next(s for s in steps if s.get("id") == "publish")
    run = publish["run"]
    assert "--source-repo" in run
    assert "--primary-ref" in run
    assert "--source-sha" in run
    # The primary ref is read from a real omnibase_infra clone, not asserted.
    assert publish["env"]["PRIMARY_REF"] == "${{ steps.primary.outputs.ref }}"
    primary = next(s for s in steps if s.get("id") == "primary")
    assert "git rev-parse HEAD" in primary["run"]


def test_the_build_context_repo_is_checked_out_at_the_workspace_root() -> None:
    """`./.github/actions/...` resolves from the workspace, not from the caller.

    The `ref` assertions below are per-job because the two self-checkouts answer
    different questions (OMN-18200):

    * ``trigger-rebuild``'s clone HEAD IS the published ``--primary-ref``, the
      omnibase_infra revision the lane is told to rebuild at, so it must be dev.
    * ``verify-sibling-converged`` only supplies the scripts its steps invoke, so
      it must be this workflow file's own commit -- at ``dev`` the YAML spelling
      ``probe-lane``'s arguments and the script reading them were different
      commits, which is what produced the FAIL compose-dev receipt on omnimarket
      run 35019423922.
    """
    workflow = _load()
    expected_ref = {
        "trigger-rebuild": "dev",
        "verify-sibling-converged": (
            "${{ inputs.infra_ref || github.job_workflow_sha }}"
        ),
    }
    assert set(workflow["jobs"]) == set(expected_ref), (
        "a job was added or renamed; decide which of the two refs above it needs "
        "rather than letting it default to an unasserted one"
    )
    for job_id, job in workflow["jobs"].items():
        first = job["steps"][0]
        assert first["with"]["repository"] == "OmniNode-ai/omnibase_infra"
        assert first["with"]["ref"] == expected_ref[job_id], job_id
        assert "path" not in first["with"]
        assert first["with"]["persist-credentials"] is False


def test_convergence_asserts_the_sibling_revision_not_the_infra_label() -> None:
    """The infra revision label does not move on a sibling-only rebuild.

    Asserting it would report converged on the first poll having proven nothing,
    which is the false green this workflow exists to remove.
    """
    workflow = _load()
    steps = workflow["jobs"]["verify-sibling-converged"]["steps"]
    converge = next(s for s in steps if s.get("id") == "converge")
    assert "check_lane_sibling_revision.py" in converge["run"]
    assert "check_dev_lane_staleness.py" not in converge["run"]
    assert GUARD.is_file()


def test_the_receipt_is_keyed_by_the_sibling_sha() -> None:
    """A receipt keyed by the infra sha would not name the merge it proves."""
    workflow = _load()
    steps = workflow["jobs"]["verify-sibling-converged"]["steps"]
    emit = next(s for s in steps if "lab_pass_receipt.py emit" in str(s.get("run", "")))
    assert '--sha "$SIBLING_SHA"' in emit["run"]
    upload = next(
        s
        for s in steps
        if str(s.get("uses", "")).startswith("actions/upload-artifact@")
    )
    assert (
        upload["with"]["name"]
        == "lab-pass-receipt-compose-dev-${{ needs.trigger-rebuild.outputs.sibling_sha }}"
    )


def test_the_receipt_is_emitted_even_when_convergence_fails() -> None:
    """ "It failed" and "nobody ran it" are the two states the receipt separates."""
    workflow = _load()
    steps = workflow["jobs"]["verify-sibling-converged"]["steps"]
    for marker in ("lab_pass_receipt.py probe-lane", "lab_pass_receipt.py emit"):
        step = next(s for s in steps if marker in str(s.get("run", "")))
        assert step["if"] == "always()", marker
    upload = next(
        s
        for s in steps
        if str(s.get("uses", "")).startswith("actions/upload-artifact@")
    )
    assert upload["if"] == "always()"
    assert upload["with"]["if-no-files-found"] == "error"


def test_the_convergence_job_runs_on_the_lane_host_fleet() -> None:
    """The lane's docker daemon is on the LAN; hosted compute cannot see it."""
    workflow = _load()
    assert workflow["jobs"]["verify-sibling-converged"]["runs-on"] == [
        "self-hosted",
        "omnibase-deploy",
    ]


def test_the_convergence_job_only_runs_for_a_published_dev_command() -> None:
    workflow = _load()
    condition = workflow["jobs"]["verify-sibling-converged"]["if"]
    assert "published == 'true'" in condition
    assert "runtime_lane == 'dev'" in condition


def test_the_guard_has_no_override_or_skip_switch() -> None:
    """A gate with a force flag is advisory, and an advisory gate is the status quo."""
    source = GUARD.read_text(encoding="utf-8")
    for forbidden in ("--force", "--skip", "--allow-stale", "SKIP_"):
        assert forbidden not in source, forbidden
