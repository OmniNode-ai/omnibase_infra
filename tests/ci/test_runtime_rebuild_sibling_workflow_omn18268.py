# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shape invariants for the sibling rebuild trigger (OMN-18268).

Each assertion here is a property that, if it silently regressed, would turn the
workflow into the false green it exists to replace. They are asserted on the
parsed workflow, never by matching prose.
"""

from __future__ import annotations

import json
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
    secrets = triggers["workflow_call"]["secrets"]
    assert set(secrets) == {
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
        "ONEXBOT_OCC_APP_ID",
        "ONEXBOT_OCC_PRIVATE_KEY",
    }, (
        "the lane declares SASL and the publisher refuses to downgrade it; no "
        "HMAC secret is a precondition of this producer. The App pair is the "
        "OMN-17057 delivery announcement, which crosses a repository boundary "
        "and so cannot use the caller's own token"
    )
    for name in ("KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"):
        assert secrets[name]["required"] is True, name
    for name in ("ONEXBOT_OCC_APP_ID", "ONEXBOT_OCC_PRIVATE_KEY"):
        # Optional at the INTERFACE so an existing caller keeps compiling, and
        # refused at RUN TIME by the announcement job itself. Making them
        # required here would break every caller in the same commit that adds
        # them; making the job skip on their absence would report a delivery
        # that did not happen, which is the failure this ticket removes.
        assert secrets[name]["required"] is False, name


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
    # None means "this job deliberately has no self-checkout". A job only needs
    # one to reach `./.github/actions/...` or a script in this repository;
    # deliver-sibling-candidate (OMN-17057) runs `gh` and `jq` and nothing from
    # the tree, so a checkout would be ceremony that reads as a dependency.
    # It is listed rather than excluded so that adding a job still forces the
    # decision this test exists to force.
    expected_ref: dict[str, str | None] = {
        "trigger-rebuild": "dev",
        "verify-sibling-converged": (
            "${{ inputs.infra_ref || github.job_workflow_sha }}"
        ),
        "deliver-sibling-candidate": None,
    }
    assert set(workflow["jobs"]) == set(expected_ref), (
        "a job was added or renamed; decide which of the refs above it needs "
        "rather than letting it default to an unasserted one"
    )
    for job_id, job in workflow["jobs"].items():
        steps = job["steps"]
        if expected_ref[job_id] is None:
            assert not any(
                str(step.get("uses", "")).startswith("./") for step in steps
            ), (
                f"{job_id} declares no self-checkout but references a local "
                "action, which resolves against the CALLER's workspace"
            )
            assert not any("scripts/" in str(step.get("run", "")) for step in steps), (
                f"{job_id} declares no self-checkout but invokes a repository "
                "script, which is not present on disk"
            )
            continue
        first = steps[0]
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
    # OMN-19507 AC2: the lane half of the name is the routed instance's
    # receipt lane (compose-dev for dev-201, which the committed table routes
    # every merge to), and the sha half is still the sibling's.
    assert upload["with"]["name"] == (
        "lab-pass-receipt-${{ needs.trigger-rebuild.outputs.receipt_lane }}"
        "-${{ needs.trigger-rebuild.outputs.sibling_sha }}"
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
    """The lane's docker daemon is on the LAN; hosted compute cannot see it.

    OMN-18602 moved this job from `omnibase-deploy` to the verify class, and
    THIS job is the reason that direction was the only safe one. It is called
    from omnimarket, so it and the release-train DEPLOY jobs sit in different
    repositories -- and a GitHub `concurrency` group is scoped to one
    repository, so no group can ever serialise them against each other.
    Relieving the queue by adding a SECOND runner to the deploy label would
    therefore have deleted the only guard that spans both callers. Moving the
    read-only work off instead leaves the deploy label as one physical runner,
    which IS the guard.

    The verify runners register into the `omnibase-deploy` runner GROUP, whose
    visibility covers omnibase_infra and omnimarket, so this caller can reach
    them; group membership grants repository visibility and the LABEL routes.
    `host-201` is required because the class has members on two other lab
    hosts that cannot see this lane.
    """
    workflow = _load()
    # OMN-19507 AC2: the runner is the routed instance's. Under the committed
    # routing table every merge routes to dev-201, whose runner is exactly the
    # host-201 verify runner this test has always required.
    from scripts.ci.deploy_lane_verify_route import job_outputs, load_table, resolve

    assert workflow["jobs"]["verify-sibling-converged"]["runs-on"] == (
        "${{ fromJSON(needs.trigger-rebuild.outputs.verify_runs_on) }}"
    )
    routed = job_outputs(
        resolve(load_table(), runtime_lane="dev", requested_by="gha/omnimarket/pr-1")
    )
    assert json.loads(routed["verify_runs_on"]) == [
        "self-hosted",
        "omnibase-verify",
        "host-201",
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
