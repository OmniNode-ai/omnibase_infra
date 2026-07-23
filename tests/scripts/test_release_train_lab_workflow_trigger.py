# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Config-as-data guards for the release-train-lab deploy trigger (OMN-14957).

Live failure being eliminated: run 29977781670's cut-tag job created
``refs/tags/lab/stability/20260723T034503Z-c16e76fd1fe4`` via ``gh api``
with the job's own GITHUB_TOKEN, and NO push-event run ever fired -- GitHub
suppresses workflow triggering for events created with the default workflow
token (documented anti-recursion behavior, OMN-9426 class). The workflow's
docstring claimed the ref creation "fires the deploy job"; it cannot, ever.

The fix chains the deploy job into the SAME workflow run after an
execute-mode cut-tag (``needs: cut-tag`` keyed off the cut tag output),
keeping ``push: tags:`` only for tags pushed with non-GITHUB_TOKEN
credentials (operator workstation). These tests pin that trigger topology so
it cannot silently regress to the push-only shape that can never fire.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-train-lab.yml"


def _load() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _triggers(data: dict[str, Any]) -> dict[str, Any]:
    # YAML 1.1 parses the bare `on:` key as boolean True.
    return data.get("on", data.get(True))  # type: ignore[no-any-return]


@pytest.mark.unit
def test_deploy_job_is_chained_onto_cut_tag() -> None:
    data = _load()
    deploy = data["jobs"]["deploy"]
    needs = deploy.get("needs")
    needs_list = needs if isinstance(needs, list) else [needs]
    assert "cut-tag" in needs_list, (
        "deploy must `needs: cut-tag`: a GITHUB_TOKEN-created tag ref delivers "
        "no push event, so in-run chaining is the ONLY trigger path for a "
        "runner-cut tag (OMN-14957)"
    )
    cond = str(deploy.get("if", ""))
    assert "!cancelled()" in cond, (
        "deploy.if must include !cancelled(): on a push event cut-tag is "
        "SKIPPED, and the implicit success() dependency would otherwise skip "
        "the deploy job too"
    )
    assert "github.event_name == 'push'" in cond
    assert "github.event_name == 'workflow_dispatch'" in cond
    assert "needs.cut-tag.result == 'success'" in cond
    assert "needs.cut-tag.outputs.tag" in cond
    assert "inputs.execute" in cond, (
        "a dry-run dispatch (execute=false) must never reach the deploy job"
    )


@pytest.mark.unit
def test_cut_tag_publishes_tag_output() -> None:
    data = _load()
    cut = data["jobs"]["cut-tag"]
    outputs = cut.get("outputs", {})
    assert "tag" in outputs, "cut-tag must publish the cut tag name as a job output"
    assert "steps.cut.outputs.tag" in str(outputs["tag"])
    # The step producing it exists and runs the cut script.
    step_ids = [s.get("id") for s in cut["steps"]]
    assert "cut" in step_ids


@pytest.mark.unit
def test_push_trigger_remains_for_operator_cut_tags() -> None:
    triggers = _triggers(_load())
    push = triggers.get("push", {})
    tags = push.get("tags", [])
    assert "lab/dev/**" in tags and "lab/stability/**" in tags, (
        "push: tags: must remain -- it is the trigger path for tags pushed "
        "with NON-GITHUB_TOKEN credentials (operator workstation)"
    )


@pytest.mark.unit
def test_deploy_resolves_triggering_tag_for_both_paths() -> None:
    data = _load()
    deploy = data["jobs"]["deploy"]
    trig = str(deploy.get("env", {}).get("TRIGGERING_TAG", ""))
    assert "github.ref_name" in trig and "needs.cut-tag.outputs.tag" in trig, (
        "TRIGGERING_TAG must resolve to the pushed ref on a push event and to "
        "the freshly cut tag on the chained path"
    )
    # The checkout must pin the tagged commit on the chained path too.
    checkout = next(
        s
        for s in deploy["steps"]
        if str(s.get("uses", "")).startswith("actions/checkout")
    )
    ref = str(checkout.get("with", {}).get("ref", ""))
    assert "needs.cut-tag.outputs.tag" in ref, (
        "deploy checkout must check out the cut tag on the chained path so the "
        "deploy scripts run AT the tagged commit"
    )


@pytest.mark.unit
def test_no_step_claims_github_token_refs_fire_push_workflows() -> None:
    """The cut script must not re-grow the disproved claim in its runtime
    output (comments aside): GITHUB_TOKEN-created refs fire nothing."""
    script = REPO_ROOT / "scripts" / "runtime_build" / "cut_release_train_tag.sh"
    active_lines = [
        line
        for line in script.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    offenders = [
        line
        for line in active_lines
        if "fires" in line and "deploy job" in line and "NO" not in line
    ]
    assert not offenders, (
        f"cut_release_train_tag.sh still claims its ref creation fires the "
        f"deploy job -- disproved live by run 29977781670: {offenders}"
    )
