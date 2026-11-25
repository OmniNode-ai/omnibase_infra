# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No pull request head is built on the dev lane (OMN-26637).

The dev deploy agent builds refs its tracking branch contains and nothing else.
Two things in this repository decided whether a pull request head could still
reach a dev build, and each is pinned here:

* the first-party source policy, whose ``open_pr_lanes`` named dev as the one
  lane that may build a DECLARED OPEN PULL REQUEST's branch (OMN-21928). It
  names none now; a pull request head is proved on a pre-PR slot;
* the workflows that publish a dev build command. ``runtime-rebuild-trigger``
  fires only on a MERGED pull request and publishes its merge commit on the base
  branch, never the head; the sibling reusable and the onex-api delivery take
  their ref from a merge as well.

The agent's own refusal of a head that does get published is pinned in
``scripts/deploy-agent/tests/unit/test_dev_queue_supersede_omn26637.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_SHARED_PATH = Path(__file__).with_name("test_first_party_slot_context_omn21928.py")
_spec = importlib.util.spec_from_file_location(
    "first_party_slot_context_fixtures_omn26637", _SHARED_PATH
)
assert _spec and _spec.loader
shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = shared
_spec.loader.exec_module(shared)

fps = shared.fps
CONTRACT = shared.CONTRACT

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"
TRIGGER = WORKFLOWS / "runtime-rebuild-trigger.yml"
SIBLING_REUSABLE = WORKFLOWS / "runtime-rebuild-trigger-reusable.yml"
DECLARED = f"omnibase_infra#5564@{'a' * 40}"


def _load(path: Path) -> dict[Any, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _trigger(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 reads the key ``on`` as True.
    block = workflow.get("on", workflow.get(True))
    assert isinstance(block, dict)
    return block


def test_pr_head_not_published_to_dev_no_lane_builds_a_declared_open_pr() -> None:
    assert CONTRACT.open_pr_lanes == ()
    with pytest.raises(ValueError, match="cannot build a declared open pull request"):
        fps.build_context_for_lane("dev", DECLARED)


def test_pr_head_not_published_to_dev_the_dev_lane_audits_in_the_merged_context() -> (
    None
):
    context, declared = fps.build_context_for_lane("dev")
    assert declared == ""
    assert context == CONTRACT.default_context
    assert CONTRACT.context(context).target_accepts == ()


def test_pr_head_not_published_to_dev_trigger_fires_only_on_a_merged_pull_request() -> (
    None
):
    workflow = _load(TRIGGER)
    pull_request = _trigger(workflow)["pull_request"]
    assert pull_request["types"] == ["closed"]
    job = workflow["jobs"]["trigger-rebuild"]
    assert "github.event.pull_request.merged == true" in job["if"]


def test_pr_head_not_published_to_dev_trigger_publishes_the_merge_commit() -> None:
    workflow = _load(TRIGGER)
    steps = workflow["jobs"]["trigger-rebuild"]["steps"]
    (publish,) = [step for step in steps if step.get("id") == "publish"]
    assert (
        publish["env"]["PR_MERGE_SHA"]
        == "${{ github.event.pull_request.merge_commit_sha }}"
    )
    assert '--source-sha "$PR_MERGE_SHA"' in publish["run"]
    for step in steps:
        rendered = yaml.safe_dump(step)
        assert "head.sha" not in rendered, step.get("name")
        assert "head_sha" not in rendered, step.get("name")


def test_pr_head_not_published_to_dev_sibling_reusable_publishes_a_merge() -> None:
    workflow = _load(SIBLING_REUSABLE)
    inputs = _trigger(workflow)["workflow_call"]["inputs"]
    assert "Merge commit SHA" in inputs["source_sha"]["description"]
    rendered = SIBLING_REUSABLE.read_text(encoding="utf-8")
    assert "head.sha" not in rendered
    assert "head_sha" not in rendered
