# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for the auto-tag -> release dispatch repair (OMN-14468).

Root cause (verified live 2026-07-12): omnibase_infra's ``auto-tag-on-merge.yml``
delegated the tag push to a reusable workflow that pushes with the default
``GITHUB_TOKEN``. GitHub suppresses new workflow runs from ``GITHUB_TOKEN``-
authored events, so the pushed tag never fired ``release.yml`` (``on: push:
tags``). v0.38.0-v0.38.3 were tagged by ``github-actions[bot]`` and NEVER
published; PyPI omnibase-infra stalled at 0.36.1 for six weeks.

Fix: after pushing the tag, dispatch ``release.yml`` explicitly via
``workflow_dispatch`` — the documented exception to the suppression rule.

These tests prove:
* the auto-tag ``if:`` still FIRES on the v0.38.0 merge condition
  (title ``chore: release v0.38.0``) — the trigger was never the problem,
* the previously-missing link is now present: an explicit ``release.yml``
  dispatch step wired to the tag, with the ``actions: write`` permission it
  needs, and
* the 8b publish-resilience wiring (retry wrapper + widened cascade window).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
AUTO_TAG_WORKFLOW = WORKFLOWS / "auto-tag-on-merge.yml"
RELEASE_WORKFLOW = WORKFLOWS / "release.yml"
CASCADE_WORKFLOW = WORKFLOWS / "dependency-cascade.yml"

# The v0.38.0 release PR title that fired auto-tag (and produced a bot tag that
# was never published) — the exact condition we replay.
V0_38_0_PR_TITLE = "chore: release v0.38.0 (OMN-12561) (#1827)"
# A core-style promotion title, which does NOT match (core is tagged manually).
CORE_PROMOTION_TITLE = "chore(OMN-13928): promote omnibase_core 0.46.5 dev->main"
# A normal feature PR title, which must not tag.
NORMAL_PR_TITLE = "fix(OMN-14376): default onex delegate to the system bus"


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _title_triggers_autotag(title: str, labels: tuple[str, ...] = ()) -> bool:
    """Pure replica of the workflow's release-title predicate.

    Mirrors the GitHub ``if:`` expression: a merged PR tags when it carries a
    ``release`` label OR its title starts with ``release:`` / ``chore: release``
    / ``chore(release)``.
    """
    return "release" in labels or title.startswith(
        ("release:", "chore: release", "chore(release)")
    )


def test_predicate_fires_on_v0_38_0_condition() -> None:
    # The trigger fires for the exact title that produced the dead v0.38.0 tag.
    assert _title_triggers_autotag(V0_38_0_PR_TITLE) is True
    # It correctly does NOT fire for core's promotion titles (core is tagged
    # manually — which is why only infra was broken by the token-suppression).
    assert _title_triggers_autotag(CORE_PROMOTION_TITLE) is False
    # And never for a normal feature PR.
    assert _title_triggers_autotag(NORMAL_PR_TITLE) is False
    # The release label path still works.
    assert _title_triggers_autotag(NORMAL_PR_TITLE, labels=("release",)) is True


def test_workflow_if_encodes_the_release_title_condition() -> None:
    workflow = _load_yaml(AUTO_TAG_WORKFLOW)
    condition = workflow["jobs"]["auto-tag"]["if"]
    assert "github.event.pull_request.merged == true" in condition
    assert "startsWith(github.event.pull_request.title, 'chore: release')" in condition
    assert "startsWith(github.event.pull_request.title, 'chore(release)')" in condition
    assert "startsWith(github.event.pull_request.title, 'release:')" in condition
    assert "contains(github.event.pull_request.labels.*.name, 'release')" in condition


def test_workflow_has_actions_write_permission_for_dispatch() -> None:
    workflow = _load_yaml(AUTO_TAG_WORKFLOW)
    permissions = workflow["permissions"]
    # contents:write to push the tag; actions:write to dispatch release.yml.
    assert permissions["contents"] == "write"
    assert permissions["actions"] == "write"


def test_workflow_pushes_tag_then_dispatches_release() -> None:
    workflow = _load_yaml(AUTO_TAG_WORKFLOW)
    steps = workflow["jobs"]["auto-tag"]["steps"]

    # The tag is still created + pushed.
    tag_step = next(
        step for step in steps if "Create and push release tag" in step.get("name", "")
    )
    assert "git tag" in tag_step["run"]
    assert "git push origin" in tag_step["run"]

    # The load-bearing new link: an explicit release.yml dispatch AFTER the tag.
    dispatch_step = next(
        step for step in steps if "Dispatch release.yml" in step.get("name", "")
    )
    run_script = dispatch_step["run"]
    assert "gh workflow run release.yml" in run_script
    assert "--ref" in run_script
    assert "-f tag=" in run_script
    assert dispatch_step["env"]["GH_TOKEN"] == "${{ github.token }}"

    # Ordering: push before dispatch.
    names = [step.get("name", "") for step in steps]
    push_idx = next(
        i for i, n in enumerate(names) if "Create and push release tag" in n
    )
    dispatch_idx = next(i for i, n in enumerate(names) if "Dispatch release.yml" in n)
    assert push_idx < dispatch_idx


def test_workflow_no_longer_delegates_to_token_pushing_reusable() -> None:
    """The reusable pushes with GITHUB_TOKEN (the root cause) — no `uses:` to it.

    A textual mention in a comment is fine (we document the history); what must
    be gone is the actual delegation: neither the job nor any step may ``uses:``
    the reusable workflow that pushes the tag with GITHUB_TOKEN.
    """
    workflow = _load_yaml(AUTO_TAG_WORKFLOW)
    job = workflow["jobs"]["auto-tag"]

    # The job runs inline steps, not a delegated reusable workflow.
    assert "uses" not in job
    assert "steps" in job

    for step in job["steps"]:
        assert "auto-tag-reusable" not in str(step.get("uses", ""))


def test_release_publish_uses_retry_wrapper() -> None:
    workflow = _load_yaml(RELEASE_WORKFLOW)
    steps = workflow["jobs"]["release"]["steps"]
    publish_step = next(
        step
        for step in steps
        if str(step.get("name", "")).startswith("Publish to PyPI")
    )
    assert "scripts/ci/publish_with_retry.py" in publish_step["run"]
    # The raw un-retried publish is gone.
    assert 'uv publish --token "$UV_PUBLISH_TOKEN"' not in publish_step["run"]
    assert publish_step["env"]["UV_PUBLISH_TOKEN"] == "${{ secrets.PYPI_TOKEN }}"


def test_publish_retry_script_exists() -> None:
    assert (REPO_ROOT / "scripts" / "ci" / "publish_with_retry.py").is_file()


def test_dependency_cascade_polls_until_pypi_visible_with_real_ceiling() -> None:
    workflow = _load_yaml(CASCADE_WORKFLOW)
    steps = workflow["jobs"]["open-bump-pr"]["steps"]
    lock_step = next(
        step
        for step in steps
        if step.get("name") == "Create branch and upgrade lockfile"
    )
    run_script = lock_step["run"]
    # Poll-until-visible against the PyPI JSON API, not a fixed 120s window.
    assert "https://pypi.org/pypi/" in run_script
    assert "CASCADE_PROPAGATION_CEILING_SECONDS" in run_script
    assert ":-900" in run_script  # default 15-minute ceiling
    # The old too-short 12x10s ceiling is gone.
    assert "seq 1 12" not in run_script
