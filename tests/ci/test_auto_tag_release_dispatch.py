# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for the auto-tag -> release dispatch repair (OMN-14468).

Root cause (verified live 2026-07-12): omnibase_infra's ``auto-tag-on-merge.yml``
delegated the tag push to a reusable workflow that pushes with the default
``GITHUB_TOKEN``. GitHub suppresses new workflow runs from ``GITHUB_TOKEN``-
authored events, so the pushed tag never fired ``release.yml`` (``on: push:
tags``). v0.38.0-v0.38.3 were tagged by ``github-actions[bot]`` and NEVER
published; PyPI omnibase-infra stalled at 0.36.1 for six weeks.

First fix (OMN-14468): push the tag with the workflow token anyway, then
dispatch ``release.yml`` explicitly via ``workflow_dispatch`` — the documented
exception to the suppression rule. That published, but every cut then arrived as
``event: workflow_dispatch``, which proves the job ran rather than that the tag
was consumable.

Second fix (OMN-18662, superseding the dispatch workaround): push the tag with an
``onexbot-occ-writer`` installation token, which DOES start push-driven CI in
this org, and delete the dispatch step. OMN-18273 established the App-token
property and showed the earlier org-wide "App tokens are suppressed too" finding
to be a credential confound; live proof is ``omnibase_spi`` run ``35134173847``
(``event: push``, ``actor: onexbot-occ-writer[bot]``).

These tests prove:
* the auto-tag ``if:`` still FIRES on the v0.38.0 merge condition
  (title ``chore: release v0.38.0``) — the trigger was never the problem,
* the link from tag to publish is present in its CURRENT form: the tag is pushed
  with a fail-closed App-token mint from a checkout that persists no workflow
  credential, and the superseded ``workflow_dispatch`` workaround is gone so it
  cannot race the push-triggered run, and
* the 8b publish-resilience wiring (retry wrapper + widened cascade window).

The two conditions in the second bullet are asserted separately because they fail
independently, and a job that gets either half wrong pushes as the workflow token
while still reporting success.
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


def test_workflow_holds_no_write_scope_of_its_own() -> None:
    """The workflow token carries no scope the tag push could accidentally use.

    OMN-18662 moved the write scope onto the App installation token. Leaving
    ``contents: write`` here would not by itself reintroduce the defect, but it
    would leave a usable workflow-token write path next to a push that must not
    take one — and ``actions: write`` existed only for the deleted dispatch.
    """
    workflow = _load_yaml(AUTO_TAG_WORKFLOW)
    permissions = workflow["permissions"]
    assert permissions["contents"] == "read", (
        "auto-tag-on-merge.yml grants the workflow token contents:write. The tag "
        "push authenticates with the onexbot-occ-writer installation token "
        "(OMN-18662); the workflow token needs read only."
    )
    assert "actions" not in permissions, (
        "actions:write existed solely to dispatch release.yml. That dispatch was "
        "deleted by OMN-18662 — a tag pushed with the App token starts release.yml "
        "from the push itself."
    )


def test_workflow_pushes_the_tag_with_a_fail_closed_app_token() -> None:
    """The tag push authenticates as the App, with no route back to the workflow token.

    A tag pushed with the workflow token is worse than no tag: ``release.yml``
    cannot consume it, and it cannot be re-pushed without deleting it first. So an
    unmintable token must stop the push rather than degrade it.
    """
    workflow = _load_yaml(AUTO_TAG_WORKFLOW)
    steps = workflow["jobs"]["auto-tag"]["steps"]

    mint = next(
        (
            step
            for step in steps
            if "actions/create-github-app-token" in str(step.get("uses", ""))
        ),
        None,
    )
    assert mint is not None, (
        "auto-tag-on-merge.yml pushes a release tag but never mints an App "
        "installation token. release.yml triggers on `push: tags:`, and a "
        "workflow-token push delivers no push event, so the tag would sit "
        "unconsumed (OMN-14468, OMN-18662)."
    )
    with_block = mint.get("with") or {}
    assert with_block.get("app-id") == "${{ secrets.ONEXBOT_OCC_APP_ID }}", (
        "the mint must use the onexbot-occ-writer app id org secret"
    )
    assert with_block.get("private-key") == "${{ secrets.ONEXBOT_OCC_PRIVATE_KEY }}", (
        "the mint must use the onexbot-occ-writer private key org secret"
    )

    tag_step = next(
        step for step in steps if "Create and push tag" in step.get("name", "")
    )
    run_script = tag_step["run"]
    assert "git tag" in run_script
    # The credential travels in the remote URL. A bare `git push origin` would
    # find no credential at all now that the checkout persists none.
    assert "x-access-token:${APP_TOKEN}" in run_script, (
        "the tag push does not carry the App token in its remote URL; with "
        "persist-credentials: false there is no other credential for it to use."
    )
    assert tag_step["env"]["APP_TOKEN"] == "${{ steps.app-token.outputs.token }}"

    job_text = yaml.safe_dump(workflow["jobs"]["auto-tag"], default_flow_style=False)
    assert "secrets.GITHUB_TOKEN" not in job_text, (
        "the tag job references secrets.GITHUB_TOKEN. The push must have no route "
        "back to the workflow token in any form — a fallback silently restores it, "
        "the push is suppressed again, and the job still reports success."
    )
    assert "github.token" not in job_text, (
        "the tag job references github.token. Same failure as above, spelled the "
        "other way."
    )


def test_workflow_checks_out_without_persisting_credentials() -> None:
    """The other half of the pair, and the one that is silently defeated.

    A default ``actions/checkout`` leaves the workflow token in the local git
    config, where it overrides the App credential in the push URL — so a job that
    mints correctly and checks out with the default still pushes as
    ``github-actions[bot]`` and still delivers no push event. Omitting the flag is
    not a smaller version of the fix; it defeats it entirely. This is the exact
    confound that produced the wrong org-wide finding in August (OMN-18273).
    """
    workflow = _load_yaml(AUTO_TAG_WORKFLOW)
    checkouts = [
        step
        for step in workflow["jobs"]["auto-tag"]["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout")
    ]
    assert checkouts, (
        "auto-tag-on-merge.yml runs no actions/checkout, so this assertion cannot "
        "mean anything. Either the job stopped checking out or the job id moved."
    )
    for step in checkouts:
        assert (step.get("with") or {}).get("persist-credentials") is False, (
            "the tag job checks out without `persist-credentials: false`. The "
            "persisted workflow-token credential overrides the App credential in "
            "the push URL, the tag is pushed as github-actions[bot], no push event "
            "is delivered, and release.yml never starts — while this job still "
            "reports success."
        )


def test_the_superseded_dispatch_workaround_is_gone() -> None:
    """A retained dispatch would race the push-triggered run and reinstate the
    ``workflow_dispatch`` event OMN-18662 exists to remove.

    Its idempotency check ("has release.yml already run for this tag") cannot
    distinguish "the push trigger worked" from "the push trigger has not fired
    yet", so keeping it as a belt-and-braces fallback would make the release event
    intermittent rather than safe.
    """
    workflow = _load_yaml(AUTO_TAG_WORKFLOW)
    job_text = yaml.safe_dump(workflow["jobs"]["auto-tag"], default_flow_style=False)
    assert "gh workflow run release.yml" not in job_text, (
        "auto-tag-on-merge.yml dispatches release.yml again. OMN-18662's proof is a "
        "release run whose event is `push`; a dispatch here races that run and "
        "makes the event intermittent."
    )


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


def test_dependency_cascade_checks_movability_before_locking() -> None:
    """OMN-15604 AC4: a git-pinned package cannot be moved by `uv lock
    --upgrade-package` (uv always prefers an explicit [tool.uv.sources]
    override over registry resolution), so the previous no-op re-lock
    silently reported "no lockfile changes -- already on latest" even when
    the repo was still stuck on the git pin. The lock step must now run the
    movability check FIRST and fail loud+explicit, not attempt the
    guaranteed-no-op lock."""
    workflow = _load_yaml(CASCADE_WORKFLOW)
    steps = workflow["jobs"]["open-bump-pr"]["steps"]

    checkout_step = next(
        step
        for step in steps
        if step.get("name") == "Checkout omnibase_infra dep-provenance script"
    )
    assert checkout_step["with"]["repository"] == "OmniNode-ai/omnibase_infra"
    assert "check_dep_provenance.py" in checkout_step["with"]["sparse-checkout"]

    lock_step = next(
        step
        for step in steps
        if step.get("name") == "Create branch and upgrade lockfile"
    )
    run_script = lock_step["run"]
    movability_check_index = run_script.index("--check-movable")
    lock_command_index = run_script.index("uv lock \\")
    # The movability pre-check must run BEFORE the (potentially no-op) lock
    # command, not after -- checking after would already have silently
    # produced the misleading "no lockfile changes" state.
    assert movability_check_index < lock_command_index
    assert "check_dep_provenance.py" in run_script

    pre_lock_block = run_script[:lock_command_index]
    assert "--check-movable" in pre_lock_block
    assert "::error::" in pre_lock_block
    assert "exit 1" in pre_lock_block
