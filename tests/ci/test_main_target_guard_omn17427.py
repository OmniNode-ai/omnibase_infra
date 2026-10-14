# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A pull request whose base is ``main`` is refused (OMN-17427).

The operator ruling of 2026-10-10T16:52:56Z re-affirms the standing rule: ``dev``
is the default branch of every code repository and a pull request that targets
``main`` violates it. ``main`` is release-synced. The release workflow advances
it by fast-forwarding the ref to the published tag over the REST API and opens
no pull request against it. The release train opens its release pull request
from ``automation/release-<version>-omn-18595`` onto the repository's default
branch, and the post-release version bump comes from
``automation/post-release-dev-bump-<version>`` onto ``dev``.

``main-target-guard`` used to let two shapes through: a ``dev`` head that carried a
``promotion-receipt:`` line and a ``hotfix/*`` head that carried
``hotfix-evidence:`` and ``backmerge:`` lines. Neither shape exists in the release
workflow any more, and the ruling allows no exception, so this module pins:

* the guard's shipped step, run as a program, refuses every base-``main`` pull
  request, the two retired shapes and the release workflow's own heads included,
  and lets the release pull requests through when they target ``dev``;
* the release workflows still open their pull requests against ``dev`` and move
  ``main`` by fast-forward, so the empty allow-list stays true to them;
* the guard is a required context (it is registered in the CI Summary poller)
  and a pre-push hook of ``.pre-commit-config.yaml`` warns locally.

The release pull request is the negative control: it is the shape the guard
must NOT refuse, so a guard that refused everything would fail here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import ci_summary_gate

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
GUARD_WORKFLOW = WORKFLOWS / "main-target-guard.yml"
RELEASE_WORKFLOW = WORKFLOWS / "release.yml"
TRAIN_WORKFLOW = WORKFLOWS / "release-train-nightly.yml"
TRAIN_POLICY = REPO_ROOT / "config" / "release_train_policy.yaml"
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"

GUARD_STEP = "Check PR target branch"
GUARD_HOOK_ID = "main-target-guard"

# The head branches the release workflows open pull requests from.
RELEASE_TRAIN_HEAD = "automation/release-1.2.3-omn-18595"
POST_RELEASE_BUMP_HEAD = "automation/post-release-dev-bump-1.2.4"

# The two bodies the retired exceptions asked for.
PROMOTION_BODY = "promotion-receipt: OCC-1234"
HOTFIX_BODY = "hotfix-evidence: OCC-1234\nbackmerge: #99"


def _guard_program() -> str:
    """The shipped guard step's ``run`` script, exactly as the workflow carries it."""
    workflow = yaml.safe_load(GUARD_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["main-target-guard"]["steps"]
    named = [step for step in steps if step.get("name") == GUARD_STEP]
    assert len(named) == 1, f"expected exactly one step named {GUARD_STEP!r}"
    return str(named[0]["run"])


def _run_guard(
    base: str, head: str, body: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run the guard the way Actions does: ``bash -e`` over the step script."""
    env = {
        "PATH": os.environ["PATH"],
        "BASE_REF": base,
        "HEAD_REF": head,
        "PR_BODY": body,
    }
    return subprocess.run(
        ["bash", "-e", "-c", _guard_program()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("head", "body"),
    [
        pytest.param("feature/OMN-1-thing", "", id="feature-branch"),
        pytest.param("dev", PROMOTION_BODY, id="promotion-receipt-from-dev"),
        pytest.param("dev", "", id="dev-without-receipt"),
        pytest.param("hotfix/outage", HOTFIX_BODY, id="hotfix-with-evidence"),
        pytest.param("hotfix/outage", "", id="hotfix-without-evidence"),
        pytest.param(RELEASE_TRAIN_HEAD, "", id="release-train-head"),
        pytest.param(POST_RELEASE_BUMP_HEAD, "", id="post-release-bump-head"),
    ],
)
def test_a_pr_with_base_main_is_refused(head: str, body: str) -> None:
    result = _run_guard("main", head, body)
    assert result.returncode != 0, (
        f"a PR from {head!r} onto main was admitted:\n{result.stdout}"
    )
    assert "::error::" in result.stdout
    assert "dev" in result.stdout, "the refusal must name dev as the base to use"


@pytest.mark.parametrize(
    "head",
    [
        pytest.param(RELEASE_TRAIN_HEAD, id="release-train-head"),
        pytest.param(POST_RELEASE_BUMP_HEAD, id="post-release-bump-head"),
        pytest.param("feature/OMN-1-thing", id="feature-branch"),
    ],
)
def test_the_release_prs_and_ordinary_prs_onto_dev_are_admitted(head: str) -> None:
    """The negative control: what the release workflows open is not refused."""
    result = _run_guard("dev", head)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::error::" not in result.stdout


def test_a_stacked_pr_base_is_left_to_the_non_dev_base_guard() -> None:
    result = _run_guard("feature/parent", "feature/child")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_guard_keeps_the_edited_trigger() -> None:
    """A retarget onto main arrives as ``edited``; losing it makes the refusal silent."""
    workflow = yaml.safe_load(GUARD_WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert "edited" in triggers["pull_request"]["types"]


def test_the_guard_reads_no_pr_body() -> None:
    """With no exception left there is no body line to read; none may creep back."""
    program = _guard_program()
    assert "PR_BODY" not in program
    assert "promotion-receipt" not in program
    assert "hotfix" not in program


def test_the_release_train_opens_its_pr_from_the_named_head_onto_the_default_branch() -> (
    None
):
    train = TRAIN_WORKFLOW.read_text(encoding="utf-8")
    assert 'branch="automation/release-${VERSION}-omn-18595"' in train
    assert '--base "${BASE_BRANCH}"' in train
    policy: dict[str, Any] = yaml.safe_load(TRAIN_POLICY.read_text(encoding="utf-8"))
    branches = {repo["default_branch"] for repo in _repos(policy)}
    assert branches, "no default_branch found in the release train policy"
    assert "main" not in branches, (
        "a release train repository lands on main; the guard's empty allow-list "
        "no longer matches the release workflow"
    )


def _repos(policy: dict[str, Any]) -> list[Any]:
    repos: dict[str, Any] = policy["repos"]
    return list(repos.values())


def test_the_post_release_bump_pr_targets_dev() -> None:
    release = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "branch=automation/post-release-dev-bump-${TARGET}" in release
    assert '--base dev --head "$BRANCH"' in release


def test_release_advances_main_by_fast_forward_and_opens_no_pr_onto_it() -> None:
    release = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "git/refs/heads/main" in release
    assert '\\"force\\":false' in release
    assert "--base main" not in release


def test_the_guard_is_registered_in_the_ci_summary_poller() -> None:
    """Dev requires one context, CI Summary; the guard is required through it."""
    assert GUARD_HOOK_ID in ci_summary_gate.EXPECTED_EXTERNAL_CONTEXTS
    workflow = yaml.safe_load(GUARD_WORKFLOW.read_text(encoding="utf-8"))
    assert workflow["jobs"]["main-target-guard"]["name"] == GUARD_HOOK_ID


def _guard_hook() -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load(
        PRECOMMIT_CONFIG.read_text(encoding="utf-8")
    )
    hooks: list[dict[str, Any]] = [
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == GUARD_HOOK_ID
    ]
    assert len(hooks) == 1, f"expected one pre-commit hook with id {GUARD_HOOK_ID!r}"
    return hooks[0]


def _run_hook(remote_branch: str | None) -> subprocess.CompletedProcess[str]:
    """Run the hook's entry the way pre-commit does at the pre-push stage."""
    hook = _guard_hook()
    env = {"PATH": os.environ["PATH"]}
    if remote_branch is not None:
        env["PRE_COMMIT_REMOTE_BRANCH"] = remote_branch
    return subprocess.run(
        ["bash", "-c", hook["entry"]],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_pre_push_hook_is_wired_to_warn_and_not_to_fail() -> None:
    hook = _guard_hook()
    assert hook["stages"] == ["pre-push"]
    assert hook["always_run"] is True
    assert hook["pass_filenames"] is False
    assert hook["verbose"] is True, "pre-commit prints a passing hook only when verbose"


def test_a_push_to_main_warns_locally_and_does_not_block() -> None:
    result = _run_hook("refs/heads/main")
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "main-target-guard" in output
    assert "dev" in output


@pytest.mark.parametrize(
    "remote_branch",
    [
        pytest.param("refs/heads/dev", id="dev"),
        pytest.param("refs/heads/feature/OMN-1-thing", id="feature"),
        pytest.param("refs/heads/main-notes", id="main-prefix"),
        pytest.param(None, id="variable-unset"),
    ],
)
def test_a_push_elsewhere_prints_nothing(remote_branch: str | None) -> None:
    result = _run_hook(remote_branch)
    assert result.returncode == 0
    assert (result.stdout + result.stderr).strip() == ""


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn17427_main_target_guard_pr_records.json"
)
def test_the_guard_replays_the_recorded_pr_records(
    recorded_response: dict[str, Any],
) -> None:
    """Replay the base and head refs of real pull requests the watcher recorded.

    Every record whose base is main in a dev-default repository is refused,
    including the hotfix and personal-branch heads; every recorded release-train
    pull request and ordinary dev-based pull request is admitted.
    """
    base_main = recorded_response["base_main"]
    admitted = (
        recorded_response["release_onto_dev"] + recorded_response["ordinary_onto_dev"]
    )
    assert base_main, "the recording holds no base-main pull request"
    assert any(r["head_ref"].startswith("automation/release-") for r in admitted)
    assert all(r["base"] == "main" for r in base_main)
    assert all(r["base"] == "dev" for r in admitted)
    for record in base_main:
        result = _run_guard(record["base"], record["head_ref"])
        assert result.returncode != 0, record["pr"]
    for record in admitted:
        result = _run_guard(record["base"], record["head_ref"])
        assert result.returncode == 0, record["pr"] + result.stdout
