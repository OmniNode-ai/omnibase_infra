# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A PR whose base is main is refused unless the release workflow opened it (OMN-17427).

Operator RULING 2026-10-10T16:52:56Z (lane dev-default-9f8a): "Dev as a default
branch has been the rule for months anything targeting Main has been violating
that rule." main is release-synced: release.yml fast-forwards it to each release
tag, and promotion PRs to main are retired. The guard therefore keeps exactly one
exemption, the release PR the release-train workflow opens, and refuses everything
else, including the dev-to-main promotion PR and the hotfix PR it used to admit.

The tests run the guard step's own shell against a PR's base, head, body and
author, so a refusal is the workflow's own exit code and not a reading of its text.
The release PR is the negative control: it reaches the same step as the refused
heads and must pass, so a guard that refuses everything cannot satisfy this file.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.ci import ci_summary_gate

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "main-target-guard.yml"
RELEASE_TRAIN_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"
)
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"

RELEASE_VERSION = "0.4.311"
BASE_ENV = {"PATH": os.environ["PATH"], "LC_ALL": "C"}


def guard_script() -> str:
    """The shell of the workflow's one guard step."""
    workflow = yaml.safe_load(GUARD_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["main-target-guard"]["steps"]
    assert len(steps) == 1, (
        "the guard is one step; a second step could skip the verdict"
    )
    return str(steps[0]["run"])


def run_guard(
    *,
    base: str,
    head: str,
    author_type: str = "User",
    body: str = "",
) -> subprocess.CompletedProcess[str]:
    env = {
        **BASE_ENV,
        "BASE_REF": base,
        "HEAD_REF": head,
        "PR_BODY": body,
        "PR_AUTHOR_TYPE": author_type,
    }
    return subprocess.run(
        ["bash", "-c", guard_script()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def release_branch(version: str = RELEASE_VERSION) -> str:
    """The head branch release-train-nightly.yml opens its release PR from."""
    text = RELEASE_TRAIN_WORKFLOW.read_text(encoding="utf-8")
    found = re.search(r'branch="(automation/release-\$\{VERSION\}-omn-\d+)"', text)
    assert found, "release-train-nightly.yml no longer names the release PR head branch"
    return found.group(1).replace("${VERSION}", version)


def test_release_branch_is_read_from_the_release_workflow() -> None:
    assert release_branch() == f"automation/release-{RELEASE_VERSION}-omn-18595"


def test_release_pr_opened_by_the_release_workflow_is_allowed() -> None:
    result = run_guard(base="main", head=release_branch(), author_type="Bot")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "head",
    [
        "jonah/omn-17427-feature",
        "dev",
        "hotfix/omn-1-urgent",
        "automation/release-latest-omn-18595",
        f"automation/release-{RELEASE_VERSION}-omn-18595-extra",
        f"x/automation/release-{RELEASE_VERSION}-omn-18595",
    ],
)
def test_every_other_head_targeting_main_is_refused(head: str) -> None:
    result = run_guard(base="main", head=head, author_type="Bot")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Retarget this PR to dev" in result.stdout


@pytest.mark.live_contact(
    "tests/ci/fixtures/main_target_guard_release_prs_omn17427.json"
)
def test_recorded_release_prs_pass_the_guard_and_a_person_with_the_same_head_does_not(
    recorded_response: dict[str, object],
) -> None:
    pull_requests = recorded_response["pull_requests"]
    assert isinstance(pull_requests, list)
    assert len(pull_requests) >= 2
    for pr in pull_requests:
        assert pr["base"] == "dev", "release PRs of a dev-default repo target dev"
        assert pr["author_is_bot"] is True
        released = run_guard(base="main", head=pr["head_ref"], author_type="Bot")
        assert released.returncode == 0, pr["pr"] + released.stdout + released.stderr
        typed = run_guard(base="main", head=pr["head_ref"], author_type="User")
        assert typed.returncode == 1, pr["pr"] + typed.stdout + typed.stderr


def test_release_branch_name_from_a_person_is_refused() -> None:
    result = run_guard(base="main", head=release_branch(), author_type="User")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Retarget this PR to dev" in result.stdout


def test_promotion_receipt_no_longer_admits_a_dev_to_main_pr() -> None:
    result = run_guard(
        base="main", head="dev", author_type="User", body="promotion-receipt: OCC-1234"
    )
    assert result.returncode == 1, result.stdout + result.stderr


def test_hotfix_evidence_no_longer_admits_a_hotfix_pr() -> None:
    result = run_guard(
        base="main",
        head="hotfix/omn-1-urgent",
        author_type="User",
        body="hotfix-evidence: OCC-1234\nbackmerge: #99",
    )
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("head", ["jonah/omn-17427-feature", "dev", release_branch()])
def test_a_pr_targeting_dev_is_not_judged(head: str) -> None:
    result = run_guard(base="dev", head=head)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "guard does not apply" in result.stdout


def test_guard_runs_on_open_retarget_push_and_reopen() -> None:
    workflow = yaml.safe_load(GUARD_WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow[True]
    types = set(triggers["pull_request"]["types"])
    assert {"opened", "edited", "synchronize", "reopened"} <= types


def test_guard_context_is_asserted_by_the_ci_summary_required_context() -> None:
    assert "main-target-guard" in ci_summary_gate.EXPECTED_EXTERNAL_CONTEXTS


def hook_entry() -> tuple[dict[str, object], list[str]]:
    config = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    hooks = [
        hook
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook.get("id") == "main-target-guard"
    ]
    assert len(hooks) == 1, (
        "pre-commit config must declare the main-target-guard hook once"
    )
    return hooks[0], shlex.split(str(hooks[0]["entry"]))


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**BASE_ENV, "HOME": str(repo)},
    )


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    repo = tmp_path / "work"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "jonah/omn-17427-feature")
    return repo


def push_hook(
    repo: Path,
    *,
    remote_branch: str,
    local_branch: str = "refs/heads/jonah/omn-17427-feature",
) -> subprocess.CompletedProcess[str]:
    _, argv = hook_entry()
    env = {
        **BASE_ENV,
        "HOME": str(repo),
        "PRE_COMMIT_REMOTE_BRANCH": remote_branch,
        "PRE_COMMIT_LOCAL_BRANCH": local_branch,
    }
    return subprocess.run(
        argv, cwd=repo, env=env, capture_output=True, text=True, check=False
    )


def test_hook_is_a_verbose_pre_push_warning_that_needs_no_filenames() -> None:
    hook, _ = hook_entry()
    assert hook["stages"] == ["pre-push"]
    assert hook["pass_filenames"] is False
    assert hook["always_run"] is True
    assert hook["verbose"] is True


def test_hook_warns_when_the_push_lands_on_main(clone: Path) -> None:
    result = push_hook(clone, remote_branch="refs/heads/main")
    assert result.returncode == 0, result.stderr
    assert "WARNING" in result.stderr
    assert "main" in result.stderr


def test_hook_warns_when_the_branch_tracks_main(clone: Path) -> None:
    git(clone, "config", "branch.jonah/omn-17427-feature.remote", "origin")
    git(clone, "config", "branch.jonah/omn-17427-feature.merge", "refs/heads/main")
    result = push_hook(clone, remote_branch="refs/heads/jonah/omn-17427-feature")
    assert result.returncode == 0, result.stderr
    assert "WARNING" in result.stderr


def test_hook_warns_when_gh_recorded_main_as_the_pr_base(clone: Path) -> None:
    git(clone, "config", "branch.jonah/omn-17427-feature.gh-merge-base", "main")
    result = push_hook(clone, remote_branch="refs/heads/jonah/omn-17427-feature")
    assert result.returncode == 0, result.stderr
    assert "WARNING" in result.stderr


def test_hook_is_silent_for_a_feature_branch_aimed_at_dev(clone: Path) -> None:
    git(clone, "config", "branch.jonah/omn-17427-feature.remote", "origin")
    git(clone, "config", "branch.jonah/omn-17427-feature.merge", "refs/heads/dev")
    git(clone, "config", "branch.jonah/omn-17427-feature.gh-merge-base", "dev")
    result = push_hook(clone, remote_branch="refs/heads/jonah/omn-17427-feature")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == ""
