# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""contracts/OMN-18595.yaml's release-cut check, run as written against real git (OMN-18595).

Loads the real contract from disk, takes its release-cut command check
verbatim, and runs it inside a real git repository that has an ``origin/dev``
ref. The contract calls ``ModelReleaseCutChangelog``; this test drives that
call through git, pyproject.toml and CHANGELOG.md rather than through the model
alone.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.integration

_CONTRACT = Path(__file__).resolve().parents[3] / "contracts" / "OMN-18595.yaml"
_CHECK_ID = "dod-omn-18595-release-cut-changelog"
_LAUNCHER = "uv run --frozen python"
_VERSION = "0.38.75"
_DEV_CHANGELOG = "# Changelog\n\n## v0.38.74\n\n- entry\n"


def _release_cut_command() -> str:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    item = next(i for i in contract["dod_evidence"] if i["id"] == _CHECK_ID)
    command = next(c for c in item["checks"] if c["check_type"] == "command")
    value: str = command["check_value"]
    assert value.startswith(_LAUNCHER), "contract launcher changed; update this test"
    return value


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)


@pytest.fixture
def repo_at_dev(tmp_path: Path) -> Path:
    """A repository whose ``origin/dev`` is its first commit; dev's changelog lags pyproject."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "dev")
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{_VERSION}"\n', encoding="utf-8"
    )
    (repo / "CHANGELOG.md").write_text(_DEV_CHANGELOG, encoding="utf-8")
    _commit_all(repo, "dev")
    _git(repo, "update-ref", "refs/remotes/origin/dev", "HEAD")
    _git(repo, "checkout", "-b", "feature")
    return repo


def _run_check(repo: Path) -> subprocess.CompletedProcess[str]:
    tokens = shlex.split(_release_cut_command())
    argv = [sys.executable, *tokens[len(_LAUNCHER.split()) :]]
    return subprocess.run(
        argv,
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(os.environ),
    )


def test_cascade_head_that_leaves_changelog_alone_passes(repo_at_dev: Path) -> None:
    (repo_at_dev / "other.txt").write_text("bump\n", encoding="utf-8")
    _commit_all(repo_at_dev, "cascade bump")
    result = _run_check(repo_at_dev)
    assert result.returncode == 0, result.stderr + result.stdout


def test_release_cut_naming_the_pyproject_version_passes(repo_at_dev: Path) -> None:
    (repo_at_dev / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## v{_VERSION}\n\n- cut\n\n"
        + _DEV_CHANGELOG.split("\n", 2)[2],
        encoding="utf-8",
    )
    _commit_all(repo_at_dev, "release cut")
    result = _run_check(repo_at_dev)
    assert result.returncode == 0, result.stderr + result.stdout


def test_release_cut_naming_another_version_fails(repo_at_dev: Path) -> None:
    (repo_at_dev / "CHANGELOG.md").write_text(
        "# Changelog\n\n## v9.9.9\n\n- cut\n\n" + _DEV_CHANGELOG.split("\n", 2)[2],
        encoding="utf-8",
    )
    _commit_all(repo_at_dev, "wrong release cut")
    result = _run_check(repo_at_dev)
    assert result.returncode == 1
    assert "9.9.9" in result.stdout


def test_unreadable_merge_base_is_read_as_a_release_cut(repo_at_dev: Path) -> None:
    # No origin/dev ref: the merge base cannot be read, the head counts as a
    # release cut, and dev's changelog (top heading v0.38.74 under pyproject
    # 0.38.75) fails the rule.
    _git(repo_at_dev, "update-ref", "-d", "refs/remotes/origin/dev")
    result = _run_check(repo_at_dev)
    assert result.returncode == 1
    assert "0.38.74" in result.stdout
