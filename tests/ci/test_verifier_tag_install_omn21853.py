# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21853: the pre-push leg installs its verifier from a dev-merge tag, as the reusable does.

The hosted receipt-gate reusable fetches the omnimarket tag, reads its commit back, refuses a
tag whose commit does not start with the tag's own suffix, and runs
``uv sync --frozen --no-dev --no-editable`` from the tag's own ``uv.lock``. These tests run
``install_verifier`` against a throwaway local remote, so no GitHub access is needed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from scripts.ci import check_deploy_scope_dod as hook

pytestmark = pytest.mark.unit

_PYPROJECT = """\
[project]
name = "fake-verifier"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = []
"""


def _run(argv: list[str], cwd: Path) -> str:
    return subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def _remote_with_commit(tmp_path: Path) -> tuple[Path, str]:
    """A local git remote holding one commit with a lockable project, and that commit."""
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    _run(["git", "init", "-q", "-b", "dev"], remote)
    _run(["git", "config", "user.email", "lane@example.invalid"], remote)
    _run(["git", "config", "user.name", "lane"], remote)
    _run(["git", "config", "commit.gpgsign", "false"], remote)
    _run(["uv", "lock", "--no-config"], remote)
    _run(["git", "add", "pyproject.toml", "uv.lock"], remote)
    _run(["git", "commit", "-q", "-m", "chore: fake verifier"], remote)
    return remote, _run(["git", "rev-parse", "HEAD"], remote)


@pytest.fixture
def remote_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Point the install script's GitHub URL for a fake repository at a local remote."""
    remote, _ = _remote_with_commit(tmp_path)
    monkeypatch.setattr(hook, "_VERIFIER_REPO", "OmniNode-ai/fake-verifier")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    gitconfig = tmp_path / "xdg" / "git" / "config"
    gitconfig.parent.mkdir(parents=True)
    gitconfig.write_text(
        f'[url "{remote}"]\n\tinsteadOf = https://github.com/OmniNode-ai/fake-verifier.git\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return str(remote)


def _tag(remote: str, name: str) -> None:
    _run(["git", "tag", name], Path(remote))


def test_a_tag_is_installed_from_its_own_lock_and_the_install_is_reused(
    remote_url: str, tmp_path: Path
) -> None:
    commit = _run(["git", "rev-parse", "HEAD"], Path(remote_url))
    tag = f"dev-20261101000200-{commit[:12]}"
    _tag(remote_url, tag)
    cache = tmp_path / "cache" / "onex-dod-verifier"

    python = hook.install_verifier(tag)

    assert python == cache / tag / "venv" / "bin" / "python"
    assert python.is_file()
    assert (cache / tag / "installed").read_text(encoding="utf-8") == (
        f"tag={tag} commit={commit}\n"
    )
    assert _run([str(python), "-I", "-c", "print('ran')"], tmp_path) == "ran"

    (cache / tag / "src" / "pyproject.toml").unlink()
    assert hook.install_verifier(tag) == python, "an installed tag is reused"


def test_a_tag_moved_to_another_commit_is_refused_and_nothing_is_installed(
    remote_url: str, tmp_path: Path
) -> None:
    tag = "dev-20261101000200-000000000000"
    _tag(remote_url, tag)
    cache = tmp_path / "cache" / "onex-dod-verifier"

    with pytest.raises(hook.DeployScopeHookError, match="was moved, refusing it"):
        hook.install_verifier(tag)

    assert not (cache / tag / "installed").exists()
    assert not (cache / tag / "venv").exists()


def test_an_absent_tag_fails_loud(remote_url: str, tmp_path: Path) -> None:
    with pytest.raises(hook.DeployScopeHookError, match="failed"):
        hook.install_verifier("dev-20261101000200-ffffffffffff")


def test_the_caller_tag_is_read_and_a_missing_or_malformed_one_is_refused(
    tmp_path: Path,
) -> None:
    caller = tmp_path / ".github" / "workflows" / "call-repo-evidence-gate.yml"
    caller.parent.mkdir(parents=True)

    caller.write_text(
        "jobs:\n  repo-evidence:\n    with:\n      verifier-tag: dev-20261101000200-949993edf3b7\n",
        encoding="utf-8",
    )
    assert hook.read_verifier_tag(tmp_path) == "dev-20261101000200-949993edf3b7"

    for with_block in ("      verifier-tag: 0.4.320\n", "      shadow: 'false'\n"):
        caller.write_text(
            f"jobs:\n  repo-evidence:\n    with:\n{with_block}", encoding="utf-8"
        )
        with pytest.raises(hook.DeployScopeHookError):
            hook.read_verifier_tag(tmp_path)
