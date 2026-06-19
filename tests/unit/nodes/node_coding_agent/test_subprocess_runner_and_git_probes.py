# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live process-group runner + git-probe coverage (OMN-13247, plan §5.4 / §5.5).

These exercise the REAL seams the EFFECT uses in production — but with benign
commands (``true`` / ``sh -c`` / ``cat``) and a real temp git repo. No
claude/codex binary is ever executed (the argv is a harmless shell, not an
agent). This proves: the runner spawns in its own process group, a timeout kills
the whole group, stdin is piped, and files_changed/diff are git-derived (tracked
edits + untracked files), never parsed from stdout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from omnibase_infra.models.coding_agent.model_subprocess_invocation import (
    ModelSubprocessInvocation,
)
from omnibase_infra.nodes.node_coding_agent_invoke_effect.handlers.handler_coding_agent_invoke import (
    _git_capture_diff,
    _git_head_sha,
    _run_subprocess_pgroup,
)

pytestmark = pytest.mark.unit

_HAS_GIT = shutil.which("git") is not None
_HAS_SH = shutil.which("sh") is not None

# The explicit child env the runner now requires (a copy of the parent env). The
# real handler overrides HOME with the contract-resolved credential home; these
# runner-level tests pass the inherited env unchanged (no agent binary is run).
_ENV = dict(os.environ)


def _invocation(
    argv: list[str],
    cwd: str,
    timeout_s: int,
    stdin: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ModelSubprocessInvocation:
    return ModelSubprocessInvocation(
        argv=argv,
        cwd=cwd,
        timeout_s=timeout_s,
        network=False,
        stdin=stdin,
        env=env if env is not None else _ENV,
    )


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)


@pytest.mark.skipif(not _HAS_SH, reason="requires a POSIX shell")
class TestProcessGroupRunner:
    def test_runner_captures_stdout_and_exit(self, tmp_path: Path) -> None:
        outcome = _run_subprocess_pgroup(
            _invocation(["sh", "-c", "printf done"], str(tmp_path), 10)
        )
        assert outcome.returncode == 0
        assert outcome.stdout == "done"
        assert outcome.timed_out is False

    def test_runner_pipes_stdin(self, tmp_path: Path) -> None:
        # Proves the prompt-via-stdin path the claude write mode relies on.
        outcome = _run_subprocess_pgroup(
            _invocation(["cat"], str(tmp_path), 10, stdin="piped-prompt")
        )
        assert outcome.returncode == 0
        assert outcome.stdout == "piped-prompt"

    def test_runner_nonzero_exit(self, tmp_path: Path) -> None:
        outcome = _run_subprocess_pgroup(
            _invocation(["sh", "-c", "exit 3"], str(tmp_path), 10)
        )
        assert outcome.returncode == 3
        assert outcome.timed_out is False

    def test_timeout_kills_group(self, tmp_path: Path) -> None:
        # A child that spawns a long-lived grandchild and sleeps; on timeout the
        # whole process group is killed (SIGTERM->SIGKILL). The call returns
        # promptly with timed_out=True rather than hanging the full sleep.
        outcome = _run_subprocess_pgroup(
            _invocation(["sh", "-c", "sleep 30 & sleep 30"], str(tmp_path), 1)
        )
        assert outcome.timed_out is True

    def test_runner_uses_explicit_env_home(self, tmp_path: Path) -> None:
        # The runner passes the explicit env verbatim to the child: the subprocess
        # sees HOME=<cred home>, NOT the parent process's HOME. This is the
        # root-HOME cred-resolution fix (OMN-13247 Phase B): the child reads its
        # ambient OAuth creds from <cred home> even when the runtime runs as root.
        cred_home = str(tmp_path / "creds")
        child_env = {**os.environ, "HOME": cred_home}
        outcome = _run_subprocess_pgroup(
            _invocation(
                ["sh", "-c", 'printf %s "$HOME"'], str(tmp_path), 10, env=child_env
            )
        )
        assert outcome.returncode == 0
        assert outcome.stdout == cred_home


@pytest.mark.skipif(not _HAS_GIT, reason="requires git")
class TestGitProbes:
    def test_head_sha_none_outside_repo(self, tmp_path: Path) -> None:
        assert _git_head_sha(str(tmp_path)) is None

    def test_head_sha_in_repo(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        (tmp_path / "a.txt").write_text("hello\n")
        subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
        sha = _git_head_sha(str(tmp_path))
        assert sha is not None
        assert len(sha) == 40

    def test_capture_diff_tracked_edit(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        (tmp_path / "a.txt").write_text("hello\n")
        subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
        # Simulate an agent edit to a tracked file.
        (tmp_path / "a.txt").write_text("hello world\n")
        files_changed, diff = _git_capture_diff(str(tmp_path))
        assert files_changed == ("a.txt",)
        assert "hello world" in diff

    def test_capture_diff_includes_untracked(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        (tmp_path / "seed.txt").write_text("seed\n")
        subprocess.run(["git", "add", "seed.txt"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
        # Simulate an agent creating a NEW (untracked) file.
        (tmp_path / "new.py").write_text("x = 1\n")
        files_changed, _diff = _git_capture_diff(str(tmp_path))
        assert "new.py" in files_changed
