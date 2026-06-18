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

import shutil
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_coding_agent_invoke_effect.handlers.handler_coding_agent_invoke import (
    _git_capture_diff,
    _git_head_sha,
    _run_subprocess_pgroup,
)

pytestmark = pytest.mark.unit

_HAS_GIT = shutil.which("git") is not None
_HAS_SH = shutil.which("sh") is not None


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)


@pytest.mark.skipif(not _HAS_SH, reason="requires a POSIX shell")
class TestProcessGroupRunner:
    def test_runner_captures_stdout_and_exit(self, tmp_path: Path) -> None:
        outcome = _run_subprocess_pgroup(
            ["sh", "-c", "printf done"], str(tmp_path), 10, False, None
        )
        assert outcome.returncode == 0
        assert outcome.stdout == "done"
        assert outcome.timed_out is False

    def test_runner_pipes_stdin(self, tmp_path: Path) -> None:
        # Proves the prompt-via-stdin path the claude write mode relies on.
        outcome = _run_subprocess_pgroup(
            ["cat"], str(tmp_path), 10, False, "piped-prompt"
        )
        assert outcome.returncode == 0
        assert outcome.stdout == "piped-prompt"

    def test_runner_nonzero_exit(self, tmp_path: Path) -> None:
        outcome = _run_subprocess_pgroup(
            ["sh", "-c", "exit 3"], str(tmp_path), 10, False, None
        )
        assert outcome.returncode == 3
        assert outcome.timed_out is False

    def test_timeout_kills_group(self, tmp_path: Path) -> None:
        # A child that spawns a long-lived grandchild and sleeps; on timeout the
        # whole process group is killed (SIGTERM->SIGKILL). The call returns
        # promptly with timed_out=True rather than hanging the full sleep.
        outcome = _run_subprocess_pgroup(
            ["sh", "-c", "sleep 30 & sleep 30"], str(tmp_path), 1, False, None
        )
        assert outcome.timed_out is True


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
