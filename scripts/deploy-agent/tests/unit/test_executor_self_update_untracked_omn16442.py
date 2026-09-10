# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16442: the self-update dirty gate must ignore UNTRACKED files.

The gate used to run a bare ``git status --porcelain`` on the agent's own code
clone and skip the update on ANY output. ``git status`` reports the whole
repository regardless of the ``-C`` subdirectory, and the deploy path dropped an
untracked byproduct (``workspace/deploy-source-refs.json``) into that clone's
root. One untracked file therefore made the clone read dirty forever, the agent
skipped every self-update, and the merged OMN-16442 tracking-ref fix could never
reach the running agent -- the exact failure the method exists to prevent.

These tests drive REAL git repositories rather than a mocked ``_run``: the whole
claim is about what ``git status --untracked-files=no`` reports, which a mock
asserts nothing about. Only ``uv sync`` and ``os.execv`` are stubbed on the pull
path, so every git decision in the method is genuine.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import DeployExecutor
from deploy_agent.executor import _run as real_run

TRACKING_BRANCH = "dev"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_clone(tmp_path: Path, *, extra_commit: bool) -> Path:
    """Build an origin repo on ``dev`` plus a clone that tracks it.

    ``extra_commit=True`` advances origin one commit past the clone, so the
    clone is genuinely BEHIND ``origin/dev``.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "--initial-branch", TRACKING_BRANCH)
    _git(origin, "config", "user.email", "agent@example.invalid")
    _git(origin, "config", "user.name", "deploy agent test")
    (origin / "agent.py").write_text("print('v1')\n", encoding="utf-8")
    _git(origin, "add", "agent.py")
    _git(origin, "commit", "-m", "v1")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone))
    _git(clone, "config", "user.email", "agent@example.invalid")
    _git(clone, "config", "user.name", "deploy agent test")

    if extra_commit:
        (origin / "agent.py").write_text("print('v2')\n", encoding="utf-8")
        _git(origin, "add", "agent.py")
        _git(origin, "commit", "-m", "v2")

    return clone


def _drop_untracked_byproduct(clone: Path) -> Path:
    """Recreate the exact byproduct that jammed the gate on the .201 clone."""
    workspace = clone / "workspace"
    workspace.mkdir()
    byproduct = workspace / "deploy-source-refs.json"
    byproduct.write_text('{"ref_pinned": true}\n', encoding="utf-8")
    assert _git(clone, "status", "--porcelain") != ""
    assert _git(clone, "status", "--porcelain", "--untracked-files=no") == ""
    return byproduct


@pytest.mark.unit
def test_untracked_byproduct_does_not_block_and_agent_reports_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An untracked-only clone at origin/dev proceeds and logs 'already at'."""
    clone = _make_clone(tmp_path, extra_commit=False)
    byproduct = _drop_untracked_byproduct(clone)
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))

    executor = DeployExecutor()
    with caplog.at_level("INFO"), patch("os.execv") as mock_execv:
        executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

    assert f"already at origin/{TRACKING_BRANCH}" in caplog.text
    assert "dirty" not in caplog.text
    assert "tracked modifications" not in caplog.text
    mock_execv.assert_not_called()
    # The gate never touches the byproduct; removing it is not this fix.
    assert byproduct.exists()


@pytest.mark.unit
def test_untracked_byproduct_does_not_block_the_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Behind + untracked-only: the agent really pulls and re-execs."""
    clone = _make_clone(tmp_path, extra_commit=True)
    _drop_untracked_byproduct(clone)
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
    before = _git(clone, "rev-parse", "HEAD")

    def _run_git_for_real(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        if cmd[0] == "uv":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )
        return real_run(cmd, timeout, **kwargs)

    executor = DeployExecutor()
    with (
        caplog.at_level("INFO"),
        patch("deploy_agent.executor._run", side_effect=_run_git_for_real),
        patch("os.execv") as mock_execv,
    ):
        executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

    assert f"behind origin/{TRACKING_BRANCH}" in caplog.text
    after = _git(clone, "rev-parse", "HEAD")
    assert after != before
    assert after == _git(clone, "rev-parse", f"origin/{TRACKING_BRANCH}")
    mock_execv.assert_called_once()


@pytest.mark.unit
def test_tracked_modification_still_skips_the_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The narrowing must not weaken the rail it was narrowed from."""
    clone = _make_clone(tmp_path, extra_commit=True)
    (clone / "agent.py").write_text("print('local edit')\n", encoding="utf-8")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
    before = _git(clone, "rev-parse", "HEAD")

    executor = DeployExecutor()
    with caplog.at_level("INFO"), patch("os.execv") as mock_execv:
        executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

    assert "tracked modifications" in caplog.text
    # The offending path is named, not just counted.
    assert "agent.py" in caplog.text
    assert _git(clone, "rev-parse", "HEAD") == before
    mock_execv.assert_not_called()


@pytest.mark.unit
def test_staged_addition_still_skips_the_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A staged new file is tracked work a pull could disturb: still a skip."""
    clone = _make_clone(tmp_path, extra_commit=True)
    (clone / "new_module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(clone, "add", "new_module.py")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
    before = _git(clone, "rev-parse", "HEAD")

    executor = DeployExecutor()
    with caplog.at_level("INFO"), patch("os.execv") as mock_execv:
        executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

    assert "tracked modifications" in caplog.text
    assert _git(clone, "rev-parse", "HEAD") == before
    mock_execv.assert_not_called()
