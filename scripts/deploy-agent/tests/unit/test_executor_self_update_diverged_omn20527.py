# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20527 AC1 and AC2: a clone that cannot fast-forward is rebuilt from origin.

Measured on .200 from 2026-09-25 to 2026-10-04: the dev-200 deploy agent's own
clone sat on a hand-made ``lab-200-proof`` branch nine commits off
``origin/dev``. ``self_update`` ran ``git pull --ff-only origin dev`` on every
idle heartbeat, git answered "Diverging branches can't be fast-forwarded" 2578
times, and the agent ran nine-day-old code that no fix to itself could reach.

These tests drive real git repositories. Only ``uv sync`` and ``os.execv`` are
stubbed, the same seams the OMN-16442 tests stub.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import (
    SELF_UPDATE_PRESERVED_REF_PREFIX,
    DeployExecutor,
)
from deploy_agent.executor import _run as real_run
from deploy_agent.loaded_code import record_loaded_code_sha

pytestmark = pytest.mark.unit

TRACKING_BRANCH = "dev"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _commit(repo: Path, text: str) -> str:
    (repo / "agent.py").write_text(text, encoding="utf-8")
    _git(repo, "add", "agent.py")
    _git(repo, "commit", "-q", "-m", text)
    return _git(repo, "rev-parse", "HEAD")


def _origin_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "--initial-branch", TRACKING_BRANCH)
    for repo in (origin,):
        _git(repo, "config", "user.email", "agent@example.invalid")
        _git(repo, "config", "user.name", "deploy agent test")
    _commit(origin, "print('v1')\n")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "agent@example.invalid")
    _git(clone, "config", "user.name", "deploy agent test")
    return origin, clone


def _git_for_real(
    cmd: list[str], timeout: int, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    if cmd[0] == "uv":
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    return real_run(cmd, timeout, **kwargs)


def _self_update(clone: Path) -> MagicMock:
    executor = DeployExecutor()
    with (
        patch("deploy_agent.executor._run", side_effect=_git_for_real),
        patch("os.execv") as mock_execv,
    ):
        executor.self_update(boundary=EnumSelfUpdateBoundary.IDLE_HEARTBEAT)
    return mock_execv


def _preserved_refs(clone: Path) -> list[str]:
    out = _git(
        clone,
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        SELF_UPDATE_PRESERVED_REF_PREFIX,
    )
    return [line for line in out.splitlines() if line]


def test_a_diverged_tracking_branch_is_rebuilt_and_its_commits_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    origin, clone = _origin_and_clone(tmp_path)
    local_only = _commit(clone, "print('local only')\n")
    remote_tip = _commit(origin, "print('v2')\n")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
    record_loaded_code_sha(str(clone))

    with caplog.at_level("INFO"):
        mock_execv = _self_update(clone)

    assert _git(clone, "rev-parse", "HEAD") == remote_tip
    assert _git(clone, "symbolic-ref", "--short", "HEAD") == TRACKING_BRANCH
    preserved = _preserved_refs(clone)
    assert len(preserved) == 1 and preserved[0].endswith(local_only), preserved
    assert "self_update_diverged_clone" in caplog.text
    mock_execv.assert_called_once()


def test_a_clone_parked_on_another_branch_returns_to_the_tracking_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The .200 shape: a side branch carrying commits, the tracking ref behind."""
    origin, clone = _origin_and_clone(tmp_path)
    _git(clone, "checkout", "-q", "-b", "lab-200-proof")
    side_tip = _commit(clone, "print('lab proof')\n")
    remote_tip = _commit(origin, "print('v2')\n")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
    record_loaded_code_sha(str(clone))

    mock_execv = _self_update(clone)

    assert _git(clone, "symbolic-ref", "--short", "HEAD") == TRACKING_BRANCH
    assert _git(clone, "rev-parse", "HEAD") == remote_tip
    # The side branch is never deleted, and its commits are pinned as well.
    assert _git(clone, "rev-parse", "lab-200-proof") == side_tip
    assert any(ref.endswith(side_tip) for ref in _preserved_refs(clone))
    mock_execv.assert_called_once()


def test_a_side_branch_with_nothing_unique_is_rebuilt_without_a_preserved_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin, clone = _origin_and_clone(tmp_path)
    _git(clone, "checkout", "-q", "-b", "parked")
    remote_tip = _commit(origin, "print('v2')\n")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
    record_loaded_code_sha(str(clone))

    _self_update(clone)

    assert _git(clone, "rev-parse", "HEAD") == remote_tip
    assert _git(clone, "symbolic-ref", "--short", "HEAD") == TRACKING_BRANCH
    assert _preserved_refs(clone) == []


def test_a_tracked_modification_still_blocks_a_diverged_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """AC2: the rebuild never runs past the dirty-check, so no edit is lost."""
    origin, clone = _origin_and_clone(tmp_path)
    local_only = _commit(clone, "print('local only')\n")
    _commit(origin, "print('v2')\n")
    (clone / "agent.py").write_text("print('uncommitted edit')\n", encoding="utf-8")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
    record_loaded_code_sha(str(clone))

    with caplog.at_level("INFO"):
        mock_execv = _self_update(clone)

    assert "tracked modifications" in caplog.text
    assert _git(clone, "rev-parse", "HEAD") == local_only
    assert (clone / "agent.py").read_text(
        encoding="utf-8"
    ) == "print('uncommitted edit')\n"
    assert _preserved_refs(clone) == []
    mock_execv.assert_not_called()
