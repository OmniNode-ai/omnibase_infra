# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TDD-first: preflight() and git_pull() must raise RuntimeError on non-zero returncodes.

Without returncode checks, the deploy-agent silently proceeds past broken environments
and failed git resets, undermining the OMN-8489 cache-bust fix.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from deploy_agent.events import Phase, PhaseStatus
from deploy_agent.executor import DeployExecutor


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


def test_preflight_git_ls_remote_fails() -> None:
    """preflight() must raise RuntimeError when git ls-remote exits non-zero (dod-001)."""
    executor = DeployExecutor()
    call_count = 0

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        nonlocal call_count
        call_count += 1
        if "ls-remote" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=2, stdout="", stderr="fatal: unable to connect"
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        import pytest

        with pytest.raises(RuntimeError) as excinfo:
            executor.preflight(on_phase_update=_noop_phase_update)
        assert "Git remote unreachable" in str(excinfo.value)
        assert "fatal: unable to connect" in str(excinfo.value)


def test_preflight_docker_info_fails() -> None:
    """preflight() must raise RuntimeError when docker info exits non-zero (dod-002)."""
    executor = DeployExecutor()

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        if "ls-remote" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )
        if "docker" in cmd and "info" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="Cannot connect to Docker daemon",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        import pytest

        with pytest.raises(RuntimeError) as excinfo:
            executor.preflight(on_phase_update=_noop_phase_update)
        assert "Docker unavailable" in str(excinfo.value)
        assert "Cannot connect to Docker daemon" in str(excinfo.value)


def test_git_pull_reset_hard_fails() -> None:
    """git_pull() must raise RuntimeError when git reset --hard exits non-zero (dod-003)."""
    executor = DeployExecutor()

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        if "fetch" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )
        if "reset" in cmd and "--hard" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=128,
                stdout="",
                stderr="fatal: ambiguous argument 'refs/heads/nonexistent-xyz'",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        import pytest

        with pytest.raises(RuntimeError) as excinfo:
            executor.git_pull(
                "refs/heads/nonexistent-xyz", on_phase_update=_noop_phase_update
            )
        assert "Git reset --hard" in str(excinfo.value)
        assert "fatal: ambiguous argument" in str(excinfo.value)


def test_happy_path() -> None:
    """Happy path: preflight succeeds and git_pull returns correct SHA (dod-004)."""
    executor = DeployExecutor()
    sentinel_sha = "abc1234def56"

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=sentinel_sha, stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        executor.preflight(on_phase_update=_noop_phase_update)
        sha = executor.git_pull("origin/main", on_phase_update=_noop_phase_update)

    assert sha == sentinel_sha
