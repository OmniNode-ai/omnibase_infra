# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for DeployExecutor.self_update().

Verifies: behind/ahead/current detection, os.execv call when behind,
--skip-self-update / kill-switch bypass, dirty-tree safety rail,
and container-mode exit(42) behavior.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import call, patch

import pytest
from deploy_agent.executor import DeployExecutor

SHA_LOCAL = "aaaaaaaabbbbbbbb"
SHA_REMOTE = "ccccccccdddddddd"


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _fail(stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


def _make_git_responses(
    *, dirty: bool = False, local: str = SHA_LOCAL, remote: str = SHA_REMOTE
):
    """Return a side_effect list for _run: status, fetch, rev-parse HEAD, rev-parse origin/main."""

    def side_effect(
        cmd: list[str], timeout: int, **kwargs
    ) -> subprocess.CompletedProcess:
        if "status" in cmd and "--porcelain" in cmd:
            return _ok("M somefile.py" if dirty else "")
        if "fetch" in cmd:
            return _ok()
        if "rev-parse" in cmd:
            if "origin/main" in cmd:
                return _ok(remote)
            return _ok(local)
        if "pull" in cmd:
            return _ok()
        if cmd[0] == "uv":
            return _ok()
        return _ok()

    return side_effect


class TestSelfUpdateSkip:
    def test_skip_flag_bypasses_all_git_calls(self) -> None:
        executor = DeployExecutor()
        with patch("deploy_agent.executor._run") as mock_run:
            executor.self_update(skip=True)
        mock_run.assert_not_called()

    def test_env_kill_switch_bypasses_all_git_calls(self, monkeypatch) -> None:
        monkeypatch.setenv("DEPLOY_AGENT_NO_SELF_UPDATE", "1")
        executor = DeployExecutor()
        with patch("deploy_agent.executor._run") as mock_run:
            executor.self_update()
        mock_run.assert_not_called()


class TestSelfUpdateDirtyTree:
    def test_dirty_tree_skips_update_without_execv(self) -> None:
        executor = DeployExecutor()
        with (
            patch(
                "deploy_agent.executor._run",
                side_effect=_make_git_responses(dirty=True),
            ),
            patch("os.execv") as mock_execv,
        ):
            executor.self_update()
        mock_execv.assert_not_called()


class TestSelfUpdateCurrent:
    def test_already_current_does_not_execv(self) -> None:
        executor = DeployExecutor()
        with (
            patch(
                "deploy_agent.executor._run",
                side_effect=_make_git_responses(local=SHA_LOCAL, remote=SHA_LOCAL),
            ),
            patch("os.execv") as mock_execv,
        ):
            executor.self_update()
        mock_execv.assert_not_called()


class TestSelfUpdateBehind:
    def test_behind_calls_execv(self) -> None:
        executor = DeployExecutor()
        with (
            patch("deploy_agent.executor._run", side_effect=_make_git_responses()),
            patch("os.execv") as mock_execv,
        ):
            executor.self_update()
        mock_execv.assert_called_once_with(sys.executable, [sys.executable] + sys.argv)

    def test_behind_container_mode_exits_42(self, monkeypatch) -> None:
        monkeypatch.setenv("DEPLOY_AGENT_MODE", "container")
        executor = DeployExecutor()
        with (
            patch("deploy_agent.executor._run", side_effect=_make_git_responses()),
            patch("sys.exit") as mock_exit,
        ):
            executor.self_update()
        mock_exit.assert_called_once_with(42)

    def test_behind_fetch_failure_skips_execv(self) -> None:
        executor = DeployExecutor()

        def side_effect(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            if "status" in cmd and "--porcelain" in cmd:
                return _ok()
            if "fetch" in cmd:
                return _fail("network error")
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=side_effect),
            patch("os.execv") as mock_execv,
        ):
            executor.self_update()
        mock_execv.assert_not_called()

    def test_behind_pull_failure_skips_execv(self) -> None:
        executor = DeployExecutor()

        def side_effect(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            if "status" in cmd and "--porcelain" in cmd:
                return _ok()
            if "fetch" in cmd:
                return _ok()
            if "rev-parse" in cmd:
                if "origin/main" in cmd:
                    return _ok(SHA_REMOTE)
                return _ok(SHA_LOCAL)
            if "pull" in cmd:
                return _fail("conflict")
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=side_effect),
            patch("os.execv") as mock_execv,
        ):
            executor.self_update()
        mock_execv.assert_not_called()


class TestSelfUpdateWiredIntoRebuildScope:
    def test_self_update_called_first_in_rebuild_scope(self) -> None:
        """self_update must be invoked before _compose_build inside rebuild_scope."""
        from deploy_agent.events import Phase, PhaseStatus, Scope

        executor = DeployExecutor()
        call_order: list[str] = []

        def fake_self_update(*, skip: bool = False) -> None:
            call_order.append("self_update")

        def fake_build(scope: Scope, sha: str, cb) -> None:
            call_order.append("build")

        def fake_up(phase: Phase, scope: Scope, services: list[str], cb) -> None:
            call_order.append("up")

        executor.self_update = fake_self_update  # type: ignore[method-assign]
        executor._compose_build = fake_build  # type: ignore[method-assign]
        executor._compose_up = fake_up  # type: ignore[method-assign]

        executor.rebuild_scope(Scope.RUNTIME, [], lambda p, s: None)

        assert call_order[0] == "self_update", (
            f"self_update must be first, got order: {call_order}"
        )

    def test_skip_self_update_flag_forwarded_from_rebuild_scope(self) -> None:
        from deploy_agent.events import Scope

        executor = DeployExecutor()
        received_skip: list[bool] = []

        def fake_self_update(*, skip: bool = False) -> None:
            received_skip.append(skip)

        def fake_build(scope, sha, cb) -> None:
            pass

        def fake_up(phase, scope, services, cb) -> None:
            pass

        executor.self_update = fake_self_update  # type: ignore[method-assign]
        executor._compose_build = fake_build  # type: ignore[method-assign]
        executor._compose_up = fake_up  # type: ignore[method-assign]

        executor.rebuild_scope(
            Scope.RUNTIME, [], lambda p, s: None, skip_self_update=True
        )

        assert received_skip == [True]
