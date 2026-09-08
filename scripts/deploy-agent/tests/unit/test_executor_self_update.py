# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for DeployExecutor.self_update().

Verifies: behind/ahead/current detection, os.execv call when behind,
--skip-self-update / kill-switch bypass, dirty-tree safety rail,
and container-mode exit(42) behavior.

The ref self_update compares against is the DECLARED tracking ref
(``DEPLOY_AGENT_TRACKING_REF``, supplied as ``dev`` by the conftest autouse
fixture), not a hardcoded ``origin/main`` -- these tests therefore match any
``origin/*`` rev-parse. That the ref is honoured, is required, and never
resolves ``main`` is asserted in ``test_tracking_ref.py`` (OMN-16442).
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import call, patch

import pytest
from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import DEPLOY_AGENT_DIR, DeployExecutor

# OMN-16442: ``self_update`` takes a required ``boundary`` so every call site
# names the job boundary it fired at. These tests exercise the method's own
# git mechanics, which are identical at either boundary, so they declare one
# and keep it constant. Which boundary each caller actually uses -- and that
# the deploy path uses none -- is asserted in
# ``test_self_update_job_boundary_omn16442.py``.
_BOUNDARY = EnumSelfUpdateBoundary.POST_TERMINAL

SHA_LOCAL = "aaaaaaaabbbbbbbb"
SHA_REMOTE = "ccccccccdddddddd"


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _fail(stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


def _make_git_responses(
    *, dirty: bool = False, local: str = SHA_LOCAL, remote: str = SHA_REMOTE
):
    """Return a side_effect list for _run: status, fetch, rev-parse HEAD, rev-parse origin/<tracking ref>."""

    def side_effect(
        cmd: list[str], timeout: int, **kwargs
    ) -> subprocess.CompletedProcess:
        if "status" in cmd and "--porcelain" in cmd:
            return _ok("M somefile.py" if dirty else "")
        if "fetch" in cmd:
            return _ok()
        if "rev-parse" in cmd:
            if any(part.startswith("origin/") for part in cmd):
                return _ok(remote)
            return _ok(local)
        if "pull" in cmd:
            return _ok()
        if cmd[0] == "uv":
            return _ok()
        return _ok()

    return side_effect


class TestSelfUpdateSkip:
    def test_default_update_source_is_canonical_repo_copy(self) -> None:
        assert DEPLOY_AGENT_DIR == "/data/omninode/omnibase_infra/scripts/deploy-agent"

    def test_skip_flag_bypasses_all_git_calls(self) -> None:
        executor = DeployExecutor()
        with patch("deploy_agent.executor._run") as mock_run:
            executor.self_update(boundary=_BOUNDARY, skip=True)
        mock_run.assert_not_called()

    def test_env_kill_switch_bypasses_all_git_calls(self, monkeypatch) -> None:
        monkeypatch.setenv("DEPLOY_AGENT_NO_SELF_UPDATE", "1")
        executor = DeployExecutor()
        with patch("deploy_agent.executor._run") as mock_run:
            executor.self_update(boundary=_BOUNDARY)
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
            executor.self_update(boundary=_BOUNDARY)
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
            executor.self_update(boundary=_BOUNDARY)
        mock_execv.assert_not_called()


class TestSelfUpdateBehind:
    def test_behind_calls_execv(self) -> None:
        executor = DeployExecutor()
        with (
            patch("deploy_agent.executor._run", side_effect=_make_git_responses()),
            patch("os.execv") as mock_execv,
        ):
            executor.self_update(boundary=_BOUNDARY)
        mock_execv.assert_called_once_with(sys.executable, [sys.executable] + sys.argv)

    def test_behind_container_mode_exits_42(self, monkeypatch) -> None:
        monkeypatch.setenv("DEPLOY_AGENT_MODE", "container")
        executor = DeployExecutor()
        with (
            patch("deploy_agent.executor._run", side_effect=_make_git_responses()),
            patch("sys.exit") as mock_exit,
        ):
            executor.self_update(boundary=_BOUNDARY)
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
            executor.self_update(boundary=_BOUNDARY)
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
                if any(part.startswith("origin/") for part in cmd):
                    return _ok(SHA_REMOTE)
                return _ok(SHA_LOCAL)
            if "pull" in cmd:
                return _fail("conflict")
            return _ok()

        with (
            patch("deploy_agent.executor._run", side_effect=side_effect),
            patch("os.execv") as mock_execv,
        ):
            executor.self_update(boundary=_BOUNDARY)
        mock_execv.assert_not_called()


class TestSelfUpdateIsNotWiredIntoTheDeployPath:
    """The mid-deploy call site is gone, not merely defaulted off (OMN-16442).

    ``rebuild_scope`` used to open with ``self.self_update(...)``. That is
    inside an accepted job, after preflight/git/compose_gen/seed; re-execing
    there aborts the deploy in flight, which is what published command
    8d0c861a-f91e-4ca2-954e-a073759dd39d as failed on 2026-09-08 after the
    replacement process recovered it as a crashed job.

    The behavioural half of this -- which boundary each caller uses, that the
    same command is processed once across a re-exec, and that a deploy running
    while behind still completes -- lives in
    ``test_self_update_job_boundary_omn16442.py``.
    """

    def test_rebuild_scope_does_not_call_self_update(self) -> None:
        from deploy_agent.events import Scope

        executor = DeployExecutor()
        calls: list[object] = []

        def fake_self_update(**kwargs: object) -> None:
            calls.append(kwargs)

        executor.self_update = fake_self_update  # type: ignore[method-assign]
        executor._compose_build = lambda *a, **k: None  # type: ignore[method-assign]
        executor._compose_up = lambda *a, **k: None  # type: ignore[method-assign]

        executor.rebuild_scope(Scope.RUNTIME, [], lambda p, s: None)

        assert calls == [], f"rebuild_scope must not self-update, got {calls}"

    def test_rebuild_scope_signature_carries_no_self_update_switch(self) -> None:
        import inspect

        assert (
            "skip_self_update"
            not in inspect.signature(DeployExecutor.rebuild_scope).parameters
        )
