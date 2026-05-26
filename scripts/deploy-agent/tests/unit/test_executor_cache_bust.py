# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TDD-first: deploy-agent rebuild phase must pass --build-arg GIT_SHA to docker compose build.

Without this, Docker's layer cache silently serves stale COPY src/ layers even when
git is at the correct SHA (root cause of PR #1231 verification failure).

Also covers OMN-10728 / OMN-11542: OMNIBASE_COMPAT_REF, OMNIMARKET_REF, and
ONEX_CHANGE_CONTROL_REF must be passed as full commit SHAs so the uv cache mount
(keyed on URL) misses when main advances.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
from deploy_agent.events import Phase, PhaseStatus, Scope

pytestmark = pytest.mark.unit
from deploy_agent.executor import DeployExecutor


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


def _make_ok_result(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TestCacheBust:
    """_compose_build must pass --build-arg GIT_SHA=<sha> so Docker invalidates COPY layers."""

    def test_compose_build_includes_git_sha_build_arg(self) -> None:
        """_compose_build must call docker compose build with --build-arg GIT_SHA=<sha>."""
        executor = DeployExecutor()
        git_sha = "abc1234def56"

        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return _make_ok_result()

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            executor._compose_build(Scope.RUNTIME, git_sha, _noop_phase_update)

        build_cmds = [c for c in captured_cmds if "build" in c]
        assert build_cmds, "Expected at least one 'docker compose build' call"
        build_cmd = build_cmds[0]

        assert "--build-arg" in build_cmd, (
            "Build command must contain --build-arg to invalidate Docker layer cache"
        )
        git_sha_arg_idx = build_cmd.index("--build-arg") + 1
        git_sha_arg = build_cmd[git_sha_arg_idx]
        assert git_sha_arg == f"GIT_SHA={git_sha}", (
            f"Expected --build-arg GIT_SHA={git_sha}, got {git_sha_arg!r}"
        )

    def test_compose_build_called_before_compose_up_in_rebuild_scope(self) -> None:
        """rebuild_scope must call _compose_build before _compose_up so images are fresh."""
        executor = DeployExecutor()
        git_sha = "abc1234def56"
        call_order: list[str] = []

        def fake_build(scope: Scope, sha: str, cb, **kwargs) -> None:
            call_order.append("build")

        def fake_up(phase: Phase, scope: Scope, services: list[str], cb) -> None:
            call_order.append("up")

        executor._compose_build = fake_build  # type: ignore[method-assign]
        executor._compose_up = fake_up  # type: ignore[method-assign]

        executor.rebuild_scope(Scope.RUNTIME, [], _noop_phase_update, git_sha=git_sha)

        assert call_order == [
            "build",
            "up",
        ], f"_compose_build must precede _compose_up, got order: {call_order}"

    def test_compose_build_called_for_full_scope(self) -> None:
        """Full scope rebuild must call _compose_build for both core and runtime."""
        executor = DeployExecutor()
        git_sha = "abc1234def56"
        build_scopes: list[Scope] = []

        def fake_build(scope: Scope, sha: str, cb, **kwargs) -> None:
            build_scopes.append(scope)

        def fake_up(phase: Phase, scope: Scope, services: list[str], cb) -> None:
            pass

        executor._compose_build = fake_build  # type: ignore[method-assign]
        executor._compose_up = fake_up  # type: ignore[method-assign]

        executor.rebuild_scope(Scope.FULL, [], _noop_phase_update, git_sha=git_sha)

        assert Scope.CORE in build_scopes, "Full scope must build core"
        assert Scope.RUNTIME in build_scopes, "Full scope must build runtime"

    def test_git_sha_forwarded_from_agent_to_executor(self) -> None:
        """The git SHA returned from git_pull must be forwarded to rebuild_scope."""
        executor = DeployExecutor()
        git_sha_seen_in_build: list[str] = []

        def fake_build(scope: Scope, sha: str, cb, **kwargs) -> None:
            git_sha_seen_in_build.append(sha)

        def fake_up(phase: Phase, scope: Scope, services: list[str], cb) -> None:
            pass

        executor._compose_build = fake_build  # type: ignore[method-assign]
        executor._compose_up = fake_up  # type: ignore[method-assign]

        sentinel_sha = "deadbeef1234"
        executor.rebuild_scope(
            Scope.RUNTIME, [], _noop_phase_update, git_sha=sentinel_sha
        )

        assert git_sha_seen_in_build == [sentinel_sha], (
            f"Expected git_sha={sentinel_sha!r} forwarded to _compose_build, "
            f"got {git_sha_seen_in_build}"
        )


class TestUvCacheBustPluginRefs:
    """Builds must pass plugin refs as full SHAs for BuildKit cache invalidation.

    OMN-10728 covered omnimarket and onex_change_control. OMN-11542 adds
    omnibase_compat because runtime evidence DTOs were stale even though the
    installed package version looked current.
    """

    def _fake_run_ok(
        self, cmd: list[str], timeout: int, **kwargs
    ) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    def test_compose_build_passes_omnimarket_ref_build_arg(self) -> None:
        """_compose_build must include --build-arg OMNIMARKET_REF=<sha>."""
        executor = DeployExecutor()
        sentinel_sha = "cafe1234abcd5678"

        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with (
            patch.dict("os.environ", {"OMNI_HOME": "/workspace/omni_home"}),
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.object(
                DeployExecutor, "_resolve_plugin_ref", return_value=sentinel_sha
            ),
        ):
            executor._compose_build(Scope.RUNTIME, "abc123", _noop_phase_update)

        build_cmds = [c for c in captured_cmds if "build" in c]
        assert build_cmds, "Expected at least one 'docker compose build' call"
        build_cmd = build_cmds[0]

        # Collect all --build-arg values from the command
        build_args = {
            build_cmd[i + 1]
            for i, tok in enumerate(build_cmd)
            if tok == "--build-arg" and i + 1 < len(build_cmd)
        }
        assert f"OMNIMARKET_REF={sentinel_sha}" in build_args, (
            f"Expected OMNIMARKET_REF={sentinel_sha!r} in build args; got {build_args}"
        )

    def test_compose_build_passes_omnibase_compat_ref_build_arg(self) -> None:
        """_compose_build must include --build-arg OMNIBASE_COMPAT_REF=<sha>."""
        executor = DeployExecutor()
        sentinel_sha = "face1234abcd5678"

        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with (
            patch.dict("os.environ", {"OMNI_HOME": "/workspace/omni_home"}),
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.object(
                DeployExecutor, "_resolve_plugin_ref", return_value=sentinel_sha
            ),
        ):
            executor._compose_build(Scope.RUNTIME, "abc123", _noop_phase_update)

        build_cmds = [c for c in captured_cmds if "build" in c]
        assert build_cmds, "Expected at least one 'docker compose build' call"
        build_cmd = build_cmds[0]
        build_args = {
            build_cmd[i + 1]
            for i, tok in enumerate(build_cmd)
            if tok == "--build-arg" and i + 1 < len(build_cmd)
        }
        assert f"OMNIBASE_COMPAT_REF={sentinel_sha}" in build_args, (
            f"Expected OMNIBASE_COMPAT_REF={sentinel_sha!r} in build args; got {build_args}"
        )

    def test_compose_build_passes_onex_change_control_ref_build_arg(self) -> None:
        """_compose_build must include --build-arg ONEX_CHANGE_CONTROL_REF=<sha>."""
        executor = DeployExecutor()
        sentinel_sha = "dead0000beef1111"

        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with (
            patch.dict("os.environ", {"OMNI_HOME": "/workspace/omni_home"}),
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.object(
                DeployExecutor, "_resolve_plugin_ref", return_value=sentinel_sha
            ),
        ):
            executor._compose_build(Scope.RUNTIME, "abc123", _noop_phase_update)

        build_cmds = [c for c in captured_cmds if "build" in c]
        assert build_cmds, "Expected at least one 'docker compose build' call"
        build_cmd = build_cmds[0]

        build_args = {
            build_cmd[i + 1]
            for i, tok in enumerate(build_cmd)
            if tok == "--build-arg" and i + 1 < len(build_cmd)
        }
        assert f"ONEX_CHANGE_CONTROL_REF={sentinel_sha}" in build_args, (
            f"Expected ONEX_CHANGE_CONTROL_REF={sentinel_sha!r} in build args; got {build_args}"
        )

    def test_resolve_plugin_ref_returns_sha_from_git(self) -> None:
        """_resolve_plugin_ref must return the SHA from git rev-parse HEAD."""
        expected_sha = "abcdef1234567890abcdef1234567890abcdef12"
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"{expected_sha}\n", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            sha = DeployExecutor._resolve_plugin_ref("/fake/repo")
        assert sha == expected_sha

    def test_resolve_plugin_ref_falls_back_to_main_on_git_failure(self) -> None:
        """_resolve_plugin_ref must return 'main' when git rev-parse fails."""
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=128, stdout="", stderr="fatal: not a git repo"
        )
        with patch("subprocess.run", return_value=mock_result):
            sha = DeployExecutor._resolve_plugin_ref("/nonexistent/repo")
        assert sha == "main"

    def test_compose_build_uses_dev_fallback_for_omnimarket_when_omni_home_unset(
        self,
    ) -> None:
        """When OMNI_HOME is not set, OMNIMARKET_REF defaults to 'dev' (OMN-12195: dev is the default branch)."""
        executor = DeployExecutor()
        captured_cmds: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            captured_cmds.append(cmd)
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        import os

        env_without_omni_home = {
            k: v for k, v in os.environ.items() if k != "OMNI_HOME"
        }

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            patch.dict("os.environ", env_without_omni_home, clear=True),
        ):
            executor._compose_build(Scope.RUNTIME, "abc123", _noop_phase_update)

        build_cmds = [c for c in captured_cmds if "build" in c]
        assert build_cmds
        build_cmd = build_cmds[0]
        build_args = {
            build_cmd[i + 1]
            for i, tok in enumerate(build_cmd)
            if tok == "--build-arg" and i + 1 < len(build_cmd)
        }
        assert "OMNIMARKET_REF=dev" in build_args, (
            f"Expected OMNIMARKET_REF=dev when OMNI_HOME unset (OMN-12195: dev is omnimarket default branch); got {build_args}"
        )
        assert "OMNIBASE_COMPAT_REF=main" in build_args, (
            f"Expected OMNIBASE_COMPAT_REF=main when OMNI_HOME unset; got {build_args}"
        )
        assert "ONEX_CHANGE_CONTROL_REF=main" in build_args, (
            f"Expected ONEX_CHANGE_CONTROL_REF=main when OMNI_HOME unset; got {build_args}"
        )
