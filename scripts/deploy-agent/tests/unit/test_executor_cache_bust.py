# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TDD-first: deploy-agent rebuild phase must pass --build-arg GIT_SHA to docker compose build.

Without this, Docker's layer cache silently serves stale COPY src/ layers even when
git is at the correct SHA (root cause of PR #1231 verification failure).
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from deploy_agent.events import BuildSource, Phase, PhaseStatus, Scope
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
            executor._compose_build(
                Scope.RUNTIME,
                git_sha,
                BuildSource.RELEASE,
                _noop_phase_update,
            )

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
        assert "BUILD_SOURCE=release" in build_cmd
        assert "RELEASE_MANIFEST_PATH=docker/runtime-release-manifest.json" in build_cmd

    def test_compose_build_called_before_compose_up_in_rebuild_scope(self) -> None:
        """rebuild_scope must call _compose_build before _compose_up so images are fresh."""
        executor = DeployExecutor()
        git_sha = "abc1234def56"
        call_order: list[str] = []

        def fake_build(scope: Scope, sha: str, build_source: BuildSource, cb) -> None:
            call_order.append("build")

        def fake_up(phase: Phase, scope: Scope, services: list[str], cb) -> None:
            call_order.append("up")

        executor._compose_build = fake_build  # type: ignore[method-assign]
        executor._compose_up = fake_up  # type: ignore[method-assign]

        executor.rebuild_scope(Scope.RUNTIME, [], _noop_phase_update, git_sha=git_sha)

        assert call_order == ["build", "up"], (
            f"_compose_build must precede _compose_up, got order: {call_order}"
        )

    def test_compose_build_called_for_full_scope(self) -> None:
        """Full scope rebuild must call _compose_build for both core and runtime."""
        executor = DeployExecutor()
        git_sha = "abc1234def56"
        build_scopes: list[Scope] = []

        def fake_build(scope: Scope, sha: str, build_source: BuildSource, cb) -> None:
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

        def fake_build(scope: Scope, sha: str, build_source: BuildSource, cb) -> None:
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
