# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prod promotion-lineage enforcement in deploy-agent builds (OMN-12626, R1).

Release-mode builds produce the digest later pinned/promoted to prod. The
deploy-agent must refuse to run a release-mode docker build from a dirty or
non-promoted (dev-only) source tree. Workspace builds (local dev iteration) are
exempt by design.

These tests stub the guard module loaded from scripts/ so the deploy-agent
behavior is verified without constructing a real git repo here (the guard's own
behavior is covered by tests/scripts/test_check_prod_promotion_lineage.py).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import BuildSource, Phase, PhaseStatus, Scope
from deploy_agent.executor import DeployExecutor

pytestmark = [pytest.mark.unit, pytest.mark.promotion_guard]


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


def _ok() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


class _GuardStub:
    """Stand-in for the scripts/ guard module."""

    class ProdLineageError(RuntimeError):
        pass

    def __init__(self, *, raises: bool) -> None:
        self.calls: list[Path] = []
        self._raises = raises

    def assert_prod_build_promoted(self, repo_dir: Path) -> str:
        self.calls.append(Path(repo_dir))
        if self._raises:
            raise self.ProdLineageError("dirty or non-promoted source tree")
        return "0123456789abcdef0123456789abcdef01234567"


def test_release_build_invokes_promotion_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _GuardStub(raises=False)
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", lambda: guard)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        captured.append(cmd)
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)
    monkeypatch.delenv("OMNI_HOME", raising=False)

    executor = DeployExecutor()
    executor._compose_build(
        Scope.RUNTIME,
        "abc1234",
        _noop_phase_update,
        build_source=BuildSource.RELEASE,
    )

    # Guard ran on the build source repo, and the build proceeded.
    assert guard.calls == [Path(executor_mod.REPO_DIR)]
    assert any("build" in cmd for cmd in captured)


def test_release_build_fails_closed_when_guard_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _GuardStub(raises=True)
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", lambda: guard)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)
    monkeypatch.delenv("OMNI_HOME", raising=False)

    executor = DeployExecutor()
    with pytest.raises(_GuardStub.ProdLineageError):
        executor._compose_build(
            Scope.RUNTIME,
            "abc1234",
            _noop_phase_update,
            build_source=BuildSource.RELEASE,
        )

    # No docker build ran — the guard fired before any side effect.
    assert calls == []
    assert guard.calls == [Path(executor_mod.REPO_DIR)]


def test_workspace_build_skips_promotion_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _GuardStub(raises=True)  # would fail if invoked
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", lambda: guard)

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)
    monkeypatch.setattr(
        DeployExecutor, "_stage_workspace", staticmethod(lambda *_: None)
    )
    monkeypatch.setenv("OMNI_HOME", "/data/omninode/omni_home")

    executor = DeployExecutor()
    # Should NOT raise — workspace builds are exempt and never touch the guard.
    executor._compose_build(
        Scope.RUNTIME,
        "abc1234",
        _noop_phase_update,
        build_source=BuildSource.WORKSPACE,
    )
    assert guard.calls == []


def test_assert_release_build_promoted_is_noop_for_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _GuardStub(raises=True)
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", lambda: guard)
    # No raise, no guard call for workspace.
    executor_mod.assert_release_build_promoted(BuildSource.WORKSPACE)
    assert guard.calls == []
