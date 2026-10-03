# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Staging stays protected through the last build; errors retain stdout."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.build_budget import EnumBuildOutcome
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import DeployExecutor
from deploy_agent.host_conditions import EnumBuildCacheState, probe_host_conditions

pytestmark = pytest.mark.unit

_LOCK_DIRNAME = ".onex-reconcile-host.lock"
_SHA = "a" * 40
_ERROR_BLOCK = (
    "------\n"
    " > [runtime builder 9/9] RUN compute_workspace_provenance.py:\n"
    "12.3 Workspace provenance FAILED:\n"
    "12.3 - Per-repo VCS provenance has no sibling entries\n"
    "------"
)


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


@pytest.mark.parametrize("scope", [Scope.FULL, Scope.RUNTIME, Scope.CORE])
def test_lock_spans_every_build_and_ends_before_compose_up(
    scope: Scope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNI_HOME", str(tmp_path))
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(tmp_path / "omnibase_infra"))
    lock = tmp_path / _LOCK_DIRNAME
    calls: list[str] = []

    def building(name: str) -> None:
        assert lock.is_dir()
        holder = (lock / "holder").read_text()
        assert f"pid={os.getpid()}\n" in holder
        assert f"host={socket.gethostname()}\n" in holder
        assert f"purpose=deploy-agent image build {_SHA[:12]}\n" in holder
        # Releasing and retaking between calls would erase this staged marker.
        marker = lock / "staged"
        if calls:
            assert marker.read_text() == "staged"
        else:
            marker.write_text("staged")
        calls.append(name)

    def compose_build(self: DeployExecutor, scope: Scope, *a: Any, **kw: Any) -> None:
        building(scope.value)

    def dev_build(self: DeployExecutor, *a: Any, **kw: Any) -> None:
        building("dev-only")

    def compose_up(self: DeployExecutor, *a: Any, **kw: Any) -> None:
        assert not lock.exists()
        calls.append("up")

    def gateway(self: DeployExecutor, *a: Any, **kw: Any) -> None:
        assert not lock.exists()
        calls.append("gateway")

    monkeypatch.setattr(DeployExecutor, "_compose_build", compose_build)
    monkeypatch.setattr(DeployExecutor, "_build_dev_lane_only_services", dev_build)
    monkeypatch.setattr(DeployExecutor, "_compose_up", compose_up)
    monkeypatch.setattr(DeployExecutor, "_deploy_gateway_lane", gateway)
    DeployExecutor().rebuild_scope(
        scope,
        [],
        _noop_phase_update,
        git_sha=_SHA,
        git_ref="origin/dev",
        build_source="workspace",
        lane=EnumRuntimeLane.DEV,
    )
    expected = {
        Scope.FULL: ["core", "runtime", "dev-only", "up", "up", "gateway"],
        Scope.RUNTIME: ["runtime", "dev-only", "up", "gateway"],
        Scope.CORE: ["core", "up", "gateway"],
    }
    assert calls == expected[scope]
    assert not lock.exists()


@pytest.mark.parametrize("scope", [Scope.FULL, Scope.RUNTIME])
def test_lock_held_on_build_context_tree_when_omni_home_differs(
    scope: Scope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_root = tmp_path / "agent"
    build_root = tmp_path / "omni_home"
    agent_root.mkdir()
    build_root.mkdir()
    monkeypatch.setenv("OMNI_HOME", str(agent_root))
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(build_root / "omnibase_infra"))
    locks = [agent_root / _LOCK_DIRNAME, build_root / _LOCK_DIRNAME]
    calls: list[str] = []

    def compose_build(self: DeployExecutor, scope: Scope, *a: Any, **kw: Any) -> None:
        assert all(lock.is_dir() for lock in locks)
        calls.append(scope.value)

    def dev_build(self: DeployExecutor, *a: Any, **kw: Any) -> None:
        assert all(lock.is_dir() for lock in locks)
        calls.append("dev-only")

    def compose_up(self: DeployExecutor, *a: Any, **kw: Any) -> None:
        assert all(not lock.exists() for lock in locks)
        calls.append("up")

    def gateway(self: DeployExecutor, *a: Any, **kw: Any) -> None:
        assert all(not lock.exists() for lock in locks)
        calls.append("gateway")

    monkeypatch.setattr(DeployExecutor, "_compose_build", compose_build)
    monkeypatch.setattr(DeployExecutor, "_build_dev_lane_only_services", dev_build)
    monkeypatch.setattr(DeployExecutor, "_compose_up", compose_up)
    monkeypatch.setattr(DeployExecutor, "_deploy_gateway_lane", gateway)
    DeployExecutor().rebuild_scope(
        scope,
        [],
        _noop_phase_update,
        git_sha=_SHA,
        git_ref="origin/dev",
        build_source="workspace",
        lane=EnumRuntimeLane.DEV,
    )
    expected = {
        Scope.FULL: ["core", "runtime", "dev-only", "up", "up", "gateway"],
        Scope.RUNTIME: ["runtime", "dev-only", "up", "gateway"],
    }
    assert calls == expected[scope]
    assert all(not lock.exists() for lock in locks)


@pytest.mark.parametrize("suffix", ["", "/."])
def test_reconcile_lock_roots_deduplicates_build_context_tree(
    suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    omni_home = f"{tmp_path}{suffix}"
    monkeypatch.setenv("OMNI_HOME", f" {omni_home} ")
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(tmp_path / "omnibase_infra"))
    assert executor_mod._reconcile_lock_roots() == [omni_home]


@pytest.mark.parametrize(
    ("scope", "fail_at"),
    [(Scope.FULL, 1), (Scope.FULL, 2), (Scope.FULL, 3), (Scope.RUNTIME, 2)],
)
def test_build_exception_releases_lock_without_recreating(
    scope: Scope, fail_at: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNI_HOME", str(tmp_path))
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(tmp_path / "omnibase_infra"))
    lock = tmp_path / _LOCK_DIRNAME
    calls = 0

    def build(self: DeployExecutor, *a: Any, **kw: Any) -> None:
        nonlocal calls
        assert lock.is_dir()
        calls += 1
        if calls == fail_at:
            raise RuntimeError("broken build")

    def unexpected(self: DeployExecutor, *a: Any, **kw: Any) -> None:
        pytest.fail("A failed build must not deploy")

    monkeypatch.setattr(DeployExecutor, "_compose_build", build)
    monkeypatch.setattr(DeployExecutor, "_build_dev_lane_only_services", build)
    monkeypatch.setattr(DeployExecutor, "_compose_up", unexpected)
    monkeypatch.setattr(DeployExecutor, "_deploy_gateway_lane", unexpected)
    with pytest.raises(RuntimeError, match="broken build"):
        DeployExecutor().rebuild_scope(
            scope, [], _noop_phase_update, git_sha=_SHA, lane=EnumRuntimeLane.DEV
        )
    assert calls == fail_at
    assert not lock.exists()


@pytest.fixture
def _no_workspace_staging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        DeployExecutor, "_stage_workspace", staticmethod(lambda *a, **kw: None)
    )
    monkeypatch.setattr(
        DeployExecutor, "_resolve_plugin_ref", lambda self, path, fallback="": "0" * 40
    )
    monkeypatch.setenv("OMNI_HOME", str(tmp_path))
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(tmp_path / "omnibase_infra"))


def _pin_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor_mod,
        "probe_host_conditions",
        lambda: probe_host_conditions(
            loadavg_reader=lambda: (1.0, 1.0, 1.0),
            io_pressure_reader=lambda: None,
            cpu_count_reader=lambda: 32,
            builder_cache_reader=lambda: EnumBuildCacheState.WARM,
        ),
    )


def _build_with_spy(monkeypatch: pytest.MonkeyPatch, runner: Any) -> list[list[str]]:
    issued: list[list[str]] = []

    def spy(cmd: list[str], **kwargs: Any) -> Any:
        issued.append(cmd)
        return runner(cmd, **kwargs)

    monkeypatch.setattr(executor_mod, "_run", spy)
    return issued


@pytest.mark.usefixtures("_no_workspace_staging")
@pytest.mark.parametrize("as_bytes", [False, True])
def test_compose_error_retains_failing_step_and_build_outcome(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    as_bytes: bool,
) -> None:
    _pin_host(monkeypatch)
    stdout = f"#1 DONE 0.1s\n{_ERROR_BLOCK}\n"
    stderr = " Image runtime Building\nfailed to solve: exit code: 1"

    def runner(cmd: list[str], **kw: Any) -> Any:
        if "compose" in cmd and "build" in cmd:
            return subprocess.CompletedProcess(
                cmd, 1, stdout.encode() if as_bytes else stdout, stderr
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    issued = _build_with_spy(monkeypatch, runner)
    phases: list[tuple[Phase, PhaseStatus]] = []
    with pytest.raises(RuntimeError) as raised:
        DeployExecutor()._compose_build(
            Scope.RUNTIME,
            _SHA,
            lambda phase, status: phases.append((phase, status)),
            build_source="workspace",
            runtime_lane=EnumRuntimeLane.DEV,
            git_ref="origin/dev",
        )
    message = str(raised.value)
    assert message.startswith("runtime_image_build_errored:")
    assert EnumBuildOutcome.classify(message) is EnumBuildOutcome.BUILD_ERRORED
    assert stderr in message
    assert (
        "--- failing build step output (compose writes BuildKit progress to stdout) ---"
        in message
    )
    for line in (
        "Workspace provenance FAILED:",
        "- Per-repo VCS provenance has no sibling entries",
    ):
        assert line in message
        assert any(
            r.levelname == "ERROR" and line in r.getMessage() for r in caplog.records
        )
    assert phases == [(Phase.RUNTIME, PhaseStatus.FAILED)]
    assert any("compose" in cmd and "build" in cmd for cmd in issued)


def test_excerpt_extracts_all_error_blocks_and_error_lines_in_order() -> None:
    error = "#9 ERROR: process did not complete successfully"
    second_block = "------\n > [other 1/1] RUN false:\n1.0 failed\n------"
    stdout = (
        f"#1 DONE 0.1s\n{error}\n{_ERROR_BLOCK}\n"
        f"#10 DONE 0.2s\n{second_block}\nDockerfile:123\n"
    )
    assert executor_mod.build_failure_excerpt(stdout) == (
        f"{error}\n{_ERROR_BLOCK}\n{second_block}"
    )


def test_excerpt_ignores_non_summary_delimiters() -> None:
    stdout = f"------\nordinary output\n------\n{_ERROR_BLOCK}\n"
    assert executor_mod.build_failure_excerpt(stdout) == _ERROR_BLOCK


def test_excerpt_falls_back_to_last_40_nonempty_lines() -> None:
    lines = [f"line {n}" for n in range(60)]
    assert executor_mod.build_failure_excerpt("\n \t\n".join(lines)) == "\n".join(
        lines[-40:]
    )


@pytest.mark.parametrize("stdout", [_ERROR_BLOCK, "fallback diagnostic output"])
def test_excerpt_truncates_to_last_limit_characters(stdout: str) -> None:
    assert executor_mod.build_failure_excerpt(stdout, limit=12) == stdout[-12:]


@pytest.mark.parametrize("stdout", ["", "\n \t\n"])
def test_excerpt_is_empty_for_empty_stdout(stdout: str) -> None:
    assert executor_mod.build_failure_excerpt(stdout) == ""
