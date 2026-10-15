# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20637: a rebuild leaves no tracked modification in the build-context clone.

``stage_workspace.sh`` and ``compute_workspace_provenance.py`` overwrite the two
committed placeholders in ``SELF_UPDATE_KNOWN_BYPRODUCT_PATHS`` inside the clone
the agent builds from. That clone is the host's canonical omnibase_infra clone,
and the canonical clone refresh refuses any clone with tracked changes, so every
merge event that landed between a build and the next reconcile tick was refused
(h201 and h202, omnibase_infra merge bded6a7). The agent restores the
placeholders itself once its last image build is done, while it still holds the
reconcile-host lock, and on a failed build too.

Real git repositories; only the build and compose steps are stubbed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import DEV_LANE_GATEWAY_SERVICES, EnumRuntimeLane, Scope
from deploy_agent.executor import SELF_UPDATE_KNOWN_BYPRODUCT_PATHS, DeployExecutor

pytestmark = pytest.mark.unit

_SHA = "b" * 40
_PLACEHOLDER = '{\n  "_note": "placeholder"\n}\n'
_REAL = '{\n  "_note": "written by a build"\n}\n'


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _make_build_clone(tmp_path: Path) -> Path:
    clone = tmp_path / "omnibase_infra"
    (clone / "workspace").mkdir(parents=True)
    _git(tmp_path, "init", "--initial-branch", "dev", str(clone))
    _git(clone, "config", "user.email", "agent@example.invalid")
    _git(clone, "config", "user.name", "deploy agent test")
    for rel_path in SELF_UPDATE_KNOWN_BYPRODUCT_PATHS:
        (clone / rel_path).write_text(_PLACEHOLDER, encoding="utf-8")
    (clone / "other.txt").write_text("operator work\n", encoding="utf-8")
    _git(clone, "add", ".")
    _git(clone, "commit", "-m", "placeholders")
    return clone


def _tracked_status(clone: Path) -> str:
    return _git(clone, "status", "--porcelain", "--untracked-files=no")


def _write_build_byproducts(clone: Path) -> None:
    for rel_path in SELF_UPDATE_KNOWN_BYPRODUCT_PATHS:
        (clone / rel_path).write_text(_REAL, encoding="utf-8")


def _rebuild(scope: Scope) -> None:
    DeployExecutor().rebuild_scope(
        scope,
        [],
        lambda phase, status: None,
        git_sha=_SHA,
        git_ref="origin/dev",
        build_source="workspace",
        lane=EnumRuntimeLane.DEV,
    )


@pytest.fixture
def build_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    clone = _make_build_clone(tmp_path)
    monkeypatch.setenv("OMNI_HOME", str(tmp_path))
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(clone))
    monkeypatch.setattr(DeployExecutor, "_compose_up", lambda self, *a, **kw: None)
    monkeypatch.setattr(
        DeployExecutor, "_build_dev_lane_only_services", lambda self, *a, **kw: None
    )
    return clone


@pytest.mark.parametrize("scope", [Scope.FULL, Scope.RUNTIME, Scope.CORE])
def test_clone_has_no_tracked_modification_after_a_rebuild(
    scope: Scope, build_clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def compose_build(self: DeployExecutor, scope: Scope, *a: Any, **kw: Any) -> None:
        _write_build_byproducts(build_clone)
        assert "sibling-pin-comparison.json" in _tracked_status(build_clone)

    monkeypatch.setattr(DeployExecutor, "_compose_build", compose_build)
    _rebuild(scope)
    assert _tracked_status(build_clone) == ""
    for rel_path in SELF_UPDATE_KNOWN_BYPRODUCT_PATHS:
        assert (build_clone / rel_path).read_text(encoding="utf-8") == _PLACEHOLDER


def test_byproducts_survive_until_the_last_build_has_consumed_them(
    build_clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_by_dev_build: list[str] = []

    def compose_build(self: DeployExecutor, scope: Scope, *a: Any, **kw: Any) -> None:
        _write_build_byproducts(build_clone)

    def dev_build(self: DeployExecutor, *a: Any, **kw: Any) -> None:
        seen_by_dev_build.extend(
            (build_clone / rel_path).read_text(encoding="utf-8")
            for rel_path in SELF_UPDATE_KNOWN_BYPRODUCT_PATHS
        )

    monkeypatch.setattr(DeployExecutor, "_compose_build", compose_build)
    monkeypatch.setattr(DeployExecutor, "_build_dev_lane_only_services", dev_build)
    _rebuild(Scope.RUNTIME)
    assert seen_by_dev_build == [_REAL, _REAL]
    assert _tracked_status(build_clone) == ""


def test_a_failed_build_still_restores_the_placeholders(
    build_clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def compose_build(self: DeployExecutor, scope: Scope, *a: Any, **kw: Any) -> None:
        _write_build_byproducts(build_clone)
        raise RuntimeError("image build failed")

    monkeypatch.setattr(DeployExecutor, "_compose_build", compose_build)
    with pytest.raises(RuntimeError, match="image build failed"):
        _rebuild(Scope.RUNTIME)
    assert _tracked_status(build_clone) == ""


def test_other_tracked_work_in_the_clone_is_left_alone(
    build_clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def compose_build(self: DeployExecutor, scope: Scope, *a: Any, **kw: Any) -> None:
        _write_build_byproducts(build_clone)
        (build_clone / "other.txt").write_text("edited by an operator\n")

    monkeypatch.setattr(DeployExecutor, "_compose_build", compose_build)
    _rebuild(Scope.RUNTIME)
    assert _tracked_status(build_clone) == "M other.txt"
    assert (build_clone / "other.txt").read_text() == "edited by an operator\n"


def test_a_build_context_that_is_not_a_git_clone_does_not_fail_the_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNI_HOME", str(tmp_path))
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(tmp_path / "omnibase_infra"))
    built: list[Scope] = []
    monkeypatch.setattr(
        DeployExecutor,
        "_compose_build",
        lambda self, scope, *a, **kw: built.append(scope),
    )
    monkeypatch.setattr(DeployExecutor, "_compose_up", lambda self, *a, **kw: None)
    _rebuild(Scope.CORE)
    assert built == [Scope.CORE]


@pytest.mark.gateway_lane
@pytest.mark.parametrize("script_exit", [0, 1])
def test_gateway_deploy_restores_what_its_own_staging_rewrote(
    script_exit: int, build_clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """deploy-gateway.sh runs stage_workspace.sh in the same clone, after the lock."""
    real_run = executor_mod._run

    def run(
        cmd: list[str], timeout: int, **kw: Any
    ) -> subprocess.CompletedProcess[str]:
        if cmd[0] != "bash":
            return real_run(cmd, timeout, **kw)
        _write_build_byproducts(build_clone)
        return subprocess.CompletedProcess(cmd, script_exit, "", "")

    monkeypatch.setattr(executor_mod, "_run", run)

    def deploy() -> None:
        DeployExecutor()._deploy_gateway_lane(
            lambda phase, status: None,
            lane=EnumRuntimeLane.DEV,
            build_source="workspace",
            targets=list(DEV_LANE_GATEWAY_SERVICES),
            git_ref="origin/dev",
        )

    if script_exit:
        with pytest.raises(RuntimeError, match="GATEWAY_DEPLOY_FAILED"):
            deploy()
    else:
        deploy()
    assert _tracked_status(build_clone) == ""
