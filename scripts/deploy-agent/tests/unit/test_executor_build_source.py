# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""BUILD_SOURCE selector contract tests for deploy-agent builds."""

from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[4]
pytestmark = pytest.mark.unit


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


def _ok() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def test_rebuild_request_accepts_canonical_build_source() -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
        build_source="workspace",
    )

    assert cmd.build_source == BuildSource.WORKSPACE


def test_rebuild_request_rejects_unknown_build_source() -> None:
    with pytest.raises(ValidationError, match="build_source"):
        ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="test",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.DEV,
            build_source="local",  # type: ignore[arg-type]
        )


def test_compose_build_passes_build_source_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNI_HOME", "/data/omninode/omni_home")
    executor = DeployExecutor()
    captured_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        captured_cmds.append(cmd)
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)
    monkeypatch.setattr(
        DeployExecutor, "_stage_workspace", staticmethod(lambda *_: None)
    )

    executor._compose_build(
        Scope.RUNTIME,
        "abc1234",
        _noop_phase_update,
        build_source=BuildSource.WORKSPACE,
    )

    build_cmd = captured_cmds[0]
    assert "--build-arg" in build_cmd
    assert "BUILD_SOURCE=workspace" in build_cmd
    assert "EXPECTED_BUILD_SOURCE=workspace" in build_cmd
    assert "OMNI_HOME=/data/omninode/omni_home" in build_cmd

    # OMN-12965: the workspace build must stamp the full OCI image-identity quad
    # so org.opencontainers.image.{version,revision,created} populate. A blank
    # revision / placeholder version degrades every proof packet.
    assert "GIT_SHA=abc1234" in build_cmd
    assert "VCS_REF=abc1234" in build_cmd
    runtime_version_args = [a for a in build_cmd if a.startswith("RUNTIME_VERSION=")]
    assert runtime_version_args, "missing RUNTIME_VERSION build-arg (OMN-12965)"
    assert runtime_version_args[0] != "RUNTIME_VERSION=0.1.0", (
        "RUNTIME_VERSION must come from pyproject, not the Dockerfile placeholder"
    )
    assert any(a.startswith("BUILD_DATE=") for a in build_cmd)


def test_unknown_build_source_fails_before_compose_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = DeployExecutor()
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)

    with pytest.raises(RuntimeError, match="Invalid deploy-agent BUILD_SOURCE"):
        executor._compose_build(
            Scope.RUNTIME,
            "abc1234",
            _noop_phase_update,
            build_source="local",
        )

    assert calls == []


def test_selector_disagreement_fails_before_compose_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = DeployExecutor()
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)

    with pytest.raises(RuntimeError, match="selector disagreement"):
        executor._compose_build(
            Scope.RUNTIME,
            "abc1234",
            _noop_phase_update,
            build_source=BuildSource.WORKSPACE,
            expected_build_source=BuildSource.RELEASE,
        )

    assert calls == []


def test_workspace_build_requires_omni_home_before_compose_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNI_HOME", raising=False)
    executor = DeployExecutor()
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], timeout: int, **kwargs) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return _ok()

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)

    with pytest.raises(RuntimeError, match="BUILD_SOURCE=workspace requires OMNI_HOME"):
        executor._compose_build(
            Scope.RUNTIME,
            "abc1234",
            _noop_phase_update,
            build_source=BuildSource.WORKSPACE,
        )

    assert calls == []


def test_runtime_compose_passes_build_source_args() -> None:
    compose_path = REPO_ROOT / "docker/docker-compose.infra.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    build_args = compose["x-runtime-base"]["build"]["args"]

    assert build_args["BUILD_SOURCE"] == "${BUILD_SOURCE:-release}"
    assert build_args["EXPECTED_BUILD_SOURCE"] == "${EXPECTED_BUILD_SOURCE:-release}"
    assert "OMNI_HOME" not in build_args


def test_runtime_dockerfile_validates_and_stamps_build_source() -> None:
    dockerfile = (REPO_ROOT / "docker/Dockerfile.runtime").read_text(encoding="utf-8")

    assert "ARG BUILD_SOURCE=release" in dockerfile
    assert "ARG EXPECTED_BUILD_SOURCE=release" in dockerfile
    assert "Invalid BUILD_SOURCE=" in dockerfile
    assert "BUILD_SOURCE selector mismatch" in dockerfile
    assert "BUILD_SOURCE=workspace requires OMNI_HOME" in dockerfile
    assert 'com.omninode.build_source="${BUILD_SOURCE}"' in dockerfile
    assert "BUILD_SOURCE=${BUILD_SOURCE}" in dockerfile
    assert "ARG OMNIBASE_COMPAT_REF=" in dockerfile
    assert (
        "https://github.com/OmniNode-ai/omnibase_compat/archive/${OMNIBASE_COMPAT_REF}.tar.gz"
        in dockerfile
    )
