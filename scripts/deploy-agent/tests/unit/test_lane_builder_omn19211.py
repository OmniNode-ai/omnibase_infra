# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane image builds run on a declared dedicated builder (OMN-19211).

dockerd on the lab host stopped three times in 33 days because the cache index of its
embedded BuildKit (``cache.db``) was damaged and dockerd loads that file before
its API. These tests pin the three behaviours that make the dedicated builder a
containment and not just a second place to build:

* an instance that names a builder builds on it (``BUILDX_BUILDER`` reaches the
  ``docker compose build`` command), and one that does not is unchanged;
* a builder that exists but does not boot is removed with its state and created
  again, which is the whole recovery from a damaged cache;
* a builder that cannot be made ready REFUSES the build, and the phase is marked
  failed. It never falls back to dockerd's embedded BuildKit.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import build_budget as lane_builder
from deploy_agent import executor as executor_mod
from deploy_agent.build_budget import (
    LANE_BUILDER_ENV,
    LaneBuilderError,
    ModelBuilderSpec,
    create_command,
    ensure_builder,
    lane_builder_env,
    load_declaration,
    resolve_spec,
)
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import DeployExecutor


def _spec() -> ModelBuilderSpec:
    return ModelBuilderSpec(
        name="b",
        driver="docker-container",
        image="moby/buildkit:v9.9.9",
        buildkitd_config="deploy/buildkit/buildkitd.toml",
        driver_opts={"network": "host"},
        bootstrap_timeout_seconds=30,
    )


class _FakeDocker:
    """A scripted ``docker`` whose builder state the test controls."""

    def __init__(
        self,
        *,
        exists: bool,
        boots: bool,
        create_ok: bool = True,
        version: str = "v9.9.9",
    ) -> None:
        self.exists = exists
        self.boots = boots
        self.version = version
        self.create_ok = create_ok
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self, argv: Sequence[str], timeout: int
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(tuple(argv))
        verb = argv[2]
        if verb == "inspect" and "--bootstrap" in argv:
            ok = self.exists and self.boots
            out = f"Driver: docker-container\nBuildKit version: {self.version}\n"
            return subprocess.CompletedProcess(
                argv, 0 if ok else 1, out if ok else "", ""
            )
        if verb == "inspect":
            return subprocess.CompletedProcess(argv, 0 if self.exists else 1, "", "")
        if verb == "rm":
            self.exists = False
            return subprocess.CompletedProcess(argv, 0, "", "")
        if verb == "create":
            if not self.create_ok:
                return subprocess.CompletedProcess(argv, 1, "", "pull failed")
            self.exists = True
            self.boots = True
            image = next(a for a in argv if a.startswith("image=") or ",image=" in a)
            self.version = image.split("image=")[1].split(",")[0].rpartition(":")[2]
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected docker call {argv}")

    def verbs(self) -> list[str]:
        return [c[2] for c in self.calls if c[2] in {"create", "rm"}]


@pytest.mark.unit
class TestEnsureBuilder:
    def test_ready_builder_is_left_alone(self) -> None:
        docker = _FakeDocker(exists=True, boots=True)
        ensure_builder(_spec(), run=docker)
        assert docker.verbs() == []

    def test_absent_builder_is_created(self) -> None:
        docker = _FakeDocker(exists=False, boots=False)
        ensure_builder(_spec(), run=docker)
        assert docker.verbs() == ["create"]

    def test_builder_that_does_not_boot_is_removed_with_state_then_recreated(
        self,
    ) -> None:
        """The recovery from a damaged cache index: only that builder is touched."""
        docker = _FakeDocker(exists=True, boots=False)
        ensure_builder(_spec(), run=docker)
        assert docker.verbs() == ["rm", "create"]
        rm = next(c for c in docker.calls if c[2] == "rm")
        assert "--keep-state" not in rm, "the damaged state must go with the builder"

    def test_builder_on_another_buildkit_version_is_recreated(self) -> None:
        docker = _FakeDocker(exists=True, boots=True, version="v1.0.0")
        ensure_builder(_spec(), run=docker)
        assert docker.verbs() == ["rm", "create"]

    def test_creation_failure_refuses(self) -> None:
        docker = _FakeDocker(exists=False, boots=False, create_ok=False)
        with pytest.raises(LaneBuilderError, match="could not be created"):
            ensure_builder(_spec(), run=docker)

    def test_create_command_carries_every_declared_value(self, tmp_path: Path) -> None:
        argv = create_command(_spec(), tmp_path)
        assert argv[:5] == ["docker", "buildx", "create", "--name", "b"]
        assert "image=moby/buildkit:v9.9.9,network=host" in argv
        assert str(tmp_path / "deploy/buildkit/buildkitd.toml") in argv
        assert "--bootstrap" in argv


@pytest.mark.unit
class TestLaneBuilderEnv:
    def test_unset_leaves_the_default_builder(self) -> None:
        docker = _FakeDocker(exists=False, boots=False)
        assert lane_builder_env({}, run=docker) == {}
        assert docker.calls == []

    def test_named_builder_is_ensured_then_selected(self) -> None:
        decl = load_declaration()
        spec = resolve_spec("lane", decl)
        docker = _FakeDocker(exists=True, boots=True, version=spec.image_version)
        env = lane_builder_env({LANE_BUILDER_ENV: "lane"}, declaration=decl, run=docker)
        assert env == {"BUILDX_BUILDER": spec.name}
        assert docker.verbs() == []

    def test_unknown_key_refuses(self) -> None:
        with pytest.raises(LaneBuilderError, match="names no builder"):
            lane_builder_env({LANE_BUILDER_ENV: "nonesuch"})


def _is_compose_build(cmd: list[str]) -> bool:
    return "build" in cmd and "compose" in cmd


@pytest.fixture
def _no_workspace_staging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        DeployExecutor, "_stage_workspace", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        DeployExecutor, "_resolve_plugin_ref", lambda self, path, fallback="": "0" * 40
    )
    monkeypatch.setenv("OMNI_HOME", str(tmp_path))


def _build(executor: DeployExecutor, updates: list[tuple[Phase, PhaseStatus]]) -> None:
    executor._compose_build(
        Scope.RUNTIME,
        "0" * 40,
        lambda phase, status: updates.append((phase, status)),
        build_source="workspace",
        runtime_lane=EnumRuntimeLane.DEV,
        git_ref="origin/dev",
    )


@pytest.mark.unit
@pytest.mark.usefixtures("_no_workspace_staging")
class TestComposeBuildUsesTheBuilder:
    def test_declared_builder_reaches_the_compose_build_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def _spy(cmd: list[str], **kwargs: Any) -> Any:
            if _is_compose_build(cmd):
                seen["env"] = kwargs.get("env")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(executor_mod, "_run", _spy)
        monkeypatch.setattr(
            executor_mod,
            "lane_builder_env",
            lambda environ: {"BUILDX_BUILDER": "onex-lane-buildkit"},
        )
        _build(DeployExecutor(), [])
        assert seen["env"]["BUILDX_BUILDER"] == "onex-lane-buildkit"

    def test_instance_without_a_builder_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def _spy(cmd: list[str], **kwargs: Any) -> Any:
            if _is_compose_build(cmd):
                seen["env"] = kwargs.get("env")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(executor_mod, "_run", _spy)
        monkeypatch.delenv(LANE_BUILDER_ENV, raising=False)
        monkeypatch.delenv("BUILDX_BUILDER", raising=False)
        _build(DeployExecutor(), [])
        assert "BUILDX_BUILDER" not in seen["env"]

    def test_unready_builder_refuses_the_build_and_never_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran: list[list[str]] = []

        def _spy(cmd: list[str], **kwargs: Any) -> Any:
            ran.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        def _refuse(environ: Any) -> dict[str, str]:
            raise LaneBuilderError("builder onex-lane-buildkit could not be created")

        monkeypatch.setattr(executor_mod, "_run", _spy)
        monkeypatch.setattr(executor_mod, "lane_builder_env", _refuse)
        updates: list[tuple[Phase, PhaseStatus]] = []
        with pytest.raises(LaneBuilderError):
            _build(DeployExecutor(), updates)
        assert not any(_is_compose_build(c) for c in ran), (
            "a refused builder must not fall back to dockerd's embedded BuildKit"
        )
        assert (Phase.RUNTIME, PhaseStatus.FAILED) in updates


@pytest.mark.unit
def test_lane_builder_module_reads_the_declaration_it_ships_with() -> None:
    assert lane_builder.DECLARATION_PATH.is_file()
    assert set(load_declaration().builders) >= {"lane", "ci"}
