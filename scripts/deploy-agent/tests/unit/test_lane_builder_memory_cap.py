# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lane builder's declared memory limit is enforced (OMN-24179).

The lane builder container on the main lab host peaked at 8288 MB within 3.5
minutes of a roll, with 1454 MB of it already swapped. The declaration now
pins a memory limit and an equal swap limit, and the deploy agent aligns a
running builder container with that declaration using ``docker update``
without ever removing the builder, its state volume or its cache.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Any

import pytest
from deploy_agent import build_budget as lane_builder
from deploy_agent.build_budget import (
    LaneBuilderError,
    ModelBuilderSpec,
    _declared_memory,
    _enforce_memory_limit,
    _size_to_bytes,
    builder_container_name,
    create_command,
    ensure_builder,
    load_declaration,
    resolve_spec,
)

pytestmark = pytest.mark.unit


def _spec(**overrides: Any) -> ModelBuilderSpec:
    base: dict[str, Any] = {
        "name": "b",
        "driver": "docker-container",
        "image": "moby/buildkit:v9.9.9",
        "buildkitd_config": "deploy/buildkit/buildkitd.toml",
        "driver_opts": {"network": "host"},
        "bootstrap_timeout_seconds": 30,
    }
    base.update(overrides)
    return ModelBuilderSpec(**base)


class _FakeDocker:
    """A scripted ``docker`` whose builder state the test controls.

    Dispatches on argv[1] for docker-level verbs (inspect/update) and on
    argv[2] for buildx verbs (inspect/rm/create), so a docker-level
    ``inspect`` (the memory-limit probe) is answered separately from the
    existing ``buildx inspect`` calls.
    """

    def __init__(
        self,
        *,
        exists: bool,
        boots: bool,
        create_ok: bool = True,
        version: str = "v9.9.9",
        memory: int = 0,
        memory_swap: int = 0,
        update_ok: bool = True,
    ) -> None:
        self.exists = exists
        self.boots = boots
        self.version = version
        self.create_ok = create_ok
        self.update_ok = update_ok
        self.memory = memory
        self.memory_swap = memory_swap
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self, argv: Sequence[str], timeout: int
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(tuple(argv))
        if argv[1] == "inspect":
            # docker-level inspect: the memory-limit probe
            return subprocess.CompletedProcess(
                argv, 0, f"{self.memory} {self.memory_swap}", ""
            )
        if argv[1] == "update":
            if not self.update_ok:
                return subprocess.CompletedProcess(argv, 1, "", "update failed")
            self.memory = int(argv[argv.index("--memory") + 1])
            self.memory_swap = int(argv[argv.index("--memory-swap") + 1])
            return subprocess.CompletedProcess(argv, 0, "", "")
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

    def updates(self) -> list[tuple[str, ...]]:
        return [c for c in self.calls if c[1] == "update"]


class TestDeclaration:
    def test_lane_builder_declares_memory_and_swap(self) -> None:
        decl = load_declaration()
        lane = decl.builders["lane"]
        assert lane.driver_opts["memory"] == "10g"
        assert lane.driver_opts["memory-swap"] == "10g"

    def test_ci_builder_declares_no_memory(self) -> None:
        decl = load_declaration()
        ci = decl.builders["ci"]
        assert "memory" not in ci.driver_opts
        assert "memory-swap" not in ci.driver_opts


class TestEnforceMemoryLimit:
    def test_ready_builder_at_zero_gets_one_update_and_no_rm_or_create(self) -> None:
        spec = _spec(driver_opts={"network": "host", "memory": "10g", "memory-swap": "10g"})
        docker = _FakeDocker(exists=True, boots=True, memory=0, memory_swap=0)
        ensure_builder(spec, run=docker)
        assert len(docker.updates()) == 1
        assert docker.updates()[0] == (
            "docker",
            "update",
            "--memory",
            "10737418240",
            "--memory-swap",
            "10737418240",
            "buildx_buildkit_b0",
        )
        assert docker.verbs() == []

    def test_ready_builder_already_at_the_pair_gets_no_update(self) -> None:
        spec = _spec(driver_opts={"network": "host", "memory": "10g", "memory-swap": "10g"})
        docker = _FakeDocker(
            exists=True, boots=True, memory=10737418240, memory_swap=10737418240
        )
        ensure_builder(spec, run=docker)
        assert docker.updates() == []
        assert docker.verbs() == []

    def test_spec_without_memory_makes_no_inspect_or_update(self) -> None:
        spec = _spec()
        docker = _FakeDocker(exists=True, boots=True)
        ensure_builder(spec, run=docker)
        assert docker.updates() == []
        assert not any(c[1] == "inspect" and "--format" in c for c in docker.calls)

    def test_failing_update_raises(self) -> None:
        spec = _spec(driver_opts={"network": "host", "memory": "10g", "memory-swap": "10g"})
        docker = _FakeDocker(
            exists=True, boots=True, memory=0, memory_swap=0, update_ok=False
        )
        with pytest.raises(LaneBuilderError, match="could not update memory limit"):
            ensure_builder(spec, run=docker)


class TestCreateCommand:
    def test_create_command_carries_memory_driver_opt(self, tmp_path: Any) -> None:
        spec = _spec(driver_opts={"network": "host", "memory": "10g"})
        argv = create_command(spec, tmp_path)
        assert "memory=10g" in argv[argv.index("--driver-opt") + 1]


class TestSizeToBytes:
    def test_parses_10g(self) -> None:
        assert _size_to_bytes("10g") == 10 * 1024**3

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(LaneBuilderError):
            _size_to_bytes("ten")


class TestHelpers:
    def test_builder_container_name(self) -> None:
        assert builder_container_name(_spec()) == "buildx_buildkit_b0"

    def test_declared_memory_returns_none_without_memory(self) -> None:
        assert _declared_memory(_spec()) is None

    def test_declared_memory_defaults_swap_to_memory(self) -> None:
        spec = _spec(driver_opts={"network": "host", "memory": "10g"})
        assert _declared_memory(spec) == (10 * 1024**3, 10 * 1024**3)
