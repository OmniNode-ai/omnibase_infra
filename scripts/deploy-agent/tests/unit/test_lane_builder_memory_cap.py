# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lane image builder carries a declared memory limit (OMN-24179).

A dev runtime image build was measured growing past 17 GB of cgroup memory with
no cap, and 96 percent of the host's swap-out fell in roll windows. The limit is
a declaration (``deploy/buildkit/builders.yaml``, ``driver_opts``) that
``create_command`` already hands to ``docker buildx create``. Seams measured:

* declaration <-> create command: the limit reaches the argv the agent runs;
* declaration <-> ensure_builder: a builder under the declared name that does
  not exist yet is created WITH the limit, and a builder under another name
  (the previous, uncapped one) is never touched;
* the name carries the limit, so a changed limit cannot leave a running builder
  under the old name reported ready.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence

import pytest
from deploy_agent.build_budget import (
    LaneBuilderError,
    create_command,
    ensure_builder,
    load_declaration,
    resolve_spec,
)

pytestmark = pytest.mark.unit

_GIB = 1024**3
_PREVIOUS_NAME = "onex-lane-buildkit"


def _bytes(text: str) -> int:
    match = re.fullmatch(r"([0-9]+)g", text)
    assert match, f"{text!r} is not a whole number of GiB written like 12g"
    return int(match.group(1)) * _GIB


def test_memory_cap_the_lane_builder_declares_a_limit_and_no_swap() -> None:
    spec = resolve_spec("lane")

    memory = spec.driver_opts.get("memory")
    swap = spec.driver_opts.get("memory-swap")
    assert memory is not None, "the lane builder declares no memory limit"
    assert swap == memory, "memory-swap must equal memory so the builder has no swap"
    assert _bytes(memory) >= 4 * _GIB, "a limit under 4 GiB would kill a healthy build"


def test_memory_cap_the_builder_name_carries_the_declared_limit() -> None:
    spec = resolve_spec("lane")

    assert spec.name != _PREVIOUS_NAME, "the uncapped builder's name is reused"
    assert spec.name.endswith(spec.driver_opts["memory"]), (
        "the name must change with the limit: ensure_builder does not read a "
        "running builder's limit"
    )


def test_memory_cap_the_ci_builder_is_unchanged() -> None:
    ci = load_declaration().builders["ci"]

    assert ci.name == "onex-shared-buildkit"
    assert "memory" not in ci.driver_opts


def test_memory_cap_create_command_carries_the_declared_limit(tmp_path) -> None:
    spec = resolve_spec("lane")

    argv = create_command(spec, tmp_path)

    opts = argv[argv.index("--driver-opt") + 1].split(",")
    assert f"memory={spec.driver_opts['memory']}" in opts
    assert f"memory-swap={spec.driver_opts['memory-swap']}" in opts


class _FakeDocker:
    """A scripted ``docker`` for the declared builder name only."""

    def __init__(self, *, existing: set[str], create_ok: bool = True) -> None:
        self.existing = set(existing)
        self.create_ok = create_ok
        self.calls: list[tuple[str, ...]] = []
        self.created_opts: str | None = None

    def __call__(
        self, argv: Sequence[str], timeout: int
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(tuple(argv))
        verb = argv[2]
        if verb == "inspect":
            name = argv[3]
            ok = name in self.existing
            out = ""
            if ok and "--bootstrap" in argv:
                spec = resolve_spec("lane")
                out = f"Driver: {spec.driver}\nBuildKit version: {spec.image_version}\n"
            return subprocess.CompletedProcess(argv, 0 if ok else 1, out, "")
        if verb == "create":
            if not self.create_ok:
                return subprocess.CompletedProcess(argv, 1, "", "pull failed")
            self.created_opts = argv[argv.index("--driver-opt") + 1]
            self.existing.add(argv[argv.index("--name") + 1])
            return subprocess.CompletedProcess(argv, 0, "", "")
        if verb == "rm":
            self.existing.discard(argv[-1])
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected docker call {argv}")

    def verbs(self) -> list[str]:
        return [c[2] for c in self.calls if c[2] in {"create", "rm"}]


def test_memory_cap_a_host_with_only_the_old_builder_gets_the_capped_one(
    tmp_path,
) -> None:
    """Golden chain: the agent finds no builder under the declared name, creates it."""
    spec = resolve_spec("lane")
    docker = _FakeDocker(existing={_PREVIOUS_NAME})

    ensure_builder(spec, repo_root=tmp_path, run=docker)

    assert docker.verbs() == ["create"], "the old builder must not be removed"
    assert docker.created_opts is not None
    assert f"memory={spec.driver_opts['memory']}" in docker.created_opts
    assert _PREVIOUS_NAME in docker.existing, "the previous builder is left alone"
    assert spec.name in docker.existing


def test_memory_cap_a_ready_capped_builder_is_left_alone(tmp_path) -> None:
    spec = resolve_spec("lane")
    docker = _FakeDocker(existing={spec.name})

    ensure_builder(spec, repo_root=tmp_path, run=docker)

    assert docker.verbs() == []


def test_memory_cap_a_failed_creation_refuses_and_never_falls_back(tmp_path) -> None:
    """Error chain: no capped builder, no build on another one."""
    spec = resolve_spec("lane")
    docker = _FakeDocker(existing={_PREVIOUS_NAME}, create_ok=False)

    with pytest.raises(LaneBuilderError, match="could not be created"):
        ensure_builder(spec, repo_root=tmp_path, run=docker)

    assert spec.name not in docker.existing
