# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's runtime recreate retires the previous hook-events writer (OMN-21288).

The dev lane deploys through the deploy agent, not the bash deploy handler. Main
now carries the claude hook-events projection (no ``ONEX_RUNTIME_NODE_EXCLUSION``),
so a previous revision's ``projection-claude-hook-events-writer`` container left
running joins main's consumer group with a stale image and splits its partitions.

This drives the real ``DeployExecutor._compose_up`` for the runtime scope, with the
committed fold policy and a fake docker that holds a running previous-revision
writer. Nothing in the agent's retire step is patched out (the sibling converge
tests patch it), so the call site, the census and the ordering are all exercised.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import DeployExecutor

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
HOOK_EVENTS_WRITER = "projection-claude-hook-events-writer"
PREVIOUS_WRITER_ID = "previouswriter0001"
SERVICE_LABEL = "label=com.docker.compose.service="


@pytest.fixture(autouse=True)
def _committed_fold_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this checkout's committed fold policy, not the deploy host's clone path."""
    monkeypatch.setattr(
        "deploy_agent.executor.RUNTIME_POLICY_ENV_FILE",
        REPO_ROOT / "docker/runtime-policy.env",
    )


def _noop(phase: Phase, status: PhaseStatus) -> None:
    return None


def _ok(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout, "")


class _FakeDockerWithRunningWriter:
    """Docker state: the previous revision's writer container is still running."""

    def __init__(self) -> None:
        self.writer_running = True
        self.calls: list[list[str]] = []
        self.writer_running_at_runtime_up: bool | None = None

    def run(self, cmd: list[str], timeout: int = 0, **_: Any) -> Any:
        self.calls.append(list(cmd))
        if cmd[:2] == ["docker", "ps"]:
            service = next(
                (a[len(SERVICE_LABEL) :] for a in cmd if a.startswith(SERVICE_LABEL)),
                "",
            )
            if service == HOOK_EVENTS_WRITER and self.writer_running:
                return _ok(f"{PREVIOUS_WRITER_ID}\n")
            return _ok("")
        if cmd[:2] == ["docker", "stop"] and PREVIOUS_WRITER_ID in cmd:
            return _ok(PREVIOUS_WRITER_ID)
        if cmd[:2] == ["docker", "rm"] and PREVIOUS_WRITER_ID in cmd:
            self.writer_running = False
            return _ok(PREVIOUS_WRITER_ID)
        if cmd[:2] == ["docker", "compose"] and "up" in cmd:
            if "omninode-runtime" in cmd:
                self.writer_running_at_runtime_up = self.writer_running
            return _ok()
        if "config" in cmd and "--hash" in cmd:
            names = cmd[cmd.index("--hash") + 1].split(",")
            return _ok("".join(f"{n} h-{n}\n" for n in names))
        return _ok()


def _recreate_main(docker: _FakeDockerWithRunningWriter) -> None:
    budget = SimpleNamespace(timeout_seconds=600, describe=lambda: "test budget")
    executor = DeployExecutor()
    with (
        patch.object(executor_mod, "_run", docker.run),
        patch.object(executor_mod, "verify_containers_up", lambda s, **k: (True, [])),
        patch.object(executor_mod, "_compose_env", lambda *a, **k: {}),
        patch.object(executor_mod, "finish_recreate_renames", lambda lane: []),
        patch.object(executor_mod, "preflight_dev_images", lambda *a, **k: None),
        patch.object(executor_mod, "runtime_compose_up_budget", lambda *a: budget),
        patch.object(executor, "_ensure_runtime_migrations_ready"),
        patch.object(executor, "rerun_changed_oneshots", return_value=[]),
    ):
        executor._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            ["omninode-runtime"],
            _noop,
            lane=EnumRuntimeLane.DEV,
        )


def test_agent_recreate_of_main_stops_and_removes_the_previous_hook_events_writer() -> (
    None
):
    docker = _FakeDockerWithRunningWriter()
    _recreate_main(docker)

    stops = [i for i, c in enumerate(docker.calls) if c[:2] == ["docker", "stop"]]
    removes = [i for i, c in enumerate(docker.calls) if c[:2] == ["docker", "rm"]]
    ups = [
        i
        for i, c in enumerate(docker.calls)
        if c[:2] == ["docker", "compose"] and "up" in c and "omninode-runtime" in c
    ]
    assert stops, "the previous revision's hook-events writer was never stopped"
    assert removes, "the previous revision's hook-events writer was never removed"
    assert ups, "the agent never recreated omninode-runtime"
    assert stops[0] < removes[0] < ups[0], "main was recreated before the retirement"
    # The one thing that splits runtime-main's partitions: a live writer container.
    assert docker.writer_running_at_runtime_up is False
    assert all(
        "-v" not in docker.calls[i] and "--volumes" not in docker.calls[i]
        for i in (*stops, *removes)
    ), "retirement must keep volumes"


def test_agent_recreate_never_starts_the_folded_writer() -> None:
    docker = _FakeDockerWithRunningWriter()
    _recreate_main(docker)
    for call in docker.calls:
        if call[:2] == ["docker", "compose"] and "up" in call:
            assert HOOK_EVENTS_WRITER not in call
