# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The agent restores a dev runtime that runs but stays unhealthy (OMN-25106).

Measured on the lab dev lane, 2026-11-16 to 2026-11-17: ``omninode-runtime``
stayed ``running`` and docker ``unhealthy`` (FailingStreak 72 at 00:13Z) after
its event bus consumers exited at 23:34:34Z. The agent's own idle restore read
it as RUNNING and never restored it, and the bus chain that could redeploy it
runs inside that runtime. The idle restore is the agent's local path: it needs
no bus to start.

The restore's rebuild rolls one compose service at a time, and compose recreates
only a service whose configuration or image moved, so a plain roll leaves a
running unhealthy container as it is and the agent would loop. The roll step
restarts a wedged container before compose waits on it, and a repeat wedge
waits out ``RESTORE_RETRY_AFTER`` before the next restore.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent import idle_converge
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import lane_config_for
from deploy_agent.idle_converge import (
    IDLE_RESTORE_REQUESTER,
    RESTORE_AFTER,
    RESTORE_RETRY_AFTER,
    EnumIdleRestoreVerdict,
    EnumRuntimePresence,
    ModelIdleRestoreInputs,
    decide_restore,
    read_lane_presence,
)
from deploy_agent.job_state import JobStore
from deploy_agent.routing import ModelLaneFlags

pytestmark = pytest.mark.unit

DEV = lane_config_for(EnumRuntimeLane.DEV)
MAIN_ONLY = DEV.model_copy(update={"disabled_services": frozenset({"runtime-effects"})})
MAIN = "omninode-runtime"
NOW = datetime(2026, 11, 17, 0, 13, tzinfo=UTC)


def _docker(answers: dict[str, str]) -> Any:
    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        name = cmd[-1]
        if name not in answers:
            return subprocess.CompletedProcess(
                cmd, 1, "", f"Error: No such object: {name}"
            )
        return subprocess.CompletedProcess(cmd, 0, f"{answers[name]}\n", "")

    return run


def _decide(presence: EnumRuntimePresence, detail: str) -> EnumIdleRestoreVerdict:
    return decide_restore(
        ModelIdleRestoreInputs(
            now=NOW,
            presence=presence,
            presence_detail=detail,
            job_active=False,
            last_activity=NOW - RESTORE_AFTER - timedelta(minutes=1),
            last_attempt=None,
        )
    ).verdict


def test_the_recorded_wedge_is_restored() -> None:
    presence, detail = read_lane_presence(
        MAIN_ONLY, run=_docker({MAIN: "running unhealthy 72"})
    )

    assert presence is EnumRuntimePresence.DOWN
    assert "unhealthy" in detail
    assert _decide(presence, detail) is EnumIdleRestoreVerdict.RESTORE


@pytest.mark.parametrize(
    "answer",
    [
        "running healthy 0",
        "running starting 4",
        "running unhealthy 3",
        "running",
        "paused unhealthy 25",
    ],
)
def test_a_runtime_that_is_not_wedged_is_left_alone(answer: str) -> None:
    presence, detail = read_lane_presence(MAIN_ONLY, run=_docker({MAIN: answer}))

    assert presence is EnumRuntimePresence.RUNNING
    assert _decide(presence, detail) is EnumIdleRestoreVerdict.NOT_DOWN


def test_the_agent_read_delegates_to_the_boot_guard_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``idle_converge.read_lane_presence`` is the handler's read, not a copy.

    The handler function has one caller outside its tests: this one, which
    ``DeployAgent._idle_read_runtime_presence`` defaults to. The agent loads the
    handler from the checkout it runs from (``load_dev_boot_guard``), so the
    agent process gets this code with its own restart.
    """
    handler = idle_converge.load_dev_boot_guard()
    seen: list[tuple[Any, ...]] = []

    def spy(*args: Any, **_kw: Any) -> tuple[str, str]:
        seen.append(args)
        return "down", "spy"

    monkeypatch.setattr(handler, "read_lane_presence", spy)

    assert read_lane_presence(DEV) == (EnumRuntimePresence.DOWN, "spy")
    assert seen == [(DEV.runtime_health_targets, DEV.disabled_services)]


class _Roll:
    """The executor boundary the roll step reads: docker, and the compose up."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, answers: dict[str, str]):
        self.calls: list[list[str]] = []
        self.answers = answers
        monkeypatch.setattr(executor_mod, "_run", self._run)
        monkeypatch.setattr(
            executor_mod, "runtime_health_probe_budget", lambda lane, service: 60
        )
        monkeypatch.setattr(executor_mod, "_compose_env", lambda extra: {})
        monkeypatch.setattr(
            executor_mod, "verify_containers_up", lambda *a, **k: (True, [])
        )

    def _run(self, cmd: list[str], timeout: int, **_kw: object) -> Any:
        self.calls.append(cmd)
        if cmd[:2] == ["docker", "inspect"]:
            return _docker(self.answers)(cmd, timeout)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def roll(self) -> list[str]:
        argv = ["docker", "compose", "up", "-d", "--no-deps", MAIN]
        guard = executor_mod.load_dev_boot_guard()
        guard.run_runtime_roll_step(argv, 60, execution=executor_mod)
        return [" ".join(c[:2]) for c in self.calls]


def test_the_restore_roll_restarts_a_running_wedged_container_before_compose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compose up alone leaves it as it is; the restart is what replaces it."""
    roll = _Roll(monkeypatch, {MAIN: "running unhealthy 72"})

    kinds = roll.roll()

    assert ["docker", "restart", MAIN] in roll.calls
    assert kinds.index("docker restart") < kinds.index("docker compose")


def test_the_restore_roll_leaves_a_healthy_container_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roll = _Roll(monkeypatch, {MAIN: "running healthy 0"})

    kinds = roll.roll()

    assert "docker restart" not in kinds
    assert "docker compose" in kinds


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.oneshot_reruns: list[object] = []
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.health_checks: list[object] = []
        self.cloud_migration_failure = ""

    def __getattr__(self, name: str) -> Any:
        def record(*_args: object, **_kwargs: object) -> Any:
            self.calls.append(name)
            if name == "git_pull":
                return "d" * 40
            if name in ("rebuild_scope", "verify"):
                return []
            return None

        return record


class _Clock:
    def __init__(self) -> None:
        self.now = NOW


def _agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> DeployAgent:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(agent, "executor", _FakeExecutor())
    # The clock is the test's own: no completed job moves the idle reading.
    monkeypatch.setattr(agent.job_store, "last_completed_at", lambda: None)
    agent._idle_converge_started_at = NOW - timedelta(hours=1)
    agent._idle_converge_now = lambda: clock.now
    # What the real read returns for ``running unhealthy 72``.
    agent._idle_read_runtime_presence = lambda: (
        EnumRuntimePresence.DOWN,
        "omninode-runtime is running but unhealthy: 72 consecutive failed health probes",
    )

    def no_running_ref(repo: str) -> str | None:
        raise AssertionError("the converge rule read a revision on a wedged lane")

    agent._idle_read_running_ref = no_running_ref
    agent._idle_read_head_ref = lambda repo: "b" * 40
    agent._idle_read_dev_tags = lambda repo: None
    monkeypatch.setattr(
        agent_mod, "publish_dev_tag_heartbeat", lambda payload, config: True
    )
    agent._idle_probe_blocker = lambda now: None
    agent._lane_flags = ModelLaneFlags()
    return agent


def _restore_jobs(tmp_path: Path) -> list[Any]:
    jobs = [
        JobStore(tmp_path / "jobs").load(UUID(p.stem))
        for p in (tmp_path / "jobs").glob("*.json")
    ]
    return [j for j in jobs if j.command["requested_by"] == IDLE_RESTORE_REQUESTER]


def test_an_idle_agent_restores_a_wedged_lane_with_a_runtime_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch, _Clock())

    agent._maybe_idle_converge()

    (job,) = _restore_jobs(tmp_path)
    assert job.command["scope"] == "runtime"
    assert job.command["runtime_lane"] == "dev"
    fake = agent.__dict__["executor"]
    assert isinstance(fake, _FakeExecutor)
    assert "rebuild_scope" in fake.calls


def test_a_repeat_wedge_backs_off_for_the_retry_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    agent = _agent(tmp_path, monkeypatch, clock)
    agent._maybe_idle_converge()
    assert len(_restore_jobs(tmp_path)) == 1

    # Still wedged ten minutes later: no second restore.
    clock.now = NOW + timedelta(minutes=10)
    agent._idle_converge_last_check = None
    agent._maybe_idle_converge()
    assert len(_restore_jobs(tmp_path)) == 1

    # Past the retry window the same wedge is restored once more.
    clock.now = NOW + RESTORE_RETRY_AFTER + timedelta(minutes=1)
    agent._idle_converge_last_check = None
    agent._maybe_idle_converge()
    assert len(_restore_jobs(tmp_path)) == 2
