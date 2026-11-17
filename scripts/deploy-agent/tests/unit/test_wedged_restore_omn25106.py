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
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from uuid import UUID

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent import idle_converge
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane, Phase, Scope
from deploy_agent.executor import DeployExecutor, lane_config_for
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
    # The handler's own read over docker's answer for the recorded wedge, so
    # the back-off below is driven by the code this change added.
    agent._idle_read_runtime_presence = lambda: read_lane_presence(
        MAIN_ONLY, run=_docker({MAIN: "running unhealthy 72"})
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


def _executor_roll(
    monkeypatch: pytest.MonkeyPatch, answers: dict[str, str]
) -> list[list[str]]:
    """Drive ``DeployExecutor._compose_up`` over the dev runtime and effects services.

    ``_run`` is the executor's one subprocess boundary and, like the real one,
    answers a non-zero exit with a result instead of raising. Docker answers
    ``answers`` for an inspect and has no other container.
    """
    services = ["omninode-runtime", "runtime-effects"]
    calls: list[list[str]] = []
    executor = DeployExecutor()

    def run(cmd: list[str], *_a: object, **_kw: object) -> Any:
        calls.append(cmd)
        if cmd[:2] == ["docker", "inspect"]:
            return _docker(answers)(cmd, 0)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(executor_mod, "_run", run)
    monkeypatch.setattr(executor_mod, "_compose_env", lambda *a, **k: {})
    monkeypatch.setattr(
        executor_mod,
        "runtime_compose_up_budget",
        lambda *a: SimpleNamespace(timeout_seconds=600, describe=lambda: "test"),
    )
    monkeypatch.setattr(executor_mod, "runtime_health_probe_budget", lambda *a: 60)
    monkeypatch.setattr(executor, "_ensure_runtime_migrations_ready", Mock())
    monkeypatch.setattr(executor, "rerun_changed_oneshots", Mock(return_value=[]))
    monkeypatch.setattr(executor, "_require_compose_up_converged", Mock())
    monkeypatch.setattr(
        executor_mod,
        "_compose_service_states",
        lambda *a: dict.fromkeys(services, ("running", None)),
    )
    monkeypatch.setattr(
        executor_mod, "verify_containers_up", lambda *a, **k: (True, [])
    )
    executor._compose_up(
        Phase.RUNTIME,
        Scope.RUNTIME,
        services,
        lambda phase, status: None,
        lane=EnumRuntimeLane.DEV,
    )
    return calls


def _index_of(calls: list[list[str]], *words: str) -> int:
    return next(i for i, c in enumerate(calls) if all(w in c for w in words))


def test_the_executor_restores_a_wedged_runtime_by_restarting_it_before_compose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole restore step: the executor's roll over a wedged docker answer."""
    calls = _executor_roll(
        monkeypatch,
        {MAIN: "running unhealthy 72", f"{MAIN}-effects": "running healthy 0"},
    )

    restarts = [c for c in calls if c[:2] == ["docker", "restart"]]
    assert restarts == [["docker", "restart", MAIN]]
    assert _index_of(calls, "restart", MAIN) < _index_of(
        calls, "up", "omninode-runtime"
    )
    # The healthy effects container is rolled by compose alone.
    assert _index_of(calls, "up", "runtime-effects") > _index_of(
        calls, "up", "omninode-runtime"
    )


def test_the_executor_run_returns_a_non_zero_exit_instead_of_raising() -> None:
    """The roll's read of an absent container relies on this: no exception breaks the roll."""
    result = executor_mod._run([sys.executable, "-c", "raise SystemExit(3)"], 30)

    assert result.returncode == 3


def test_the_executor_roll_goes_on_when_the_container_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _executor_roll(monkeypatch, {})

    assert not [c for c in calls if c[:2] == ["docker", "restart"]]
    assert _index_of(calls, "up", "omninode-runtime") < _index_of(
        calls, "up", "runtime-effects"
    )
