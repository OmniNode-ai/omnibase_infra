# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A recreate retry waits for dockerd's removals and leaves nothing Dead (OMN-18475).

Measured on the compose dev lane, idle-converge job ``77dcfaf4`` at
2026-10-22T01:00Z, load1 48 on 32 cores:

1. ``up -d --force-recreate forward-migration`` timed out at 300 s and left the
   new container Created as ``a9570f27d439_omnibase-infra-forward-migration``;
2. the OMN-20864 cleanup's ``docker rm -f`` of it timed out at 60 s, and dockerd
   went on removing it;
3. the retry ran at once, compose tried to recreate from that container, and
   dockerd answered ``removal of container 5da809605ebd... is already in
   progress``. The job failed and left a Dead and a Created container.

``_FakeDaemon`` reproduces that sequence: a removal the client gave up on keeps
going for a number of polls, and a recreate that meets it fails the same way.
"""

from __future__ import annotations

import itertools
import subprocess
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import (
    REMOVAL_IN_PROGRESS,
    DeployExecutor,
    lane_config_for,
    run_compose_recreate,
    settle_service_containers,
)

pytestmark = pytest.mark.unit

PROJECT = lane_config_for(EnumRuntimeLane.DEV).compose_project
SERVICE = "forward-migration"
REAL_NAME = f"{PROJECT}-{SERVICE}"
UP = ["docker", "compose", "-p", PROJECT, "up", "-d", "--force-recreate", SERVICE]


@dataclass
class _Container:
    id: str
    name: str
    state: str
    service: str
    project: str = PROJECT
    removal_polls_left: int = 0


@dataclass
class _FakeDaemon:
    """Just enough of dockerd and compose to replay the 01:00Z race."""

    containers: list[_Container] = field(default_factory=list)
    up_hangs: int = 0
    rm_hangs: int = 0
    removal_polls: int = 2
    ups: list[list[str]] = field(default_factory=list)
    _ids: Any = field(default_factory=lambda: itertools.count(0x5DA809605EBD))

    def _new_id(self) -> str:
        return f"{next(self._ids):012x}"

    def tick(self, _seconds: float = 0) -> None:
        for container in list(self.containers):
            if container.state == "removing":
                container.removal_polls_left -= 1
                if container.removal_polls_left <= 0:
                    self.containers.remove(container)

    def of(self, service: str) -> list[_Container]:
        return [c for c in self.containers if c.service == service]

    def __call__(
        self, cmd: list[str], timeout: int, **_kw: object
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["docker", "compose"]:
            return self._up(cmd, timeout)
        if cmd[:3] == ["docker", "ps", "-a"]:
            return self._ps(cmd)
        if cmd[:3] == ["docker", "rm", "-f"]:
            return self._rm(cmd, timeout)
        # The migration preflight's projection-table probe: the tables exist.
        stdout = "t\n" if "omnidash_analytics" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    def _up(self, cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        self.ups.append(list(cmd))
        service = cmd[-1]
        if self.up_hangs:
            self.up_hangs -= 1
            for old in self.of(service):
                self.containers.remove(old)
            temp_id = self._new_id()
            self.containers.append(
                _Container(temp_id, f"a9570f27d439_{REAL_NAME}", "created", service)
            )
            raise subprocess.TimeoutExpired(cmd, timeout)
        for old in self.of(service):
            if old.state == "removing":
                old.state = "dead"
                return subprocess.CompletedProcess(
                    cmd,
                    1,
                    stdout="",
                    stderr=(
                        f"Error response from daemon: removal of container "
                        f"{old.id} {REMOVAL_IN_PROGRESS}"
                    ),
                )
        for old in self.of(service):
            self.containers.remove(old)
        self.containers.append(_Container(self._new_id(), REAL_NAME, "exited", service))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _ps(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        rows = [
            c
            for c in self.containers
            if f"label=com.docker.compose.project={c.project}" in cmd
        ]
        if "status=created" in cmd:
            out = "".join(
                f"{c.id}\t{c.name}\t{c.service}\n" for c in rows if c.state == "created"
            )
        else:
            out = "".join(f"{c.id}\t{c.name}\t{c.state}\t{c.service}\n" for c in rows)
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    def _rm(self, cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        target = next((c for c in self.containers if c.id == cmd[3]), None)
        if target is None:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if target.state == "removing":
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=f"removal of container {target.id} {REMOVAL_IN_PROGRESS}",
            )
        if self.rm_hangs:
            self.rm_hangs -= 1
            target.state = "removing"
            target.removal_polls_left = self.removal_polls
            raise subprocess.TimeoutExpired(cmd, timeout)
        self.containers.remove(target)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def _drive(daemon: _FakeDaemon) -> Any:
    return (
        patch.object(executor_mod, "_run", side_effect=daemon),
        patch.object(executor_mod, "_removal_wait_sleep", side_effect=daemon.tick),
    )


def test_replay_77dcfaf4_the_retry_waits_for_the_removal_and_succeeds() -> None:
    """RED on the tree before OMN-18475: the retry met the removal and failed."""
    daemon = _FakeDaemon(up_hangs=1, rm_hangs=1, removal_polls=2)
    run, sleep = _drive(daemon)

    with run, sleep:
        result = run_compose_recreate(
            UP, timeout=300, lane=EnumRuntimeLane.DEV, services=[SERVICE]
        )

    assert result.returncode == 0, result.stderr
    assert len(daemon.ups) == 2
    assert [(c.name, c.state) for c in daemon.of(SERVICE)] == [(REAL_NAME, "exited")]


def test_an_attempt_that_meets_a_removal_in_progress_is_settled_and_retried() -> None:
    daemon = _FakeDaemon(
        containers=[
            _Container(
                "5da809605ebd",
                f"a9570f27d439_{REAL_NAME}",
                "removing",
                SERVICE,
                removal_polls_left=3,
            )
        ]
    )
    run, sleep = _drive(daemon)

    with run, sleep:
        result = run_compose_recreate(
            UP, timeout=300, lane=EnumRuntimeLane.DEV, services=[SERVICE]
        )

    assert result.returncode == 0, result.stderr
    assert len(daemon.ups) == 2
    assert [(c.name, c.state) for c in daemon.of(SERVICE)] == [(REAL_NAME, "exited")]


def test_a_failing_recreate_leaves_no_dead_or_created_container() -> None:
    """The removal outlives the wait: the job fails, and still cleans up."""
    daemon = _FakeDaemon(
        containers=[
            _Container("0000000000aa", REAL_NAME, "dead", SERVICE),
            _Container("0000000000bb", f"a9570f27d439_{REAL_NAME}", "created", SERVICE),
        ]
    )

    def failing(
        cmd: list[str], timeout: int, **kw: object
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["docker", "compose"]:
            daemon.ups.append(cmd)
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="image not found"
            )
        return daemon(cmd, timeout, **kw)

    with patch.object(executor_mod, "_run", side_effect=failing):
        removed: list[str] = []
        result = run_compose_recreate(
            UP,
            timeout=300,
            lane=EnumRuntimeLane.DEV,
            services=[SERVICE],
            removed=removed,
        )

    assert result.returncode == 1
    assert len(daemon.ups) == 1
    assert daemon.of(SERVICE) == []
    assert sorted(removed) == sorted([REAL_NAME, f"a9570f27d439_{REAL_NAME}"])


def test_settle_touches_only_dead_and_leftover_containers_of_the_named_services() -> (
    None
):
    daemon = _FakeDaemon(
        containers=[
            _Container("0000000000a1", REAL_NAME, "dead", SERVICE),
            _Container(
                "0000000000a2", f"{PROJECT}-migration-gate", "dead", "migration-gate"
            ),
            _Container(
                "0000000000a3", f"{PROJECT}-{SERVICE}-plain", "created", SERVICE
            ),
            _Container(
                "0000000000a4",
                "proof-forward-migration-1",
                "dead",
                SERVICE,
                project="omn18496-proof",
            ),
            _Container(
                "0000000000a5",
                f"{PROJECT}-omninode-runtime",
                "running",
                "omninode-runtime",
            ),
        ]
    )
    run, sleep = _drive(daemon)

    with run, sleep:
        settled = settle_service_containers(EnumRuntimeLane.DEV, [SERVICE])

    assert settled.removed == [REAL_NAME]
    assert settled.still_removing == []
    assert {c.id for c in daemon.containers} == {
        "0000000000a2",
        "0000000000a3",
        "0000000000a4",
        "0000000000a5",
    }


def test_a_removal_that_never_finishes_is_bounded() -> None:
    daemon = _FakeDaemon(
        containers=[
            _Container(
                "5da809605ebd", REAL_NAME, "removing", SERVICE, removal_polls_left=10**6
            )
        ]
    )
    sleeps: list[float] = []
    clock = itertools.count(0, executor_mod.CONTAINER_REMOVAL_POLL_SECONDS)

    with (
        patch.object(executor_mod, "_run", side_effect=daemon),
        patch.object(executor_mod, "_removal_wait_sleep", side_effect=sleeps.append),
        patch.object(executor_mod.time, "monotonic", side_effect=lambda: next(clock)),
    ):
        settled = settle_service_containers(
            EnumRuntimeLane.DEV, [SERVICE], wait_seconds=30
        )

    assert settled.still_removing == [REAL_NAME]
    assert len(sleeps) <= 30 / executor_mod.CONTAINER_REMOVAL_POLL_SECONDS + 1


def test_forward_migration_twice_back_to_back_without_the_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proof the lane runs on the lab, in miniature: run, run again at once."""
    monkeypatch.setattr(executor_mod, "DEV_LANE_ONLY_MIGRATION_SERVICES", ())
    monkeypatch.setattr(
        executor_mod, "verify_containers_up", lambda *_a, **_k: (True, [])
    )
    monkeypatch.setattr(executor_mod, "_compose_env", lambda *_a, **_k: {})
    # The first run hits the measured sequence; the second follows it at once.
    daemon = _FakeDaemon(up_hangs=1, rm_hangs=1, removal_polls=2)
    run, sleep = _drive(daemon)

    with run, sleep:
        DeployExecutor()._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV)
        DeployExecutor()._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV)

    assert [(c.name, c.state) for c in daemon.of(SERVICE)] == [(REAL_NAME, "exited")]
    assert not [
        c for c in daemon.containers if c.state in {"dead", "created", "removing"}
    ]
