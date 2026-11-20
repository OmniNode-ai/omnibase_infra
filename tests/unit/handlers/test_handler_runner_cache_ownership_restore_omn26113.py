# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26113: the cache ownership restore changes ownership of idle runners' declared cache mounts, and only those.

A root-owned cache mount fails every job's first npm or pre-commit call with EACCES. The restore chowns the declared
cache bind mounts of a container to the runner user, and refuses to write anywhere a ``Runner.Worker`` is running, the
process list is unreadable, or no ``Runner.Listener`` is there to prove the container is a runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from omnibase_infra.handlers.handler_runner_cache_ownership_restore import (
    DockerCacheCli,
    HandlerRunnerCacheOwnershipRestore,
)
from omnibase_infra.models.model_runner_cache_ownership_restore_request import (
    ModelRunnerCacheOwnershipRestoreRequest,
)
from omnibase_infra.nodes.node_deploy_runners.runtime_runner_cache_ownership_restore import (
    main,
)

pytestmark = pytest.mark.unit

HOME = "/home/runner"
SCRATCH = "tmp"
CACHE_MOUNTS = [
    f"{HOME}/.npm",
    f"{HOME}/.cache/pip",
    f"{HOME}/.cache/pre-commit",
    f"{HOME}/.cache/pnpm",
    f"{HOME}/.cache/ms-playwright",
    f"{HOME}/.cache/omni/ci-envs",
]
# Binds that are not caches: the restore must never name one of them.
OTHER_BINDS = [
    f"{HOME}/actions-runner/_work",
    f"{HOME}/{SCRATCH}",
    f"{HOME}/.lab-credentials",
    f"{HOME}/.onex-disk-admission-pause",
    "/var/run/docker.sock",
    "/opt/onex-runner-hooks",
]
IDLE = ["bash", "Runner.Listener", "ps"]
BUSY = ["bash", "Runner.Listener", "Runner.Worker", "node"]


@dataclass
class FakeContainer:
    processes: list[str] | None
    owner: int = 0  # who owns the cache mounts now
    binds: list[str] = field(default_factory=lambda: [*CACHE_MOUNTS, *OTHER_BINDS])
    job_starts_after_reads: int | None = (
        None  # the Worker appears after this many process reads
    )
    chown_has_effect: bool = True


class FakeDocker(DockerCacheCli):
    """docker replaced by an in-memory host that records every write."""

    def __init__(self, containers: dict[str, FakeContainer]) -> None:
        self.containers = containers
        self.chown_calls: list[tuple[str, int, int, list[str]]] = []
        self._reads: dict[str, int] = {}

    def container_names(self) -> list[str]:
        return sorted([*self.containers, "unrelated-service"])

    def bind_destinations(self, name: str) -> list[str]:
        return list(self.containers[name].binds)

    def process_names(self, name: str) -> list[str] | None:
        box = self.containers[name]
        self._reads[name] = self._reads.get(name, 0) + 1
        if box.processes is None:
            return None
        if (
            box.job_starts_after_reads is not None
            and self._reads[name] > box.job_starts_after_reads
        ):
            return BUSY
        return box.processes

    def unowned_entry(self, name: str, uid: int, gid: int, paths: list[str]) -> str:
        return paths[0] if self.containers[name].owner != uid else ""

    def chown(self, name: str, uid: int, gid: int, paths: list[str]) -> None:
        self.chown_calls.append((name, uid, gid, paths))
        if self.containers[name].chown_has_effect:
            self.containers[name].owner = uid


def _run(docker: FakeDocker):
    class Injected(HandlerRunnerCacheOwnershipRestore):
        def make_docker(self, request: ModelRunnerCacheOwnershipRestoreRequest):
            return docker

    return Injected().handle(
        ModelRunnerCacheOwnershipRestoreRequest(container_match="^hp051-")
    )


def test_runner_cache_ownership_restore_chowns_an_idle_runners_declared_cache_mounts_only() -> (
    None
):
    docker = FakeDocker({"hp051-omnibase-ci-runner-9": FakeContainer(IDLE)})
    result = _run(docker)

    assert result.outcome == "ok"
    assert [f.outcome for f in result.findings] == ["restored"]
    assert docker.containers["hp051-omnibase-ci-runner-9"].owner == 1001
    [(name, uid, gid, paths)] = docker.chown_calls
    assert (name, uid, gid) == ("hp051-omnibase-ci-runner-9", 1001, 1001)
    assert sorted(paths) == sorted(CACHE_MOUNTS)
    for other in (*OTHER_BINDS, HOME, f"{HOME}/.cache"):
        assert other not in paths


def test_runner_cache_ownership_restore_never_runs_under_a_running_runner_worker() -> (
    None
):
    """The falsifier: a restore that changes ownership under a running Runner.Worker."""
    docker = FakeDocker({"hp051-omnibase-ci-runner-14": FakeContainer(BUSY)})
    result = _run(docker)

    assert docker.chown_calls == []
    assert docker.containers["hp051-omnibase-ci-runner-14"].owner == 0
    assert result.busy == ("hp051-omnibase-ci-runner-14",)
    assert result.outcome == "ok"


def test_runner_cache_ownership_restore_treats_an_unreadable_process_list_as_busy() -> (
    None
):
    docker = FakeDocker({"hp051-omnibase-ci-runner-1": FakeContainer(None)})
    result = _run(docker)

    assert docker.chown_calls == []
    assert result.busy == ("hp051-omnibase-ci-runner-1",)


def test_runner_cache_ownership_restore_needs_the_listener_to_call_a_container_idle() -> (
    None
):
    """No Runner.Worker is not enough: a process list with no Runner.Listener is not a running runner."""
    docker = FakeDocker({"hp051-omnibase-ci-runner-2": FakeContainer(["bash", "ps"])})
    result = _run(docker)

    assert docker.chown_calls == []
    assert result.busy == ("hp051-omnibase-ci-runner-2",)


def test_runner_cache_ownership_restore_rereads_the_process_list_right_before_the_chown() -> (
    None
):
    """A job that starts between the first idle read and the chown stops the chown."""
    box = FakeContainer(IDLE, job_starts_after_reads=1)
    docker = FakeDocker({"hp051-omnibase-ci-runner-3": box})
    result = _run(docker)

    assert docker.chown_calls == []
    assert box.owner == 0
    assert result.busy == ("hp051-omnibase-ci-runner-3",)


def test_runner_cache_ownership_restore_leaves_an_already_owned_runner_unwritten() -> (
    None
):
    docker = FakeDocker({"hp051-omnibase-ci-runner-4": FakeContainer(IDLE, owner=1001)})
    result = _run(docker)

    assert docker.chown_calls == []
    assert result.already_owned == ("hp051-omnibase-ci-runner-4",)


def test_runner_cache_ownership_restore_skips_a_container_with_no_cache_mounts() -> (
    None
):
    docker = FakeDocker(
        {"hp051-ci-mq-runner-1": FakeContainer(IDLE, binds=list(OTHER_BINDS))}
    )
    result = _run(docker)

    assert docker.chown_calls == []
    assert result.no_cache_mounts == ("hp051-ci-mq-runner-1",)


def test_runner_cache_ownership_restore_busy_runner_does_not_block_an_idle_sibling() -> (
    None
):
    """Positive control for the zero-chown assertions above: the same handler does write where it may."""
    docker = FakeDocker(
        {
            "hp051-omnibase-ci-runner-5": FakeContainer(BUSY),
            "hp051-omnibase-ci-runner-6": FakeContainer(IDLE),
        }
    )
    result = _run(docker)

    assert [c[0] for c in docker.chown_calls] == ["hp051-omnibase-ci-runner-6"]
    assert result.busy == ("hp051-omnibase-ci-runner-5",)


def test_runner_cache_ownership_restore_reports_a_chown_that_did_not_hold() -> None:
    docker = FakeDocker(
        {"hp051-omnibase-ci-runner-7": FakeContainer(IDLE, chown_has_effect=False)}
    )
    result = _run(docker)

    assert result.outcome == "failed"
    assert result.findings[0].outcome == "failed"


def test_runner_cache_ownership_restore_cli_prints_the_result_and_exits_by_outcome(
    capsys: pytest.CaptureFixture[str],
) -> None:
    docker = FakeDocker(
        {
            "hp051-omnibase-ci-runner-8": FakeContainer(IDLE),
            "hp051-omnibase-ci-runner-9": FakeContainer(BUSY),
        }
    )
    code = main(
        [
            "--container-match",
            "^hp051-",
            "--docker-timeout",
            "60",
            "--chown-timeout",
            "900",
        ],
        docker=docker,
    )

    line = capsys.readouterr().out.strip()
    prefix = "[runner-cache-ownership-restore] "
    assert code == 0
    assert line.startswith(prefix)
    payload = json.loads(line[len(prefix) :])
    assert payload["busy"] == ["hp051-omnibase-ci-runner-9"]
    assert payload["findings"][0]["container"] == "hp051-omnibase-ci-runner-8"
