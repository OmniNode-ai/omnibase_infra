# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22151: a changed dev runtime is replaced start-first.

Deploy-agent job 0c915304 (2026-11-02) replaced the dev runtime stop-first and
:8085/ready read refused and 503 for about three minutes. The start-first effect
starts the replacement beside the old container, stops the old one only after
the replacement reads /ready 200, and when the replacement never gets there it
removes it, leaves the old one running and fails with a named reason.

The effect is loaded from this checkout beside the agent, the way the agent
loads its other host effect, and driven over the agent's own executor module
with ``_run`` answering Docker the way a dev host does. No Docker daemon is
touched.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest
from deploy_agent import executor as executor_mod

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra"
OLD = "omnibase-infra-omninode-runtime"
NEW = f"{OLD}-start-first"
SERVICE = "omninode-runtime"
READY_URL = "http://localhost:8085/ready"
COMPOSE = ["docker", "compose", "-p", "omnibase-infra"]


def _load_effect() -> ModuleType:
    name = "omnibase_infra.handlers.handler_runtime_start_first"
    # Sibling source models are needed while executing the handler's imports.
    # Restore the installed package's cache, including on a failed load.
    with pytest.MonkeyPatch.context() as patch:
        for relative in (
            "models/model_runtime_start_first_request.py",
            "models/model_runtime_start_first_result.py",
            "handlers/handler_runtime_start_first.py",
        ):
            module_name = "omnibase_infra." + relative.removesuffix(".py").replace(
                "/", "."
            )
            spec = importlib.util.spec_from_file_location(module_name, SRC / relative)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, module_name, module)
            spec.loader.exec_module(module)
        return sys.modules[name]


@dataclass
class FakeDevHost:
    """Containers on a fake dev host, with a clock the poll advances."""

    ready_after_probes: int | None = 3
    exits_after_probes: int | None = None
    start_exit: int = 0
    stop_ignored: bool = False
    running: set[str] = field(default_factory=lambda: {OLD})
    events: list[str] = field(default_factory=list)
    old_running_at_first_200: bool | None = None
    probes: int = 0
    now: float = 0.0
    argv: list[list[str]] = field(default_factory=list)

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def run(
        self,
        cmd: list[str],
        timeout: int,
        *,
        cwd: object = None,
        env: object = None,
    ) -> subprocess.CompletedProcess[str]:
        self.argv.append(cmd)
        out, code = "", 0
        if cmd[:2] == ["docker", "inspect"]:
            out = "true" if cmd[-1] in self.running else ""
            code = 0 if cmd[-1] in self.running else 1
        elif cmd[: len(COMPOSE)] == COMPOSE and "run" in cmd:
            name = cmd[cmd.index("--name") + 1]
            self.events.append(f"start {name}")
            code = self.start_exit
            if code == 0:
                self.running.add(name)
        elif cmd[:2] == ["docker", "exec"]:
            out = str(self._ready(cmd[2]))
        elif cmd[:2] == ["docker", "stop"]:
            self.events.append(f"stop {cmd[-1]}")
            if not self.stop_ignored:
                self.running.discard(cmd[-1])
        elif cmd[:3] == ["docker", "rm", "-f"]:
            self.events.append(f"remove {cmd[-1]}")
            self.running.discard(cmd[-1])
        else:
            raise AssertionError(f"unexpected command {cmd}")
        return subprocess.CompletedProcess(cmd, code, stdout=out, stderr="")

    def _ready(self, container: str) -> int:
        self.probes += 1
        if (
            self.exits_after_probes is not None
            and self.probes >= self.exits_after_probes
        ):
            self.running.discard(container)
        if container not in self.running:
            return 0
        if (
            self.ready_after_probes is not None
            and self.probes >= self.ready_after_probes
        ):
            if self.old_running_at_first_200 is None:
                self.old_running_at_first_200 = OLD in self.running
                self.events.append("first 200")
            return 200
        return 503


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> FakeDevHost:
    fake = FakeDevHost()
    monkeypatch.setattr(executor_mod, "_run", fake.run)
    monkeypatch.setattr(executor_mod, "_compose_argv", lambda lane: list(COMPOSE))
    monkeypatch.setattr(executor_mod, "_compose_env", lambda extra=None: {})
    return fake


def _replace(host: FakeDevHost, bound: float = 60.0) -> object:
    effect = _load_effect()
    handler = effect.HandlerRuntimeStartFirst(
        execution=executor_mod, clock=host.clock, sleep=host.sleep
    )
    return handler.handle(
        effect.ModelRuntimeStartFirstRequest(
            service=SERVICE,
            old_container=OLD,
            ready_url=READY_URL,
            bound_seconds=bound,
            poll_seconds=5,
        )
    )


def test_runtime_start_first_old_still_running_when_replacement_first_ready(
    host: FakeDevHost,
) -> None:
    host.ready_after_probes = 4
    result = _replace(host)
    assert result.status == "pass"
    # The fake host read the old container's state at the first 200.
    assert host.old_running_at_first_200 is True
    assert result.old_running_at_first_ready is True
    assert result.probes == 4
    assert host.running == {NEW}


def test_runtime_start_first_stops_old_only_after_first_ready(
    host: FakeDevHost,
) -> None:
    _replace(host)
    started = host.events.index(f"start {NEW}")
    ready = host.events.index("first 200")
    stopped = host.events.index(f"stop {OLD}")
    assert started < ready < stopped
    assert host.events.count(f"stop {OLD}") == 1


def test_runtime_start_first_replacement_publishes_no_host_port(
    host: FakeDevHost,
) -> None:
    _replace(host)
    start = next(cmd for cmd in host.argv if "run" in cmd)
    assert start == [*COMPOSE, "run", "-d", "--no-deps", "--name", NEW, SERVICE]
    assert "--service-ports" not in start
    probe = next(cmd for cmd in host.argv if cmd[:2] == ["docker", "exec"])
    assert probe[:3] == ["docker", "exec", NEW]
    assert probe[-1] == "http://localhost:8085/ready"


def test_runtime_start_first_never_ready_removes_replacement_keeps_old(
    host: FakeDevHost,
) -> None:
    host.ready_after_probes = None
    result = _replace(host, bound=60.0)
    assert result.status == "fail"
    assert result.failure == "replacement-never-ready"
    assert host.running == {OLD}
    assert f"stop {OLD}" not in host.events
    assert host.events[-1] == f"remove {NEW}"
    # The poll ended inside the declared bound.
    assert host.now <= 60.0


def test_runtime_start_first_never_ready_when_replacement_exits(
    host: FakeDevHost,
) -> None:
    host.ready_after_probes = None
    host.exits_after_probes = 2
    result = _replace(host)
    assert result.failure == "replacement-exited-before-ready"
    assert host.running == {OLD}
    assert f"stop {OLD}" not in host.events


def test_runtime_start_first_never_ready_when_replacement_does_not_start(
    host: FakeDevHost,
) -> None:
    host.start_exit = 1
    result = _replace(host)
    assert result.failure == "replacement-not-started"
    assert host.running == {OLD}
    assert f"stop {OLD}" not in host.events


def test_runtime_start_first_refuses_when_old_is_not_running(
    host: FakeDevHost,
) -> None:
    host.running = set()
    result = _replace(host)
    assert result.failure == "old-runtime-not-running"
    assert host.events == []


def test_runtime_start_first_clears_a_leftover_replacement_first(
    host: FakeDevHost,
) -> None:
    host.running = {OLD, NEW}
    _replace(host)
    assert host.events[:2] == [f"remove {NEW}", f"start {NEW}"]


def test_runtime_start_first_fails_when_old_does_not_stop(
    host: FakeDevHost,
) -> None:
    host.stop_ignored = True
    result = _replace(host)
    assert result.status == "fail"
    assert result.failure == "old-runtime-not-stopped"


@pytest.mark.parametrize("already_loaded", [False, True])
@pytest.mark.parametrize("load_fails", [False, True])
def test_runtime_start_first_loader_restores_module_cache(
    monkeypatch: pytest.MonkeyPatch, already_loaded: bool, load_fails: bool
) -> None:
    names = (
        "omnibase_infra.models.model_runtime_start_first_request",
        "omnibase_infra.models.model_runtime_start_first_result",
        "omnibase_infra.handlers.handler_runtime_start_first",
    )
    for name in names:
        if already_loaded:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        else:
            monkeypatch.delitem(sys.modules, name, raising=False)
    before = {name: sys.modules.get(name) for name in names}
    if load_fails:
        real_spec = importlib.util.spec_from_file_location

        def spec(name: str, location: Path) -> object:
            if name == names[-1]:
                raise ImportError("handler load failed")
            return real_spec(name, location)

        monkeypatch.setattr(importlib.util, "spec_from_file_location", spec)
        with pytest.raises(ImportError, match="handler load failed"):
            _load_effect()
    else:
        loaded = _load_effect()
        assert loaded.HandlerRuntimeStartFirst is not None
    assert {name: sys.modules.get(name) for name in names} == before


@pytest.mark.parametrize(
    "error",
    [
        OSError("Docker unavailable"),
        subprocess.CalledProcessError(1, "docker"),
        subprocess.TimeoutExpired("docker", 15),
    ],
)
def test_runtime_start_first_probe_error_keeps_old_and_removes_replacement(
    host: FakeDevHost, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def run(cmd: list[str], *args: object, **kwargs: object) -> object:
        if cmd[:2] == ["docker", "exec"]:
            raise error
        return host.run(cmd, *args, **kwargs)

    monkeypatch.setattr(executor_mod, "_run", run)
    result = _replace(host, bound=10)
    assert result.failure == "replacement-never-ready"
    assert host.running == {OLD}
    assert f"stop {OLD}" not in host.events


@pytest.mark.parametrize(
    "error", [OSError("Docker unavailable"), subprocess.TimeoutExpired("compose", 300)]
)
def test_runtime_start_first_start_error_keeps_old_and_removes_replacement(
    host: FakeDevHost, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def run(cmd: list[str], *args: object, **kwargs: object) -> object:
        if cmd[: len(COMPOSE)] == COMPOSE and "run" in cmd:
            # A start can time out after Docker has created the container.
            host.running.add(NEW)
            raise error
        return host.run(cmd, *args, **kwargs)

    monkeypatch.setattr(executor_mod, "_run", run)
    result = _replace(host)
    assert result.failure == "replacement-not-started"
    assert host.running == {OLD}
    assert f"stop {OLD}" not in host.events


def test_runtime_start_first_name_collision_never_adopts_stale_replacement(
    host: FakeDevHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host.running.add(NEW)

    def run(cmd: list[str], *args: object, **kwargs: object) -> object:
        if cmd[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="busy")
        if cmd[: len(COMPOSE)] == COMPOSE and "run" in cmd:
            assert NEW in host.running
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="name in use")
        return host.run(cmd, *args, **kwargs)

    monkeypatch.setattr(executor_mod, "_run", run)
    result = _replace(host)
    assert result.failure == "replacement-not-started"
    assert OLD in host.running
    assert host.probes == 0
    assert f"stop {OLD}" not in host.events


@pytest.mark.parametrize(("output", "exit_code"), [("garbage", 0), ("200", 1)])
def test_runtime_start_first_failed_probe_does_not_stop_old(
    host: FakeDevHost,
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    exit_code: int,
) -> None:
    def run(cmd: list[str], *args: object, **kwargs: object) -> object:
        if cmd[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, exit_code, stdout=output, stderr="")
        return host.run(cmd, *args, **kwargs)

    monkeypatch.setattr(executor_mod, "_run", run)
    result = _replace(host, bound=10)
    assert result.failure == "replacement-never-ready"
    assert host.running == {OLD}
    assert f"stop {OLD}" not in host.events
