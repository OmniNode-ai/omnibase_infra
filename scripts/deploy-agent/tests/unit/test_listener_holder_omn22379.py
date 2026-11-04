# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22379 AC1: the deploy agent's listener holder keeps the runtime host listeners.

A connection queued on a held listener is answered after the agent re-execs and
after a systemd-style restart, because the listener is recovered by name from
``LISTEN_FDS`` / ``LISTEN_FDNAMES``. Real sockets throughout; the re-exec test
runs a real ``os.execv`` in a subprocess. systemd cannot be started in CI, so
``_SystemdDouble`` plays its two roles: it receives ``FDSTORE=1`` datagrams with
the descriptors, and it starts a process with ``LISTEN_PID`` set to that
process's own pid and the stored descriptors on 3, 4 in ``LISTEN_FDNAMES`` order.
"""

from __future__ import annotations

import array
import contextlib
import fcntl
import json
import os
import signal
import socket
import sys
import textwrap
import time
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pytest
from deploy_agent.agent import DeployAgent

import omnibase_infra
from omnibase_infra.handlers.handler_runtime_listener_holder import (
    HandlerRuntimeListenerHolder,
)
from omnibase_infra.models.model_runtime_listener_holder_request import (
    ModelRuntimeListenerHolderRequest,
)
from omnibase_infra.models.model_runtime_listener_holder_result import (
    ModelRuntimeListenerHolderResult,
)
from omnibase_infra.models.model_runtime_listener_registry import (
    ModelRuntimeListenerRegistry,
)

pytestmark = pytest.mark.unit


def _holder(environ: MutableMapping[str, str]) -> HandlerRuntimeListenerHolder:
    return HandlerRuntimeListenerHolder(ModelRuntimeListenerRegistry.assemble(environ))


_UNIT = Path(__file__).resolve().parents[2] / "deploy" / "deploy-agent-dev.service"
_SRC = str(Path(omnibase_infra.__file__).resolve().parent.parent)

_CHILD = textwrap.dedent(
    """
    import json, os, sys
    from omnibase_infra.handlers.handler_runtime_listener_holder import (
        HandlerRuntimeListenerHolder,
    )
    from omnibase_infra.models.model_runtime_listener_holder_request import (
        ModelRuntimeListenerHolderRequest as Request,
    )
    from omnibase_infra.models.model_runtime_listener_registry import (
        ModelRuntimeListenerRegistry,
    )

    def say(result):
        print(json.dumps({"outcome": result.outcome, "pools": list(result.pools),
                          "ports": list(result.ports), "pid": os.getpid()}), flush=True)

    def answer_one_per_pool(holder):
        for pool in holder.fd_store.pools():
            conn, _ = holder.fd_store.socket_for(pool).accept()
            conn.recv(4096)
            conn.sendall(("HTTP/1.0 200 OK\\r\\n\\r\\n%s:%d" % (pool, os.getpid())).encode())
            conn.close()

    mode = sys.argv[1]
    services = ModelRuntimeListenerRegistry.assemble(os.environ)
    holder = HandlerRuntimeListenerHolder(services)
    if mode == "bind":
        say(holder.handle(Request(operation="install_bind", main_port=0, effects_port=0)))
        sys.stdin.readline()
        if sys.argv[2] == "reexec":
            services.reexec(sys.executable, [sys.executable, __file__, "serve"])
    elif mode == "serve":
        result = holder.handle(Request(operation="recover"))
        say(result)
        if result.ok:
            answer_one_per_pool(holder)
    """
)


@dataclass(frozen=True)
class _Report:
    outcome: str
    pools: list[str]
    ports: list[int]
    pid: int


class _Child:
    """A spawned process with a stdin pipe and a stdout pipe."""

    def __init__(self, pid: int, stdin: IO[str], stdout: IO[str]) -> None:
        self.pid = pid
        self.stdin = stdin
        self.stdout = stdout

    def report(self) -> _Report:
        line = self.stdout.readline()
        assert line, "child produced no output"
        raw = json.loads(line)
        return _Report(raw["outcome"], raw["pools"], raw["ports"], raw["pid"])

    def say(self, text: str) -> None:
        self.stdin.write(text)
        self.stdin.flush()

    def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.pid, signal.SIGKILL)

    def wait(self, timeout: float = 30.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                pid, status = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                return -1
            if pid:
                return os.waitstatus_to_exitcode(status)
            time.sleep(0.02)
        raise AssertionError("child did not exit")


def _spawn(
    argv: list[str], env: dict[str, str], descriptors: dict[int, int] | None = None
) -> _Child:
    """posix_spawn with stdin/stdout pipes; ``descriptors`` maps target fd to source fd."""
    stdin_read, stdin_write = os.pipe()
    stdout_read, stdout_write = os.pipe()
    actions = [
        (os.POSIX_SPAWN_DUP2, stdin_read, 0),
        (os.POSIX_SPAWN_DUP2, stdout_write, 1),
        *(
            (os.POSIX_SPAWN_DUP2, source, target)
            for target, source in (descriptors or {}).items()
        ),
    ]
    pid = os.posix_spawn(argv[0], argv, env, file_actions=actions)
    os.close(stdin_read)
    os.close(stdout_write)
    return _Child(pid, os.fdopen(stdin_write, "w"), os.fdopen(stdout_read, "r"))


@pytest.fixture
def child_script(tmp_path: Path) -> Path:
    script = tmp_path / "holder_child.py"
    script.write_text(_CHILD)
    return script


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LISTEN_")}
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra)
    return env


def _queue_request(port: int) -> socket.socket:
    """Connect over IPv4 (the listener is dual-stack) and send, with nobody accepting."""
    client = socket.create_connection(("127.0.0.1", port), timeout=15)
    client.sendall(b"GET /ready HTTP/1.1\r\nHost: x\r\n\r\n")
    return client


def _queue_both(ports: list[int]) -> dict[str, socket.socket]:
    main_port, effects_port = ports
    return {
        "runtime-main": _queue_request(main_port),
        "runtime-effects": _queue_request(effects_port),
    }


def _answer(client: socket.socket) -> str:
    chunks = []
    while chunk := client.recv(4096):
        chunks.append(chunk)
    return b"".join(chunks).decode()


class _SystemdDouble:
    """Receives FDSTORE=1 with descriptors; starts a service with LISTEN_FDS."""

    def __init__(self, directory: Path) -> None:
        self.path = str(directory / "notify.sock")
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.bind(self.path)
        self._sock.settimeout(15)
        self.store: dict[str, int] = {}

    def collect(self, count: int) -> None:
        for _ in range(count):
            data, ancillary, _, _ = self._sock.recvmsg(
                1024, socket.CMSG_SPACE(array.array("i", [0]).itemsize)
            )
            fields = dict(line.split("=", 1) for line in data.decode().split("\n"))
            assert fields["FDSTORE"] == "1"
            payload = ancillary[0][2]
            fd = array.array("i", payload[: array.array("i").itemsize])[0]
            # park above the range the service will receive them on
            self.store[fields["FDNAME"]] = fcntl.fcntl(fd, fcntl.F_DUPFD, 50)
            os.close(fd)

    def start(self, script: Path, names: tuple[str, ...], mode: str) -> _Child:
        """Start the service as systemd does: LISTEN_PID is the service's own pid."""
        first, second = (self.store[name] for name in names)
        return _spawn(
            [
                "/bin/sh",
                "-c",
                'LISTEN_PID=$$ LISTEN_FDS=2 LISTEN_FDNAMES="$0" exec "$1" "$2" "$3"',
                ":".join(names),
                sys.executable,
                str(script),
                mode,
            ],
            _env(),
            {3: first, 4: second},
        )

    def close(self) -> None:
        self._sock.close()
        for fd in self.store.values():
            os.close(fd)


def _binder(script: Path, mode: str, **env: str) -> _Child:
    return _spawn([sys.executable, str(script), "bind", mode], _env(**env))


def test_listener_holder_connection_queued_across_reexec_is_answered(
    child_script: Path,
) -> None:
    proc = _binder(child_script, "reexec")
    try:
        bound = proc.report()
        assert bound.outcome == "bound:no_fd_store"
        clients = _queue_both(bound.ports)
        proc.say("go\n")
        recovered = proc.report()  # printed by the re-exec'd image
        assert recovered.outcome == "recovered"
        assert recovered.pid == bound.pid, "a re-exec keeps the pid"
        assert recovered.pools == ["runtime-main", "runtime-effects"]
        assert recovered.ports == bound.ports
        for pool, client in clients.items():
            response = _answer(client)
            assert response.startswith("HTTP/1.0 200 OK"), response
            assert response.endswith(f"{pool}:{bound.pid}"), response
        assert proc.wait() == 0
    finally:
        proc.kill()
        proc.wait()


def test_listener_holder_connection_queued_across_systemd_restart_is_answered(
    child_script: Path, tmp_path: Path
) -> None:
    systemd = _SystemdDouble(tmp_path)
    first = _binder(child_script, "park", NOTIFY_SOCKET=systemd.path)
    second: _Child | None = None
    try:
        bound = first.report()
        assert bound.outcome == "bound", "the fd store accepted both listeners"
        systemd.collect(2)
        assert sorted(systemd.store) == ["runtime-effects", "runtime-main"]
        first.kill()  # the agent dies; only the manager's copies of the sockets remain
        first.wait()
        clients = _queue_both(bound.ports)
        # The manager restarts the agent with the stored descriptors, effects first:
        # recovery is by name, not by position.
        second = systemd.start(
            child_script, ("runtime-effects", "runtime-main"), "serve"
        )
        recovered = second.report()
        assert recovered.outcome == "recovered"
        assert recovered.pid != bound.pid, "a restart is a new process"
        assert sorted(recovered.pools) == ["runtime-effects", "runtime-main"]
        assert recovered.ports == bound.ports, "same sockets, matched to their names"
        for pool, client in clients.items():
            response = _answer(client)
            assert response.startswith("HTTP/1.0 200 OK"), response
            assert response.endswith(f"{pool}:{recovered.pid}"), response
    finally:
        first.kill()
        first.wait()
        if second is not None:
            second.kill()
            second.wait()
        systemd.close()


def test_listener_holder_reexec_remap_swaps_descriptors_without_losing_either(
    child_script: Path, tmp_path: Path
) -> None:
    """Stored order effects,main puts effects on 3; the remap must put main there."""
    systemd = _SystemdDouble(tmp_path)
    binder = _binder(child_script, "park", NOTIFY_SOCKET=systemd.path)
    swap_script = tmp_path / "swap_child.py"
    swap_script.write_text(
        _CHILD
        + textwrap.dedent(
            """
            if mode == "swap":
                say(holder.handle(Request(operation="recover")))
                holder.prepare_for_reexec()
                os.execv(sys.executable, [sys.executable, __file__, "serve"])
            """
        )
    )
    proc: _Child | None = None
    try:
        bound = binder.report()
        systemd.collect(2)
        binder.kill()
        binder.wait()
        clients = _queue_both(bound.ports)
        proc = systemd.start(swap_script, ("runtime-effects", "runtime-main"), "swap")
        assert proc.report().outcome == "recovered"
        again = proc.report()  # after the real execv
        assert again.outcome == "recovered"
        assert again.pools == ["runtime-main", "runtime-effects"]
        assert again.ports == bound.ports
        for pool, client in clients.items():
            assert _answer(client).endswith(f"{pool}:{again.pid}")
    finally:
        binder.kill()
        binder.wait()
        if proc is not None:
            proc.kill()
            proc.wait()
        systemd.close()


def test_listener_holder_ignores_descriptors_meant_for_another_pid() -> None:
    environ = {
        "LISTEN_FDS": "2",
        "LISTEN_FDNAMES": "runtime-main:runtime-effects",
        "LISTEN_PID": str(os.getpid() + 1),
    }
    holder = _holder(environ)
    result = holder.handle(ModelRuntimeListenerHolderRequest(operation="recover"))
    assert result.outcome == "inert:listen_pid_mismatch"
    assert result.pools == ()
    assert holder.fd_store.pools() == ()


def test_listener_holder_refuses_a_stored_descriptor_that_does_not_listen(
    child_script: Path, tmp_path: Path
) -> None:
    systemd = _SystemdDouble(tmp_path)
    left, right = socket.socketpair()
    systemd.store["runtime-main"] = fcntl.fcntl(left.fileno(), fcntl.F_DUPFD, 50)
    systemd.store["runtime-effects"] = fcntl.fcntl(right.fileno(), fcntl.F_DUPFD, 50)
    proc = systemd.start(child_script, ("runtime-main", "runtime-effects"), "serve")
    try:
        result = proc.report()
        assert result.outcome == "refused:not_a_listener:runtime-main"
        assert result.pools == []
    finally:
        proc.kill()
        proc.wait()
        left.close()
        right.close()
        systemd.close()


def test_listener_holder_binds_dual_stack_with_reuseaddr_and_policy_backlog() -> None:
    holder = _holder({})
    try:
        result = holder.handle(
            ModelRuntimeListenerHolderRequest(
                operation="install_bind", main_port=0, effects_port=0, backlog=77
            )
        )
        assert result.ok and result.outcome == "bound:no_fd_store"
        assert result.pools == ("runtime-main", "runtime-effects")
        for pool in result.pools:
            sock = holder.fd_store.socket_for(pool)
            assert sock is not None
            assert sock.family == socket.AF_INET6
            assert sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 1
            assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
            assert not sock.get_inheritable()
            with socket.create_connection(("127.0.0.1", sock.getsockname()[1])):
                pass  # an IPv4 client reaches the [::] listener
    finally:
        holder.handle(ModelRuntimeListenerHolderRequest(operation="drop"))
    assert holder.fd_store.pools() == ()


def _rebind(
    holder: HandlerRuntimeListenerHolder, record: Path, effects_port: int = 0
) -> ModelRuntimeListenerHolderResult:
    return holder.handle(
        ModelRuntimeListenerHolderRequest(
            operation="reboot_rebind",
            install_record_path=str(record),
            main_port=0,
            effects_port=effects_port,
        )
    )


def test_listener_holder_rebinds_after_reboot_only_for_a_completed_install_and_free_ports(
    tmp_path: Path,
) -> None:
    record = tmp_path / "install.json"
    holder = _holder({})
    assert _rebind(holder, record).outcome == "inert:no_install_record"
    record.write_text(json.dumps({"state": "in_progress"}))
    assert _rebind(holder, record).outcome == "inert:install_not_completed"
    record.write_text("{not json")
    assert _rebind(holder, record).outcome == "refused:install_record_unreadable"
    assert holder.fd_store.pools() == (), (
        "nothing bound while the record does not read completed"
    )

    record.write_text(json.dumps({"state": "completed"}))
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as squatter:
        squatter.bind(("::", 0))
        squatter.listen(1)
        refused = _rebind(holder, record, effects_port=squatter.getsockname()[1])
        assert refused.outcome == "refused:port_in_use" and not refused.ok
        assert holder.fd_store.pools() == (), "an all-or-nothing bind never keeps half"
    try:
        bound = _rebind(holder, record)
        assert bound.ok and bound.pools == ("runtime-main", "runtime-effects")
        assert _rebind(holder, record).outcome == "inert:already_held"
    finally:
        holder.handle(ModelRuntimeListenerHolderRequest(operation="drop"))


def test_listener_holder_is_inert_until_an_install_binds(tmp_path: Path) -> None:
    assert ModelRuntimeListenerRegistry.start({}) is None, "no unit key, no socket"
    handover = tmp_path / "handover"
    services = ModelRuntimeListenerRegistry.start(
        {"DEV_RUNTIME_HANDOVER_DIR": str(handover)}
    )
    assert services is not None
    holder = HandlerRuntimeListenerHolder(services)
    assert holder.fd_store.pools() == ()
    assert not (handover / "broker.sock").exists(), "no broker before an install binds"
    result = holder.handle(
        ModelRuntimeListenerHolderRequest(operation="prepare_reexec")
    )
    assert result.outcome == "inert:nothing_held"


def test_listener_holder_agent_wiring_is_a_noop_without_listeners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DeployAgent, "_settle_in_flight", lambda self: False)
    monkeypatch.setattr(DeployAgent, "drain_settle", lambda self, timeout=None: None)
    calls: list[str] = []

    class _Holder:
        def prepare_for_reexec(self) -> None:
            calls.append("prepare_for_reexec")

    agent = object.__new__(DeployAgent)
    agent._runtime_listeners = None
    agent._await_settle_before_reexec()
    assert calls == []
    agent._runtime_listeners = _Holder()
    agent._await_settle_before_reexec()
    assert calls == [], "settling must not remap descriptors before an exec decision"


def test_listener_holder_dev_unit_declares_the_fd_store_keys() -> None:
    service: dict[str, str] = {}
    in_service = False
    for raw in _UNIT.read_text().splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_service = line == "[Service]"
        elif in_service and "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            service.setdefault(key, value)
    assert service["FileDescriptorStoreMax"] == "4"
    assert service["FileDescriptorStorePreserve"] == "yes"
    assert service["NotifyAccess"] == "all"
    assert "WatchdogSec" in service


@pytest.mark.parametrize("failure", ["stage", "dup2", "exec"])
def test_listener_holder_failed_reexec_preserves_fds_and_environment(
    failure: str,
) -> None:
    import subprocess

    code = r"""
import fcntl, json, os, socket, sys
import omnibase_infra.handlers.handler_runtime_listener_holder as module
from omnibase_infra.nodes.node_runtime_listener_holder_effect import runtime_listener_holder as io_module
from omnibase_infra.models.model_runtime_listener_registry import ModelRuntimeListenerRegistry
from omnibase_infra.models.model_runtime_listener_holder_request import ModelRuntimeListenerHolderRequest
# Occupy the target descriptors with unrelated listeners, just as an agent's
# event loop or health server can occupy them. Keep their queues intact too.
targets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(2)]
for s in targets:
    s.bind(("127.0.0.1", 0)); s.listen(8)
assert [s.fileno() for s in targets] == [3, 4]
env = {"LISTEN_PID": "before", "LISTEN_FDS": "old", "LISTEN_FDNAMES": "old"}
services = ModelRuntimeListenerRegistry.assemble(env)
holder = module.HandlerRuntimeListenerHolder(services)
assert holder.handle(ModelRuntimeListenerHolderRequest(operation="install_bind", main_port=0, effects_port=0)).ok
held = [holder.fd_store.socket_for(p) for p in ("runtime-main", "runtime-effects")]
all_sockets = targets + held
before = [(s.fileno(), s.getsockname(), s.get_inheritable()) for s in all_sockets]
clients = [socket.create_connection(("127.0.0.1", s.getsockname()[1]), timeout=3) for s in all_sockets]
for c in clients: c.sendall(b"queued")
original_fcntl, original_dup2 = io_module.fcntl.fcntl, io_module.os.dup2
calls = 0
failure = sys.argv[1]
def stage(fd, command, arg=0):
    global calls
    if command == fcntl.F_DUPFD_CLOEXEC:
        calls += 1
        if calls == 2: raise OSError("injected staging failure")
    return original_fcntl(fd, command, arg)
def remap(fd, target, *, inheritable=True):
    global calls
    calls += 1
    if calls == 2: raise OSError("injected remap failure")
    return original_dup2(fd, target, inheritable=inheritable)
if failure == "stage": io_module.fcntl.fcntl = stage
if failure == "dup2": io_module.os.dup2 = remap
try:
    if failure == "exec":
        services.reexec("/this-executable-does-not-exist", ["missing"])
    else:
        holder.prepare_for_reexec()
except OSError:
    pass
else:
    raise AssertionError("fault did not fire")
finally:
    io_module.fcntl.fcntl, io_module.os.dup2 = original_fcntl, original_dup2
assert env == {"LISTEN_PID": "before", "LISTEN_FDS": "old", "LISTEN_FDNAMES": "old"}, env
assert [(s.fileno(), s.getsockname(), s.get_inheritable()) for s in all_sockets] == before
for s in all_sockets:
    s.settimeout(3)
    accepted, _ = s.accept()
    assert accepted.recv(6) == b"queued"
    accepted.close()
for c in clients: c.close()
for s in all_sockets: s.close()
# fcntl copies use the high staging range and must all have been closed.
for fd in range(100, 120):
    try: os.fstat(fd)
    except OSError: continue
    raise AssertionError(f"leaked descriptor {fd}")
print("original descriptors, flags, queues and environment preserved")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, failure],
        env=_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "original descriptors, flags, queues and environment preserved" in result.stdout
    )


@pytest.mark.parametrize("restart", ["none", "succeeded", "failed"])
def test_listener_holder_executor_remaps_only_at_the_exec_boundary(
    monkeypatch: pytest.MonkeyPatch,
    declare_loaded_code_sha: Callable[[str], None],
    restart: str,
) -> None:
    import subprocess

    import deploy_agent.executor as module
    from deploy_agent.events import EnumSelfUpdateBoundary
    from deploy_agent.executor import DeployExecutor

    declare_loaded_code_sha("a" * 40)
    monkeypatch.setenv("DEPLOY_AGENT_MODE", "host")
    monkeypatch.delenv("DEPLOY_AGENT_NO_SELF_UPDATE", raising=False)
    monkeypatch.delenv("DEPLOY_AGENT_LAUNCHER", raising=False)
    events: list[str] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        value = "b" * 40 if "rev-parse" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout=value, stderr="")

    def restart_unit(*args: object, **kwargs: object) -> None:
        events.append("restart")
        if restart == "failed":
            raise OSError("restart refused")

    def execute(path: str, argv: list[str]) -> None:
        assert path == sys.executable and argv[0] == sys.executable
        events.append("reexec")

    monkeypatch.setattr(module, "_run", run)
    monkeypatch.setattr(
        module,
        "_sync_own_unit_after_pull",
        lambda _: None if restart == "none" else "agent.service",
    )
    monkeypatch.setattr(module, "_uv_sync_after_pull", lambda *_: None)
    monkeypatch.setattr(module.subprocess, "run", restart_unit)
    executor = DeployExecutor()
    executor.reexec = execute
    executor.self_update(
        boundary=EnumSelfUpdateBoundary.POST_TERMINAL,
        on_before_reexec=lambda: events.append("settle"),
    )
    expected = {
        "none": ["settle", "reexec"],
        "succeeded": ["settle", "restart"],
        "failed": ["settle", "restart", "settle", "reexec"],
    }
    assert events == expected[restart]
