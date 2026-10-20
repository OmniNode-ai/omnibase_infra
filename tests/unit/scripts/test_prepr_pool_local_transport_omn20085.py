# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Local pool commands and host routing without ssh (OMN-20085)."""

from __future__ import annotations

import getpass
import importlib.util
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PY = REPO_ROOT / "scripts" / "runtime_build" / "prepr_runtime_pool.py"

pytestmark = pytest.mark.unit


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pool = _load("prepr_pool_local_transport_omn20085", POOL_PY)


def _host(target: str = "127.0.0.1", name: str = "local-slot") -> Any:
    return pool.PoolHost(
        name=name,
        ssh_target=target,
        surface=name,
        os="linux",
        status="pool",
    )


def test_is_this_host_checks_address_and_user() -> None:
    user = getpass.getuser()
    other = "someoneelse" if user != "someoneelse" else "anotheruser"
    assert pool.is_this_host(f"{user}@127.0.0.1")
    assert not pool.is_this_host(f"{other}@127.0.0.1")
    assert not pool.is_this_host(f"{user}@192.0.2.55")
    assert pool.is_this_host("127.0.0.1")
    assert not pool.is_this_host(f"{user}@not-an-ip.example")
    assert not pool.is_this_host(f"{other}@127.0.0.1", bind=lambda a: True)


def test_is_this_host_accepts_injected_user_and_bind() -> None:
    checked = []

    def bind(addr: str) -> bool:
        checked.append(addr)
        return True

    assert pool.is_this_host("lane@192.0.2.55", user="lane", bind=bind)
    assert not pool.is_this_host("other@192.0.2.55", user="lane", bind=bind)
    assert checked == ["192.0.2.55"]


def test_local_run_captures_output_and_exit_code() -> None:
    rc, out = pool.LocalTransport().run(
        _host(), "echo hi; echo err >&2; exit 3", timeout=5
    )
    assert rc == 3
    assert "hi" in out
    assert "err" in out


def test_local_run_starts_in_home() -> None:
    rc, out = pool.LocalTransport().run(_host(), "pwd", timeout=5)
    assert rc == 0
    assert out.strip() == str(Path.home())


def test_local_run_does_not_leak_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_LEAK_MARKER", "x")
    rc, out = pool.LocalTransport().run(
        _host(), 'echo "[${ONEX_LEAK_MARKER:-}]"', timeout=5
    )
    assert rc == 0
    assert out.strip() == "[]"


def test_local_run_times_out_and_preserves_output() -> None:
    started = time.monotonic()
    rc, out = pool.LocalTransport().run(
        _host(), "echo before; echo error >&2; sleep 30", timeout=1
    )
    assert time.monotonic() - started < 10
    assert rc == 124
    assert "before" in out
    assert "error" in out
    assert "[timeout after 1s]" in out


def test_local_timeout_kills_background_child(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    rc, out = pool.LocalTransport().run(
        _host(), f"sleep 300 & echo $! > {shlex.quote(str(pidfile))}; wait", timeout=1
    )
    assert rc == 124
    assert "[timeout after 1s]" in out
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_local_put_copies_and_reports_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("pool script\n", encoding="utf-8")
    dest = tmp_path / "dest"
    transport = pool.LocalTransport()
    assert transport.put(_host(), source, str(dest)) == 0
    assert dest.read_bytes() == source.read_bytes()
    assert transport.put(_host(), source, str(tmp_path / "missing" / "dest")) == 1


class RecordingTransport:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[tuple[Any, ...]] = []

    def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
        self.calls.append(("run", host, command, timeout))
        return 7, self.label

    def put(self, host: Any, local: Path, remote: str) -> int:
        self.calls.append(("put", host, local, remote))
        return 8


def test_routing_delegates_and_caches_each_target(tmp_path: Path) -> None:
    local = RecordingTransport("local")
    ssh = RecordingTransport("ssh")
    checked = []

    def is_local(target: str) -> bool:
        checked.append(target)
        return target == "127.0.0.1"

    transport = pool.HostRoutingTransport(ssh=ssh, local=local, is_local=is_local)
    source = tmp_path / "source"
    dest = str(tmp_path / "dest")
    for target, expected in (("127.0.0.1", local), ("192.0.2.55", ssh)):
        first = _host(target, "slot-1")
        second = _host(target, "slot-2")
        assert transport.run(first, "echo test", timeout=3) == (7, expected.label)
        assert transport.put(second, source, dest) == 8
        assert transport.run(second, "pwd", timeout=5) == (7, expected.label)
        assert expected.calls == [
            ("run", first, "echo test", 3),
            ("put", second, source, dest),
            ("run", second, "pwd", 5),
        ]
    assert checked == ["127.0.0.1", "192.0.2.55"]


def test_main_constructs_routing_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "pool.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "slot": {
                    "compose_project": "omnibase-infra-local",
                    "ports": [8085, 8086],
                    "lease_dir": ".onex-prepr-pool/lease",
                    "work_root_prefix": "prove-pool-",
                    "model_endpoint": "http://192.0.2.10:8000",
                    "max_load_ratio": 1.0,
                },
                "hosts": [
                    {
                        "name": "local-slot",
                        "ssh_target": "127.0.0.1",
                        "surface": "local-slot",
                        "os": "linux",
                        "status": "pool",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    constructed = []

    def routing() -> RecordingTransport:
        transport = RecordingTransport("fixed probe failure")
        constructed.append(transport)
        return transport

    monkeypatch.setattr(pool, "HostRoutingTransport", routing)
    monkeypatch.delenv("ONEX_LEDGER_PATH", raising=False)
    assert pool.main(["--config", str(config), "status"]) == pool.EXIT_PASS
    assert len(constructed) == 1
    assert len(constructed[0].calls) == 1
    assert constructed[0].calls[0][0] == "run"
    output = capsys.readouterr().out
    assert "OFFLINE" in output
    assert "fixed probe failure" in output
