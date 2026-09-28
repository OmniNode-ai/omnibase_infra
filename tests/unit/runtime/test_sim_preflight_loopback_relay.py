# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reject changed identity and egress before any relay connection."""

from __future__ import annotations

import asyncio
import os
import signal
import stat
import struct
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from omnibase_infra.runtime import sim_preflight_loopback_relay as relay
from omnibase_infra.runtime.sim_preflight_loopback_relay import (
    NETWORK,
    PROJECT,
    ModelSimPreflightRelayReceipt,
    validate_inspection,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field", ["Config", "HostConfig", "State", "NetworkSettings", "Mounts"]
)
def test_malformed_docker_structures_fail_closed(field: str) -> None:
    containers, network = fixture()
    containers[0][field] = "not-an-object-or-list"
    with pytest.raises(ValueError, match="malformed"):
        validate_inspection(containers, network, services=("postgres", "redpanda"))


@pytest.mark.unit
def test_external_docker_hashes_remain_constrained() -> None:
    _, ids = validate_inspection(*fixture(), services=("postgres", "redpanda"))
    for field in ("network_id", "transport_image_id"):
        fields = {
            "pid": 1,
            "network_id": "a" * 64,
            "transport_image_id": "sha256:" + "b" * 64,
            "container_ids": ids,
            "services": tuple(relay.ENDPOINTS),
            "nonce": "c" * 64,
        }
        fields[field] = "not-a-docker-hash"
        with pytest.raises(ValueError):
            ModelSimPreflightRelayReceipt.model_validate(fields)


@pytest.mark.unit
def test_sigterm_sets_graceful_shutdown_event(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = Mock()
    stop = asyncio.Event()
    monkeypatch.setattr(relay.signal, "getsignal", lambda _: signal.SIG_IGN)
    restore_signal = Mock()
    monkeypatch.setattr(relay.signal, "signal", restore_signal)
    restore = relay._install_shutdown_signals(loop, stop)
    callbacks = {
        call.args[0]: call.args[1] for call in loop.add_signal_handler.call_args_list
    }
    assert set(callbacks) == {signal.SIGTERM, signal.SIGINT}
    callbacks[signal.SIGTERM]()
    assert stop.is_set()
    restore()
    assert loop.remove_signal_handler.call_count == 2
    assert restore_signal.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_bounds_stuck_writer_and_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stuck() -> None:
        await asyncio.Future()

    monkeypatch.setattr(relay, "CLEANUP_TIMEOUT_SECONDS", 0.01, raising=False)
    writer = Mock()
    writer.wait_closed = AsyncMock(side_effect=stuck)
    await asyncio.wait_for(relay._close_writer(writer), 0.2)
    writer.transport.abort.assert_called_once()
    process = Mock(returncode=None)
    process.communicate = AsyncMock(side_effect=stuck)
    await asyncio.wait_for(relay._stop_process(process), 0.2)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_retains_child_when_handler_cleanup_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = Mock(returncode=None)
    active = {process}
    entered = asyncio.Event()

    async def handler() -> None:
        try:
            entered.set()
            await asyncio.Future()
        finally:
            active.discard(process)

    task = asyncio.create_task(handler())
    await entered.wait()
    stop_process = AsyncMock()
    monkeypatch.setattr(relay, "_stop_process", stop_process)
    await asyncio.wait_for(relay._cleanup_connections({task}, active), 0.2)
    assert task.done()
    assert not active
    stop_process.assert_awaited_once_with(process)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_bounds_handler_that_delays_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler() -> None:
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(handler())
    await entered.wait()
    process = Mock(returncode=None)
    stop_process = AsyncMock()
    monkeypatch.setattr(relay, "_stop_process", stop_process)
    monkeypatch.setattr(relay, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    try:
        await asyncio.wait_for(relay._cleanup_connections({task}, {process}), 0.2)
        stop_process.assert_awaited_once_with(process)
    finally:
        release.set()
        await task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_has_one_cleanup_reader_for_actual_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "print('ready',flush=True); time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    await asyncio.wait_for(process.stdout.readline(), 2)
    monkeypatch.setattr(relay, "CLEANUP_TIMEOUT_SECONDS", 0.03)
    cleanups: dict[asyncio.subprocess.Process, asyncio.Task[None]] = {}
    active = {process}
    calls = 0
    original = relay._stop_process

    async def stop(child: asyncio.subprocess.Process) -> None:
        nonlocal calls
        calls += 1
        await original(child)

    monkeypatch.setattr(relay, "_stop_process", stop)
    try:
        task = asyncio.create_task(relay._cleanup_process_once(process, cleanups))
        await asyncio.sleep(0)
        await asyncio.wait_for(relay._cleanup_connections({task}, active, cleanups), 1)
        assert calls == 1
        assert process.returncode is not None
    finally:
        if process.returncode is None:
            process.kill()
        await process.communicate()


@pytest.mark.unit
def test_inspection_remains_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock(return_value=SimpleNamespace(stdout="[]"))
    monkeypatch.setattr(relay.subprocess, "run", run)
    assert relay._inspect("container", ["fixed-container"]) == []
    assert run.call_args.kwargs["timeout"] == relay.DOCKER_INSPECT_TIMEOUT_SECONDS == 30
    assert (
        relay.CONTROL_RESPONSE_TIMEOUT_SECONDS
        == 2 * relay.DOCKER_INSPECT_TIMEOUT_SECONDS + 5
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "fault",
    ["none", "identity", "transport", "loopback", "ipv6", "missing", "unselected"],
)
def test_connection_uses_validated_subset_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    _, receipt = private_receipt(tmp_path)
    containers, network = fixture()
    selected = [containers[0], containers[2]]
    selected[0]["NetworkSettings"]["Networks"][NETWORK]["IPAddress"] = "172.18.0.5"
    if fault == "identity":
        selected[0]["Id"] = "d" * 64
    elif fault == "transport":
        selected[1]["Image"] = "sha256:" + "e" * 64
    elif fault == "loopback":
        selected[0]["NetworkSettings"]["Networks"][NETWORK]["IPAddress"] = "127.0.0.1"
    elif fault == "ipv6":
        selected[0]["NetworkSettings"]["Networks"][NETWORK]["IPAddress"] = "fd00::1"
    elif fault == "missing":
        selected[0]["NetworkSettings"]["Networks"][NETWORK].pop("IPAddress")
    calls: list[tuple[str, list[str]]] = []

    def inspect(kind: str, names: list[str]) -> list[dict[str, Any]]:
        calls.append((kind, names))
        return selected if kind == "container" else [network]

    monkeypatch.setattr(relay, "_inspect", inspect)
    if fault == "none":
        assert relay.inspect_connection_target(receipt, "postgres") == "172.18.0.5"
        assert calls == [
            ("container", [f"{PROJECT}-postgres", f"{PROJECT}-relay-transport"]),
            ("network", [NETWORK]),
        ]
    else:
        with pytest.raises(ValueError):
            relay.inspect_connection_target(
                receipt, "keycloak" if fault == "unselected" else "postgres"
            )
        if fault == "unselected":
            assert calls == []


@pytest.mark.unit
def test_cli_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "relay",
            "--private-directory",
            "/private/tmp/example",
            "--service",
            "keycloak",
            "--transport-image-id",
            "sha256:" + "b" * 64,
        ],
    )
    monkeypatch.setattr(
        relay, "serve", AsyncMock(side_effect=ValueError("private-token-example"))
    )
    with pytest.raises(SystemExit) as failure:
        relay.main()
    assert failure.value.code == 65
    assert "sim_relay_startup_refused" in caplog.text
    assert "private-token-example" not in caplog.text


def private_receipt(tmp_path: Path) -> tuple[Path, ModelSimPreflightRelayReceipt]:
    tmp_path.chmod(0o700)
    _, ids = validate_inspection(*fixture(), services=("postgres", "redpanda"))
    receipt = ModelSimPreflightRelayReceipt(
        pid=os.getpid(),
        network_id="a" * 64,
        container_ids=ids,
        transport_image_id="sha256:" + "b" * 64,
        services=("postgres", "redpanda"),
        nonce="c" * 64,
    )
    path = tmp_path / "relay-receipt.json"
    path.write_text(receipt.model_dump_json())
    path.chmod(0o600)
    return path, receipt


class FakeControl:
    def __init__(self, response: bytes, peer_pid: int) -> None:
        self.parts = [response[:17], response[17:], b""]
        self.peer_pid = peer_pid

    def __enter__(self) -> FakeControl:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def settimeout(self, _: int) -> None:
        assert _ == relay.CONTROL_RESPONSE_TIMEOUT_SECONDS

    def connect(self, _: str) -> None:
        pass

    def sendall(self, _: bytes) -> None:
        pass

    def getsockopt(self, *args: int) -> bytes:
        return (
            struct.pack("3i", self.peer_pid, os.getuid(), 0)
            if args[-1] == 12
            else struct.pack("i", self.peer_pid)
        )

    def recv(self, _: int) -> bytes:
        return self.parts.pop(0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "peer",
        "truncated",
        "public-file",
        "missing-service",
        "public-socket",
        "regular-socket",
        "foreign-socket",
    ],
)
def test_live_receipt_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    path, receipt = private_receipt(tmp_path)
    response = receipt.model_dump_json().encode()
    if fault == "truncated":
        response = response[:-5]
    if fault == "public-file":
        path.chmod(0o644)
    control = FakeControl(response, receipt.pid + (1 if fault == "peer" else 0))
    original_lstat = Path.lstat

    def fake_lstat(item: Path) -> Any:
        if item.name != "relay-control.sock":
            return original_lstat(item)
        mode = (stat.S_IFREG if fault == "regular-socket" else stat.S_IFSOCK) | (
            0o644 if fault == "public-socket" else 0o600
        )
        return SimpleNamespace(
            st_mode=mode, st_uid=os.getuid() + (1 if fault == "foreign-socket" else 0)
        )

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    monkeypatch.setattr(relay.socket, "socket", lambda *_: control)
    calls: list[ModelSimPreflightRelayReceipt] = []
    monkeypatch.setattr(relay, "inspect_targets", lambda item: calls.append(item))
    required = ("onex-api",) if fault == "missing-service" else ("postgres",)
    if fault == "none":
        assert relay.verify_live_relay_receipt(path, required) == receipt
        assert calls == [receipt]
    else:
        with pytest.raises(ValueError):
            relay.verify_live_relay_receipt(path, required)
        assert calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_output_race_preserves_other_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.chmod(0o700)
    _, ids = validate_inspection(*fixture(), services=("postgres", "redpanda"))
    monkeypatch.setattr(relay, "inspect_targets", lambda *_, **__: ("a" * 64, ids))
    server = type(
        "Server", (), {"close": lambda self: None, "wait_closed": AsyncMock()}
    )()
    monkeypatch.setattr(relay.asyncio, "start_server", AsyncMock(return_value=server))
    monkeypatch.setattr(
        relay.asyncio, "start_unix_server", AsyncMock(return_value=server)
    )

    class FakeSocket:
        def bind(self, name: str) -> None:
            Path(name).write_text("")

        def setblocking(self, _: bool) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(relay.socket, "socket", lambda *_: FakeSocket())

    def race(path: str, *_: object) -> int:
        Path(path).write_text("other process receipt")
        raise FileExistsError

    monkeypatch.setattr(relay.os, "open", race)
    with pytest.raises(FileExistsError):
        await relay.serve(tmp_path, ("postgres", "redpanda"), "sha256:" + "b" * 64)
    assert (tmp_path / "relay-receipt.json").read_text() == "other process receipt"
    assert not (tmp_path / "relay-control.sock").exists()


def fixture() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    network = {
        "Id": "a" * 64,
        "Name": NETWORK,
        "Internal": True,
        "Driver": "bridge",
        "Labels": {"com.docker.compose.project": PROJECT},
    }
    containers = []
    for index, service in enumerate(("postgres", "redpanda", "relay-transport")):
        containers.append(
            {
                "Id": str(index + 1) * 64,
                "Image": "sha256:" + "b" * 64,
                "Name": f"/{PROJECT}-{service}",
                "Config": {
                    "User": "1000:1000",
                    "Entrypoint": ["python"],
                    "Cmd": ["-c", "import signal; signal.pause()"],
                    "Labels": {
                        "com.docker.compose.project": PROJECT,
                        "com.docker.compose.service": service,
                    },
                },
                "State": {"Running": True, "Status": "running"},
                "NetworkSettings": {
                    "Networks": {NETWORK: {"NetworkID": "a" * 64, "Gateway": ""}}
                },
                "HostConfig": {
                    "ReadonlyRootfs": True,
                    "CapDrop": ["ALL"],
                    "Dns": ["127.0.0.1"],
                    "SecurityOpt": ["no-new-privileges:true"],
                },
                "Mounts": [],
            }
        )
    return containers, network


@pytest.mark.unit
def test_exact_subset() -> None:
    containers, network = fixture()
    net, ids = validate_inspection(
        containers, network, services=("postgres", "redpanda")
    )
    receipt = ModelSimPreflightRelayReceipt(
        pid=1,
        network_id=net,
        container_ids=ids,
        transport_image_id="sha256:" + "b" * 64,
        services=("postgres", "redpanda"),
        nonce="c" * 64,
    )
    assert validate_inspection(containers, network, receipt, receipt.services) == (
        net,
        ids,
    )
    containers[0]["Id"] = "d" * 64
    with pytest.raises(ValueError, match="pinned identity"):
        validate_inspection(containers, network, receipt, receipt.services)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation",
    [
        "egress",
        "network",
        "project",
        "privileged",
        "socket",
        "writable",
        "caps",
        "image",
        "stopped",
        "lan-bind",
        "ipv6-egress",
        "cap-add",
        "entrypoint",
        "command",
        "dns",
        "transport-publish",
    ],
)
def test_refuses_unsafe_targets(mutation: str) -> None:
    containers, network = fixture()
    _, ids = validate_inspection(containers, network, services=("postgres", "redpanda"))
    receipt = ModelSimPreflightRelayReceipt(
        pid=1,
        network_id="a" * 64,
        container_ids=ids,
        transport_image_id="sha256:" + "b" * 64,
        services=("postgres", "redpanda"),
        nonce="c" * 64,
    )
    if mutation == "egress":
        network["Internal"] = False
    elif mutation == "network":
        containers[0]["NetworkSettings"]["Networks"]["other"] = {}
    elif mutation == "project":
        containers[0]["Config"]["Labels"]["com.docker.compose.project"] = "shared"
    elif mutation == "privileged":
        containers[0]["HostConfig"]["Privileged"] = True
    elif mutation == "socket":
        containers[2]["Mounts"] = [{"Destination": "/var/run/docker.sock"}]
    elif mutation == "writable":
        containers[2]["HostConfig"]["ReadonlyRootfs"] = False
    elif mutation == "caps":
        containers[2]["HostConfig"]["CapDrop"] = []
    elif mutation == "image":
        containers[2]["Image"] = "sha256:" + "e" * 64
    elif mutation == "stopped":
        containers[0]["State"]["Running"] = False
    elif mutation == "lan-bind":
        containers[0]["HostConfig"]["PortBindings"] = {
            "5432/tcp": [{"HostIp": "192.0.2.1", "HostPort": "65036"}]
        }
    elif mutation == "ipv6-egress":
        containers[0]["NetworkSettings"]["Networks"][NETWORK]["IPv6Gateway"] = "fd00::1"
    elif mutation == "cap-add":
        containers[2]["HostConfig"]["CapAdd"] = ["NET_ADMIN"]
    elif mutation == "entrypoint":
        containers[2]["Config"]["Entrypoint"] = ["sh"]
    elif mutation == "command":
        containers[2]["Config"]["Cmd"] = ["-c", "other"]
    elif mutation == "dns":
        containers[2]["HostConfig"]["Dns"] = ["8.8.8.8"]
    elif mutation == "transport-publish":
        containers[2]["HostConfig"]["PortBindings"] = {
            "8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "9000"}]
        }
    with pytest.raises(ValueError):
        validate_inspection(containers, network, receipt, receipt.services)


@pytest.mark.unit
@pytest.mark.parametrize("services", [(), ("postgres", "postgres"), ("arbitrary",)])
def test_refuses_unbounded_selection(services: tuple[str, ...]) -> None:
    containers, network = fixture()
    with pytest.raises(ValueError):
        validate_inspection(deepcopy(containers), network, services=services)
