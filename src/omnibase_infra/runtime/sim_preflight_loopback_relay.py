# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Temporary host TCP access to five pinned services on the internal sim bridge."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
from collections.abc import Callable
from ipaddress import IPv4Address
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

type DockerNetworkHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type DockerImageDigest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

PROJECT = "omnibase-infra-sim-preflight"
NETWORK = f"{PROJECT}-network"
TRANSPORT = "relay-transport"
DOCKER_INSPECT_TIMEOUT_SECONDS = 30
CONTROL_RESPONSE_TIMEOUT_SECONDS = 65
CLEANUP_TIMEOUT_SECONDS = 3
# No caller-supplied destination, listener address, command, or network.
ENDPOINTS = {
    "keycloak": (28080, 28080),
    "onex-api": (8090, 8000),
    "postgres": (65036, 5432),
    "redpanda": (65092, 19092),
    "valkey": (65379, 6379),
}


class ModelSimPreflightRelayReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    pid: int = Field(gt=0)
    network_id: DockerNetworkHash
    container_ids: dict[str, str]
    transport_image_id: DockerImageDigest
    services: tuple[str, ...]
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")


def _mapping(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("sim relay inspection object malformed")
    return value


def _sequence(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError("sim relay inspection list malformed")
    return value


def _inspect(kind: str, names: list[str]) -> list[dict[str, JsonValue]]:
    try:
        result = subprocess.run(
            ["docker", kind, "inspect", *names],
            check=True,
            capture_output=True,
            text=True,
            timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
        )
        return TypeAdapter(list[dict[str, JsonValue]]).validate_json(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError("sim relay Docker inspection failed") from exc


def validate_inspection(
    containers: list[dict[str, JsonValue]],
    network: dict[str, JsonValue],
    pinned: ModelSimPreflightRelayReceipt | None = None,
    services: tuple[str, ...] = tuple(ENDPOINTS),
) -> tuple[str, dict[str, str]]:
    """Fail closed on replacement containers, secondary networks, or routed egress."""
    network_id = network.get("Id")
    if (
        network.get("Name") != NETWORK
        or network.get("Internal") is not True
        or network.get("Driver") != "bridge"
        or not isinstance(network_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", network_id) is None
        or _mapping(network.get("Labels", {})).get("com.docker.compose.project")
        != PROJECT
    ):
        raise ValueError("sim relay network identity or isolation differs")
    if (
        not services
        or len(set(services)) != len(services)
        or not set(services) <= set(ENDPOINTS)
    ):
        raise ValueError("sim relay service selection differs")
    expected = {*services, TRANSPORT}
    if len(containers) != len(expected):
        raise ValueError("sim relay service inspection incomplete")
    ids: dict[str, str] = {}
    for item in containers:
        config = _mapping(item.get("Config", {}))
        labels = _mapping(config.get("Labels", {}))
        service = labels.get("com.docker.compose.service")
        container_id = item.get("Id")
        settings = _mapping(item.get("NetworkSettings", {}))
        networks = _mapping(settings.get("Networks", {}))
        state = _mapping(item.get("State", {}))
        host = _mapping(item.get("HostConfig", {}))
        if (
            not isinstance(service, str)
            or service not in expected
            or service in ids
            or labels.get("com.docker.compose.project") != PROJECT
            or item.get("Name") != f"/{PROJECT}-{service}"
            or not isinstance(container_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
            or state.get("Running") is not True
            or state.get("Status") != "running"
            or set(networks) != {NETWORK}
            or _mapping(networks[NETWORK]).get("NetworkID") != network_id
            or _mapping(networks[NETWORK]).get("Gateway") not in ("", None)
            or _mapping(networks[NETWORK]).get("IPv6Gateway") not in ("", None)
            or host.get("NetworkMode") in ("host", "none")
            or host.get("Privileged") is True
            or any(
                _mapping(binding).get("HostIp") != "127.0.0.1"
                for bindings in _mapping(host.get("PortBindings") or {}).values()
                for binding in _sequence(bindings or [])
            )
            or any(
                _mapping(m).get("Destination") == "/var/run/docker.sock"
                for m in _sequence(item.get("Mounts", []))
            )
        ):
            raise ValueError("sim relay container identity or isolation differs")
        if service == TRANSPORT:
            if (
                host.get("ReadonlyRootfs") is not True
                or host.get("CapDrop") != ["ALL"]
                or host.get("CapAdd")
                or host.get("Dns") != ["127.0.0.1"]
                or host.get("PortBindings")
                or any(_mapping(settings.get("Ports") or {}).values())
                or config.get("Entrypoint") != ["python"]
                or config.get("Cmd") != ["-c", "import signal; signal.pause()"]
                or "no-new-privileges:true"
                not in _sequence(host.get("SecurityOpt", []))
                or item.get("Mounts")
                or config.get("User") != "1000:1000"
                or (
                    pinned is not None
                    and item.get("Image") != pinned.transport_image_id
                )
            ):
                raise ValueError("sim relay transport restriction differs")
        ids[service] = container_id
    if pinned is not None and (
        pinned.network_id != network_id or pinned.container_ids != ids
    ):
        raise ValueError("sim relay pinned identity changed")
    return network_id, ids


def inspect_targets(
    pinned: ModelSimPreflightRelayReceipt | None = None,
    services: tuple[str, ...] = tuple(ENDPOINTS),
) -> tuple[str, dict[str, str]]:
    if pinned is not None:
        services = pinned.services
    containers = _inspect(
        "container", [f"{PROJECT}-{s}" for s in (*services, TRANSPORT)]
    )
    networks = _inspect("network", [NETWORK])
    if len(networks) != 1:
        raise ValueError("sim relay network inspection incomplete")
    return validate_inspection(containers, networks[0], pinned, services)


def inspect_connection_target(
    pinned: ModelSimPreflightRelayReceipt, service: str
) -> str:
    """Validate this destination and transport, and reuse their exact live snapshot."""
    if service not in pinned.services or service not in ENDPOINTS:
        raise ValueError("sim relay connection service differs")
    subset = pinned.model_copy(
        update={
            "services": (service,),
            "container_ids": {
                name: pinned.container_ids[name] for name in (service, TRANSPORT)
            },
        }
    )
    containers = _inspect(
        "container", [f"{PROJECT}-{service}", f"{PROJECT}-{TRANSPORT}"]
    )
    networks = _inspect("network", [NETWORK])
    if len(networks) != 1:
        raise ValueError("sim relay network inspection incomplete")
    validate_inspection(containers, networks[0], subset, (service,))
    target = next(
        item for item in containers if item["Id"] == pinned.container_ids[service]
    )
    destination = _mapping(
        _mapping(_mapping(target["NetworkSettings"])["Networks"])[NETWORK]
    ).get("IPAddress")
    if not isinstance(destination, str) or not destination:
        raise ValueError("sim relay target address missing")
    address = IPv4Address(destination)
    if address.is_loopback or address.is_unspecified or address.is_multicast:
        raise ValueError("sim relay target address differs")
    return str(address)


def _private_directory(path: Path) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("sim relay requires an absolute private directory")
    info = path.stat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_uid != os.getuid()
    ):
        raise ValueError("sim relay directory must be owned and mode 0700")


def verify_live_relay_receipt(
    path: Path, required_services: tuple[str, ...] = ("postgres", "redpanda")
) -> ModelSimPreflightRelayReceipt:
    """Challenge the running relay, then independently recheck pinned Docker state."""
    _private_directory(path.parent)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.getuid()
    ):
        raise ValueError("sim relay receipt must be owned and mode 0600")
    receipt = ModelSimPreflightRelayReceipt.model_validate_json(path.read_bytes())
    if set(receipt.container_ids) != {*receipt.services, TRANSPORT} or not set(
        required_services
    ) <= set(receipt.services):
        raise ValueError("sim relay receipt service set differs")
    control_path = path.parent / "relay-control.sock"
    control_info = control_path.lstat()
    if (
        not stat.S_ISSOCK(control_info.st_mode)
        or stat.S_IMODE(control_info.st_mode) != 0o600
        or control_info.st_uid != os.getuid()
    ):
        raise ValueError("sim relay control socket must be owned and mode 0600")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as control:
        control.settimeout(CONTROL_RESPONSE_TIMEOUT_SECONDS)
        control.connect(str(control_path))
        # macOS LOCAL_PEERPID (SOL_LOCAL=0, option=2); Linux SO_PEERCRED.
        if hasattr(socket, "SO_PEERCRED"):
            import struct

            peer_pid, peer_uid, _ = struct.unpack(
                "3i", control.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            if peer_pid != receipt.pid or peer_uid != os.getuid():
                raise ValueError("sim relay control peer differs")
        else:
            import struct

            peer_pid = struct.unpack("i", control.getsockopt(0, 2, 4))[0]
            if peer_pid != receipt.pid:
                raise ValueError("sim relay control peer differs")
        control.sendall((receipt.nonce + "\n").encode())
        chunks: list[bytes] = []
        total = 0
        while chunk := control.recv(4096):
            total += len(chunk)
            if total > 65536:
                raise ValueError("sim relay challenge exceeds bound")
            chunks.append(chunk)
        answer = b"".join(chunks)
    if ModelSimPreflightRelayReceipt.model_validate_json(answer) != receipt:
        raise ValueError("sim relay live challenge differs")
    inspect_targets(receipt)
    return receipt


# Executed only in the pinned Gateway container; arguments are fixed ENDPOINTS.
_BRIDGE = """import socket,sys,threading
s=socket.create_connection((sys.argv[1],int(sys.argv[2])),10)
s.settimeout(None)
def upload():
 try:
  while True:
   b=sys.stdin.buffer.read1(65536)
   if not b: break
   s.sendall(b)
 finally:
  s.shutdown(socket.SHUT_WR)
threading.Thread(target=upload,daemon=True).start()
while True:
 b=s.recv(65536)
 if not b: break
 sys.stdout.buffer.write(b);sys.stdout.buffer.flush()
s.close()
"""


def _install_shutdown_signals(
    loop: asyncio.AbstractEventLoop, stop: asyncio.Event
) -> Callable[[], None]:
    """Explicitly handle termination even when the launcher inherited ignored INT."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous:
        loop.add_signal_handler(sig, stop.set)

    def restore() -> None:
        for sig, handler in previous.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, handler)

    return restore


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), CLEANUP_TIMEOUT_SECONDS)
    except (TimeoutError, OSError):
        writer.transport.abort()
        logging.getLogger(__name__).log(
            logging.ERROR, "sim_relay_writer_cleanup_forced"
        )


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        # Drain pipe output as well as reaping the process; wait() alone can
        # remain blocked after exit when a cancelled reader left full pipes.
        await asyncio.wait_for(process.communicate(), CLEANUP_TIMEOUT_SECONDS)
    except (TimeoutError, OSError):
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.communicate(), CLEANUP_TIMEOUT_SECONDS)
        except (TimeoutError, OSError):
            logging.getLogger(__name__).log(
                logging.ERROR, "sim_relay_process_cleanup_incomplete"
            )


async def _cleanup_process_once(
    process: asyncio.subprocess.Process,
    cleanups: dict[asyncio.subprocess.Process, asyncio.Task[None]],
) -> None:
    task = cleanups.get(process)
    if task is None:
        task = asyncio.create_task(_stop_process(process))
        cleanups[process] = task
    await asyncio.shield(task)


async def _cleanup_connections(
    handlers: set[asyncio.Task[None]],
    active: set[asyncio.subprocess.Process],
    cleanups: dict[asyncio.subprocess.Process, asyncio.Task[None]] | None = None,
) -> None:
    if cleanups is None:
        cleanups = {}
    # Snapshot ownership before cancellation can run a handler's finally block.
    process_snapshot = tuple(active)
    handler_snapshot = tuple(handlers)
    for task in handler_snapshot:
        task.cancel()
    try:
        if handler_snapshot:
            _, pending = await asyncio.wait(
                handler_snapshot, timeout=CLEANUP_TIMEOUT_SECONDS
            )
            if pending:
                logging.getLogger(__name__).error(
                    "sim_relay_handler_cleanup_incomplete"
                )
    finally:
        await asyncio.gather(
            *(_cleanup_process_once(process, cleanups) for process in process_snapshot),
            *cleanups.values(),
        )


async def serve(
    directory: Path, services: tuple[str, ...], transport_image_id: str
) -> None:
    _private_directory(directory)
    network_id, ids = inspect_targets(services=services)
    receipt = ModelSimPreflightRelayReceipt(
        pid=os.getpid(),
        network_id=network_id,
        container_ids=ids,
        nonce=secrets.token_hex(32),
        services=services,
        transport_image_id=transport_image_id,
    )
    inspect_targets(receipt)
    receipt_path = directory / "relay-receipt.json"
    control_path = directory / "relay-control.sock"
    if receipt_path.exists() or control_path.exists():
        raise ValueError("sim relay output already exists")
    active: set[asyncio.subprocess.Process] = set()
    handlers: set[asyncio.Task[None]] = set()
    cleanups: dict[asyncio.subprocess.Process, asyncio.Task[None]] = {}

    async def connection(
        service: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        process = None
        try:
            destination = await asyncio.to_thread(
                inspect_connection_target, receipt, service
            )
            process = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                "-i",
                ids[TRANSPORT],
                "python",
                "-c",
                _BRIDGE,
                destination,
                str(ENDPOINTS[service][1]),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            active.add(process)
            assert process.stdin is not None and process.stdout is not None

            async def upload() -> None:
                assert process is not None and process.stdin is not None
                while data := await reader.read(65536):
                    process.stdin.write(data)
                    await process.stdin.drain()
                process.stdin.close()

            async def download() -> None:
                assert process is not None and process.stdout is not None
                while data := await process.stdout.read(65536):
                    writer.write(data)
                    await writer.drain()

            tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
            try:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()
                # Client half-close may precede a response; backend EOF ends the relay.
                if tasks[1] in pending:
                    await asyncio.wait_for(tasks[1], 30)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except (OSError, ValueError, ConnectionError, TimeoutError):
            logging.getLogger(__name__).log(
                logging.ERROR, "sim_relay_connection_refused service=%s", service
            )
        finally:
            if process is not None:
                try:
                    await _cleanup_process_once(process, cleanups)
                finally:
                    active.discard(process)
                    cleanup = cleanups.get(process)
                    if cleanup is not None and cleanup.done():
                        cleanups.pop(process, None)
            await _close_writer(writer)

    async def challenge(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            nonce = await asyncio.wait_for(reader.readline(), 5)
            if secrets.compare_digest(nonce.strip(), receipt.nonce.encode()):
                await asyncio.to_thread(inspect_targets, receipt)
                writer.write(receipt.model_dump_json().encode())
                await writer.drain()
            else:
                logging.getLogger(__name__).error("sim_relay_challenge_refused")
        except (OSError, ValueError, TimeoutError):
            logging.getLogger(__name__).log(
                logging.ERROR, "sim_relay_challenge_refused"
            )
        finally:
            await _close_writer(writer)

    servers: list[asyncio.Server] = []

    async def control_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        handlers.add(task)
        try:
            await challenge(reader, writer)
        finally:
            handlers.discard(task)

    owned_receipt = False
    owned_control = False
    stop = asyncio.Event()
    restore_signals = _install_shutdown_signals(asyncio.get_running_loop(), stop)
    try:
        for service in services:
            host_port = ENDPOINTS[service][0]

            async def handler(
                reader: asyncio.StreamReader,
                writer: asyncio.StreamWriter,
                target: str = service,
            ) -> None:
                task = asyncio.current_task()
                assert task is not None
                handlers.add(task)
                try:
                    await connection(target, reader, writer)
                finally:
                    handlers.discard(task)

            servers.append(
                await asyncio.start_server(handler, "127.0.0.1", host_port, limit=65536)
            )
        control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            control_socket.bind(str(control_path))
            owned_control = True
            control_socket.setblocking(False)
            servers.append(
                await asyncio.start_unix_server(
                    control_handler, sock=control_socket, limit=256
                )
            )
        except BaseException:
            control_socket.close()
            raise
        control_path.chmod(0o600)
        with open(
            receipt_path, "x", opener=lambda path, flags: os.open(path, flags, 0o600)
        ) as output:
            owned_receipt = True
            output.write(receipt.model_dump_json())
        logging.getLogger(__name__).info(
            "sim relay ready: %s fixed loopback listeners", len(services)
        )
        await stop.wait()
    finally:
        for server in servers:
            server.close()
        await asyncio.gather(*(server.wait_closed() for server in servers))
        # Invalidate proof before waiting for any connection subprocess cleanup.
        if owned_receipt:
            receipt_path.unlink(missing_ok=True)
        if owned_control:
            control_path.unlink(missing_ok=True)
        await _cleanup_connections(handlers, active, cleanups)
        restore_signals()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-directory", type=Path, required=True)
    parser.add_argument(
        "--service", choices=tuple(ENDPOINTS), action="append", required=True
    )
    parser.add_argument("--transport-image-id", required=True)
    args = parser.parse_args()
    try:
        asyncio.run(
            serve(args.private_directory, tuple(args.service), args.transport_image_id)
        )
    except (OSError, ValueError):
        logging.getLogger(__name__).log(logging.ERROR, "sim_relay_startup_refused")
        raise SystemExit(65) from None


if __name__ == "__main__":
    main()
