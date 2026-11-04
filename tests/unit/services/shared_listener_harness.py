# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-socket harness for the shared listener tests (OMN-22377).

A fake broker listens on a unix socket and hands the test's listening socket to
every runtime that asks, by SCM_RIGHTS. Nothing on the socket path is mocked.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

from omnibase_infra.models.model_shared_listener_policy import (
    ModelSharedListenerPolicy,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.services.health_checker import ServiceHealth

ANSWER_GRANTED = "granted"
ANSWER_WAIT = "wait"
ANSWER_ALONE = "alone"
ANSWER_REFUSE = "refuse"


def make_policy(**overrides: object) -> ModelSharedListenerPolicy:
    values: dict[str, object] = {
        "pool": "main",
        "listener_first_request_seconds": 2.0,
        "last_attached_hold_seconds": 1.0,
        "detach_successor_wait_seconds": 1.0,
        "passive_ready_watch_ms": 20,
        "probe_client_timeout_seconds": 5.0,
        "response_margin_seconds": 0.5,
    }
    values.update(overrides)
    return ModelSharedListenerPolicy.model_validate(values)


def policy_environ(policy: ModelSharedListenerPolicy) -> dict[str, str]:
    return {
        "ONEX_RUNTIME_HANDOVER_POOL": policy.pool,
        "ONEX_RUNTIME_LISTENER_FIRST_REQUEST_SECONDS": str(
            policy.listener_first_request_seconds
        ),
        "ONEX_RUNTIME_LAST_ATTACHED_HOLD_SECONDS": str(
            policy.last_attached_hold_seconds
        ),
        "ONEX_RUNTIME_DETACH_SUCCESSOR_WAIT_SECONDS": str(
            policy.detach_successor_wait_seconds
        ),
        "ONEX_RUNTIME_PASSIVE_READY_WATCH_MS": str(policy.passive_ready_watch_ms),
        "ONEX_RUNTIME_PROBE_CLIENT_TIMEOUT_SECONDS": str(
            policy.probe_client_timeout_seconds
        ),
        "ONEX_RUNTIME_LISTENER_RESPONSE_MARGIN_SECONDS": str(
            policy.response_margin_seconds
        ),
    }


class FakeRuntime:
    """Just enough of RuntimeHostProcess for /ready."""

    def __init__(self, name: str, *, role: str = "active") -> None:
        self.name = name
        self.role = role
        self.ready = True

    async def readiness_check(self, correlation_id: object = None) -> dict[str, object]:
        return {"ready": self.ready, "instance": self.name}


class BrokerSession:
    def __init__(self) -> None:
        self.provisional = threading.Event()
        self.confirmed = threading.Event()
        self.closed = threading.Event()
        self.detach_asks = 0


class FakeBroker:
    """Unix-socket broker following the wire shape in shared_listener_broker_client."""

    def __init__(self, listener: socket.socket) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="hov", dir="/tmp"))
        self.listener = listener
        self.sessions: list[BrokerSession] = []
        self.detach_answer: Callable[[BrokerSession], str] = self._by_peers
        self.handover_in_progress = False
        self.get_gate = threading.Event()
        self.get_gate.set()
        self.get_received = threading.Event()
        self._granted: set[BrokerSession] = set()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.dir / "broker.sock"))
        self._server.listen(16)
        self._stop = False
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _by_peers(self, session: BrokerSession) -> str:
        peers = [
            s
            for s in self.sessions
            if s is not session
            and s.confirmed.is_set()
            and not s.closed.is_set()
            and s not in self._granted
        ]
        if peers and not self._granted:
            self._granted.add(session)
            return ANSWER_GRANTED
        return ANSWER_WAIT if self.handover_in_progress else ANSWER_ALONE

    def _accept_loop(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            session = BrokerSession()
            self.sessions.append(session)
            threading.Thread(
                target=self._serve, args=(conn, session), daemon=True
            ).start()

    def _serve(self, conn: socket.socket, session: BrokerSession) -> None:
        try:
            reader = conn.makefile("rb")
            for line in reader:
                message = json.loads(line)
                op = message["op"]
                if op == "get":
                    self.get_received.set()
                    self.get_gate.wait()
                    session.provisional.set()
                    socket.send_fds(conn, [b'{"ok": true}\n'], [self.listener.fileno()])
                elif op == "confirmed":
                    session.confirmed.set()
                    conn.sendall(b'{"ok": true}\n')
                elif op == "detach":
                    session.detach_asks += 1
                    answer = self.detach_answer(session)
                    if answer == ANSWER_REFUSE:
                        conn.sendall(b'{"ok": false, "error": "refused"}\n')
                    else:
                        conn.sendall(
                            json.dumps({"ok": True, "answer": answer}).encode() + b"\n"
                        )
        except OSError:
            pass
        finally:
            session.closed.set()
            conn.close()

    def close(self) -> None:
        self._stop = True
        self._server.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def make_server(runtime: FakeRuntime, port: int | None = None) -> ServiceHealth:
    """A real ServiceHealth around the fake runtime, on a free in-container port."""
    return ServiceHealth(
        runtime=cast("RuntimeHostProcess", runtime),
        port=port or free_port(),
        host="127.0.0.1",
    )


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def make_listening_socket() -> socket.socket:
    return socket.create_server(("127.0.0.1", 0), backlog=128)


async def wait_until(
    predicate: Callable[[], bool], *, timeout: float = 5.0, what: str = "condition"
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


async def open_connection(
    port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_connection("127.0.0.1", port)


async def send_ready_request(writer: asyncio.StreamWriter) -> None:
    writer.write(b"GET /ready HTTP/1.1\r\nHost: probe\r\n\r\n")
    await writer.drain()


async def read_response(
    reader: asyncio.StreamReader, timeout: float = 5.0
) -> tuple[int, dict[str, str], dict[str, object]]:
    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout)
    lines = head.decode().split("\r\n")
    status = int(lines[0].split()[1])
    headers = {
        k.lower(): v for k, _, v in (ln.partition(": ") for ln in lines[1:] if ln)
    }
    body = await asyncio.wait_for(
        reader.readexactly(int(headers.get("content-length", "0"))), timeout
    )
    return status, headers, json.loads(body) if body else {}


def details(body: dict[str, object]) -> dict[str, object]:
    inner = body.get("details", body)
    assert isinstance(inner, dict)
    return inner


def env_without_mount() -> None:
    os.environ.pop("ONEX_RUNTIME_HANDOVER_POOL", None)
