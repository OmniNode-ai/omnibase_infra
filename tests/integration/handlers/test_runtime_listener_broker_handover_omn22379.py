# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A runtime replacement through the real listener broker socket (OMN-22379).

``scripts/deploy-agent/tests/unit`` pins the session table's rules over random
sequences. This module runs the replacement an operator sees, over the real unix
socket server and real SCM_RIGHTS: the old and the new runtime each fetch the one
listening socket, the new one confirms, the old one is granted its detach, and a
TCP connection queued after the old duplicate closes is accepted through the
new runtime's duplicate. Nothing on the wire is mocked.

Falsifier: grant the old runtime's detach before the new session is confirmed, or
let the broker drop the held listener, and the verdict or the accept fails.
"""

from __future__ import annotations

import array
import json
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

from omnibase_infra.enums.enum_runtime_listener_broker_op import (
    EnumRuntimeListenerBrokerOp,
)
from omnibase_infra.enums.enum_runtime_listener_pool import EnumRuntimeListenerPool
from omnibase_infra.handlers.handler_runtime_listener_broker import (
    HandlerRuntimeListenerBroker,
)
from omnibase_infra.handlers.handler_runtime_listener_broker_server import (
    HandlerRuntimeListenerBrokerServer,
)
from omnibase_infra.models.model_runtime_listener_broker_request import (
    ModelRuntimeListenerBrokerRequest,
)
from omnibase_infra.models.model_runtime_listener_fd_store import (
    ModelRuntimeListenerFdStore,
)
from omnibase_infra.models.model_runtime_listener_registry import (
    ModelRuntimeListenerRegistry,
)

pytestmark = pytest.mark.integration

POOL = "runtime-main"


class _Client:
    """One runtime's connection to the broker socket."""

    def __init__(self, path: Path) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(10)
        self.sock.connect(str(path))
        self.reader = self.sock.makefile("rb")

    def ask(self, **fields: object) -> dict[str, object]:
        self.sock.sendall(json.dumps(fields).encode() + b"\n")
        parsed: dict[str, object] = json.loads(self.reader.readline())
        return parsed

    def get_listener(self, pool: str) -> tuple[dict[str, object], socket.socket | None]:
        self.sock.sendall(json.dumps({"op": "get", "pool": pool}).encode() + b"\n")
        data, ancillary, _, _ = self.sock.recvmsg(
            4096, socket.CMSG_SPACE(array.array("i", [0]).itemsize)
        )
        answer: dict[str, object] = json.loads(data)
        if not ancillary:
            return answer, None
        fd = array.array("i", ancillary[0][2][: array.array("i").itemsize])[0]
        return answer, socket.socket(fileno=fd)

    def close(self) -> None:
        self.reader.close()
        self.sock.close()


class _Served(NamedTuple):
    broker: HandlerRuntimeListenerBroker
    listener: socket.socket
    connect: Callable[[], _Client]


@pytest.fixture
def served(tmp_path: Path) -> Iterator[_Served]:
    path = tmp_path / "handover" / "broker.sock"
    services = ModelRuntimeListenerRegistry.assemble({}, path)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    services.get_service(ModelRuntimeListenerFdStore).keep(POOL, listener)
    broker = HandlerRuntimeListenerBroker(services)
    server = HandlerRuntimeListenerBrokerServer(services)
    server.start()
    clients: list[_Client] = []

    def connect() -> _Client:
        client = _Client(path)
        clients.append(client)
        return client

    try:
        yield _Served(broker, listener, connect)
    finally:
        for client in clients:
            client.close()
        server.stop()
        listener.close()


def _handover(begin: bool) -> ModelRuntimeListenerBrokerRequest:
    return ModelRuntimeListenerBrokerRequest(
        op=EnumRuntimeListenerBrokerOp.HANDOVER_BEGIN
        if begin
        else EnumRuntimeListenerBrokerOp.HANDOVER_END,
        pool=EnumRuntimeListenerPool.MAIN,
    )


def test_replacement_hands_traffic_to_the_new_runtime_duplicate(
    served: _Served,
) -> None:
    broker, listener, connect = served
    port = listener.getsockname()[1]
    old, new = connect(), connect()

    answer, old_socket = old.get_listener(POOL)
    assert answer["ok"] is True and old_socket is not None
    assert old.ask(op="attach", pool=POOL, instance="old")["state"] == "provisional"
    assert old.ask(op="confirmed", pool=POOL)["state"] == "confirmed"

    # The agent begins a handover: the only confirmed runtime may not leave yet.
    broker.handle(_handover(True))
    assert old.ask(op="detach_request", pool=POOL)["verdict"] == "wait"

    answer, new_socket = new.get_listener(POOL)
    assert answer["ok"] is True and new_socket is not None
    try:
        assert new.ask(op="attach", pool=POOL, instance="new")["state"] == "provisional"
        # A provisional successor is no acceptor: the old runtime still waits.
        assert old.ask(op="detach_request", pool=POOL)["verdict"] == "wait"
        assert new.ask(op="confirmed", pool=POOL)["state"] == "confirmed"
        assert old.ask(op="detach_request", pool=POOL)["verdict"] == "granted"

        # The old runtime closes its duplicate; the queue is the kernel's, so a
        # connection made afterwards is accepted through the new duplicate.
        old_socket.close()
        new_socket.settimeout(5)
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            client.sendall(b"ping")
            accepted, _ = new_socket.accept()
            try:
                accepted.settimeout(5)
                assert accepted.recv(4) == b"ping"
            finally:
                accepted.close()
    finally:
        new_socket.close()
        old_socket.close()


def test_positive_control_a_second_confirmed_runtime_is_required_for_a_grant(
    served: _Served,
) -> None:
    """The ``granted`` above is not the broker's default answer."""
    broker, _, connect = served
    only = connect()
    only.ask(op="attach", pool=POOL, instance="only")
    only.ask(op="confirmed", pool=POOL)
    broker.handle(_handover(True))
    assert only.ask(op="detach_request", pool=POOL)["verdict"] == "wait"
    broker.handle(_handover(False))
    assert only.ask(op="detach_request", pool=POOL)["verdict"] == "alone"
