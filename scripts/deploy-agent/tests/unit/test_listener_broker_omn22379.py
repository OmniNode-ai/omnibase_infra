# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22379 AC2: the listener broker never strands a handover and counts a provisional session for nothing.

Rule A6: a detach is ``granted`` only while another CONFIRMED session of the pool is
attached and holds no outstanding grant; otherwise ``wait`` (handover in progress)
or ``alone``. The invariant is checked over thousands of random operation
sequences against a model of the table kept in this file, and a mutant table that
counts provisional sessions is shown to violate it (the positive control). The
socket tests drive the real unix-socket server with SCM_RIGHTS.
"""

from __future__ import annotations

import array
import json
import random
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
from omnibase_infra.models.model_runtime_listener_session_table import (
    ModelRuntimeListenerSessionTable,
)

pytestmark = pytest.mark.unit

POOL = "runtime-main"


class _MutantCountsProvisional(ModelRuntimeListenerSessionTable):
    """Seeded defect: a provisional session counts as an acceptor."""

    def decide_detach(self, session_id: str) -> tuple[str, str]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.confirmed:
                return "", "no_session"
            if any(
                o.pool == session.pool and not o.granted
                for k, o in self._sessions.items()
                if k != session_id
            ):
                session.granted = True
                return "granted", ""
            return ("wait" if session.pool in self._handover else "alone"), ""


def _table_with(
    *sessions: tuple[str, bool], handover: bool
) -> ModelRuntimeListenerSessionTable:
    table = ModelRuntimeListenerSessionTable()
    for name, confirmed in sessions:
        table.open_session(name, POOL, name)
        if confirmed:
            table.confirm_session(name)
    table.mark_handover(POOL, handover)
    return table


def test_listener_broker_provisional_session_counts_for_nothing() -> None:
    table = _table_with(("old", True), ("new", False), handover=True)
    assert table.decide_detach("old") == ("wait", ""), (
        "a provisional successor is no acceptor"
    )
    assert table.confirmed_acceptors(POOL) == 1
    table.confirm_session("new")
    assert table.decide_detach("old") == ("granted", "")
    assert table.decide_detach("new") == ("wait", ""), (
        "old holds the grant and is leaving"
    )


def test_listener_broker_provisional_session_cannot_ask_to_detach() -> None:
    table = _table_with(("a", True), ("b", False), handover=True)
    assert table.decide_detach("b") == ("", "not_confirmed")
    assert table.decide_detach("ghost") == ("", "no_session")


def test_listener_broker_last_confirmed_acceptor_waits_during_handover_and_is_alone_otherwise() -> (
    None
):
    table = _table_with(("only", True), handover=True)
    assert table.decide_detach("only") == ("wait", "")
    table.mark_handover(POOL, False)
    assert table.decide_detach("only") == ("alone", "")


def test_listener_broker_grants_are_serialized_per_pool() -> None:
    table = _table_with(("a", True), ("b", True), handover=True)
    assert table.decide_detach("a") == ("granted", "")
    assert table.decide_detach("a") == ("granted", ""), (
        "asking again keeps the same grant"
    )
    assert table.decide_detach("b") == ("wait", ""), (
        "a is already leaving; b is the last acceptor"
    )
    table.end_session("a")
    assert table.decide_detach("b") == ("wait", "")


def test_listener_broker_a_grant_is_withdrawn_when_the_acceptor_it_relied_on_ends() -> (
    None
):
    table = _table_with(("a", True), ("b", True), handover=True)
    assert table.decide_detach("a") == ("granted", "")
    table.end_session("b")  # b crashes after the grant
    assert table.decide_detach("a") == ("wait", ""), (
        "asking again must not repeat a stale grant"
    )
    assert table.confirmed_acceptors(POOL) == 1, "a is an acceptor again"


def test_listener_broker_a_crashed_session_stops_counting() -> None:
    table = _table_with(("a", True), ("b", True), handover=True)
    table.end_session("b")  # the connection died
    assert table.decide_detach("a") == ("wait", "")


def test_listener_broker_pools_are_independent() -> None:
    table = _table_with(("a", True), handover=True)
    table.open_session("e", "runtime-effects", "e")
    table.confirm_session("e")
    assert table.decide_detach("a") == ("wait", ""), (
        "another pool's session is no acceptor"
    )


def _run_sequence(
    table: ModelRuntimeListenerSessionTable, rng: random.Random
) -> str | None:
    """Drive random operations; return a description of the first violation, or None."""
    names = [f"s{i}" for i in range(5)]
    state: dict[
        str, dict[str, bool]
    ] = {}  # the checker's own model: name -> confirmed/granted
    handover = False
    for step in range(60):
        name = rng.choice(names)
        action = rng.choice(["open", "confirm", "detach", "detach", "end", "flag"])
        if action == "open":
            table.open_session(name, POOL, name)
            state[name] = {"confirmed": False, "granted": False}
        elif action == "confirm":
            table.confirm_session(name)
            if name in state:
                state[name]["confirmed"] = True
        elif action == "end":
            table.end_session(name)
            state.pop(name, None)
        elif action == "flag":
            handover = not handover
            table.mark_handover(POOL, handover)
        else:
            verdict, _error = table.decide_detach(name)
            mine = state.get(name)
            if mine is None or not mine["confirmed"]:
                if verdict:
                    return (
                        f"step {step}: verdict {verdict!r} for a non-confirmed {name}"
                    )
                continue
            mine["granted"] = verdict == "granted"  # every ask re-decides the grant
            if verdict == "granted":
                remaining = [
                    k for k, v in state.items() if v["confirmed"] and not v["granted"]
                ]
                if not remaining:
                    return (
                        f"step {step}: granted {name} leaving zero confirmed acceptors "
                        f"(handover={handover}, sessions={state})"
                    )
            elif verdict == "alone" and handover:
                return f"step {step}: alone answered during a handover"
            elif verdict == "wait" and not handover:
                return f"step {step}: wait answered with no handover"
    return None


def test_listener_broker_never_grants_the_last_confirmed_acceptor_over_random_sequences() -> (
    None
):
    for seed in range(1500):
        violation = _run_sequence(
            ModelRuntimeListenerSessionTable(), random.Random(seed)
        )
        assert violation is None, f"seed {seed}: {violation}"


def test_listener_broker_invariant_check_catches_a_table_that_counts_provisional_sessions() -> (
    None
):
    violations = [
        _run_sequence(_MutantCountsProvisional(), random.Random(seed))
        for seed in range(1500)
    ]
    assert any(v is not None for v in violations), "the checker must be able to fail"


# --------------------------------------------------------------------- the socket


class _Client:
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
    listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    listener.bind(("::", 0))
    listener.listen(8)
    services.get_service(ModelRuntimeListenerFdStore).keep(POOL, listener)
    broker = HandlerRuntimeListenerBroker(services)
    server = HandlerRuntimeListenerBrokerServer(services)
    server.start()
    server.start()  # idempotent
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


def _wait_for_sessions(broker: HandlerRuntimeListenerBroker, expected: int) -> None:
    import time

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if broker.sessions.session_count(POOL) == expected:
            return
        time.sleep(0.01)
    raise AssertionError(f"sessions never reached {expected}")


def test_listener_broker_hands_the_listener_over_by_scm_rights(served: _Served) -> None:
    _, listener, connect = served
    answer, received = connect().get_listener(POOL)
    assert answer["ok"] is True and received is not None
    try:
        assert received.getsockname() == listener.getsockname()
        assert received.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
        with socket.create_connection(("127.0.0.1", listener.getsockname()[1])):
            conn, _ = received.accept()  # the duplicate shares the one accept queue
            conn.close()
    finally:
        received.close()


def test_listener_broker_refuses_a_pool_the_agent_does_not_hold(
    served: _Served,
) -> None:
    _, _, connect = served
    client = connect()
    answer, received = client.get_listener("runtime-effects")
    assert received is None
    assert answer["ok"] is False and answer["error"] == "not_installed"
    assert (
        client.ask(op="attach", pool="runtime-effects", instance="x")["error"]
        == "not_installed"
    )


def test_listener_broker_session_lifecycle_over_the_socket(served: _Served) -> None:
    broker, _, connect = served
    old, new = connect(), connect()
    assert (
        old.ask(op="attach", pool=POOL, instance="runtime-main-a")["state"]
        == "provisional"
    )
    assert old.ask(op="detach_request", pool=POOL)["error"] == "not_confirmed"
    assert old.ask(op="confirmed", pool=POOL)["state"] == "confirmed"
    broker.handle(_flag(True))
    assert old.ask(op="detach_request", pool=POOL)["verdict"] == "wait"
    assert (
        new.ask(op="attach", pool=POOL, instance="runtime-main-b")["state"]
        == "provisional"
    )
    assert old.ask(op="detach_request", pool=POOL)["verdict"] == "wait", (
        "provisional counts for nothing"
    )
    assert new.ask(op="confirmed", pool=POOL)["state"] == "confirmed"
    assert old.ask(op="detach_request", pool=POOL)["verdict"] == "granted"
    assert new.ask(op="detach_request", pool=POOL)["verdict"] == "wait"
    old.close()  # the old runtime leaves
    _wait_for_sessions(broker, 1)
    assert new.ask(op="detach_request", pool=POOL)["verdict"] == "wait", (
        "still the last one"
    )
    broker.handle(_flag(False))
    assert new.ask(op="detach_request", pool=POOL)["verdict"] == "alone"


def test_listener_broker_a_crashed_runtime_connection_ends_its_session(
    served: _Served,
) -> None:
    broker, _, connect = served
    first, second = connect(), connect()
    for client in (first, second):
        client.ask(op="attach", pool=POOL, instance="i")
        client.ask(op="confirmed", pool=POOL)
    broker.handle(_flag(True))
    _wait_for_sessions(broker, 2)
    second.close()  # no goodbye: the process is gone
    _wait_for_sessions(broker, 1)
    assert first.ask(op="detach_request", pool=POOL)["verdict"] == "wait"


def test_listener_broker_refuses_what_a_client_must_not_send(served: _Served) -> None:
    broker, _, connect = served
    client = connect()
    for bad in (
        {"op": "handover_begin", "pool": POOL},
        {"op": "end", "pool": POOL},
        {"op": "attach", "pool": POOL, "session_id": "forged"},
        {"op": "get", "pool": "nope"},
        {"op": "teleport", "pool": POOL},
        [1, 2],
    ):
        answer = _raw(client, bad)
        assert answer["ok"] is False and answer["error"] == "bad_request", bad
    assert not broker.sessions.handover_in_progress(POOL), (
        "a client cannot raise the agent's flag"
    )
    client.sock.sendall(b"not json\n")
    assert json.loads(client.reader.readline())["error"] == "bad_request"
    assert client.ask(op="attach", pool=POOL, instance="x")["ok"] is True, (
        "the connection survives"
    )


def _raw(client: _Client, value: object) -> dict[str, object]:
    client.sock.sendall(json.dumps(value).encode() + b"\n")
    parsed: dict[str, object] = json.loads(client.reader.readline())
    return parsed


def _flag(begin: bool) -> ModelRuntimeListenerBrokerRequest:
    return ModelRuntimeListenerBrokerRequest(
        op=EnumRuntimeListenerBrokerOp.HANDOVER_BEGIN
        if begin
        else EnumRuntimeListenerBrokerOp.HANDOVER_END,
        pool=EnumRuntimeListenerPool.MAIN,
    )
