# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""AC2 of OMN-22377: two attached runtimes share one accept queue."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from omnibase_infra.enums.enum_shared_listener_state import EnumSharedListenerState
from omnibase_infra.handlers import mixin_shared_listener
from omnibase_infra.services.health_checker import ServiceHealth
from tests.unit.services.shared_listener_harness import (
    FakeBroker,
    FakeRuntime,
    details,
    free_port,
    make_listening_socket,
    make_policy,
    make_server,
    open_connection,
    policy_environ,
    read_response,
    send_ready_request,
    wait_until,
)

pytestmark = pytest.mark.unit

ATTACHED = EnumSharedListenerState.ATTACHED
DETACHED = EnumSharedListenerState.DETACHED


@pytest.fixture
async def mounted(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeBroker]:
    listener_socket = make_listening_socket()
    broker = FakeBroker(listener_socket)
    monkeypatch.setattr(mixin_shared_listener, "HANDOVER_MOUNT", broker.dir)
    for key, value in policy_environ(make_policy()).items():
        monkeypatch.setenv(key, value)
    yield broker
    broker.close()
    listener_socket.close()


def _server(name: str, *, role: str = "active") -> tuple[ServiceHealth, FakeRuntime]:
    runtime = FakeRuntime(name, role=role)
    server = make_server(runtime)
    return server, runtime


async def _probe(port: int) -> tuple[int, str]:
    reader, writer = await open_connection(port)
    await send_ready_request(writer)
    status, _headers, body = await read_response(reader)
    writer.close()
    return status, str(details(body)["instance"])


async def test_shared_listener_handover_queued_connections_answered_by_other(
    mounted: FakeBroker,
) -> None:
    old, _ = _server("old")
    new, _ = _server("new")
    await old.start()
    await new.start()
    try:
        old_listener, new_listener = old.shared_listener, new.shared_listener
        assert old_listener is not None and new_listener is not None
        await wait_until(
            lambda: old_listener.state is ATTACHED and new_listener.state is ATTACHED,
            what="both attached",
        )
        port = mounted.listener.getsockname()[1]
        old.mark_stopping()
        drain = asyncio.ensure_future(old.drain_shared_listener())
        await wait_until(lambda: not old_listener.accepting, what="old stops accepting")
        # Queued while the old runtime detaches: the other one answers.
        during = await asyncio.gather(*(_probe(port) for _ in range(25)))
        await drain
        assert old_listener.state is DETACHED
        after = await asyncio.gather(*(_probe(port) for _ in range(25)))
        assert all(status == 200 for status, _ in during + after)
        assert {who for _, who in during + after} == {"new"}
    finally:
        await old.stop()
        await new.stop()


async def test_shared_listener_handover_attach_racing_sigterm_never_accepts(
    mounted: FakeBroker,
) -> None:
    mounted.get_gate.clear()  # the broker holds its reply to the fd request
    server, _ = _server("A")
    await server.start()
    listener = server.shared_listener
    assert listener is not None
    try:
        await asyncio.get_running_loop().run_in_executor(
            None, mounted.get_received.wait, 5
        )
        assert listener.state is EnumSharedListenerState.ATTACHING
        server.mark_stopping()  # SIGTERM lands mid-attach
        stop = asyncio.ensure_future(server.drain_shared_listener())
        await asyncio.sleep(0.05)
        mounted.get_gate.set()  # the fd arrives after the latch
        await stop
        assert listener.state is DETACHED
        assert not listener.accepting
        assert listener._site is None
        assert not any(s.confirmed.is_set() for s in mounted.sessions)
        # The late fd and the provisional session are closed, not served.
        loop = asyncio.get_running_loop()
        closed = await loop.run_in_executor(None, mounted.sessions[0].closed.wait, 5)
        assert closed
        reader, writer = await open_connection(mounted.listener.getsockname()[1])
        await send_ready_request(writer)
        with pytest.raises(asyncio.TimeoutError):
            await read_response(reader, timeout=0.3)
        writer.close()
        assert listener.state is DETACHED
    finally:
        mounted.get_gate.set()
        await server.stop()


async def test_shared_listener_handover_latch_blocks_attach_after_sigterm(
    mounted: FakeBroker,
) -> None:
    server, _ = _server("A")
    await server.start()
    listener = server.shared_listener
    assert listener is not None
    server.mark_stopping()
    await server.drain_shared_listener()
    assert await listener.attach() is False
    assert listener.state is DETACHED
    await server.stop()


async def test_shared_listener_handover_revocation_granted_detaches(
    mounted: FakeBroker,
) -> None:
    first, _ = _server("first")
    second, _ = _server("second")
    await first.start()
    await second.start()
    try:
        a, b = first.shared_listener, second.shared_listener
        assert a is not None and b is not None
        await wait_until(
            lambda: a.state is ATTACHED and b.state is ATTACHED, what="both attached"
        )
        await a.begin_revocation()  # latch, ask B, drain on granted
        assert a.revoking
        assert a.state is DETACHED
        await asyncio.sleep(0.2)
        assert a.state is DETACHED  # the latch forbids a re-attach
        a.end_revocation()
        await wait_until(lambda: a.state is ATTACHED, what="re-attach after latch")
    finally:
        await first.stop()
        await second.stop()
