# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rolling replacement over one shared host listener (OMN-22377).

The unit modules under ``tests/unit/services`` pin each rule (A1 to A6) on its own.
This one runs the sequence an operator sees when a runtime is replaced on a host:
the old runtime and its successor both attach the one listening socket by
SCM_RIGHTS, the old one receives SIGTERM, and every probe sent during and after
its drain is answered 200 by the successor, then the successor is the only
process left holding the socket. The socket, the broker wire and the aiohttp
servers are real; only the runtime behind ``/ready`` is a stand-in.

Falsifier: make the old runtime keep accepting after ``mark_stopping``, or let
``drain_shared_listener`` return before it closes its duplicate, and a probe is
answered by "old" or the drain assertion fails.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from omnibase_infra.enums.enum_shared_listener_state import EnumSharedListenerState
from omnibase_infra.handlers import mixin_shared_listener
from tests.unit.services.shared_listener_harness import (
    FakeBroker,
    FakeRuntime,
    details,
    make_listening_socket,
    make_policy,
    make_server,
    open_connection,
    policy_environ,
    read_response,
    send_ready_request,
    wait_until,
)

pytestmark = pytest.mark.integration

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


async def _probe(port: int) -> tuple[int, str]:
    reader, writer = await open_connection(port)
    await send_ready_request(writer)
    status, _headers, body = await read_response(reader)
    writer.close()
    return status, str(details(body)["instance"])


async def test_rolling_replacement_hands_the_listener_to_the_successor(
    mounted: FakeBroker,
) -> None:
    old = make_server(FakeRuntime("old"))
    new = make_server(FakeRuntime("new"))
    await old.start()
    await new.start()
    try:
        old_listener, new_listener = old.shared_listener, new.shared_listener
        assert old_listener is not None and new_listener is not None
        await wait_until(
            lambda: old_listener.state is ATTACHED and new_listener.state is ATTACHED,
            what="both runtimes attached",
        )
        assert all(session.confirmed.is_set() for session in mounted.sessions)
        port = mounted.listener.getsockname()[1]

        # Positive control: while both are attached the socket answers, and the
        # answer names one of the two runtimes, so the instance field is a real
        # discriminator for the assertions below.
        before = await asyncio.gather(*(_probe(port) for _ in range(20)))
        assert all(status == 200 for status, _ in before)
        assert {who for _, who in before} <= {"old", "new"}

        # SIGTERM reaches the old runtime: latch, then drain.
        old.mark_stopping()
        drain = asyncio.ensure_future(old.drain_shared_listener())
        await wait_until(lambda: not old_listener.accepting, what="old stops accepting")

        during = await asyncio.gather(*(_probe(port) for _ in range(25)))
        await drain
        assert old_listener.state is DETACHED
        assert not old_listener.accepting

        after = await asyncio.gather(*(_probe(port) for _ in range(25)))

        assert all(status == 200 for status, _ in during + after)
        assert {who for _, who in during + after} == {"new"}
        assert new_listener.state is ATTACHED
        assert new_listener.accepting
    finally:
        await old.stop()
        await new.stop()
