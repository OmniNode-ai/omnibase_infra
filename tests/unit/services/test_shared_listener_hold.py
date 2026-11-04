# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""AC3 of OMN-22377: the last acceptor holds instead of leaving (rule A6)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from omnibase_infra.enums.enum_shared_listener_state import EnumSharedListenerState
from omnibase_infra.handlers import mixin_shared_listener
from omnibase_infra.services.health_checker import ServiceHealth
from tests.unit.services.shared_listener_harness import (
    ANSWER_REFUSE,
    ANSWER_WAIT,
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
HOLDING = EnumSharedListenerState.HOLDING
POLICY = make_policy()


@pytest.fixture
async def mounted(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeBroker]:
    listener_socket = make_listening_socket()
    broker = FakeBroker(listener_socket)
    monkeypatch.setattr(mixin_shared_listener, "HANDOVER_MOUNT", broker.dir)
    for key, value in policy_environ(POLICY).items():
        monkeypatch.setenv(key, value)
    yield broker
    broker.close()
    listener_socket.close()


async def _started(role: str) -> tuple[ServiceHealth, FakeRuntime]:
    runtime = FakeRuntime("A", role=role)
    server = make_server(runtime)
    await server.start()
    listener = server.shared_listener
    assert listener is not None
    await wait_until(lambda: listener.state is ATTACHED, what="attach")
    return server, runtime


async def test_shared_listener_hold_wait_answer_keeps_accepting(
    mounted: FakeBroker,
) -> None:
    mounted.handover_in_progress = True
    mounted.detach_answer = lambda _session: ANSWER_WAIT
    server, runtime = await _started("passive")
    listener = server.shared_listener
    assert listener is not None
    try:
        runtime.ready = False
        await wait_until(lambda: listener.state is HOLDING, what="hold")
        assert listener.accepting  # `wait` keeps the runtime accepting
        assert mounted.sessions[0].detach_asks >= 1
        port = mounted.listener.getsockname()[1]
        reader, writer = await open_connection(port)
        await send_ready_request(writer)
        started = time.monotonic()
        await asyncio.sleep(0.2)
        runtime.ready = True  # readiness returns inside the hold
        status, _headers, body = await read_response(reader)
        elapsed = time.monotonic() - started
        writer.close()
        assert status == 200
        assert details(body)["ready"] is True
        assert elapsed >= 0.15  # it was held, not answered 503 at once
    finally:
        await server.stop()


async def test_shared_listener_hold_refused_detach_keeps_accepting(
    mounted: FakeBroker,
) -> None:
    mounted.detach_answer = lambda _session: ANSWER_REFUSE
    server, runtime = await _started("passive")
    listener = server.shared_listener
    assert listener is not None
    try:
        runtime.ready = False
        await wait_until(lambda: listener.state is HOLDING, what="hold")
        assert listener.accepting
    finally:
        await server.stop()


async def test_shared_listener_hold_bound_ends_with_the_readiness_of_that_moment(
    mounted: FakeBroker,
) -> None:
    mounted.detach_answer = lambda _session: ANSWER_WAIT
    server, runtime = await _started("passive")
    listener = server.shared_listener
    assert listener is not None
    try:
        runtime.ready = False
        await wait_until(lambda: listener.state is HOLDING, what="hold")
        reader, writer = await open_connection(mounted.listener.getsockname()[1])
        started = time.monotonic()
        await send_ready_request(writer)
        status, _headers, _body = await read_response(reader)
        elapsed = time.monotonic() - started
        writer.close()
        assert status == 503
        assert POLICY.last_attached_hold_seconds * 0.9 <= elapsed
        assert elapsed < POLICY.probe_client_timeout_seconds
    finally:
        await server.stop()


async def test_shared_listener_hold_revocation_answers_200_inside_probe_timeout(
    mounted: FakeBroker,
) -> None:
    mounted.detach_answer = lambda _session: ANSWER_WAIT
    server, runtime = await _started("active")
    listener = server.shared_listener
    assert listener is not None
    try:
        await listener.begin_revocation()  # on_partitions_revoked
        assert listener.state is HOLDING
        assert listener.accepting
        runtime.ready = False  # the assignment is now cleared
        reader, writer = await open_connection(mounted.listener.getsockname()[1])
        started = time.monotonic()
        await send_ready_request(writer)
        await asyncio.sleep(0.3)  # reassignment takes a while
        runtime.ready = True  # on_partitions_assigned finished
        listener.end_revocation()
        status, _headers, body = await read_response(reader)
        elapsed = time.monotonic() - started
        writer.close()
        assert status == 200
        assert details(body)["ready"] is True
        assert elapsed < POLICY.probe_client_timeout_seconds
        await wait_until(lambda: listener.state is ATTACHED, what="back to attached")
    finally:
        await server.stop()


def test_shared_listener_hold_larger_than_probe_client_timeout_is_refused() -> None:
    with pytest.raises(ValidationError, match="probe_client_timeout_seconds"):
        make_policy(last_attached_hold_seconds=4.8, response_margin_seconds=0.5)
    with pytest.raises(ValidationError, match="probe_client_timeout_seconds"):
        make_policy(last_attached_hold_seconds=5.0)
    make_policy(last_attached_hold_seconds=4.0, response_margin_seconds=0.5)


def test_shared_listener_hold_policy_missing_value_fails_loudly() -> None:
    environ = policy_environ(POLICY)
    del environ["ONEX_RUNTIME_PASSIVE_READY_WATCH_MS"]
    from omnibase_infra.models.model_shared_listener_policy import (
        ModelSharedListenerPolicy,
    )

    with pytest.raises(ValueError, match="PASSIVE_READY_WATCH_MS"):
        ModelSharedListenerPolicy.from_environ(environ)
