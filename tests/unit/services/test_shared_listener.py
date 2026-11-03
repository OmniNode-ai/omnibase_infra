# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""AC1 and AC4 of OMN-22377: a runtime serves a socket it received by SCM_RIGHTS."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from omnibase_infra.enums.enum_shared_listener_state import EnumSharedListenerState
from omnibase_infra.handlers import mixin_shared_listener
from omnibase_infra.runtime import service_kernel
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


async def test_shared_listener_serves_ready_on_received_socket(
    mounted: FakeBroker,
) -> None:
    runtime = FakeRuntime("A")
    server = make_server(runtime)
    await server.start()
    try:
        listener = server.shared_listener
        assert listener is not None
        await wait_until(
            lambda: listener.state is EnumSharedListenerState.ATTACHED,
            what="attach",
        )
        assert mounted.sessions[0].confirmed.is_set()
        reader, writer = await open_connection(mounted.listener.getsockname()[1])
        await send_ready_request(writer)
        status, _headers, body = await read_response(reader)
        writer.close()
        assert status == 200
        fields = details(body)
        assert fields["instance"] == "A"
        assert fields["role"] == "active"
        assert fields["listener_attached"] is True
        assert fields["consuming"] is True
        assert "epoch" in fields
    finally:
        await server.stop()


async def test_shared_listener_sigterm_stops_accepting_before_readiness_flips(
    mounted: FakeBroker,
) -> None:
    runtime = FakeRuntime("A")
    server = make_server(runtime)
    await server.start()
    listener = server.shared_listener
    assert listener is not None
    await wait_until(
        lambda: listener.state is EnumSharedListenerState.ATTACHED, what="attach"
    )
    port = mounted.listener.getsockname()[1]
    # A connection the runtime accepted before any request byte arrived.
    early_reader, early_writer = await open_connection(port)
    await wait_until(
        lambda: listener.accepted_connections == 1,
        what="accept of the silent connection",
    )

    server.mark_stopping()  # what the kernel's signal handler does first
    assert listener.stopping
    drain = asyncio.ensure_future(server.drain_shared_listener())
    await wait_until(
        lambda: listener.state is not EnumSharedListenerState.ATTACHED, what="drain"
    )
    await wait_until(lambda: not listener.accepting, what="accepting stopped")

    # Accepting has stopped: a new connection stays in the shared queue.
    late_reader, late_writer = await open_connection(port)
    await send_ready_request(late_writer)

    # The silent connection is still answered, while readiness still reads true.
    await send_ready_request(early_writer)
    status, headers, body = await read_response(early_reader)
    assert status == 200
    assert details(body)["ready"] is True
    assert headers.get("connection") == "close"
    await drain
    assert listener.state is EnumSharedListenerState.DETACHED

    # Only now does the teardown change readiness (the kernel's next steps).
    runtime.ready = False
    with pytest.raises(asyncio.TimeoutError):
        await read_response(late_reader, timeout=0.3)  # never accepted by A
    late_writer.close()
    early_writer.close()
    await server.stop()


async def test_shared_listener_unmounted_runtime_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mixin_shared_listener, "HANDOVER_MOUNT", tmp_path / "absent")
    port = free_port()
    server = make_server(FakeRuntime("A"), port)
    await server.start()
    try:
        assert server.shared_listener is None
        reader, writer = await open_connection(port)
        await send_ready_request(writer)
        status, _headers, body = await read_response(reader)
        writer.close()
        assert status == 200
        assert not {"role", "epoch", "listener_attached", "consuming"} & set(
            details(body)
        )
        server.mark_stopping()
        await server.drain_shared_listener()
    finally:
        await server.stop()


def test_shared_listener_kernel_latches_then_drains_before_teardown() -> None:
    source = inspect.getsource(service_kernel)
    latch = source.index("health_server.mark_stopping()")
    set_event = source.index("shutdown_event.set()", latch)
    drain = source.index("await health_server.drain_shared_listener()")
    monitor_stop = source.index("await runtime_health_monitor.stop()", drain)
    runtime_stop = source.index("runtime.stop(), timeout=grace_period", drain)
    assert latch < set_event < drain < monitor_stop < runtime_stop
