# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execute the kernel signal and shutdown paths with a delayed listener drain."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers import mixin_shared_listener
from omnibase_infra.runtime import service_kernel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mounted", [True, False])
async def test_shared_listener_kernel_latches_then_drains_before_teardown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_wire_infrastructure: MagicMock,
    mock_inmemory_runtime_config: MagicMock,
    mounted: bool,
) -> None:
    mount = tmp_path / "handover"
    if mounted:
        mount.mkdir()
    monkeypatch.setattr(mixin_shared_listener, "HANDOVER_MOUNT", mount)
    monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
    order: list[str] = []
    callbacks: dict[signal.Signals, tuple[Callable[..., None], tuple[object, ...]]] = {}
    drain_started, release_drain = asyncio.Event(), asyncio.Event()
    signal_active = False
    original_event = asyncio.Event

    class ShutdownEvent(original_event):
        def set(self) -> None:
            if signal_active:
                order.append("wake")
            super().set()

    def register(
        sig: signal.Signals, callback: Callable[..., None], *args: object
    ) -> None:
        callbacks[sig] = (callback, args)

    async def start_runtime() -> None:
        nonlocal signal_active
        callback, args = callbacks[signal.SIGTERM]
        signal_active = True
        callback(*args)
        signal_active = False

    async def drain() -> None:
        order.append("drain-start")
        drain_started.set()
        await release_drain.wait()
        order.append("drain-end")

    health = MagicMock()
    health.start = AsyncMock()
    health.stop = AsyncMock()
    health.mark_stopping.side_effect = lambda: order.append("latch")
    health.drain_shared_listener = AsyncMock(side_effect=drain)
    runtime = MagicMock()
    runtime.start = AsyncMock(side_effect=start_runtime)
    runtime.stop = AsyncMock(side_effect=lambda: order.append("runtime-stop"))
    bus = MagicMock(spec=EventBusInmemory)
    bus.start = AsyncMock()
    bus.close = AsyncMock()
    bus.subscribe = AsyncMock(return_value=AsyncMock())
    bus.publish_envelope = AsyncMock()
    loop = asyncio.get_running_loop()

    with (
        patch.object(loop, "add_signal_handler", side_effect=register),
        patch.object(loop, "remove_signal_handler"),
        patch.object(service_kernel.asyncio, "Event", ShutdownEvent),
        patch.object(service_kernel, "RuntimeHostProcess", return_value=runtime),
        patch(
            "omnibase_infra.backends.auto_configure.select_event_bus", return_value=bus
        ),
        patch(
            "omnibase_infra.services.health_checker.ServiceHealth", return_value=health
        ),
    ):
        task = asyncio.create_task(service_kernel.bootstrap())
        try:
            if mounted:
                await asyncio.wait_for(drain_started.wait(), timeout=10)
                assert order == ["latch", "wake", "drain-start"]
                runtime.stop.assert_not_awaited()
                health.stop.assert_not_awaited()
                release_drain.set()
            assert await asyncio.wait_for(task, timeout=10) == 0
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    if mounted:
        assert order == ["latch", "wake", "drain-start", "drain-end", "runtime-stop"]
        health.drain_shared_listener.assert_awaited_once()
    else:
        assert order == ["wake", "runtime-stop"]
        health.mark_stopping.assert_not_called()
        health.drain_shared_listener.assert_not_awaited()
