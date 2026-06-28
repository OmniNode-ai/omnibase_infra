# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Loop-affinity regression for the auto-wired sync event_publisher (OMN-13658).

Legacy sync handlers (e.g. ``HandlerContextRoiRunner``) run on a
``ThreadPoolExecutor`` worker thread because the dispatch engine offloads
blocking sync handlers via ``run_in_executor``. A publish issued from such a
thread must be scheduled back onto the runtime kernel's owning event loop —
where the publish awaitable's internal Futures are bound — NOT executed on a
fresh throwaway ``asyncio.run`` loop in the worker thread. The latter produced
the ``got Future attached to a different loop`` warning and the 2-3 minute
terminal-emission retry delay this ticket removes.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from unittest.mock import patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_sync_event_publisher,
)

_TERMINAL_TOPIC = "onex.evt.platform.context-roi.v1"


class _RecordingEventBus:
    """Async event bus that records the loop its publish coroutine ran on."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes | None, bytes]] = []
        self.publish_loop: asyncio.AbstractEventLoop | None = None

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> None:
        # Capturing the running loop proves which loop actually executed the
        # coroutine: a fresh worker-thread loop would differ from the kernel loop.
        self.publish_loop = asyncio.get_running_loop()
        self.published.append((topic, key, value))


async def _build_publisher_on_loop(
    event_bus: object,
) -> Callable[[str, bytes], None]:
    # Constructed on the kernel loop, exactly as wire_from_manifest does.
    return _make_sync_event_publisher(
        event_bus=event_bus,
        handler_name="HandlerContextRoiRunner",
    )


@pytest.mark.unit
def test_sync_publisher_from_worker_thread_runs_on_kernel_loop() -> None:
    """A publish from a non-event-loop thread runs on the kernel loop.

    Asserts the DoD: completes without error, does NOT spawn a new
    ``asyncio.run`` loop, and the publish coroutine executes on the kernel loop
    rather than a throwaway worker-thread loop.
    """
    kernel_loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=kernel_loop.run_forever, daemon=True)
    loop_thread.start()
    try:
        event_bus = _RecordingEventBus()
        # Build the publisher ON the kernel loop's own thread.
        publisher = asyncio.run_coroutine_threadsafe(
            _build_publisher_on_loop(event_bus), kernel_loop
        ).result(timeout=5)

        worker_errors: list[BaseException] = []

        # DoD guard: the sync publisher must NOT spin a new asyncio.run loop.
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.asyncio.run"
        ) as mock_asyncio_run:

            def _worker() -> None:
                try:
                    publisher(_TERMINAL_TOPIC, b'{"ok":true}')
                except BaseException as exc:  # noqa: BLE001 — surface to assertion
                    worker_errors.append(exc)

            # A genuine worker thread with no running event loop, mirroring the
            # ThreadPoolExecutor worker the dispatch engine uses.
            worker = threading.Thread(target=_worker)
            worker.start()
            worker.join(timeout=5)

            assert not worker.is_alive(), "sync publisher call hung on a worker thread"
            mock_asyncio_run.assert_not_called()

        assert worker_errors == []

        # Wait for the coroutine scheduled onto the kernel loop to complete.
        deadline = time.monotonic() + 5.0
        while not event_bus.published and time.monotonic() < deadline:
            time.sleep(0.01)

        assert event_bus.published == [(_TERMINAL_TOPIC, None, b'{"ok":true}')]
        # The publish coroutine ran on the kernel loop — not a worker-thread loop.
        assert event_bus.publish_loop is kernel_loop
    finally:
        kernel_loop.call_soon_threadsafe(kernel_loop.stop)
        loop_thread.join(timeout=5)
        kernel_loop.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_publisher_from_kernel_loop_thread_schedules_task() -> None:
    """A publish from the kernel loop thread schedules a task on that loop.

    The async handler path (publishing while already on the kernel loop) keeps
    using ``create_task`` and must still deliver the event.
    """
    event_bus = _RecordingEventBus()
    publisher = _make_sync_event_publisher(
        event_bus=event_bus,
        handler_name="HandlerContextRoiRunner",
    )

    publisher(_TERMINAL_TOPIC, b'{"ok":true}')
    # Let the scheduled task run to completion on this loop.
    await asyncio.sleep(0)

    assert event_bus.published == [(_TERMINAL_TOPIC, None, b'{"ok":true}')]
    assert event_bus.publish_loop is asyncio.get_running_loop()
