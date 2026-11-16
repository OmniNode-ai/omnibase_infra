# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A plain-function handler entry method runs off the event loop (OMN-24491).

Root cause: ``_make_dispatch_callback`` called a handler's plain ``handle``
directly inside the async dispatch callback, so a job that blocks on the
network, a database or a subprocess froze the one runtime event loop (readiness
endpoint and Kafka heartbeats included) for as long as it blocked.

The fix runs a plain-function entry method on a worker thread and leaves an
async entry method (``async def handle`` or ``handle_async``) unchanged.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef

pytestmark = pytest.mark.unit

_BLOCK_SECONDS = 0.6
_TICK_SECONDS = 0.02


class _StubPayload:
    @classmethod
    def model_validate(cls, data: object) -> _StubPayload:
        return cls()


def _envelope() -> object:
    envelope = MagicMock()
    envelope.payload = {"k": "v"}
    envelope.correlation_id = "00000000-0000-0000-0000-000000000001"
    envelope.event_type = "ModelTestPayload"
    return envelope


def _event_model() -> ModelHandlerRef:
    return ModelHandlerRef(name="ModelTestPayload", module="fake.models")


async def _dispatch(handler: Any) -> None:
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_event_model_class",
        return_value=_StubPayload,
    ):
        callback = _make_dispatch_callback(
            handler,
            event_model=_event_model(),
        )
        await callback(_envelope())


class _Heartbeat:
    """Sample the event loop every tick and keep the largest gap between ticks."""

    def __init__(self) -> None:
        self.max_gap = 0.0
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(_TICK_SECONDS)
            now = time.monotonic()
            self.max_gap = max(self.max_gap, now - last)
            last = now

    async def __aenter__(self) -> _Heartbeat:
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(_TICK_SECONDS * 3)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._task is not None
        # Let the loop tick once more so a stall that just ended is measured.
        await asyncio.sleep(_TICK_SECONDS * 3)
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


class _SyncHandler:
    def __init__(self) -> None:
        self.thread_id: int | None = None

    def handle(self, payload: object) -> object:
        self.thread_id = threading.get_ident()
        return {"from": "sync"}


class _AsyncHandleHandler:
    def __init__(self) -> None:
        self.thread_id: int | None = None

    async def handle(self, payload: object) -> object:
        self.thread_id = threading.get_ident()
        return {"from": "async-handle"}


class _HandleAsyncHandler:
    def __init__(self) -> None:
        self.sync_thread_id: int | None = None
        self.async_thread_id: int | None = None

    def handle(self, payload: object) -> object:
        self.sync_thread_id = threading.get_ident()
        return {"from": "sync"}

    async def handle_async(self, payload: object) -> object:
        self.async_thread_id = threading.get_ident()
        return {"from": "handle_async"}


class _BlockingHandler:
    def handle(self, payload: object) -> object:
        time.sleep(_BLOCK_SECONDS)
        return {"from": "blocking"}


class _FailingHandler:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.thread_id: int | None = None

    def handle(self, payload: object) -> object:
        self.thread_id = threading.get_ident()
        raise self.error


@pytest.mark.asyncio
async def test_entry_method_plain_function_runs_on_worker_thread() -> None:
    handler = _SyncHandler()
    await _dispatch(handler)
    assert handler.thread_id is not None
    assert handler.thread_id != threading.get_ident()


@pytest.mark.asyncio
async def test_entry_method_async_handle_stays_on_loop_thread() -> None:
    handler = _AsyncHandleHandler()
    await _dispatch(handler)
    assert handler.thread_id == threading.get_ident()


@pytest.mark.asyncio
async def test_entry_method_handle_async_stays_on_loop_thread() -> None:
    handler = _HandleAsyncHandler()
    await _dispatch(handler)
    assert handler.async_thread_id == threading.get_ident()
    assert handler.sync_thread_id is None


@pytest.mark.asyncio
async def test_heartbeat_gap_stays_small_while_plain_handler_blocks() -> None:
    async with _Heartbeat() as beat:
        await _dispatch(_BlockingHandler())
    assert beat.max_gap < 0.25


@pytest.mark.asyncio
async def test_heartbeat_control_direct_call_blocks_the_loop() -> None:
    """Positive control: the old direct call freezes the loop for the block time."""
    handler = _BlockingHandler()
    async with _Heartbeat() as beat:
        handler.handle(object())
    assert beat.max_gap >= 0.48


@pytest.mark.asyncio
async def test_error_chain_plain_handler_error_reaches_dispatch_unchanged() -> None:
    cause = KeyError("root cause")
    error = ValueError("boom")
    error.__cause__ = cause
    handler = _FailingHandler(error)
    with pytest.raises(ValueError, match="boom") as excinfo:
        await _dispatch(handler)
    assert excinfo.value is error
    assert excinfo.value.__cause__ is cause
    assert handler.thread_id != threading.get_ident()
