# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration: a blocking plain-function entry method leaves the loop live (OMN-24491).

Runs the real ``invoke_entry_method`` on a real event loop with a real blocking
call (no mocks) and reads the loop heartbeat while the handler blocks.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from omnibase_infra.handlers.entry_method_offload import invoke_entry_method

pytestmark = pytest.mark.integration

_BLOCK_SECONDS = 0.5
_TICK_SECONDS = 0.02


class _BlockingHandler:
    def __init__(self) -> None:
        self.thread_id: int | None = None

    def handle(self, arg: object) -> str:
        self.thread_id = threading.get_ident()
        time.sleep(_BLOCK_SECONDS)
        return f"done:{arg}"


async def _max_gap_while(task: asyncio.Task[object]) -> float:
    gap = 0.0
    last = time.monotonic()
    while not task.done():
        await asyncio.sleep(_TICK_SECONDS)
        now = time.monotonic()
        gap = max(gap, now - last)
        last = now
    return gap


@pytest.mark.asyncio
async def test_blocking_plain_handler_runs_off_loop_and_loop_stays_live() -> None:
    handler = _BlockingHandler()
    task = asyncio.create_task(invoke_entry_method(handler.handle, "x"))
    gap = await _max_gap_while(task)
    assert await task == "done:x"
    assert handler.thread_id != threading.get_ident()
    assert gap < _BLOCK_SECONDS / 2


@pytest.mark.asyncio
async def test_plain_handler_error_is_the_same_object() -> None:
    boom = RuntimeError("boom")

    def handle(arg: object) -> object:
        raise boom

    with pytest.raises(RuntimeError) as excinfo:
        await invoke_entry_method(handle, None)
    assert excinfo.value is boom
