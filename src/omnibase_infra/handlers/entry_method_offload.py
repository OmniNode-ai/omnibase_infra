# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Run a plain-function handler entry method off the runtime event loop (OMN-24491).

The auto-wired dispatch callback runs on the one runtime event loop. A handler
whose entry method is a plain function blocks whatever thread calls it, so a job
waiting on the network, a database or a subprocess froze the readiness endpoint
and every Kafka heartbeat for as long as it waited. Such a method is run on a
worker thread instead. An ``async def`` entry method (``handle`` or
``handle_async``) awaits cooperatively and stays on the loop unchanged.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import cast


async def invoke_entry_method(
    handle_method: Callable[[object], object],
    dispatch_arg: object,
) -> object:
    """Call the entry method and await a coroutine result.

    A plain function (not an ``async def``) runs through ``asyncio.to_thread``,
    which copies the current context into the worker and re-raises the worker's
    exception object on the loop, so the dispatch path sees the same error it
    saw before.
    """
    if not inspect.iscoroutinefunction(handle_method):
        result = await asyncio.to_thread(handle_method, dispatch_arg)
    else:
        result = handle_method(dispatch_arg)
    if asyncio.iscoroutine(result):
        return await cast("Awaitable[object]", result)
    return result
