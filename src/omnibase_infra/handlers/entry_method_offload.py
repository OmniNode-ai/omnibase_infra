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
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import cast

from omnibase_infra.handlers.contract_dispatch_cost import run_in_worker_timed


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
        result = await run_in_worker_timed(handle_method, dispatch_arg)
    else:
        result = handle_method(dispatch_arg)
    if asyncio.iscoroutine(result):
        return await cast("Awaitable[object]", result)
    return result


async def yield_each[T](items: Iterable[T]) -> AsyncIterator[T]:
    """Yield each item, then give the event loop one turn before the next (OMN-26320).

    ``wire_from_manifest`` prepares every discovered contract and imports every
    handler class in plain loops whose steps are synchronous, so the one runtime
    event loop was held for the whole pass: 12.4 s on a lab run of the real
    kernel with the dev contract set (807 contracts, 1834 handler classes),
    during which it served neither /health nor any bus consumer. Used as
    ``async for item in yield_each(items):`` in place of ``for item in items:``
    it hands the loop back after every item, so the longest hold becomes the
    slowest single item instead of the sum.

    Items run in the same order, on the loop thread, and an exception raised
    while an item is handled propagates to the awaiting caller unchanged. The
    work stays on the loop thread on purpose: preparing a contract constructs
    handler instances and imports their modules, and none of that code has been
    audited for loop or thread affinity.
    """
    for item in items:
        yield item
        await asyncio.sleep(0)
