# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Admit one message at a time to a projection handler that owns its adapter (OMN-26430).

A runner-shaped projection handler builds its database adapter in ``__init__`` and
connects, uses and closes it inside one ``asyncio.run`` per message. The adapter
lives on the INSTANCE, so two threads inside ``handle`` together are one adapter
connected, used and closed on two loops at once. On the dev lane on 2026-11-23 that
state ended the runtime process twice with a segfault in the asyncpg protocol
extension and no log line, and the deploy agent rolled the lane back both times.

The runtime dispatches a projection handler with ``asyncio.to_thread`` and admits up
to ``PROJECTION_HANDLER_MAX_INFLIGHT`` threads. This module is the per-handler lock
in front of that gate: an adapter owner is admitted one message at a time, and a
handler that is handed its adapter per call (it shares nothing between threads) is
admitted as before.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from weakref import WeakKeyDictionary

# Held strongly beside its locks: a handler lives as long as the runtime, and the
# strong reference keeps ``id`` from naming a different handler after a collection.
_OWNER_LOCKS: dict[
    int, tuple[object, WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]]
] = {}


def serves_its_own_database(handler_instance: object) -> bool:
    """Whether the handler carries a DB adapter it opened itself.

    A handler on the DB-injection arm receives its adapter from the runtime as
    ``input_data["_db"]`` and holds none of its own. A runner-shaped handler
    constructs an async pool adapter in ``__init__`` and reaches it through
    ``self.db`` -- the runtime never sees that adapter's construction and cannot
    scope its lifetime.
    """
    db = getattr(handler_instance, "db", None)
    if db is None:
        return False
    return callable(getattr(db, "connect", None)) and callable(
        getattr(db, "close", None)
    )


@contextlib.asynccontextmanager
async def owned_database_gate(handler_instance: object) -> AsyncIterator[None]:
    """Hold the handler's lock for the call when it owns its adapter; else pass."""
    if not serves_its_own_database(handler_instance):
        yield
        return
    _, by_loop = _OWNER_LOCKS.setdefault(
        id(handler_instance), (handler_instance, WeakKeyDictionary())
    )
    loop = asyncio.get_running_loop()
    lock = by_loop.get(loop)
    if lock is None:
        lock = by_loop[loop] = asyncio.Lock()
    async with lock:
        yield
