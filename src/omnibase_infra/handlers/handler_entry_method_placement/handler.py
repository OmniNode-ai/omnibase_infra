# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Entry-method placement for runtime dispatch (OMN-24492).

Why this exists
---------------
The runtime's dispatch callback used to invoke a handler's entry method
(``handle`` / ``handle_async``) directly on the event loop. When that entry
method is a plain ``def``, the whole loop — heartbeats, readiness endpoint —
froze for as long as the handler ran: on 2026-11-14 a plain entry function
froze the dev runtime loop for ~70 s every 5 min.

This handler decides, once per wiring, where an entry method belongs: an
``async def`` entry method cooperates with the loop and stays ON the loop
exactly as before; a plain ``def`` one is dispatched onto a worker thread
through ``asyncio.to_thread``. ``to_thread`` copies the contextvars context
— which the runtime's ``current_dispatch_envelope()`` relies on — and
propagates the entry method's exception as the very same object, unwrapped
and unlogged.

No new setting and no new constant: the worker threads are the loop's
default executor, the one the dispatch engine already uses for its sync
dispatchers.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_entry_method_placement import (
    EnumEntryMethodPlacement,
)
from omnibase_infra.models.entry_method_placement.model_entry_method_placement import (
    ModelEntryMethodPlacement,
)
from omnibase_infra.models.entry_method_placement.model_entry_method_placement_request import (
    ModelEntryMethodPlacementRequest,
)


class HandlerEntryMethodPlacement:
    """Decide where a handler's dispatch entry method runs. Pure."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Classification: a pure compute handler, not an infra/IO one."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """COMPUTE, and deterministically so — no clock, no I/O, no randomness."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelEntryMethodPlacementRequest
    ) -> ModelEntryMethodPlacement:
        """Classify the entry method: ON_LOOP for ``async def``, else worker."""
        if inspect.iscoroutinefunction(request.entry):
            return ModelEntryMethodPlacement(placement=EnumEntryMethodPlacement.ON_LOOP)
        return ModelEntryMethodPlacement(
            placement=EnumEntryMethodPlacement.WORKER_THREAD
        )

    def caller(
        self, entry: Callable[[object], object]
    ) -> Callable[[object], Awaitable[object]]:
        """Wrap ``entry`` so callers can ``await`` it at its decided placement.

        The placement is decided ONCE, here, through :meth:`handle` — the
        returned wrapper sits on the per-message dispatch hot path and must
        not re-introspect. An exception raised by ``entry`` propagates as the
        SAME exception object, with no wrapping and no logging, whatever
        placement ran it.
        """
        placement = self.handle(ModelEntryMethodPlacementRequest(entry=entry)).placement

        async def call(argument: object) -> object:
            if placement is EnumEntryMethodPlacement.WORKER_THREAD:
                # to_thread copies the contextvars context (what the runtime's
                # current_dispatch_envelope() relies on) and lets the entry's
                # exception propagate unchanged.
                result = await asyncio.to_thread(entry, argument)
            else:
                result = entry(argument)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        return call


__all__ = ["HandlerEntryMethodPlacement"]
