# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the entry-method placement handler (OMN-24492).

The runtime froze its event loop whenever a handler's dispatch entry method
was a plain (blocking) function: ``_make_dispatch_callback`` invoked the
entry method directly on the loop at all three dispatch sites, so the loop —
heartbeats, readiness endpoint — stalled for as long as the handler ran
(~70 s every 5 min on the dev runtime on 2026-11-14).

``HandlerEntryMethodPlacement`` classifies an entry method once (``async
def`` -> ON_LOOP, plain ``def`` -> WORKER_THREAD) and its ``caller`` wraps
the entry method so the dispatch sites can simply ``await`` it: a plain
entry method runs through ``asyncio.to_thread`` on a worker thread (with
the contextvars context copied, which ``current_dispatch_envelope()``
relies on), an async entry method stays on the loop. Any exception the
entry method raises propagates as the SAME object.

These are plain (non-async) test functions driving the async caller with
``asyncio.run`` — no pytest-asyncio plugin needed.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from pydantic import ValidationError

from omnibase_infra.enums.enum_entry_method_placement import (
    EnumEntryMethodPlacement,
)
from omnibase_infra.handlers.handler_entry_method_placement.handler import (
    HandlerEntryMethodPlacement,
)
from omnibase_infra.models.entry_method_placement.model_entry_method_placement_request import (
    ModelEntryMethodPlacementRequest,
)

pytestmark = pytest.mark.unit


class _EntryBoomError(Exception):
    """Raised by stub entry methods to prove exception identity is preserved."""


def _plain_entry(argument: object) -> object:
    """A plain (blocking-shaped) entry function."""
    return {"from": "plain-entry"}


async def _async_entry(argument: object) -> object:
    """An ``async def`` entry function."""
    return {"from": "async-entry"}


class _DualEntryHandler:
    """Handler exposing BOTH a plain ``handle`` and an async ``handle_async``."""

    def handle(self, payload: object) -> object:
        return {"from": "bound-plain-handle"}

    async def handle_async(self, payload: object) -> object:
        return {"from": "bound-async-handle-async"}


# ---------------------------------------------------------------------------
# Golden: the placement decision
# ---------------------------------------------------------------------------


class TestGoldenPlacementDecision:
    """async def stays on the loop; a plain def goes to a worker thread."""

    def test_golden_plain_function_places_on_worker_thread(self) -> None:
        placement = (
            HandlerEntryMethodPlacement()
            .handle(ModelEntryMethodPlacementRequest(entry=_plain_entry))
            .placement
        )

        assert placement is EnumEntryMethodPlacement.WORKER_THREAD

    def test_golden_async_function_places_on_loop(self) -> None:
        placement = (
            HandlerEntryMethodPlacement()
            .handle(ModelEntryMethodPlacementRequest(entry=_async_entry))
            .placement
        )

        assert placement is EnumEntryMethodPlacement.ON_LOOP

    def test_golden_bound_async_method_places_on_loop(self) -> None:
        placement = (
            HandlerEntryMethodPlacement()
            .handle(
                ModelEntryMethodPlacementRequest(entry=_DualEntryHandler().handle_async)
            )
            .placement
        )

        assert placement is EnumEntryMethodPlacement.ON_LOOP

    def test_golden_bound_plain_method_places_on_worker_thread(self) -> None:
        placement = (
            HandlerEntryMethodPlacement()
            .handle(ModelEntryMethodPlacementRequest(entry=_DualEntryHandler().handle))
            .placement
        )

        assert placement is EnumEntryMethodPlacement.WORKER_THREAD


# ---------------------------------------------------------------------------
# Golden: the caller invokes at the decided placement
# ---------------------------------------------------------------------------


class TestGoldenCallerInvocation:
    """``caller`` runs the entry method where its placement says to."""

    def test_golden_caller_of_plain_function_runs_off_loop_and_returns_value(
        self,
    ) -> None:
        seen_thread_ids: list[int] = []

        def entry(argument: object) -> object:
            seen_thread_ids.append(threading.get_ident())
            return "plain-value"

        call = HandlerEntryMethodPlacement().caller(entry)

        async def _run() -> tuple[object, int]:
            loop_thread_id = threading.get_ident()
            return await call({}), loop_thread_id

        result, loop_thread_id = asyncio.run(_run())

        assert result == "plain-value"
        assert len(seen_thread_ids) == 1
        assert seen_thread_ids[0] != loop_thread_id

    def test_golden_caller_of_async_function_runs_on_loop_thread(self) -> None:
        seen_thread_ids: list[int] = []

        async def entry(argument: object) -> object:
            seen_thread_ids.append(threading.get_ident())
            return "async-value"

        call = HandlerEntryMethodPlacement().caller(entry)

        async def _run() -> tuple[object, int]:
            loop_thread_id = threading.get_ident()
            return await call({}), loop_thread_id

        result, loop_thread_id = asyncio.run(_run())

        assert result == "async-value"
        assert len(seen_thread_ids) == 1
        assert seen_thread_ids[0] == loop_thread_id

    def test_golden_caller_of_plain_function_returning_coroutine_awaits_once(
        self,
    ) -> None:
        ran_on_thread_ids: list[int] = []

        async def inner() -> str:
            ran_on_thread_ids.append(threading.get_ident())
            return "final-value"

        def entry(argument: object) -> object:
            return inner()

        call = HandlerEntryMethodPlacement().caller(entry)

        async def _run() -> object:
            return await call({})

        result = asyncio.run(_run())

        assert result == "final-value"
        # The coroutine body ran exactly once — awaiting the same coroutine a
        # second time would raise RuntimeError instead of returning a value.
        assert len(ran_on_thread_ids) == 1


# ---------------------------------------------------------------------------
# Error: parse-time rejection and exception identity
# ---------------------------------------------------------------------------


class TestErrorChain:
    """A non-callable entry fails at parse time; entry errors keep identity."""

    def test_error_non_callable_entry_is_rejected_at_parse_time(self) -> None:
        with pytest.raises(ValidationError):
            ModelEntryMethodPlacementRequest(entry="not callable")

    def test_error_plain_entry_exception_arrives_as_same_object(self) -> None:
        raised = _EntryBoomError("plain entry boom")

        def entry(argument: object) -> object:
            raise raised

        call = HandlerEntryMethodPlacement().caller(entry)

        async def _run() -> _EntryBoomError:
            with pytest.raises(_EntryBoomError) as excinfo:
                await call({})
            return excinfo.value

        assert asyncio.run(_run()) is raised

    def test_error_async_entry_exception_arrives_as_same_object(self) -> None:
        raised = _EntryBoomError("async entry boom")

        async def entry(argument: object) -> object:
            raise raised

        call = HandlerEntryMethodPlacement().caller(entry)

        async def _run() -> _EntryBoomError:
            with pytest.raises(_EntryBoomError) as excinfo:
                await call({})
            return excinfo.value

        assert asyncio.run(_run()) is raised
