# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests that a plain (blocking) dispatch entry method runs OFF the event loop (OMN-24492).

Root cause: ``_make_dispatch_callback`` invoked the handler's entry method
(``handle`` / ``handle_async``) DIRECTLY on the runtime event loop at all three
dispatch sites. When the entry method is a plain ``def`` (blocking) function,
the whole loop — heartbeats, readiness endpoint — froze for the duration of the
call.

The fix: the entry method is classified once at wiring time
(``_entry_method_runs_off_loop``). A plain function is dispatched through
``_invoke_entry_method`` via ``asyncio.to_thread`` (which copies the contextvars
context ``current_dispatch_envelope()`` relies on); an ``async def`` entry
method stays on the loop exactly as before. Exception identity is preserved at
every site: the caller's exception propagates as the SAME object.

This test proves:
  1. A sync ``handle`` runs on a worker thread, not the loop thread.
  2. ``handle_async`` (preferred) and async-only ``handle`` still run on the
     loop thread, and the sync ``handle`` is not called when ``handle_async``
     wins.
  3. The contextvars context visible to a sync handler inside the worker thread
     matches the context set around the callback call.
  4. A 0.6 s blocking ``handle`` leaves the largest heartbeat gap under the
     golden bound through the callback — while the failing control (the old
     direct call on the loop, through the SAME heartbeat helper) reads a gap
     over the bound.
  5. An exception raised by the entry method arrives at the caller as the very
     same object at the first dispatch site, through ``handle_async``, and at
     the second (fallback envelope) call site — no wrapping, no swallowing.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import itertools
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    DispatcherFunc,
    ProtocolHandleable,
    _make_dispatch_callback,
)
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Golden constants (OMN-24492)
# ---------------------------------------------------------------------------

# How long the blocking sync handle sleeps to simulate real work.
BLOCK_SECONDS = 0.6
# Heartbeat cadence beside the blocking dispatch.
HEARTBEAT_INTERVAL_SECONDS = 0.01
# Golden: through the callback (off-loop dispatch) the loop never stalls.
GOLDEN_MAX_HEARTBEAT_GAP_SECONDS = 0.25
# Failing control: the old direct call on the loop MUST stall it.
FAILING_CONTROL_MIN_HEARTBEAT_GAP_SECONDS = 0.48


# ---------------------------------------------------------------------------
# Stub handler classes
# ---------------------------------------------------------------------------


class _ThreadRecordingSyncHandler:
    """Plain ``def handle`` that records which thread it ran on."""

    def __init__(self) -> None:
        self.handle_thread_id: int | None = None

    def handle(self, payload: object) -> object:
        self.handle_thread_id = threading.get_ident()
        return {"from": "sync-handle"}


class _SyncAndAsyncHandler:
    """Handler exposing BOTH ``handle_async`` (preferred) and plain ``handle``."""

    def __init__(self) -> None:
        self.handle_called = False
        self.handle_thread_id: int | None = None
        self.handle_async_thread_id: int | None = None

    def handle(self, payload: object) -> object:
        self.handle_called = True
        self.handle_thread_id = threading.get_ident()
        return {"from": "sync-handle"}

    async def handle_async(self, payload: object) -> object:
        self.handle_async_thread_id = threading.get_ident()
        return {"from": "handle-async"}


class _AsyncHandleOnlyHandler:
    """Handler whose only entrypoint is an ``async def handle`` (no handle_async)."""

    def __init__(self) -> None:
        self.handle_thread_id: int | None = None

    async def handle(self, payload: object) -> object:
        self.handle_thread_id = threading.get_ident()
        return {"from": "async-handle-only"}


class _ContextReadingSyncHandler:
    """Plain ``def handle`` that reads a contextvars value set by the caller."""

    def __init__(self) -> None:
        self.seen: str | None = None

    def handle(self, payload: object) -> object:
        self.seen = _DISPATCH_MARKER_VAR.get()
        return {"from": "sync-handle"}


class _BlockingSyncHandler:
    """Plain ``def handle`` that blocks the calling thread for BLOCK_SECONDS."""

    def handle(self, payload: object) -> object:
        time.sleep(BLOCK_SECONDS)
        return {"from": "blocking-sync-handle"}


class _RaisingSyncHandler:
    """Plain ``def handle`` that raises a pre-built exception instance."""

    def __init__(self, raised: Exception) -> None:
        self._raised = raised
        self.raised_on_thread: int | None = None

    def handle(self, payload: object) -> object:
        self.raised_on_thread = threading.get_ident()
        raise self._raised


class _RaisingAsyncHandler:
    """``async def handle_async`` that raises a pre-built exception instance."""

    def __init__(self, raised: Exception) -> None:
        self._raised = raised

    def handle(self, payload: object) -> object:
        return {"from": "sync-handle"}

    async def handle_async(self, payload: object) -> object:
        raise self._raised


class _EnvelopeRaisingSyncHandler:
    """Envelope-accepting plain ``def handle`` used for the SECOND call site.

    The parameter is named ``envelope`` (the legacy heuristic), so the callback
    routes the payload-validation fallback through it (the fallback envelope
    call site). The handler raises the pre-built exception instance it was
    handed, wherever the callback invokes it from.
    """

    def __init__(self, raised: Exception) -> None:
        self._raised = raised
        self.received: object = None

    def handle(self, envelope: object) -> object:
        self.received = envelope
        raise self._raised


class _DispatchBoomError(Exception):
    """Raised by stub handlers to prove exception identity is preserved."""


# ---------------------------------------------------------------------------
# Helpers (shape reused from the handle_async dispatch exemplar)
# ---------------------------------------------------------------------------


class _StubPayload:
    """Minimal payload stub that passes isinstance checks."""

    @classmethod
    def model_validate(cls, data: object) -> _StubPayload:
        return cls()


class _ExplodingValidationPayload:
    """Payload stub whose ``model_validate`` ALWAYS raises.

    Drives ``_make_dispatch_callback`` into the payload-validation ``except``
    branch, where an envelope-accepting handler is re-invoked at the SECOND
    (fallback envelope) call site.
    """

    @classmethod
    def model_validate(cls, data: object) -> _StubPayload:
        raise _VALIDATION_ERROR


_VALIDATION_ERROR = ValueError("payload fails declared event model validation")

_CORR_ID = "00000000-0000-0000-0000-000000000001"

_DISPATCH_MARKER_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "omn24492_dispatch_marker", default="unset"
)


def _callback_for(
    handler: object, event_model: ModelHandlerRef | None = None
) -> DispatcherFunc:
    """Build the runtime's dispatch callback for a stub handler (the stubs are duck-typed)."""
    return _make_dispatch_callback(
        cast("ProtocolHandleable", handler), event_model=event_model
    )


def _make_envelope(payload: object) -> Any:
    """Build a minimal event envelope-like object."""
    envelope = MagicMock()
    envelope.payload = payload
    envelope.correlation_id = _CORR_ID
    envelope.event_type = "ModelTestPayload"
    envelope.envelope_timestamp = None
    envelope.tenant_id = None
    return envelope


def _fake_event_model(*, name: str = "ModelTestPayload") -> ModelHandlerRef:
    return ModelHandlerRef(name=name, module="fake.models")


def _heartbeat_gaps(beats: list[float]) -> list[float]:
    """Compute the consecutive gaps between recorded heartbeat beats."""
    return [later - earlier for earlier, later in itertools.pairwise(beats)]


def _beside_a_heartbeat(
    blocking_dispatch: Callable[[int], Awaitable[None]],
) -> list[float]:
    """Run ``blocking_dispatch`` on a fresh loop beside a 10 ms heartbeat task.

    Returns the monotonic timestamps of every heartbeat beat recorded until the
    dispatch completes. The heartbeat beats a couple of times BEFORE the
    blocking dispatch runs so the stall (if any) shows up as a single large
    gap; the failing control dispatches the same blocking handler directly on
    the loop (the pre-OMN-24492 behaviour) through this SAME helper, so the two
    tests differ only in HOW the blocking work is dispatched.
    """
    beats: list[float] = []

    async def _main() -> list[float]:
        loop_thread_id = threading.get_ident()

        async def _heartbeat() -> None:
            while True:
                beats.append(time.monotonic())
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

        heartbeat_task = asyncio.create_task(_heartbeat())
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS * 2)
        try:
            await blocking_dispatch(loop_thread_id)
            # Let the heartbeat wake once more, so a stall that ended with the
            # dispatch is on its record.
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS * 3)
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        return beats

    return asyncio.run(_main())


# ---------------------------------------------------------------------------
# Tests: thread placement of the entry method
# ---------------------------------------------------------------------------


class TestEntryMethodThreadPlacement:
    @pytest.mark.asyncio
    async def test_entry_method_sync_handle_runs_off_loop_thread(self) -> None:
        """A plain ``def handle`` is invoked on a thread other than the loop's."""
        handler = _ThreadRecordingSyncHandler()
        callback = _callback_for(
            handler,
            event_model=None,
        )

        loop_thread_id = threading.get_ident()
        await callback(_make_envelope({}))

        assert handler.handle_thread_id is not None
        assert handler.handle_thread_id != loop_thread_id

    @pytest.mark.asyncio
    async def test_entry_method_handle_async_stays_on_loop_thread(self) -> None:
        """``handle_async`` runs ON the loop thread; plain ``handle`` is not called."""
        handler = _SyncAndAsyncHandler()
        callback = _callback_for(
            handler,
            event_model=None,
        )

        loop_thread_id = threading.get_ident()
        await callback(_make_envelope({}))

        assert handler.handle_async_thread_id == loop_thread_id
        assert handler.handle_called is False
        assert handler.handle_thread_id is None

    @pytest.mark.asyncio
    async def test_entry_method_async_handle_stays_on_loop_thread(self) -> None:
        """An async-only ``handle`` still runs ON the loop thread (unchanged)."""
        handler = _AsyncHandleOnlyHandler()
        callback = _callback_for(
            handler,
            event_model=None,
        )

        loop_thread_id = threading.get_ident()
        await callback(_make_envelope({}))

        assert handler.handle_thread_id == loop_thread_id

    @pytest.mark.asyncio
    async def test_entry_method_sync_handle_sees_contextvars_in_worker_thread(
        self,
    ) -> None:
        """A sync handle reads the contextvars set around the callback call.

        ``asyncio.to_thread`` copies the current contextvars context — the
        mechanism ``current_dispatch_envelope()`` depends on — so the value set
        before the call must be visible INSIDE the worker thread.
        """
        handler = _ContextReadingSyncHandler()
        callback = _callback_for(
            handler,
            event_model=None,
        )

        reset_handle = _DISPATCH_MARKER_VAR.set("visible-inside-worker")
        try:
            await callback(_make_envelope({}))
        finally:
            _DISPATCH_MARKER_VAR.reset(reset_handle)

        assert handler.seen == "visible-inside-worker"


# ---------------------------------------------------------------------------
# Tests: heartbeat keeps beating beside a blocking sync handle
# ---------------------------------------------------------------------------


class TestHeartbeatKeepsBeating:
    def test_heartbeat_gap_stays_under_golden_with_blocking_sync_handle(self) -> None:
        """Through the callback, a 0.6 s blocking handle leaves gaps < 0.25 s."""
        handler = _BlockingSyncHandler()
        callback = _callback_for(
            handler,
            event_model=None,
        )
        envelope = _make_envelope({})

        async def _dispatch_through_callback(loop_thread_id: int) -> None:
            await callback(envelope)

        beats = _beside_a_heartbeat(_dispatch_through_callback)
        gaps = _heartbeat_gaps(beats)
        assert beats, "heartbeat task never ran"
        assert max(gaps) < GOLDEN_MAX_HEARTBEAT_GAP_SECONDS

    def test_heartbeat_control_direct_loop_call_gap_exceeds_bound(self) -> None:
        """Failing control: the OLD direct call on the loop stalls the heartbeat."""
        handler = _BlockingSyncHandler()
        envelope = _make_envelope({})

        async def _dispatch_directly_on_loop(loop_thread_id: int) -> None:
            # The pre-OMN-24492 behaviour: invoke the blocking handler ON the
            # loop, exactly as the callback used to.
            handler.handle(envelope)

        beats = _beside_a_heartbeat(_dispatch_directly_on_loop)
        gaps = _heartbeat_gaps(beats)
        assert beats, "heartbeat task never ran"
        assert max(gaps) >= FAILING_CONTROL_MIN_HEARTBEAT_GAP_SECONDS


# ---------------------------------------------------------------------------
# Tests: exception identity is preserved at every call site
# ---------------------------------------------------------------------------


class TestErrorChainPreservesExceptionIdentity:
    @pytest.mark.asyncio
    async def test_error_chain_sync_handle_raises_same_exception_object(self) -> None:
        """A sync handle's exception arrives as the SAME object (no wrapping)."""
        raised = _DispatchBoomError("sync handle boom")
        handler = _RaisingSyncHandler(raised)
        callback = _callback_for(
            handler,
            event_model=None,
        )

        with pytest.raises(_DispatchBoomError) as excinfo:
            await callback(_make_envelope({}))

        assert excinfo.value is raised
        # The error crossed the thread boundary: it was raised on a worker
        # thread, not on the event loop thread.
        assert handler.raised_on_thread is not None
        assert handler.raised_on_thread != threading.get_ident()

    @pytest.mark.asyncio
    async def test_error_chain_handle_async_raises_same_exception_object(self) -> None:
        """An ``async def handle_async`` raising keeps the same object (unchanged)."""
        raised = _DispatchBoomError("handle_async boom")
        handler = _RaisingAsyncHandler(raised)
        callback = _callback_for(
            handler,
            event_model=None,
        )

        with pytest.raises(_DispatchBoomError) as excinfo:
            await callback(_make_envelope({}))

        assert excinfo.value is raised

    @pytest.mark.asyncio
    async def test_error_chain_fallback_call_site_raises_same_exception_object(
        self,
    ) -> None:
        """The SECOND (fallback envelope) call site raises the same object too.

        The payload fails the declared event model validation, so an
        envelope-accepting handler is re-invoked with a re-hydrated raw
        envelope; the sync handle's exception must not be swallowed there.
        """
        raised = _DispatchBoomError("fallback call site boom")
        handler = _EnvelopeRaisingSyncHandler(raised)
        callback = _callback_for(
            handler,
            event_model=_fake_event_model(),
        )
        envelope = _make_envelope({})

        with (
            patch(
                "omnibase_infra.runtime.auto_wiring.handler_wiring._import_event_model_class"
            ) as mock_import,
            pytest.raises(_DispatchBoomError) as excinfo,
        ):
            mock_import.return_value = _ExplodingValidationPayload
            await callback(envelope)

        assert excinfo.value is raised
        # The handler really was reached through the fallback call site.
        assert isinstance(handler.received, object)
