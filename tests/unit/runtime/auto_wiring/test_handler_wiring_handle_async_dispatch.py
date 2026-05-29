# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that _make_dispatch_callback prefers handle_async over handle (OMN-12002).

Root cause: auto-wired dispatch callbacks previously called handler_instance.handle
unconditionally. Orchestrator handlers (e.g. HandlerSwarmDispatchOrchestrator) expose
handle_async as the runtime-publish entry point; handle is the no-publish test/
standalone path. Messages dispatched via the event-bus callback loop therefore never
triggered the handler's Kafka publishes.

The fix: _make_dispatch_callback prefers handle_async when callable, falling back
to handle for handlers that only implement the sync interface.

This test proves:
  1. When a handler declares handle_async, the dispatch callback calls handle_async.
  2. When a handler only declares handle (no handle_async), handle is still called.
  3. Side effects (e.g. bus.publish calls) made inside handle_async are executed.
  4. The callback still awaits coroutine results from handle_async.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stub handler classes
# ---------------------------------------------------------------------------


class _HandlerWithHandleAsync:
    """Simulates an FSM orchestrator that publishes during handle_async."""

    def __init__(self, bus: Any) -> None:
        self._bus = bus
        self.handle_called = False
        self.handle_async_called = False

    def handle(self, payload: object) -> object:
        self.handle_called = True
        return {"from": "sync-handle"}

    async def handle_async(self, payload: object) -> None:
        self.handle_async_called = True
        # Side effect: publish a command to downstream topic
        await self._bus.publish(
            topic="onex.cmd.omnimarket.swarm-check-endpoint-health.v1",
            key=None,
            value=b'{"run_id": "r1"}',
        )


class _HandlerSyncOnly:
    """Simulates a legacy handler with only handle (no handle_async)."""

    def __init__(self) -> None:
        self.handle_called = False

    def handle(self, payload: object) -> object:
        self.handle_called = True
        return {"from": "sync-only"}


class _HandlerAsyncHandleOnly:
    """Handler where handle itself is async (no handle_async)."""

    def __init__(self) -> None:
        self.handle_called = False

    async def handle(self, payload: object) -> object:
        self.handle_called = True
        return {"from": "async-handle-only"}


class _HandlerWithoutHandle:
    """Handler that has only a domain-specific method name."""

    def execute(self, payload: object) -> object:
        return {"from": "execute"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubPayload:
    """Minimal payload stub that passes isinstance checks."""

    @classmethod
    def model_validate(cls, data: object) -> _StubPayload:
        return cls()


_CORR_ID = "00000000-0000-0000-0000-000000000001"


def _make_envelope(payload: object) -> object:
    """Build a minimal event envelope-like object."""
    envelope = MagicMock()
    envelope.payload = payload
    envelope.correlation_id = _CORR_ID
    envelope.event_type = "ModelTestPayload"
    return envelope


from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef


def _fake_event_model(*, name: str = "ModelTestPayload") -> ModelHandlerRef:
    return ModelHandlerRef(name=name, module="fake.models")


# ---------------------------------------------------------------------------
# Tests: handle_async is preferred
# ---------------------------------------------------------------------------


class TestHandleAsyncPreference:
    @pytest.mark.asyncio
    async def test_handle_async_called_when_present(self) -> None:
        """Dispatch callback calls handle_async, not handle, when handle_async exists."""
        mock_bus = AsyncMock()
        handler = _HandlerWithHandleAsync(bus=mock_bus)

        payload_obj = {
            "task": "test-task",
            "endpoint_ids": ["ep-1"],
            "run_id": "r1",
            "correlation_id": "c1",
        }

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_event_model_class"
        ) as mock_import:
            # Make model_validate return the dict as-is
            mock_import.return_value = _StubPayload

            callback = _make_dispatch_callback(
                handler,  # type: ignore[arg-type]
                event_model=_fake_event_model(),
            )
            envelope = _make_envelope(payload_obj)
            await callback(envelope)  # type: ignore[arg-type]

        # handle_async was called; handle was NOT called
        assert handler.handle_async_called is True
        assert handler.handle_called is False

    @pytest.mark.asyncio
    async def test_side_effect_publishes_execute(self) -> None:
        """Bus.publish inside handle_async is actually called (not dropped)."""
        mock_bus = AsyncMock()
        handler = _HandlerWithHandleAsync(bus=mock_bus)

        payload_obj = {
            "task": "test-task",
            "endpoint_ids": [],
            "run_id": "r1",
            "correlation_id": "c1",
        }

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_event_model_class"
        ) as mock_import:
            mock_import.return_value = _StubPayload

            callback = _make_dispatch_callback(
                handler,  # type: ignore[arg-type]
                event_model=_fake_event_model(),
            )
            envelope = _make_envelope(payload_obj)
            await callback(envelope)  # type: ignore[arg-type]

        # The downstream publish inside handle_async was executed
        mock_bus.publish.assert_awaited_once_with(
            topic="onex.cmd.omnimarket.swarm-check-endpoint-health.v1",
            key=None,
            value=b'{"run_id": "r1"}',
        )

    @pytest.mark.asyncio
    async def test_multi_topic_publishes_all_execute(self) -> None:
        """All bus.publish calls inside handle_async fire (not just first/primary)."""

        class _MultiPublishHandler:
            def __init__(self, bus: Any) -> None:
                self._bus = bus

            def handle(self, payload: object) -> object:
                return {}

            async def handle_async(self, payload: object) -> object:
                for topic in [
                    "onex.cmd.omnimarket.swarm-check-endpoint-health.v1",
                    "onex.cmd.omnimarket.swarm-decompose.v1",
                    "onex.cmd.omnimarket.swarm-select-endpoints.v1",
                ]:
                    await self._bus.publish(topic=topic, key=None, value=b"{}")
                return {}

        mock_bus = AsyncMock()
        handler = _MultiPublishHandler(bus=mock_bus)

        payload_obj = {}

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_event_model_class"
        ) as mock_import:
            mock_import.return_value = _StubPayload

            callback = _make_dispatch_callback(
                handler,  # type: ignore[arg-type]
                event_model=_fake_event_model(),
            )
            envelope = _make_envelope(payload_obj)
            await callback(envelope)  # type: ignore[arg-type]

        # All three publish calls fired
        assert mock_bus.publish.await_count == 3
        published_topics = [
            call.kwargs["topic"] for call in mock_bus.publish.await_args_list
        ]
        assert "onex.cmd.omnimarket.swarm-check-endpoint-health.v1" in published_topics
        assert "onex.cmd.omnimarket.swarm-decompose.v1" in published_topics
        assert "onex.cmd.omnimarket.swarm-select-endpoints.v1" in published_topics


# ---------------------------------------------------------------------------
# Tests: fallback to handle when handle_async absent
# ---------------------------------------------------------------------------


class TestHandleFallback:
    @pytest.mark.asyncio
    async def test_handle_called_when_no_handle_async(self) -> None:
        """Dispatch callback falls back to handle when handle_async is absent."""
        handler = _HandlerSyncOnly()

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_event_model_class"
        ) as mock_import:
            mock_import.return_value = _StubPayload

            callback = _make_dispatch_callback(
                handler,  # type: ignore[arg-type]
                event_model=_fake_event_model(),
            )
            envelope = _make_envelope({})
            await callback(envelope)  # type: ignore[arg-type]

        assert handler.handle_called is True

    @pytest.mark.asyncio
    async def test_async_handle_without_handle_async(self) -> None:
        """Handler with async handle (but no handle_async) still works correctly."""
        handler = _HandlerAsyncHandleOnly()

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_event_model_class"
        ) as mock_import:
            mock_import.return_value = _StubPayload

            callback = _make_dispatch_callback(
                handler,  # type: ignore[arg-type]
                event_model=_fake_event_model(),
            )
            envelope = _make_envelope({})
            await callback(envelope)  # type: ignore[arg-type]

        assert handler.handle_called is True

    @pytest.mark.asyncio
    async def test_non_callable_handle_async_attribute_is_ignored(self) -> None:
        """A non-callable handle_async attribute (e.g. None) falls through to handle."""

        class _HandlerWithNonCallableHandleAsync:
            handle_async = None  # not callable

            def __init__(self) -> None:
                self.handle_called = False

            def handle(self, payload: object) -> object:
                self.handle_called = True
                return {}

        handler = _HandlerWithNonCallableHandleAsync()

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_event_model_class"
        ) as mock_import:
            mock_import.return_value = _StubPayload

            callback = _make_dispatch_callback(
                handler,  # type: ignore[arg-type]
                event_model=_fake_event_model(),
            )
            envelope = _make_envelope({})
            await callback(envelope)  # type: ignore[arg-type]

        assert handler.handle_called is True

    @pytest.mark.asyncio
    async def test_missing_handle_fails_at_dispatch_not_wiring(self) -> None:
        """Missing legacy handle does not fail startup-time callback construction."""
        from omnibase_core.models.errors import ModelOnexError

        handler = _HandlerWithoutHandle()
        callback = _make_dispatch_callback(
            handler,  # type: ignore[arg-type]
            event_model=None,
        )

        with pytest.raises(ModelOnexError, match="does not expose a callable"):
            await callback(_make_envelope({}))  # type: ignore[arg-type]
