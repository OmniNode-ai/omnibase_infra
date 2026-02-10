# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for IntentExecutor (service_intent_executor).

Tests the contract-driven intent executor which routes intents to effect
layer handlers based on the payload's ``intent_type`` field.

Test Coverage:
- Handler registration (store, overwrite)
- Single execute routing, early return on None payload, protocol checks
- Error propagation (RuntimeHostError passthrough, generic exception re-raise)
- Batch execute_all (sequential order, shared correlation_id, fail-fast)
- Correlation ID generation and propagation

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - IntentExecutor: Implementation under test
    - ProtocolIntentEffect: Handler protocol
    - ProtocolIntentPayload: Payload protocol
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.runtime.protocols.protocol_intent_payload import (
    ProtocolIntentPayload,
)
from omnibase_infra.runtime.service_intent_executor import (
    IntentExecutor,
    ProtocolIntentEffect,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockPayload:
    """Minimal payload implementing ProtocolIntentPayload."""

    intent_type: str = "test.operation"

    def __init__(self, intent_type: str = "test.operation") -> None:
        self.intent_type = intent_type


class _PayloadWithoutIntentType:
    """Payload missing intent_type -- does NOT satisfy ProtocolIntentPayload."""


def _make_intent(
    payload: object | None = None,
    *,
    intent_type: str = "extension",
) -> MagicMock:
    """Create a MagicMock that behaves like ModelIntent."""
    intent = MagicMock()
    intent.payload = payload
    intent.intent_type = intent_type
    return intent


def _make_handler() -> MagicMock:
    """Create an AsyncMock effect handler satisfying ProtocolIntentEffect."""
    handler = MagicMock(spec=ProtocolIntentEffect)
    handler.execute = AsyncMock()
    return handler


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegisterHandler:
    """Tests for IntentExecutor.register_handler."""

    def test_register_handler_stores_handler(self) -> None:
        """Should store the handler under the given intent_type key."""
        container = MagicMock()
        executor = IntentExecutor(container=container)
        handler = _make_handler()

        executor.register_handler("consul.register", handler)

        assert executor._effect_handlers["consul.register"] is handler

    def test_register_handler_overwrites_existing(self) -> None:
        """Second register for the same intent_type should replace the first."""
        container = MagicMock()
        executor = IntentExecutor(container=container)
        handler_a = _make_handler()
        handler_b = _make_handler()

        executor.register_handler("consul.register", handler_a)
        executor.register_handler("consul.register", handler_b)

        assert executor._effect_handlers["consul.register"] is handler_b


# ---------------------------------------------------------------------------
# Single execute tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecute:
    """Tests for IntentExecutor.execute."""

    @pytest.fixture
    def executor(self) -> IntentExecutor:
        """Create an IntentExecutor with a mock container."""
        return IntentExecutor(container=MagicMock())

    @pytest.mark.asyncio
    async def test_execute_routes_to_correct_handler(
        self, executor: IntentExecutor
    ) -> None:
        """Should call handler.execute with payload and correlation_id."""
        handler = _make_handler()
        executor.register_handler("test.operation", handler)

        payload = _MockPayload("test.operation")
        intent = _make_intent(payload=payload)
        cid = uuid4()

        await executor.execute(intent, correlation_id=cid)

        handler.execute.assert_awaited_once_with(payload, correlation_id=cid)

    @pytest.mark.asyncio
    async def test_execute_returns_early_on_none_payload(
        self, executor: IntentExecutor
    ) -> None:
        """Intent with payload=None should return without error."""
        intent = _make_intent(payload=None)

        # Should not raise
        await executor.execute(intent, correlation_id=uuid4())

    @pytest.mark.asyncio
    async def test_execute_raises_on_non_protocol_payload(
        self, executor: IntentExecutor
    ) -> None:
        """Payload without intent_type should raise RuntimeHostError."""
        payload = _PayloadWithoutIntentType()
        intent = _make_intent(payload=payload)

        with pytest.raises(RuntimeHostError, match="has no intent_type field"):
            await executor.execute(intent, correlation_id=uuid4())

    @pytest.mark.asyncio
    async def test_execute_raises_on_none_intent_type(
        self, executor: IntentExecutor
    ) -> None:
        """Payload with intent_type=None should raise RuntimeHostError."""
        payload = _MockPayload("test.operation")
        payload.intent_type = None  # type: ignore[assignment]
        intent = _make_intent(payload=payload)

        with pytest.raises(RuntimeHostError, match="no intent_type"):
            await executor.execute(intent, correlation_id=uuid4())

    @pytest.mark.asyncio
    async def test_execute_raises_on_missing_handler(
        self, executor: IntentExecutor
    ) -> None:
        """No handler registered for intent_type should raise RuntimeHostError."""
        payload = _MockPayload("unregistered.type")
        intent = _make_intent(payload=payload)

        with pytest.raises(RuntimeHostError, match="No effect handler registered"):
            await executor.execute(intent, correlation_id=uuid4())

    @pytest.mark.asyncio
    async def test_execute_propagates_runtime_host_error(
        self, executor: IntentExecutor
    ) -> None:
        """RuntimeHostError from handler should be re-raised unchanged."""
        handler = _make_handler()
        original_error = RuntimeHostError("handler boom")
        handler.execute.side_effect = original_error
        executor.register_handler("test.operation", handler)

        payload = _MockPayload("test.operation")
        intent = _make_intent(payload=payload)

        with pytest.raises(RuntimeHostError, match="handler boom") as exc_info:
            await executor.execute(intent, correlation_id=uuid4())

        assert exc_info.value is original_error

    @pytest.mark.asyncio
    async def test_execute_wraps_generic_exception(
        self, executor: IntentExecutor
    ) -> None:
        """Generic Exception from handler should still propagate (re-raised)."""
        handler = _make_handler()
        handler.execute.side_effect = ValueError("unexpected")
        executor.register_handler("test.operation", handler)

        payload = _MockPayload("test.operation")
        intent = _make_intent(payload=payload)

        with pytest.raises(ValueError, match="unexpected"):
            await executor.execute(intent, correlation_id=uuid4())


# ---------------------------------------------------------------------------
# Batch execute_all tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecuteAll:
    """Tests for IntentExecutor.execute_all."""

    @pytest.fixture
    def executor(self) -> IntentExecutor:
        """Create an IntentExecutor with a mock container."""
        return IntentExecutor(container=MagicMock())

    @pytest.mark.asyncio
    async def test_execute_all_processes_sequentially(
        self, executor: IntentExecutor
    ) -> None:
        """Handlers should be called in the order of the intent list."""
        call_order: list[str] = []

        handler_a = _make_handler()
        handler_a.execute = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append("a")
        )
        handler_b = _make_handler()
        handler_b.execute = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append("b")
        )

        executor.register_handler("op.a", handler_a)
        executor.register_handler("op.b", handler_b)

        intents = [
            _make_intent(payload=_MockPayload("op.a")),
            _make_intent(payload=_MockPayload("op.b")),
        ]

        await executor.execute_all(intents, correlation_id=uuid4())

        assert call_order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_execute_all_shares_correlation_id(
        self, executor: IntentExecutor
    ) -> None:
        """A single correlation_id should be shared across all intents."""
        captured_cids: list[UUID] = []

        async def _capture(
            payload: object, *, correlation_id: UUID | None = None
        ) -> None:
            if correlation_id is not None:
                captured_cids.append(correlation_id)

        handler = _make_handler()
        handler.execute = AsyncMock(side_effect=_capture)
        executor.register_handler("test.operation", handler)

        intents = [
            _make_intent(payload=_MockPayload("test.operation")),
            _make_intent(payload=_MockPayload("test.operation")),
        ]

        cid = uuid4()
        await executor.execute_all(intents, correlation_id=cid)

        assert len(captured_cids) == 2
        assert captured_cids[0] == cid
        assert captured_cids[1] == cid

    @pytest.mark.asyncio
    async def test_execute_all_stops_on_first_failure(
        self, executor: IntentExecutor
    ) -> None:
        """Second intent should NOT be called if the first raises."""
        handler_a = _make_handler()
        handler_a.execute.side_effect = RuntimeHostError("fail early")
        handler_b = _make_handler()

        executor.register_handler("op.a", handler_a)
        executor.register_handler("op.b", handler_b)

        intents = [
            _make_intent(payload=_MockPayload("op.a")),
            _make_intent(payload=_MockPayload("op.b")),
        ]

        with pytest.raises(RuntimeHostError, match="fail early"):
            await executor.execute_all(intents, correlation_id=uuid4())

        handler_b.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_all_empty_list_is_noop(
        self, executor: IntentExecutor
    ) -> None:
        """Empty intent list should return without error."""
        # Should not raise
        await executor.execute_all([], correlation_id=uuid4())


# ---------------------------------------------------------------------------
# Correlation ID tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorrelationId:
    """Tests for correlation_id generation and propagation."""

    @pytest.fixture
    def executor(self) -> IntentExecutor:
        """Create an IntentExecutor with a mock container."""
        return IntentExecutor(container=MagicMock())

    @pytest.mark.asyncio
    async def test_execute_uses_provided_correlation_id(
        self, executor: IntentExecutor
    ) -> None:
        """Explicit correlation_id should be passed through to the handler."""
        handler = _make_handler()
        executor.register_handler("test.operation", handler)

        payload = _MockPayload("test.operation")
        intent = _make_intent(payload=payload)
        explicit_cid = uuid4()

        await executor.execute(intent, correlation_id=explicit_cid)

        handler.execute.assert_awaited_once()
        _, kwargs = handler.execute.call_args
        assert kwargs["correlation_id"] == explicit_cid

    @pytest.mark.asyncio
    async def test_execute_generates_fallback_correlation_id(
        self, executor: IntentExecutor
    ) -> None:
        """When correlation_id is None, a uuid4 should be generated and used."""
        handler = _make_handler()
        executor.register_handler("test.operation", handler)

        payload = _MockPayload("test.operation")
        intent = _make_intent(payload=payload)

        await executor.execute(intent, correlation_id=None)

        handler.execute.assert_awaited_once()
        _, kwargs = handler.execute.call_args
        generated_cid = kwargs["correlation_id"]
        assert isinstance(generated_cid, UUID)
