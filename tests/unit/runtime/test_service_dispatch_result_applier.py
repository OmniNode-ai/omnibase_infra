# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for DispatchResultApplier.

Tests the runtime service responsible for processing ModelDispatchResult outputs:
publishing output events to the event bus and delegating intents to the
IntentExecutor.

Key behaviors validated:
- Early exit on non-success statuses
- Ordering contract: intents execute BEFORE events publish
- Intent delegation and error propagation
- Deterministic envelope IDs via uuid5
- Clock injection for envelope timestamps
- Correlation ID propagation
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, call
from uuid import UUID, uuid4, uuid5

import pytest
from pydantic import BaseModel

from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubPayload(BaseModel):
    """Concrete BaseModel subclass satisfying ProtocolIntentPayload."""

    intent_type: str = "test.stub"


class _StubEvent(BaseModel):
    """Minimal event model for testing output event publishing."""

    value: str = "test"


def _make_result(**overrides: object) -> ModelDispatchResult:
    """Build a ModelDispatchResult with sensible defaults.

    All keyword arguments are forwarded as field overrides.
    """
    defaults: dict[str, object] = {
        "status": EnumDispatchStatus.SUCCESS,
        "topic": "test.topic",
        "started_at": datetime.now(UTC),
        "correlation_id": uuid4(),
        "dispatcher_id": "test-dispatcher",
    }
    defaults.update(overrides)
    return ModelDispatchResult(**defaults)


def _make_intent(**overrides: object) -> ModelIntent:
    """Build a minimal ModelIntent for testing."""
    defaults: dict[str, object] = {
        "intent_type": "test.intent",
        "target": "test://target",
        "payload": _StubPayload(),
    }
    defaults.update(overrides)
    return ModelIntent(**defaults)


# ---------------------------------------------------------------------------
# Early exit tests
# ---------------------------------------------------------------------------


class TestEarlyExit:
    """Verify that non-success results are skipped without side effects."""

    @pytest.mark.asyncio
    async def test_skips_non_success_result(self) -> None:
        """HANDLER_ERROR status should not trigger publish or intent execution."""
        bus = AsyncMock()
        executor = AsyncMock()
        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            intent_executor=executor,
        )

        result = _make_result(status=EnumDispatchStatus.HANDLER_ERROR)
        await applier.apply(result)

        bus.publish_envelope.assert_not_called()
        executor.execute_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_no_dispatcher_result(self) -> None:
        """NO_DISPATCHER status should not trigger publish or intent execution."""
        bus = AsyncMock()
        executor = AsyncMock()
        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            intent_executor=executor,
        )

        result = _make_result(status=EnumDispatchStatus.NO_DISPATCHER)
        await applier.apply(result)

        bus.publish_envelope.assert_not_called()
        executor.execute_all.assert_not_called()


# ---------------------------------------------------------------------------
# Ordering contract tests
# ---------------------------------------------------------------------------


class TestOrderingContract:
    """Verify that intents execute BEFORE events are published."""

    @pytest.mark.asyncio
    async def test_intents_execute_before_events_publish(self) -> None:
        """execute_all must be called BEFORE any publish_envelope call.

        Uses a shared call-order tracker to prove the ordering contract:
        writes (intents) commit before reads (output events) are observable.
        """
        call_order: list[str] = []

        async def _track_execute_all(*args: object, **kwargs: object) -> None:
            call_order.append("execute_all")

        async def _track_publish(*args: object, **kwargs: object) -> None:
            call_order.append("publish_envelope")

        bus = AsyncMock()
        bus.publish_envelope.side_effect = _track_publish

        executor = AsyncMock()
        executor.execute_all.side_effect = _track_execute_all

        intent = _make_intent()
        event = _StubEvent(value="ordering-test")
        result = _make_result(
            output_intents=(intent,),
            output_events=[event],
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            intent_executor=executor,
        )
        await applier.apply(result)

        assert call_order == ["execute_all", "publish_envelope"], (
            f"Expected intents before events, got: {call_order}"
        )


# ---------------------------------------------------------------------------
# Intent execution tests
# ---------------------------------------------------------------------------


class TestIntentExecution:
    """Verify intent delegation and error handling."""

    @pytest.mark.asyncio
    async def test_raises_when_intents_but_no_executor(self) -> None:
        """Intents present without an IntentExecutor must raise RuntimeHostError."""
        bus = AsyncMock()
        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            intent_executor=None,
        )

        intent = _make_intent()
        result = _make_result(output_intents=(intent,))

        with pytest.raises(RuntimeHostError, match="IntentExecutor"):
            await applier.apply(result)

        bus.publish_envelope.assert_not_called()

    @pytest.mark.asyncio
    async def test_propagates_intent_execution_failure(self) -> None:
        """If intent_executor.execute_all raises, the error must propagate."""
        bus = AsyncMock()
        executor = AsyncMock()
        executor.execute_all.side_effect = RuntimeError("intent boom")

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            intent_executor=executor,
        )

        intent = _make_intent()
        result = _make_result(output_intents=(intent,))

        with pytest.raises(RuntimeError, match="intent boom"):
            await applier.apply(result)

        # Events must NOT be published when intents fail (ordering contract).
        bus.publish_envelope.assert_not_called()

    @pytest.mark.asyncio
    async def test_delegates_intents_to_executor(self) -> None:
        """execute_all receives the correct intents and correlation_id."""
        bus = AsyncMock()
        executor = AsyncMock()

        cid = uuid4()
        intent_a = _make_intent(intent_type="a.intent")
        intent_b = _make_intent(intent_type="b.intent")
        result = _make_result(
            output_intents=(intent_a, intent_b),
            correlation_id=cid,
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            intent_executor=executor,
        )
        await applier.apply(result)

        executor.execute_all.assert_awaited_once_with(
            (intent_a, intent_b),
            correlation_id=cid,
        )


# ---------------------------------------------------------------------------
# Event publishing tests
# ---------------------------------------------------------------------------


class TestEventPublishing:
    """Verify output events are published correctly."""

    @pytest.mark.asyncio
    async def test_publishes_output_events(self) -> None:
        """Each output event should be published to the configured output topic."""
        bus = AsyncMock()
        event_a = _StubEvent(value="a")
        event_b = _StubEvent(value="b")
        result = _make_result(output_events=[event_a, event_b])

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="my.output.topic",
        )
        await applier.apply(result)

        assert bus.publish_envelope.await_count == 2
        for c in bus.publish_envelope.call_args_list:
            assert c.kwargs["topic"] == "my.output.topic"

    @pytest.mark.asyncio
    async def test_deterministic_envelope_id(self) -> None:
        """Envelope IDs must be uuid5(correlation_id, 'ClassName:idx')."""
        bus = AsyncMock()
        cid = uuid4()
        event = _StubEvent(value="det")
        result = _make_result(
            output_events=[event],
            correlation_id=cid,
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
        )
        await applier.apply(result)

        expected_id = uuid5(cid, "_StubEvent:0")
        published_envelope = bus.publish_envelope.call_args.kwargs["envelope"]
        assert published_envelope.envelope_id == expected_id

    @pytest.mark.asyncio
    async def test_no_publish_when_no_events(self) -> None:
        """Empty output_events list should result in zero publish calls."""
        bus = AsyncMock()
        result = _make_result(output_events=[])

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
        )
        await applier.apply(result)

        bus.publish_envelope.assert_not_called()


# ---------------------------------------------------------------------------
# Clock injection tests
# ---------------------------------------------------------------------------


class TestClockInjection:
    """Verify that a custom clock controls envelope timestamps."""

    @pytest.mark.asyncio
    async def test_custom_clock_for_envelope_timestamp(self) -> None:
        """Injected clock should determine the envelope_timestamp value."""
        frozen_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        bus = AsyncMock()
        event = _StubEvent(value="clock")
        result = _make_result(output_events=[event])

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            clock=lambda: frozen_time,
        )
        await applier.apply(result)

        published_envelope = bus.publish_envelope.call_args.kwargs["envelope"]
        assert published_envelope.envelope_timestamp == frozen_time


# ---------------------------------------------------------------------------
# Correlation ID tests
# ---------------------------------------------------------------------------


class TestCorrelationId:
    """Verify correlation ID propagation and defaults."""

    @pytest.mark.asyncio
    async def test_uses_result_correlation_id(self) -> None:
        """When no explicit correlation_id is passed, result.correlation_id is used."""
        bus = AsyncMock()
        cid = uuid4()
        event = _StubEvent(value="cid-test")
        result = _make_result(
            output_events=[event],
            correlation_id=cid,
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
        )
        await applier.apply(result)

        published_envelope = bus.publish_envelope.call_args.kwargs["envelope"]
        assert published_envelope.correlation_id == cid

    @pytest.mark.asyncio
    async def test_correlation_id_always_present(self) -> None:
        """ModelDispatchResult.correlation_id has default_factory=uuid4.

        This means the uuid4() fallback in apply() is dead code -- the
        correlation_id on the result object is always non-None.  Verify
        this invariant so the dead-code path stays documented.
        """
        result = _make_result()
        # correlation_id is auto-generated even when not explicitly provided
        assert result.correlation_id is not None
        assert isinstance(result.correlation_id, UUID)

        # Double-check with a fresh instance that omits correlation_id
        bare_result = ModelDispatchResult(
            status=EnumDispatchStatus.SUCCESS,
            topic="t",
            started_at=datetime.now(UTC),
        )
        assert bare_result.correlation_id is not None
        assert isinstance(bare_result.correlation_id, UUID)
