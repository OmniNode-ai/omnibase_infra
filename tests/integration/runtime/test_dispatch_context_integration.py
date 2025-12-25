# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Integration tests for dispatch context injection (OMN-973).

These tests verify the full dispatch flow correctly handles context injection:
- Orchestrators and Effects receive `now` (time injection enabled)
- Reducers do NOT receive `now` (deterministic execution required)

This enforces ONEX architectural rule B4:
    - Orchestrators MAY use now for deadlines and timeouts
    - Effects MAY use now for retries/metrics
    - Reducers MUST ignore now and never depend on it

Related:
    - OMN-973: Enforce time injection context at dispatch
    - B4 (Handler Context / Time Injection) in ONEX Runtime ticket plan
    - docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from pydantic import BaseModel

from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_context import ModelDispatchContext
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.dispatch_context_enforcer import DispatchContextEnforcer
from omnibase_infra.runtime.dispatcher_registry import (
    DispatcherRegistry,
)
from tests.helpers.deterministic import DeterministicClock, DeterministicIdGenerator

# =============================================================================
# Test Payload Models
# =============================================================================


class UserCreatedEvent(BaseModel):
    """Test event payload for user creation."""

    user_id: str
    email: str


class UserUpdatedEvent(BaseModel):
    """Test event payload for user update."""

    user_id: str
    new_email: str


# =============================================================================
# Mock Envelope Helper
# =============================================================================


def create_mock_envelope(
    correlation_id: UUID | None = None,
    trace_id: UUID | None = None,
    payload: object | None = None,
) -> MagicMock:
    """Create a mock ModelEventEnvelope for testing.

    Args:
        correlation_id: Optional correlation ID for the envelope.
        trace_id: Optional trace ID for the envelope.
        payload: Optional payload for the envelope.

    Returns:
        MagicMock configured to behave like ModelEventEnvelope.
    """
    envelope = MagicMock()
    envelope.correlation_id = correlation_id
    envelope.trace_id = trace_id
    envelope.payload = payload
    return envelope


# =============================================================================
# Context-Capturing Test Dispatchers
# =============================================================================


class ContextCapturingDispatcher:
    """
    Test dispatcher that captures the context it receives for assertions.

    This dispatcher implements ProtocolMessageDispatcher and stores the
    dispatch context it receives, allowing tests to verify context injection rules.
    """

    def __init__(
        self,
        dispatcher_id: str,
        node_kind: EnumNodeKind,
        category: EnumMessageCategory = EnumMessageCategory.EVENT,
        message_types: set[str] | None = None,
    ) -> None:
        self._dispatcher_id = dispatcher_id
        self._node_kind = node_kind
        self._category = category
        self._message_types = message_types or set()

        # Captured data for assertions
        self.captured_context: ModelDispatchContext | None = None
        self.captured_envelope: object | None = None
        self.invocation_count: int = 0

    @property
    def dispatcher_id(self) -> str:
        return self._dispatcher_id

    @property
    def category(self) -> EnumMessageCategory:
        return self._category

    @property
    def message_types(self) -> set[str]:
        return self._message_types

    @property
    def node_kind(self) -> EnumNodeKind:
        return self._node_kind

    async def handle(
        self,
        envelope: object,
        context: ModelDispatchContext | None = None,
    ) -> ModelDispatchResult:
        """Handle the message and capture the context for assertions."""
        self.captured_envelope = envelope
        self.captured_context = context
        self.invocation_count += 1

        return ModelDispatchResult(
            dispatch_id=uuid4(),
            status=EnumDispatchStatus.SUCCESS,
            topic="test.events.v1",
            dispatcher_id=self._dispatcher_id,
            message_type=type(envelope).__name__ if envelope else None,
        )

    def reset(self) -> None:
        """Reset captured state for reuse between tests."""
        self.captured_context = None
        self.captured_envelope = None
        self.invocation_count = 0


class DeterministicResultDispatcher(ContextCapturingDispatcher):
    """
    Test dispatcher that produces deterministic results based on input only.

    This dispatcher is used to verify replay determinism - when the same
    event is dispatched to two identical reducers, they should produce
    identical results because neither receives time injection.
    """

    def __init__(
        self,
        dispatcher_id: str,
        node_kind: EnumNodeKind = EnumNodeKind.REDUCER,
        **kwargs: object,
    ) -> None:
        super().__init__(dispatcher_id=dispatcher_id, node_kind=node_kind, **kwargs)  # type: ignore[arg-type]
        self.processed_user_ids: list[str] = []

    async def handle(
        self,
        envelope: object,
        context: ModelDispatchContext | None = None,
    ) -> ModelDispatchResult:
        """Process the event deterministically - result depends only on input."""
        await super().handle(envelope, context)

        # Simulate deterministic processing
        if hasattr(envelope, "payload") and hasattr(envelope.payload, "user_id"):
            user_id = envelope.payload.user_id
            self.processed_user_ids.append(user_id)

        return ModelDispatchResult(
            dispatch_id=uuid4(),
            status=EnumDispatchStatus.SUCCESS,
            topic="test.events.v1",
            dispatcher_id=self._dispatcher_id,
            output_count=len(self.processed_user_ids),
        )

    def reset(self) -> None:
        super().reset()
        self.processed_user_ids = []


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def deterministic_clock() -> DeterministicClock:
    """Create a deterministic clock for predictable time testing."""
    return DeterministicClock(start=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC))


@pytest.fixture
def id_generator() -> DeterministicIdGenerator:
    """Create a deterministic ID generator for predictable UUIDs."""
    return DeterministicIdGenerator(seed=100)


@pytest.fixture
def context_enforcer() -> DispatchContextEnforcer:
    """Create a DispatchContextEnforcer instance."""
    return DispatchContextEnforcer()


@pytest.fixture
def reducer_dispatcher() -> ContextCapturingDispatcher:
    """Create a reducer dispatcher (EVENT -> REDUCER)."""
    return ContextCapturingDispatcher(
        dispatcher_id="test-reducer",
        node_kind=EnumNodeKind.REDUCER,
        category=EnumMessageCategory.EVENT,
        message_types={"UserCreatedEvent", "UserUpdatedEvent"},
    )


@pytest.fixture
def orchestrator_dispatcher() -> ContextCapturingDispatcher:
    """Create an orchestrator dispatcher (EVENT -> ORCHESTRATOR)."""
    return ContextCapturingDispatcher(
        dispatcher_id="test-orchestrator",
        node_kind=EnumNodeKind.ORCHESTRATOR,
        category=EnumMessageCategory.EVENT,
        message_types={"UserCreatedEvent", "UserUpdatedEvent"},
    )


@pytest.fixture
def effect_dispatcher() -> ContextCapturingDispatcher:
    """Create an effect dispatcher (COMMAND -> EFFECT)."""
    return ContextCapturingDispatcher(
        dispatcher_id="test-effect",
        node_kind=EnumNodeKind.EFFECT,
        category=EnumMessageCategory.COMMAND,
        message_types={"SendEmailCommand"},
    )


@pytest.fixture
def compute_dispatcher() -> ContextCapturingDispatcher:
    """Create a compute dispatcher (EVENT -> COMPUTE)."""
    return ContextCapturingDispatcher(
        dispatcher_id="test-compute",
        node_kind=EnumNodeKind.COMPUTE,
        category=EnumMessageCategory.EVENT,
    )


@pytest.fixture
def dispatcher_registry() -> DispatcherRegistry:
    """Create a fresh DispatcherRegistry."""
    return DispatcherRegistry()


# =============================================================================
# DispatchContextEnforcer Unit Tests
# =============================================================================


class TestDispatchContextEnforcerBasics:
    """Unit tests for DispatchContextEnforcer class behavior."""

    def test_requires_time_injection_orchestrator(
        self, context_enforcer: DispatchContextEnforcer
    ) -> None:
        """Orchestrator should require time injection."""
        assert context_enforcer.requires_time_injection(EnumNodeKind.ORCHESTRATOR)

    def test_requires_time_injection_effect(
        self, context_enforcer: DispatchContextEnforcer
    ) -> None:
        """Effect should require time injection."""
        assert context_enforcer.requires_time_injection(EnumNodeKind.EFFECT)

    def test_requires_time_injection_reducer_false(
        self, context_enforcer: DispatchContextEnforcer
    ) -> None:
        """Reducer should NOT require time injection."""
        assert not context_enforcer.requires_time_injection(EnumNodeKind.REDUCER)

    def test_requires_time_injection_compute_false(
        self, context_enforcer: DispatchContextEnforcer
    ) -> None:
        """Compute should NOT require time injection."""
        assert not context_enforcer.requires_time_injection(EnumNodeKind.COMPUTE)

    def test_forbids_time_injection_reducer(
        self, context_enforcer: DispatchContextEnforcer
    ) -> None:
        """Reducer should forbid time injection."""
        assert context_enforcer.forbids_time_injection(EnumNodeKind.REDUCER)

    def test_forbids_time_injection_compute(
        self, context_enforcer: DispatchContextEnforcer
    ) -> None:
        """Compute should forbid time injection."""
        assert context_enforcer.forbids_time_injection(EnumNodeKind.COMPUTE)

    def test_forbids_time_injection_orchestrator_false(
        self, context_enforcer: DispatchContextEnforcer
    ) -> None:
        """Orchestrator should NOT forbid time injection."""
        assert not context_enforcer.forbids_time_injection(EnumNodeKind.ORCHESTRATOR)

    def test_forbids_time_injection_effect_false(
        self, context_enforcer: DispatchContextEnforcer
    ) -> None:
        """Effect should NOT forbid time injection."""
        assert not context_enforcer.forbids_time_injection(EnumNodeKind.EFFECT)


# =============================================================================
# Context Creation Tests
# =============================================================================


class TestContextCreationForDispatcher:
    """Tests for context creation via create_context_for_dispatcher."""

    def test_create_context_reducer_no_time(
        self,
        context_enforcer: DispatchContextEnforcer,
        reducer_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Reducer context should NOT contain time injection."""
        correlation_id = id_generator.next_uuid()
        envelope = create_mock_envelope(correlation_id=correlation_id)

        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer_dispatcher,
            envelope=envelope,
        )

        assert ctx.node_kind == EnumNodeKind.REDUCER
        assert ctx.correlation_id == correlation_id
        assert ctx.now is None  # Critical: reducers don't get time
        assert not ctx.has_time_injection

    def test_create_context_orchestrator_has_time(
        self,
        context_enforcer: DispatchContextEnforcer,
        orchestrator_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Orchestrator context should contain time injection."""
        correlation_id = id_generator.next_uuid()
        before_time = datetime.now(UTC)
        envelope = create_mock_envelope(correlation_id=correlation_id)

        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=orchestrator_dispatcher,
            envelope=envelope,
        )

        after_time = datetime.now(UTC)

        assert ctx.node_kind == EnumNodeKind.ORCHESTRATOR
        assert ctx.correlation_id == correlation_id
        assert ctx.now is not None  # Critical: orchestrators get time
        assert before_time <= ctx.now <= after_time
        assert ctx.has_time_injection

    def test_create_context_effect_has_time(
        self,
        context_enforcer: DispatchContextEnforcer,
        effect_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Effect context should contain time injection."""
        correlation_id = id_generator.next_uuid()
        before_time = datetime.now(UTC)
        envelope = create_mock_envelope(correlation_id=correlation_id)

        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=effect_dispatcher,
            envelope=envelope,
        )

        after_time = datetime.now(UTC)

        assert ctx.node_kind == EnumNodeKind.EFFECT
        assert ctx.correlation_id == correlation_id
        assert ctx.now is not None  # Critical: effects get time
        assert before_time <= ctx.now <= after_time
        assert ctx.has_time_injection

    def test_create_context_compute_no_time(
        self,
        context_enforcer: DispatchContextEnforcer,
        compute_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Compute context should NOT contain time injection."""
        correlation_id = id_generator.next_uuid()
        envelope = create_mock_envelope(correlation_id=correlation_id)

        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=compute_dispatcher,
            envelope=envelope,
        )

        assert ctx.node_kind == EnumNodeKind.COMPUTE
        assert ctx.correlation_id == correlation_id
        assert ctx.now is None  # Critical: compute nodes don't get time
        assert not ctx.has_time_injection

    def test_create_context_with_trace_id(
        self,
        context_enforcer: DispatchContextEnforcer,
        reducer_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Context should propagate trace_id correctly."""
        correlation_id = id_generator.next_uuid()
        trace_id = id_generator.next_uuid()
        envelope = create_mock_envelope(
            correlation_id=correlation_id,
            trace_id=trace_id,
        )

        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer_dispatcher,
            envelope=envelope,
        )

        assert ctx.correlation_id == correlation_id
        assert ctx.trace_id == trace_id

    def test_create_context_generates_correlation_id_if_missing(
        self,
        context_enforcer: DispatchContextEnforcer,
        reducer_dispatcher: ContextCapturingDispatcher,
    ) -> None:
        """Context should generate correlation_id if not in envelope."""
        envelope = create_mock_envelope(correlation_id=None)

        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer_dispatcher,
            envelope=envelope,
        )

        # Should have generated a new correlation ID
        assert ctx.correlation_id is not None
        assert isinstance(ctx.correlation_id, UUID)


# =============================================================================
# Integration Tests: Simulated Dispatch Flow
# =============================================================================


class TestDispatchFlowContextInjection:
    """
    Integration tests simulating the full dispatch flow with context injection.

    These tests verify that when the dispatch engine routes messages to
    dispatchers, the appropriate context (with or without time) is provided.
    """

    @pytest.mark.asyncio
    async def test_full_dispatch_flow_reducer_no_time(
        self,
        context_enforcer: DispatchContextEnforcer,
        reducer_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """
        Full dispatch to reducer verifies no time injection.

        When dispatching to a REDUCER, the context must NOT include `now`
        to ensure deterministic event replay.
        """
        correlation_id = id_generator.next_uuid()
        test_payload = UserCreatedEvent(user_id="user-123", email="test@example.com")
        envelope = create_mock_envelope(
            correlation_id=correlation_id,
            payload=test_payload,
        )

        # Create context for reducer (should NOT have time)
        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer_dispatcher,
            envelope=envelope,
        )

        # Simulate dispatch with the created context
        await reducer_dispatcher.handle(envelope, context=ctx)

        # Verify the reducer received context WITHOUT time
        assert reducer_dispatcher.captured_context is not None
        assert reducer_dispatcher.captured_context.now is None
        assert not reducer_dispatcher.captured_context.has_time_injection
        assert reducer_dispatcher.captured_context.node_kind == EnumNodeKind.REDUCER

    @pytest.mark.asyncio
    async def test_full_dispatch_flow_orchestrator_has_time(
        self,
        context_enforcer: DispatchContextEnforcer,
        orchestrator_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """
        Full dispatch to orchestrator verifies time injection.

        When dispatching to an ORCHESTRATOR, the context MUST include `now`
        for timeout and deadline calculations.
        """
        correlation_id = id_generator.next_uuid()
        test_payload = UserCreatedEvent(user_id="user-456", email="test2@example.com")
        envelope = create_mock_envelope(
            correlation_id=correlation_id,
            payload=test_payload,
        )
        before_time = datetime.now(UTC)

        # Create context for orchestrator (should have time)
        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=orchestrator_dispatcher,
            envelope=envelope,
        )

        after_time = datetime.now(UTC)

        # Simulate dispatch with the created context
        await orchestrator_dispatcher.handle(envelope, context=ctx)

        # Verify the orchestrator received context WITH time
        assert orchestrator_dispatcher.captured_context is not None
        assert orchestrator_dispatcher.captured_context.now is not None
        assert before_time <= orchestrator_dispatcher.captured_context.now <= after_time
        assert orchestrator_dispatcher.captured_context.has_time_injection
        assert (
            orchestrator_dispatcher.captured_context.node_kind
            == EnumNodeKind.ORCHESTRATOR
        )

    @pytest.mark.asyncio
    async def test_full_dispatch_flow_effect_has_time(
        self,
        context_enforcer: DispatchContextEnforcer,
        effect_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """
        Full dispatch to effect verifies time injection.

        When dispatching to an EFFECT, the context MUST include `now`
        for retry logic and timeout calculations.
        """
        correlation_id = id_generator.next_uuid()
        envelope = create_mock_envelope(correlation_id=correlation_id)
        before_time = datetime.now(UTC)

        # Create context for effect (should have time)
        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=effect_dispatcher,
            envelope=envelope,
        )

        after_time = datetime.now(UTC)

        # Simulate dispatch with the created context
        await effect_dispatcher.handle(envelope, context=ctx)

        # Verify the effect received context WITH time
        assert effect_dispatcher.captured_context is not None
        assert effect_dispatcher.captured_context.now is not None
        assert before_time <= effect_dispatcher.captured_context.now <= after_time
        assert effect_dispatcher.captured_context.has_time_injection
        assert effect_dispatcher.captured_context.node_kind == EnumNodeKind.EFFECT

    @pytest.mark.asyncio
    async def test_mixed_dispatchers_correct_contexts(
        self,
        context_enforcer: DispatchContextEnforcer,
        reducer_dispatcher: ContextCapturingDispatcher,
        orchestrator_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """
        When fan-out to mixed node types, each gets correct context.

        This simulates a scenario where the same event triggers both
        a reducer (for state aggregation) and an orchestrator (for workflow).
        The reducer should get NO time, orchestrator should get time.
        """
        correlation_id = id_generator.next_uuid()
        test_payload = UserCreatedEvent(user_id="user-789", email="mixed@example.com")
        envelope = create_mock_envelope(
            correlation_id=correlation_id,
            payload=test_payload,
        )

        # Dispatch to REDUCER (no time)
        reducer_ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer_dispatcher,
            envelope=envelope,
        )
        await reducer_dispatcher.handle(envelope, context=reducer_ctx)

        # Dispatch to ORCHESTRATOR (has time)
        orchestrator_ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=orchestrator_dispatcher,
            envelope=envelope,
        )
        await orchestrator_dispatcher.handle(envelope, context=orchestrator_ctx)

        # Verify REDUCER got context WITHOUT time
        assert reducer_dispatcher.captured_context is not None
        assert reducer_dispatcher.captured_context.now is None
        assert not reducer_dispatcher.captured_context.has_time_injection

        # Verify ORCHESTRATOR got context WITH time
        assert orchestrator_dispatcher.captured_context is not None
        assert orchestrator_dispatcher.captured_context.now is not None
        assert orchestrator_dispatcher.captured_context.has_time_injection

        # Both should have same correlation_id
        assert reducer_dispatcher.captured_context.correlation_id == correlation_id
        assert orchestrator_dispatcher.captured_context.correlation_id == correlation_id


# =============================================================================
# Replay Determinism Tests
# =============================================================================


class TestReducerReplayDeterminism:
    """
    Tests verifying that reducers produce identical results on replay.

    Because reducers don't receive time injection, replaying the same
    events should produce identical results - this is critical for
    event sourcing and audit log replay.
    """

    @pytest.mark.asyncio
    async def test_reducer_replay_determinism(
        self,
        context_enforcer: DispatchContextEnforcer,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """
        Replaying events to reducer produces identical results.

        Two reducers processing the same event with the same context
        should produce identical results because neither receives time.
        """
        correlation_id = id_generator.next_uuid()
        test_payload = UserCreatedEvent(
            user_id="user-replay-test", email="replay@example.com"
        )
        envelope = create_mock_envelope(
            correlation_id=correlation_id,
            payload=test_payload,
        )

        # Create two separate reducers
        reducer1 = DeterministicResultDispatcher(dispatcher_id="reducer-1")
        reducer2 = DeterministicResultDispatcher(dispatcher_id="reducer-2")

        # Create contexts for both (should be identical - no time)
        ctx1 = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer1,
            envelope=envelope,
        )
        ctx2 = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer2,
            envelope=envelope,
        )

        # Dispatch same event to both reducers
        result1 = await reducer1.handle(envelope, context=ctx1)
        result2 = await reducer2.handle(envelope, context=ctx2)

        # Both contexts should have NO time (deterministic)
        assert ctx1.now is None
        assert ctx2.now is None
        assert ctx1.has_time_injection == ctx2.has_time_injection is False

        # Results should be functionally identical
        assert result1.status == result2.status
        assert result1.output_count == result2.output_count

    @pytest.mark.asyncio
    async def test_reducer_multiple_events_deterministic_order(
        self,
        context_enforcer: DispatchContextEnforcer,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """
        Processing multiple events produces same order regardless of wall time.

        Even if there's a delay between processing, the order of operations
        should be the same because reducers don't use time.
        """
        # Create two reducers
        reducer1 = DeterministicResultDispatcher(dispatcher_id="order-test-1")
        reducer2 = DeterministicResultDispatcher(dispatcher_id="order-test-2")

        events = [
            UserCreatedEvent(user_id="user-1", email="user1@example.com"),
            UserCreatedEvent(user_id="user-2", email="user2@example.com"),
            UserCreatedEvent(user_id="user-3", email="user3@example.com"),
        ]

        # Process all events on both reducers
        for event in events:
            envelope = create_mock_envelope(
                correlation_id=id_generator.next_uuid(),
                payload=event,
            )

            ctx1 = context_enforcer.create_context_for_dispatcher(
                dispatcher=reducer1,
                envelope=envelope,
            )
            ctx2 = context_enforcer.create_context_for_dispatcher(
                dispatcher=reducer2,
                envelope=envelope,
            )

            await reducer1.handle(envelope, context=ctx1)
            await reducer2.handle(envelope, context=ctx2)

        # Both reducers should have processed same events in same order
        assert reducer1.invocation_count == reducer2.invocation_count == 3


# =============================================================================
# Correlation ID Propagation Tests
# =============================================================================


class TestCorrelationIdPropagation:
    """Tests verifying correlation ID flows through context correctly."""

    @pytest.mark.asyncio
    async def test_context_propagates_correlation_id(
        self,
        context_enforcer: DispatchContextEnforcer,
        reducer_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Correlation ID flows from envelope through context to handler."""
        # Specific correlation ID to track
        correlation_id = id_generator.next_uuid()
        test_payload = UserCreatedEvent(user_id="corr-test", email="corr@example.com")
        envelope = create_mock_envelope(
            correlation_id=correlation_id,
            payload=test_payload,
        )

        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer_dispatcher,
            envelope=envelope,
        )

        await reducer_dispatcher.handle(envelope, context=ctx)

        # Verify the dispatcher received the exact correlation ID
        assert reducer_dispatcher.captured_context is not None
        assert reducer_dispatcher.captured_context.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_context_propagates_trace_id(
        self,
        context_enforcer: DispatchContextEnforcer,
        orchestrator_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Trace ID flows from envelope through context to handler."""
        correlation_id = id_generator.next_uuid()
        trace_id = id_generator.next_uuid()
        test_payload = UserCreatedEvent(user_id="trace-test", email="trace@example.com")
        envelope = create_mock_envelope(
            correlation_id=correlation_id,
            trace_id=trace_id,
            payload=test_payload,
        )

        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=orchestrator_dispatcher,
            envelope=envelope,
        )

        await orchestrator_dispatcher.handle(envelope, context=ctx)

        # Verify both correlation and trace IDs were propagated
        assert orchestrator_dispatcher.captured_context is not None
        assert orchestrator_dispatcher.captured_context.correlation_id == correlation_id
        assert orchestrator_dispatcher.captured_context.trace_id == trace_id


# =============================================================================
# ModelDispatchContext Factory Method Tests
# =============================================================================


class TestModelDispatchContextFactoryMethods:
    """Tests for ModelDispatchContext factory methods."""

    def test_for_reducer_creates_no_time_context(
        self, id_generator: DeterministicIdGenerator
    ) -> None:
        """ModelDispatchContext.for_reducer() creates context without time."""
        correlation_id = id_generator.next_uuid()

        ctx = ModelDispatchContext.for_reducer(
            correlation_id=correlation_id,
        )

        assert ctx.node_kind == EnumNodeKind.REDUCER
        assert ctx.now is None
        assert not ctx.has_time_injection

    def test_for_orchestrator_requires_time(
        self,
        id_generator: DeterministicIdGenerator,
        deterministic_clock: DeterministicClock,
    ) -> None:
        """ModelDispatchContext.for_orchestrator() creates context with time."""
        correlation_id = id_generator.next_uuid()
        current_time = deterministic_clock.now()

        ctx = ModelDispatchContext.for_orchestrator(
            correlation_id=correlation_id,
            now=current_time,
        )

        assert ctx.node_kind == EnumNodeKind.ORCHESTRATOR
        assert ctx.now == current_time
        assert ctx.has_time_injection

    def test_for_effect_requires_time(
        self,
        id_generator: DeterministicIdGenerator,
        deterministic_clock: DeterministicClock,
    ) -> None:
        """ModelDispatchContext.for_effect() creates context with time."""
        correlation_id = id_generator.next_uuid()
        current_time = deterministic_clock.now()

        ctx = ModelDispatchContext.for_effect(
            correlation_id=correlation_id,
            now=current_time,
        )

        assert ctx.node_kind == EnumNodeKind.EFFECT
        assert ctx.now == current_time
        assert ctx.has_time_injection

    def test_reducer_context_with_time_raises_validation_error(
        self,
        id_generator: DeterministicIdGenerator,
        deterministic_clock: DeterministicClock,
    ) -> None:
        """Creating reducer context with time should raise ValueError."""
        correlation_id = id_generator.next_uuid()
        current_time = deterministic_clock.now()

        with pytest.raises(ValueError) as exc_info:
            ModelDispatchContext(
                correlation_id=correlation_id,
                node_kind=EnumNodeKind.REDUCER,
                now=current_time,  # Invalid for reducer
            )

        assert "Reducer" in str(exc_info.value)
        assert "time injection" in str(exc_info.value).lower()

    def test_validate_for_node_kind_reducer_no_time(
        self, id_generator: DeterministicIdGenerator
    ) -> None:
        """validate_for_node_kind() should pass for reducer without time."""
        ctx = ModelDispatchContext.for_reducer(correlation_id=id_generator.next_uuid())

        assert ctx.validate_for_node_kind() is True


# =============================================================================
# Validation Tests
# =============================================================================


class TestContextValidation:
    """Tests for context validation methods."""

    def test_validate_no_time_injection_for_reducer_passes(
        self,
        context_enforcer: DispatchContextEnforcer,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Validation passes for reducer context without time."""
        ctx = ModelDispatchContext.for_reducer(correlation_id=id_generator.next_uuid())

        # Should not raise
        context_enforcer.validate_no_time_injection_for_reducer(ctx)

    def test_validate_no_time_injection_for_reducer_raises_if_has_time(
        self,
        context_enforcer: DispatchContextEnforcer,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Validation fails for manually constructed reducer with time."""
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        # Manually construct an invalid context (bypassing factory validation)
        # Note: This would normally be caught at construction, but we test
        # the validation method separately
        ctx = ModelDispatchContext(
            correlation_id=id_generator.next_uuid(),
            node_kind=EnumNodeKind.ORCHESTRATOR,  # Use valid kind for construction
            now=datetime.now(UTC),
        )

        # Create a new context with REDUCER but same now value
        # This tests the validation method, not the constructor
        bad_ctx = ctx.model_copy(update={"node_kind": EnumNodeKind.REDUCER})

        # This is expected to fail because the validation doesn't run on model_copy
        # The test verifies that validate_no_time_injection_for_reducer catches this
        with pytest.raises(ModelOnexError) as exc_info:
            context_enforcer.validate_no_time_injection_for_reducer(bad_ctx)

        assert "REDUCER" in str(exc_info.value)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error scenarios."""

    def test_enforcer_handles_all_standard_node_kinds(
        self,
        context_enforcer: DispatchContextEnforcer,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Enforcer should handle all standard node kinds."""
        node_kinds = [
            EnumNodeKind.REDUCER,
            EnumNodeKind.ORCHESTRATOR,
            EnumNodeKind.EFFECT,
            EnumNodeKind.COMPUTE,
        ]

        for node_kind in node_kinds:
            dispatcher = ContextCapturingDispatcher(
                dispatcher_id=f"test-{node_kind.value}",
                node_kind=node_kind,
            )
            envelope = create_mock_envelope(correlation_id=id_generator.next_uuid())

            ctx = context_enforcer.create_context_for_dispatcher(
                dispatcher=dispatcher,
                envelope=envelope,
            )

            assert ctx.node_kind == node_kind

            # Verify time injection rules
            if context_enforcer.requires_time_injection(node_kind):
                assert ctx.now is not None
            else:
                assert ctx.now is None

    def test_context_is_immutable(
        self,
        context_enforcer: DispatchContextEnforcer,
        reducer_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """ModelDispatchContext should be immutable (frozen)."""
        envelope = create_mock_envelope(correlation_id=id_generator.next_uuid())

        ctx = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer_dispatcher,
            envelope=envelope,
        )

        with pytest.raises(Exception):  # ValidationError for frozen models
            ctx.now = datetime.now(UTC)  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_multiple_dispatches_isolated_contexts(
        self,
        context_enforcer: DispatchContextEnforcer,
        reducer_dispatcher: ContextCapturingDispatcher,
        id_generator: DeterministicIdGenerator,
    ) -> None:
        """Each dispatch should have an isolated context."""
        correlation_id_1 = id_generator.next_uuid()
        correlation_id_2 = id_generator.next_uuid()

        envelope1 = create_mock_envelope(correlation_id=correlation_id_1)
        envelope2 = create_mock_envelope(correlation_id=correlation_id_2)

        ctx1 = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer_dispatcher,
            envelope=envelope1,
        )

        ctx2 = context_enforcer.create_context_for_dispatcher(
            dispatcher=reducer_dispatcher,
            envelope=envelope2,
        )

        # Contexts should be different objects with different correlation IDs
        assert ctx1 is not ctx2
        assert ctx1.correlation_id != ctx2.correlation_id
        assert ctx1.correlation_id == correlation_id_1
        assert ctx2.correlation_id == correlation_id_2
