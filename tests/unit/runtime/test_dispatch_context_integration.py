# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for MessageDispatchEngine + DispatchContextEnforcer integration.

Tests verify that when dispatchers are registered with a `node_kind` parameter,
the MessageDispatchEngine correctly:
- Creates ModelDispatchContext with appropriate time injection
- Passes context to dispatchers that accept 2+ parameters
- Maintains backwards compatibility for dispatchers without node_kind

Time Injection Rules (ONEX Architecture):
- REDUCER: now=None (deterministic execution required)
- COMPUTE: now=None (pure transformation, deterministic)
- ORCHESTRATOR: now=datetime.now(UTC) (coordination needs time)
- EFFECT: now=datetime.now(UTC) (I/O operations need time)
- RUNTIME_HOST: now=datetime.now(UTC) (infrastructure needs time)

Related:
    - OMN-990: Add tests for MessageDispatchEngine + DispatchContextEnforcer integration
    - OMN-973: Time injection enforcement at dispatch
    - src/omnibase_infra/runtime/message_dispatch_engine.py
    - src/omnibase_infra/runtime/dispatch_context_enforcer.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from pydantic import BaseModel

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_context import ModelDispatchContext
from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

# =============================================================================
# Test Payload Classes
# =============================================================================


class SampleEventPayload(BaseModel):
    """Simple test event payload using Pydantic for proper type handling."""

    data: str = "test"


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_envelope(
    correlation_id: UUID | None = None,
    trace_id: UUID | None = None,
    payload: object | None = None,
) -> MagicMock:
    """Create a mock test envelope for dispatch tests.

    Using MagicMock to avoid circular import issues with ModelEventEnvelope
    while still providing the interface needed by the dispatch engine.

    Args:
        correlation_id: Optional correlation ID for the envelope.
        trace_id: Optional trace ID for the envelope.
        payload: Optional payload for the envelope.

    Returns:
        MagicMock configured to behave like ModelEventEnvelope.
    """
    envelope = MagicMock()
    envelope.correlation_id = correlation_id or uuid4()
    envelope.trace_id = trace_id
    envelope.payload = payload or SampleEventPayload()
    envelope.span_id = None
    return envelope


def setup_engine_with_dispatcher(
    dispatcher_id: str,
    dispatcher: object,
    category: EnumMessageCategory = EnumMessageCategory.EVENT,
    node_kind: EnumNodeKind | None = None,
    topic_pattern: str = "test.*.events.*",
) -> MessageDispatchEngine:
    """Set up a MessageDispatchEngine with a single dispatcher and route.

    Args:
        dispatcher_id: Unique identifier for the dispatcher.
        dispatcher: The dispatcher callable.
        category: Message category for the dispatcher.
        node_kind: Optional ONEX node kind for time injection.
        topic_pattern: Topic pattern for the route.

    Returns:
        Configured and frozen MessageDispatchEngine.
    """
    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id=dispatcher_id,
        dispatcher=dispatcher,
        category=category,
        node_kind=node_kind,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id=f"route-{dispatcher_id}",
            topic_pattern=topic_pattern,
            message_category=category,
            dispatcher_id=dispatcher_id,
        )
    )
    engine.freeze()
    return engine


# =============================================================================
# REDUCER Dispatcher Tests (NO time injection)
# =============================================================================


class TestReducerDispatcherReceivesNoTime:
    """Tests verifying REDUCER dispatchers receive context with now=None."""

    @pytest.mark.asyncio
    async def test_reducer_dispatcher_receives_no_time_async(self) -> None:
        """Verify async REDUCER dispatchers receive context with now=None."""
        received_context: ModelDispatchContext | None = None

        async def reducer_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-reducer",
            dispatcher=reducer_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.REDUCER,
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.events.v1", envelope)

        # PRIMARY ASSERTION: Reducer context must have no time injection
        assert received_context is not None, "Dispatcher was not called"
        assert received_context.now is None, (
            "CRITICAL VIOLATION: Reducer received time injection! "
            f"Got now={received_context.now}. Reducers must be deterministic."
        )
        assert received_context.node_kind == EnumNodeKind.REDUCER
        assert received_context.has_time_injection is False

    @pytest.mark.asyncio
    async def test_reducer_dispatcher_receives_correlation_id(self) -> None:
        """Verify REDUCER context contains envelope's correlation_id."""
        received_context: ModelDispatchContext | None = None
        expected_correlation_id = UUID("12345678-1234-5678-1234-567812345678")

        async def reducer_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-reducer-corr",
            dispatcher=reducer_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.REDUCER,
        )

        envelope = create_test_envelope(correlation_id=expected_correlation_id)
        await engine.dispatch("test.user.events.v1", envelope)

        assert received_context is not None
        assert received_context.correlation_id == expected_correlation_id


# =============================================================================
# COMPUTE Dispatcher Tests (NO time injection)
# =============================================================================


class TestComputeDispatcherReceivesNoTime:
    """Tests verifying COMPUTE dispatchers receive context with now=None."""

    @pytest.mark.asyncio
    async def test_compute_dispatcher_receives_no_time_async(self) -> None:
        """Verify async COMPUTE dispatchers receive context with now=None."""
        received_context: ModelDispatchContext | None = None

        async def compute_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-compute",
            dispatcher=compute_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.COMPUTE,
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.events.v1", envelope)

        # PRIMARY ASSERTION: Compute context must have no time injection
        assert received_context is not None, "Dispatcher was not called"
        assert received_context.now is None, (
            "CRITICAL VIOLATION: Compute node received time injection! "
            f"Got now={received_context.now}. Compute nodes must be deterministic."
        )
        assert received_context.node_kind == EnumNodeKind.COMPUTE
        assert received_context.has_time_injection is False


# =============================================================================
# ORCHESTRATOR Dispatcher Tests (WITH time injection)
# =============================================================================


class TestOrchestratorDispatcherReceivesTime:
    """Tests verifying ORCHESTRATOR dispatchers receive context with now set."""

    @pytest.mark.asyncio
    async def test_orchestrator_dispatcher_receives_time_async(self) -> None:
        """Verify async ORCHESTRATOR dispatchers receive context with now set."""
        received_context: ModelDispatchContext | None = None
        before_dispatch = datetime.now(UTC)

        async def orchestrator_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-orchestrator",
            dispatcher=orchestrator_dispatcher,
            category=EnumMessageCategory.COMMAND,
            node_kind=EnumNodeKind.ORCHESTRATOR,
            topic_pattern="test.*.commands.*",
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.commands.v1", envelope)

        after_dispatch = datetime.now(UTC)

        # PRIMARY ASSERTION: Orchestrator context must have time injection
        assert received_context is not None, "Dispatcher was not called"
        assert received_context.now is not None, (
            "Orchestrator did not receive time injection! "
            "Orchestrators need `now` for deadline and timeout calculations."
        )
        assert received_context.node_kind == EnumNodeKind.ORCHESTRATOR
        assert received_context.has_time_injection is True

        # Verify time is within expected range
        assert before_dispatch <= received_context.now <= after_dispatch


# =============================================================================
# EFFECT Dispatcher Tests (WITH time injection)
# =============================================================================


class TestEffectDispatcherReceivesTime:
    """Tests verifying EFFECT dispatchers receive context with now set."""

    @pytest.mark.asyncio
    async def test_effect_dispatcher_receives_time_async(self) -> None:
        """Verify async EFFECT dispatchers receive context with now set."""
        received_context: ModelDispatchContext | None = None
        before_dispatch = datetime.now(UTC)

        async def effect_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-effect",
            dispatcher=effect_dispatcher,
            category=EnumMessageCategory.COMMAND,
            node_kind=EnumNodeKind.EFFECT,
            topic_pattern="test.*.commands.*",
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.commands.v1", envelope)

        after_dispatch = datetime.now(UTC)

        # PRIMARY ASSERTION: Effect context must have time injection
        assert received_context is not None, "Dispatcher was not called"
        assert received_context.now is not None, (
            "Effect did not receive time injection! "
            "Effects need `now` for retry logic and timeout calculations."
        )
        assert received_context.node_kind == EnumNodeKind.EFFECT
        assert received_context.has_time_injection is True

        # Verify time is within expected range
        assert before_dispatch <= received_context.now <= after_dispatch


# =============================================================================
# RUNTIME_HOST Dispatcher Tests (WITH time injection)
# =============================================================================


class TestRuntimeHostDispatcherReceivesTime:
    """Tests verifying RUNTIME_HOST dispatchers receive context with now set."""

    @pytest.mark.asyncio
    async def test_runtime_host_dispatcher_receives_time_async(self) -> None:
        """Verify async RUNTIME_HOST dispatchers receive context with now set."""
        received_context: ModelDispatchContext | None = None
        before_dispatch = datetime.now(UTC)

        async def runtime_host_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-runtime-host",
            dispatcher=runtime_host_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.RUNTIME_HOST,
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.infra.events.v1", envelope)

        after_dispatch = datetime.now(UTC)

        # PRIMARY ASSERTION: Runtime host context must have time injection
        assert received_context is not None, "Dispatcher was not called"
        assert received_context.now is not None, (
            "Runtime host did not receive time injection! "
            "Runtime hosts need `now` for infrastructure operations."
        )
        assert received_context.node_kind == EnumNodeKind.RUNTIME_HOST
        assert received_context.has_time_injection is True

        # Verify time is within expected range
        assert before_dispatch <= received_context.now <= after_dispatch


# =============================================================================
# Backwards Compatibility Tests (no node_kind)
# =============================================================================


class TestBackwardsCompatibility:
    """Tests verifying dispatchers without node_kind still work."""

    @pytest.mark.asyncio
    async def test_dispatcher_without_node_kind_works(self) -> None:
        """Verify dispatchers without node_kind still work (backwards compatible)."""
        was_called = False

        async def simple_dispatcher(
            envelope: object,
        ) -> str | None:
            nonlocal was_called
            was_called = True
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-simple",
            dispatcher=simple_dispatcher,
            category=EnumMessageCategory.EVENT,
            # No node_kind parameter - backwards compatibility
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.events.v1", envelope)

        assert was_called, "Simple dispatcher without node_kind was not called"

    @pytest.mark.asyncio
    async def test_sync_dispatcher_without_node_kind_works(self) -> None:
        """Verify sync dispatchers without node_kind still work."""
        was_called = False

        def sync_simple_dispatcher(
            envelope: object,
        ) -> str | None:
            nonlocal was_called
            was_called = True
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-sync-simple",
            dispatcher=sync_simple_dispatcher,
            category=EnumMessageCategory.EVENT,
            # No node_kind parameter
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.events.v1", envelope)

        assert was_called, "Sync dispatcher without node_kind was not called"


# =============================================================================
# Sync Dispatcher with Context Tests
# =============================================================================


class TestSyncDispatcherWithContext:
    """Tests verifying sync dispatchers also receive context when node_kind is set."""

    @pytest.mark.asyncio
    async def test_sync_effect_dispatcher_receives_context(self) -> None:
        """Verify sync dispatchers also receive context when node_kind is set."""
        received_context: ModelDispatchContext | None = None

        def sync_effect_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-sync-effect",
            dispatcher=sync_effect_dispatcher,
            category=EnumMessageCategory.COMMAND,
            node_kind=EnumNodeKind.EFFECT,
            topic_pattern="test.*.commands.*",
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.commands.v1", envelope)

        assert received_context is not None, "Sync dispatcher was not called"
        assert received_context.now is not None, (
            "Sync effect dispatcher did not receive time injection"
        )
        assert received_context.node_kind == EnumNodeKind.EFFECT

    @pytest.mark.asyncio
    async def test_sync_reducer_dispatcher_receives_no_time(self) -> None:
        """Verify sync REDUCER dispatchers receive context with now=None."""
        received_context: ModelDispatchContext | None = None

        def sync_reducer_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-sync-reducer",
            dispatcher=sync_reducer_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.REDUCER,
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.events.v1", envelope)

        assert received_context is not None, "Sync reducer dispatcher was not called"
        assert received_context.now is None, (
            "CRITICAL: Sync reducer received time injection! "
            f"Got now={received_context.now}"
        )
        assert received_context.node_kind == EnumNodeKind.REDUCER


# =============================================================================
# Correlation ID Propagation Tests
# =============================================================================


class TestCorrelationIdPropagation:
    """Tests verifying context receives correlation_id from envelope."""

    @pytest.mark.asyncio
    async def test_context_contains_envelope_correlation_id(self) -> None:
        """Verify context receives correlation_id from envelope."""
        received_context: ModelDispatchContext | None = None
        expected_correlation_id = UUID("12345678-1234-5678-1234-567812345678")

        async def orchestrator_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-corr-id",
            dispatcher=orchestrator_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.ORCHESTRATOR,
        )

        envelope = create_test_envelope(correlation_id=expected_correlation_id)
        await engine.dispatch("test.user.events.v1", envelope)

        assert received_context is not None
        assert received_context.correlation_id == expected_correlation_id

    @pytest.mark.asyncio
    async def test_context_contains_envelope_trace_id(self) -> None:
        """Verify context receives trace_id from envelope."""
        received_context: ModelDispatchContext | None = None
        expected_trace_id = UUID("87654321-4321-8765-4321-876543218765")

        async def effect_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-trace-id",
            dispatcher=effect_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.EFFECT,
        )

        envelope = create_test_envelope(trace_id=expected_trace_id)
        await engine.dispatch("test.user.events.v1", envelope)

        assert received_context is not None
        assert received_context.trace_id == expected_trace_id


# =============================================================================
# Comprehensive Node Kind Matrix Test
# =============================================================================


class TestAllNodeKindsTimeInjectionMatrix:
    """Matrix test covering all node kinds for time injection rules."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("node_kind", "expects_time"),
        [
            (EnumNodeKind.REDUCER, False),
            (EnumNodeKind.COMPUTE, False),
            (EnumNodeKind.ORCHESTRATOR, True),
            (EnumNodeKind.EFFECT, True),
            (EnumNodeKind.RUNTIME_HOST, True),
        ],
        ids=[
            "reducer-no-time",
            "compute-no-time",
            "orchestrator-with-time",
            "effect-with-time",
            "runtime-host-with-time",
        ],
    )
    async def test_node_kind_time_injection_matrix(
        self,
        node_kind: EnumNodeKind,
        expects_time: bool,
    ) -> None:
        """Verify each node kind receives correct time injection based on ONEX rules."""
        received_context: ModelDispatchContext | None = None

        async def test_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal received_context
            received_context = context
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id=f"test-{node_kind.value}",
            dispatcher=test_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=node_kind,
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.events.v1", envelope)

        assert received_context is not None, (
            f"Dispatcher for {node_kind} was not called"
        )
        assert received_context.node_kind == node_kind

        if expects_time:
            assert received_context.now is not None, (
                f"{node_kind.value} should receive time injection but got now=None"
            )
            assert received_context.has_time_injection is True
        else:
            assert received_context.now is None, (
                f"{node_kind.value} should NOT receive time injection but got "
                f"now={received_context.now}"
            )
            assert received_context.has_time_injection is False


# =============================================================================
# Context Not Passed When Dispatcher Has Single Parameter
# =============================================================================


class TestContextNotPassedToSingleParamDispatcher:
    """Tests verifying single-param dispatchers work even with node_kind set."""

    @pytest.mark.asyncio
    async def test_single_param_dispatcher_with_node_kind_still_works(self) -> None:
        """Verify dispatcher with 1 param works even when node_kind is set.

        When a dispatcher only accepts 1 parameter (envelope only),
        the engine should NOT attempt to pass context, avoiding TypeError.
        """
        was_called = False
        received_envelope: object | None = None

        async def single_param_dispatcher(
            envelope: object,
        ) -> str | None:
            nonlocal was_called, received_envelope
            was_called = True
            received_envelope = envelope
            return None

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-single-param",
            dispatcher=single_param_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.ORCHESTRATOR,  # node_kind set, but dispatcher has 1 param
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.events.v1", envelope)

        assert was_called, "Single-param dispatcher was not called"
        assert received_envelope is not None


# =============================================================================
# Multiple Dispatchers with Different Node Kinds
# =============================================================================


class TestMultipleDispatchersWithDifferentNodeKinds:
    """Tests for fan-out to multiple dispatchers with different node kinds."""

    @pytest.mark.asyncio
    async def test_fanout_to_mixed_node_kinds(self) -> None:
        """Verify fan-out correctly injects time per dispatcher's node_kind."""
        reducer_context: ModelDispatchContext | None = None
        orchestrator_context: ModelDispatchContext | None = None

        async def reducer_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal reducer_context
            reducer_context = context
            return None

        async def orchestrator_dispatcher(
            envelope: object,
            context: ModelDispatchContext,
        ) -> str | None:
            nonlocal orchestrator_context
            orchestrator_context = context
            return None

        engine = MessageDispatchEngine()

        # Register REDUCER dispatcher
        engine.register_dispatcher(
            dispatcher_id="reducer",
            dispatcher=reducer_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.REDUCER,
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="route-reducer",
                topic_pattern="test.*.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="reducer",
            )
        )

        # Register ORCHESTRATOR dispatcher for same topic
        engine.register_dispatcher(
            dispatcher_id="orchestrator",
            dispatcher=orchestrator_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.ORCHESTRATOR,
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="route-orchestrator",
                topic_pattern="test.*.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="orchestrator",
            )
        )

        engine.freeze()

        envelope = create_test_envelope()
        await engine.dispatch("test.user.events.v1", envelope)

        # REDUCER should NOT have time
        assert reducer_context is not None
        assert reducer_context.now is None
        assert reducer_context.node_kind == EnumNodeKind.REDUCER

        # ORCHESTRATOR should have time
        assert orchestrator_context is not None
        assert orchestrator_context.now is not None
        assert orchestrator_context.node_kind == EnumNodeKind.ORCHESTRATOR

        # Both should have same correlation_id
        assert reducer_context.correlation_id == orchestrator_context.correlation_id


# =============================================================================
# Uninspectable Dispatcher Edge Case Tests
# =============================================================================


class TestUninspectableDispatcherFallback:
    """Tests for dispatchers whose signature cannot be inspected.

    When inspect.signature() fails (e.g., C extensions, certain decorators,
    wrapped functions), the engine should gracefully fall back to treating
    the dispatcher as not accepting context, calling it with only the envelope.

    This covers the edge case identified in CodeRabbit review for PR #73.

    Related:
        - OMN-990: Dispatcher signature inspection edge cases
        - src/omnibase_infra/runtime/message_dispatch_engine.py lines 1664-1672
    """

    @pytest.mark.asyncio
    async def test_uninspectable_dispatcher_called_with_envelope_only(self) -> None:
        """Verify dispatchers that cannot be inspected are called with envelope only.

        When inspect.signature() raises ValueError or TypeError (which can happen
        with C extensions, built-in functions, or certain decorators), the engine
        should:
        1. Log a warning about the signature inspection failure
        2. Treat the dispatcher as not accepting context (accepts_context=False)
        3. Call the dispatcher with only the envelope parameter
        4. Not raise any exceptions

        This test uses a sync callable class with __signature__ property that raises
        ValueError, simulating the behavior of uninspectable callables like
        C extensions.
        """
        # Track what the dispatcher receives
        received_args: list[object] = []

        # Create a sync callable that breaks signature inspection.
        # The engine uses inspect.iscoroutinefunction() which returns False for
        # class instances with __call__, so this is treated as a sync dispatcher
        # and executed via run_in_executor.
        class UninspectableDispatcher:
            """A dispatcher class that cannot be inspected.

            Setting __signature__ to raise ValueError mimics C extensions
            and other uninspectable callables.
            """

            @property
            def __signature__(self) -> None:
                """Raise ValueError to simulate uninspectable callable."""
                raise ValueError("No signature available")

            def __call__(self, *args: object) -> str | None:
                """Process the envelope and return None."""
                received_args.extend(args)
                return None

        uninspectable_dispatcher = UninspectableDispatcher()

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-uninspectable",
            dispatcher=uninspectable_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.ORCHESTRATOR,  # Would normally inject time
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.events.v1", envelope)

        # Dispatcher should be called with only the envelope (no context)
        assert len(received_args) == 1, (
            f"Expected dispatcher to receive 1 argument (envelope only), "
            f"but received {len(received_args)} arguments"
        )
        assert received_args[0] is envelope, "Dispatcher should receive the envelope"

    @pytest.mark.asyncio
    async def test_uninspectable_dispatcher_registration_succeeds(self) -> None:
        """Verify uninspectable dispatchers can be registered without error.

        Even if signature inspection fails, the dispatcher should be registered
        successfully with accepts_context=False in its internal entry.
        """

        class UninspectableDispatcher:
            """A dispatcher class that raises TypeError on signature inspection."""

            @property
            def __signature__(self) -> None:
                """Raise TypeError to simulate certain built-in functions."""
                raise TypeError("No signature available for built-in")

            def __call__(self, envelope: object) -> str | None:
                """Process envelope."""
                return None

        uninspectable_dispatcher = UninspectableDispatcher()

        # Registration should succeed without raising exceptions
        engine = MessageDispatchEngine()
        engine.register_dispatcher(
            dispatcher_id="test-uninspectable-reg",
            dispatcher=uninspectable_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.EFFECT,
        )

        # Verify dispatcher is registered by checking dispatcher_count increased
        assert engine.dispatcher_count == 1, (
            "Uninspectable dispatcher should be registered successfully"
        )

    @pytest.mark.asyncio
    async def test_uninspectable_dispatcher_with_context_param_works(self) -> None:
        """Verify uninspectable dispatcher with context param falls back gracefully.

        Even if a dispatcher actually accepts a context parameter, if its signature
        cannot be inspected, it will only receive the envelope. This is the
        documented fallback behavior - the dispatcher must handle this gracefully.

        This test documents this edge case: if you have an uninspectable dispatcher
        that needs context, you must wrap it in an inspectable function.
        """
        received_args: list[object] = []

        class UninspectableWithContextDispatcher:
            """Dispatcher that could accept context but can't be inspected."""

            @property
            def __signature__(self) -> None:
                """Raise ValueError to simulate uninspectable callable."""
                raise ValueError("No signature available")

            def __call__(
                self,
                envelope: object,
                context: ModelDispatchContext | None = None,
            ) -> str | None:
                """Accept envelope and optional context.

                Due to uninspectable signature, context will always be None
                when called by the engine (engine passes envelope only).
                """
                received_args.append(envelope)
                received_args.append(context)
                return None

        dispatcher = UninspectableWithContextDispatcher()

        engine = setup_engine_with_dispatcher(
            dispatcher_id="test-uninspectable-with-context",
            dispatcher=dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.ORCHESTRATOR,
        )

        envelope = create_test_envelope()
        await engine.dispatch("test.user.events.v1", envelope)

        # Dispatcher receives only envelope - context is not passed
        # The dispatcher's default value (None) applies
        assert len(received_args) == 2, (
            "Dispatcher should record 2 args (envelope, context default)"
        )
        assert received_args[0] is envelope
        # Context is None because engine didn't pass it (uninspectable fallback)
        assert received_args[1] is None, (
            "Uninspectable dispatcher should not receive context from engine"
        )
