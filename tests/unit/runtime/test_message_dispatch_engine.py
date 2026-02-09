# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: N803
# N803 disabled: Uppercase parameter names are intentional for testing case sensitivity
"""
Comprehensive tests for MessageDispatchEngine.

Tests cover:
- Route registration (valid, duplicate, after freeze)
- Handler registration (valid, with message types, after freeze)
- Freeze pattern (freeze, is_frozen, double freeze)
- Dispatch success (single handler, multiple handlers, fan-out)
- Dispatch errors (no handlers, category mismatch, invalid topic, handler exception)
- Async handlers
- Metrics collection
- Deterministic routing (same input -> same handlers)
- Concurrent dispatch thread safety (freeze-after-init pattern)

OMN-934: Message dispatch engine implementation
"""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_outputs import ModelDispatchOutputs
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.runtime.service_message_dispatch_engine import MessageDispatchEngine

# ============================================================================
# Test Event Types (for category inference)
# ============================================================================


class UserCreatedEvent:
    """Test event class that ends with 'Event'."""

    def __init__(self, user_id: str, name: str) -> None:
        self.user_id = user_id
        self.name = name


class CreateUserCommand:
    """Test command class that ends with 'Command'."""

    def __init__(self, name: str) -> None:
        self.name = name


class ProvisionUserIntent:
    """Test intent class that ends with 'Intent'."""

    def __init__(self, user_type: str) -> None:
        self.user_type = user_type


class SomeGenericPayload:
    """Generic payload class - defaults to EVENT category."""

    def __init__(self, data: str) -> None:
        self.data = data


class OrderSummaryProjection:
    """Test projection class that ends with 'Projection'.

    Note on PROJECTION semantics (OMN-985 resolution):
        PROJECTION is NOT a message category for routing. This class exists
        to demonstrate the distinction between node output types and message
        categories. Projections are:

        - Produced by REDUCER nodes as local state outputs
        - NOT routed via MessageDispatchEngine
        - NOT part of EnumMessageCategory
        - Applied locally by the runtime to a projection sink

        The MessageDispatchEngine only routes EVENT, COMMAND, and INTENT
        message categories. Projection handling is separate from message
        dispatch and is the responsibility of the runtime's projection sink.

        See EnumNodeOutputType.PROJECTION for the node output type and
        CLAUDE.md "Enum Usage" section for the full distinction.
    """

    def __init__(self, order_id: str, total: float) -> None:
        self.order_id = order_id
        self.total = total


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def dispatch_engine() -> MessageDispatchEngine:
    """Create a fresh MessageDispatchEngine for each test."""
    return MessageDispatchEngine()


@pytest.fixture
def event_envelope() -> ModelEventEnvelope[UserCreatedEvent]:
    """Create a test event envelope."""
    return ModelEventEnvelope(
        payload=UserCreatedEvent(user_id="user-123", name="Test User"),
        correlation_id=uuid4(),
    )


@pytest.fixture
def command_envelope() -> ModelEventEnvelope[CreateUserCommand]:
    """Create a test command envelope."""
    return ModelEventEnvelope(
        payload=CreateUserCommand(name="New User"),
        correlation_id=uuid4(),
    )


@pytest.fixture
def intent_envelope() -> ModelEventEnvelope[ProvisionUserIntent]:
    """Create a test intent envelope."""
    return ModelEventEnvelope(
        payload=ProvisionUserIntent(user_type="admin"),
        correlation_id=uuid4(),
    )


# ============================================================================
# Route Registration Tests
# ============================================================================


@pytest.mark.unit
class TestRouteRegistration:
    """Tests for route registration functionality."""

    def test_register_route_valid(self, dispatch_engine: MessageDispatchEngine) -> None:
        """Test successful route registration."""
        route = ModelDispatchRoute(
            route_id="user-events-route",
            topic_pattern="*.user.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="user-handler",
        )

        dispatch_engine.register_route(route)

        assert dispatch_engine.route_count == 1

    def test_register_route_multiple(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test registering multiple routes."""
        routes = [
            ModelDispatchRoute(
                route_id=f"route-{i}",
                topic_pattern=f"*.domain{i}.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id=f"handler-{i}",
            )
            for i in range(5)
        ]

        for route in routes:
            dispatch_engine.register_route(route)

        assert dispatch_engine.route_count == 5

    def test_register_route_duplicate_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that duplicate route_id raises DUPLICATE_REGISTRATION error."""
        route = ModelDispatchRoute(
            route_id="duplicate-route",
            topic_pattern="*.user.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="handler",
        )

        dispatch_engine.register_route(route)

        # Try to register with same route_id
        duplicate = ModelDispatchRoute(
            route_id="duplicate-route",  # Same ID
            topic_pattern="*.order.events.*",  # Different pattern
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="other-handler",
        )

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_route(duplicate)

        assert exc_info.value.error_code == EnumCoreErrorCode.DUPLICATE_REGISTRATION
        assert "duplicate-route" in exc_info.value.message

    def test_register_route_none_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that registering None raises INVALID_PARAMETER error."""
        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_route(None)  # type: ignore[arg-type]

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_route_after_freeze_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that route registration after freeze raises INVALID_STATE error."""
        dispatch_engine.freeze()

        route = ModelDispatchRoute(
            route_id="late-route",
            topic_pattern="*.user.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="handler",
        )

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_route(route)

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE
        assert "frozen" in exc_info.value.message.lower()


# ============================================================================
# Handler Registration Tests
# ============================================================================


@pytest.mark.unit
class TestHandlerRegistration:
    """Tests for handler registration functionality."""

    def test_register_handler_valid_sync(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test successful sync handler registration."""

        def sync_handler(envelope: ModelEventEnvelope[object]) -> str:
            return "handled"

        dispatch_engine.register_dispatcher(
            dispatcher_id="sync-handler",
            dispatcher=sync_handler,
            category=EnumMessageCategory.EVENT,
        )

        assert dispatch_engine.dispatcher_count == 1

    def test_register_handler_valid_async(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test successful async handler registration."""

        async def async_handler(envelope: ModelEventEnvelope[object]) -> str:
            return "handled"

        dispatch_engine.register_dispatcher(
            dispatcher_id="async-handler",
            dispatcher=async_handler,
            category=EnumMessageCategory.EVENT,
        )

        assert dispatch_engine.dispatcher_count == 1

    def test_register_handler_with_message_types(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test handler registration with specific message types."""

        def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "handled"

        dispatch_engine.register_dispatcher(
            dispatcher_id="typed-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserCreatedEvent", "UserUpdatedEvent"},
        )

        assert dispatch_engine.dispatcher_count == 1

    def test_register_handler_multiple_categories(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test registering handlers for different categories."""

        def event_handler(envelope: ModelEventEnvelope[object]) -> str:
            return "event"

        def command_handler(envelope: ModelEventEnvelope[object]) -> str:
            return "command"

        def intent_handler(envelope: ModelEventEnvelope[object]) -> str:
            return "intent"

        dispatch_engine.register_dispatcher(
            dispatcher_id="event-handler",
            dispatcher=event_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="command-handler",
            dispatcher=command_handler,
            category=EnumMessageCategory.COMMAND,
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="intent-handler",
            dispatcher=intent_handler,
            category=EnumMessageCategory.INTENT,
        )

        assert dispatch_engine.dispatcher_count == 3

    def test_register_handler_duplicate_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that duplicate dispatcher_id raises DUPLICATE_REGISTRATION error."""

        def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "handled"

        dispatch_engine.register_dispatcher(
            dispatcher_id="dup-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_dispatcher(
                dispatcher_id="dup-handler",  # Same ID
                dispatcher=handler,
                category=EnumMessageCategory.COMMAND,  # Different category
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.DUPLICATE_REGISTRATION
        assert "dup-handler" in exc_info.value.message

    def test_register_handler_empty_id_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that empty dispatcher_id raises INVALID_PARAMETER error."""

        def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "handled"

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_dispatcher(
                dispatcher_id="",
                dispatcher=handler,
                category=EnumMessageCategory.EVENT,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_handler_whitespace_id_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that whitespace-only dispatcher_id raises INVALID_PARAMETER error."""

        def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "handled"

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_dispatcher(
                dispatcher_id="   ",
                dispatcher=handler,
                category=EnumMessageCategory.EVENT,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_handler_none_callable_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that non-callable handler raises INVALID_PARAMETER error."""
        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_dispatcher(
                dispatcher_id="bad-handler",
                dispatcher=None,  # type: ignore[arg-type]
                category=EnumMessageCategory.EVENT,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_handler_non_callable_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that non-callable object raises INVALID_PARAMETER error."""
        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_dispatcher(
                dispatcher_id="bad-handler",
                dispatcher="not a function",  # type: ignore[arg-type]
                category=EnumMessageCategory.EVENT,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_handler_invalid_category_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that invalid category raises INVALID_PARAMETER error."""

        def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "handled"

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_dispatcher(
                dispatcher_id="handler",
                dispatcher=handler,
                category="not_a_category",  # type: ignore[arg-type]
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_handler_after_freeze_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that handler registration after freeze raises INVALID_STATE error."""
        dispatch_engine.freeze()

        def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "handled"

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_dispatcher(
                dispatcher_id="late-handler",
                dispatcher=handler,
                category=EnumMessageCategory.EVENT,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE
        assert "frozen" in exc_info.value.message.lower()


# ============================================================================
# Freeze Pattern Tests
# ============================================================================


@pytest.mark.unit
class TestFreezePattern:
    """Tests for the freeze-after-init pattern."""

    def test_freeze_sets_frozen_flag(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that freeze() sets the frozen flag."""
        assert not dispatch_engine.is_frozen

        dispatch_engine.freeze()

        assert dispatch_engine.is_frozen

    def test_freeze_double_freeze_is_idempotent(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that calling freeze() multiple times is idempotent."""
        dispatch_engine.freeze()
        assert dispatch_engine.is_frozen

        # Second freeze should not raise
        dispatch_engine.freeze()
        assert dispatch_engine.is_frozen

    def test_freeze_validates_route_handler_references(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that freeze validates all routes reference existing handlers."""
        # Register a route without a matching handler
        route = ModelDispatchRoute(
            route_id="orphan-route",
            topic_pattern="*.user.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="nonexistent-handler",
        )
        dispatch_engine.register_route(route)

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.freeze()

        assert exc_info.value.error_code == EnumCoreErrorCode.ITEM_NOT_REGISTERED
        assert "nonexistent-handler" in exc_info.value.message

    def test_freeze_with_valid_configuration(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test successful freeze with valid route-handler configuration."""

        def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "handled"

        dispatch_engine.register_dispatcher(
            dispatcher_id="user-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="user-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="user-handler",
            )
        )

        # Should not raise
        dispatch_engine.freeze()

        assert dispatch_engine.is_frozen

    def test_freeze_empty_engine(self, dispatch_engine: MessageDispatchEngine) -> None:
        """Test freeze with no routes or handlers."""
        # Should not raise - empty engine is valid
        dispatch_engine.freeze()

        assert dispatch_engine.is_frozen
        assert dispatch_engine.route_count == 0
        assert dispatch_engine.dispatcher_count == 0


# ============================================================================
# Dispatch Success Tests
# ============================================================================


@pytest.mark.unit
class TestDispatchSuccess:
    """Tests for successful dispatch operations."""

    @pytest.mark.asyncio
    async def test_dispatch_single_handler(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test dispatch with a single matching handler."""
        results: list[str] = []

        async def handler(envelope: ModelEventEnvelope[object]) -> str:
            results.append("handled")
            return "output.topic.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="event-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="event-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="event-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1
        assert result.outputs is not None
        assert "output.topic.v1" in result.outputs

    @pytest.mark.asyncio
    async def test_dispatch_sync_handler(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test dispatch with a sync handler (runs in executor)."""
        results: list[str] = []

        def sync_handler(envelope: ModelEventEnvelope[object]) -> str:
            results.append("sync_handled")
            return "sync.output.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="sync-handler",
            dispatcher=sync_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="sync-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="sync-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dispatch_multiple_handlers_fan_out(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test fan-out dispatch to multiple handlers via multiple routes."""
        results: list[str] = []

        async def handler1(envelope: ModelEventEnvelope[object]) -> str:
            results.append("handler1")
            return "output1.v1"

        async def handler2(envelope: ModelEventEnvelope[object]) -> str:
            results.append("handler2")
            return "output2.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler-1",
            dispatcher=handler1,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="handler-2",
            dispatcher=handler2,
            category=EnumMessageCategory.EVENT,
        )

        # Two routes pointing to different handlers, both match the topic
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route-1",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler-1",
            )
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route-2",
                topic_pattern="dev.**",  # Also matches
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler-2",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 2
        assert "handler1" in results
        assert "handler2" in results
        assert result.output_count == 2

    @pytest.mark.asyncio
    async def test_dispatch_handler_returning_list_of_outputs(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test handler that returns list of output topics."""

        async def handler(envelope: ModelEventEnvelope[object]) -> list[str]:
            return ["output1.v1", "output2.v1", "output3.v1"]

        dispatch_engine.register_dispatcher(
            dispatcher_id="multi-output-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="multi-output-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert result.output_count == 3
        assert result.outputs is not None
        assert len(result.outputs) == 3

    @pytest.mark.asyncio
    async def test_dispatch_handler_returning_none(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test handler that returns None (no outputs)."""

        async def handler(envelope: ModelEventEnvelope[object]) -> None:
            pass  # No return value

        dispatch_engine.register_dispatcher(
            dispatcher_id="void-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="void-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert result.output_count == 0

    @pytest.mark.asyncio
    async def test_dispatch_with_message_type_filter(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test dispatch with message type filtering."""
        results: list[str] = []

        async def user_created_handler(envelope: ModelEventEnvelope[object]) -> str:
            results.append("user_created")
            return "created.output"

        async def user_updated_handler(envelope: ModelEventEnvelope[object]) -> str:
            results.append("user_updated")
            return "updated.output"

        dispatch_engine.register_dispatcher(
            dispatcher_id="created-handler",
            dispatcher=user_created_handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserCreatedEvent"},  # Only handles UserCreatedEvent
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="updated-handler",
            dispatcher=user_updated_handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserUpdatedEvent"},  # Only handles UserUpdatedEvent
        )

        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="created-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="created-handler",
            )
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="updated-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="updated-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        # Only created-handler should be invoked
        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1
        assert "user_created" in results

    @pytest.mark.asyncio
    async def test_dispatch_preserves_correlation_id(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that dispatch result preserves envelope correlation_id."""

        async def handler(envelope: ModelEventEnvelope[object]) -> None:
            pass

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.correlation_id == event_envelope.correlation_id


# ============================================================================
# Dispatch Error Tests
# ============================================================================


@pytest.mark.unit
class TestDispatchErrors:
    """Tests for dispatch error scenarios."""

    @pytest.mark.asyncio
    async def test_dispatch_before_freeze_raises_error(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that dispatch before freeze raises INVALID_STATE error."""
        # Don't call freeze()

        with pytest.raises(ModelOnexError) as exc_info:
            await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE
        assert "freeze" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_dispatch_empty_topic_raises_error(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that empty topic raises INVALID_PARAMETER error."""
        dispatch_engine.freeze()

        with pytest.raises(ModelOnexError) as exc_info:
            await dispatch_engine.dispatch("", event_envelope)

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    @pytest.mark.asyncio
    async def test_dispatch_whitespace_topic_raises_error(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that whitespace-only topic raises INVALID_PARAMETER error."""
        dispatch_engine.freeze()

        with pytest.raises(ModelOnexError) as exc_info:
            await dispatch_engine.dispatch("   ", event_envelope)

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    @pytest.mark.asyncio
    async def test_dispatch_none_envelope_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that None envelope raises INVALID_PARAMETER error."""
        dispatch_engine.freeze()

        with pytest.raises(ModelOnexError) as exc_info:
            await dispatch_engine.dispatch(
                "dev.user.events.v1",
                None,  # type: ignore[arg-type]
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    @pytest.mark.asyncio
    async def test_dispatch_no_handlers_returns_no_dispatcher_status(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test dispatch with no matching handlers returns NO_DISPATCHER status."""
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.NO_DISPATCHER
        assert result.error_message is not None
        assert "No dispatcher" in result.error_message

    @pytest.mark.asyncio
    async def test_dispatch_invalid_topic_returns_invalid_message(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test dispatch with invalid topic (no category) returns INVALID_MESSAGE."""
        dispatch_engine.freeze()

        # Topic without events/commands/intents segment
        result = await dispatch_engine.dispatch("invalid.topic.here", event_envelope)

        assert result.status == EnumDispatchStatus.INVALID_MESSAGE
        assert result.error_message is not None
        assert "category" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_dispatch_projection_topic_returns_invalid_message(
        self,
        dispatch_engine: MessageDispatchEngine,
    ) -> None:
        """Test that PROJECTION topics are NOT routable via MessageDispatchEngine.

        Architectural Decision (OMN-985):
            PROJECTION is NOT a message category for routing. Projections are:
            - Node output types (EnumNodeOutputType.PROJECTION), not message categories
            - Produced by REDUCER nodes as local state outputs
            - Applied locally by the runtime to a projection sink
            - NOT routed via Kafka topics or MessageDispatchEngine

            Topics containing ".projections" segment are therefore invalid for
            dispatch because EnumMessageCategory.from_topic() does not recognize
            "projections" as a valid category suffix.

        See Also:
            - EnumMessageCategory: Only EVENT, COMMAND, INTENT are valid
            - EnumNodeOutputType: PROJECTION exists here for node validation
            - CLAUDE.md "Enum Usage" section for full distinction
        """
        dispatch_engine.freeze()

        # Create envelope with projection payload
        projection_envelope = ModelEventEnvelope(
            payload=OrderSummaryProjection(order_id="order-123", total=99.99),
            correlation_id=uuid4(),
        )

        # Topic with .projections segment - NOT a valid routable category
        result = await dispatch_engine.dispatch(
            "dev.order.projections.v1", projection_envelope
        )

        # PROJECTION is not recognized as a message category
        assert result.status == EnumDispatchStatus.INVALID_MESSAGE
        assert result.error_message is not None
        assert "category" in result.error_message.lower()
        # Verify the topic is mentioned in the error for debugging
        assert "projections" in result.error_message.lower()

    @pytest.mark.skip(
        reason="TODO(OMN-934): Re-enable when ModelEventEnvelope.infer_category() is implemented in omnibase_core"
    )
    @pytest.mark.asyncio
    async def test_dispatch_category_mismatch_returns_invalid_message(
        self,
        dispatch_engine: MessageDispatchEngine,
        command_envelope: ModelEventEnvelope[CreateUserCommand],
    ) -> None:
        """Test dispatch where envelope category doesn't match topic category."""
        dispatch_engine.freeze()

        # Sending a COMMAND envelope to an events topic
        result = await dispatch_engine.dispatch("dev.user.events.v1", command_envelope)

        assert result.status == EnumDispatchStatus.INVALID_MESSAGE
        assert result.error_message is not None
        assert "mismatch" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception_returns_handler_error(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that handler exception results in HANDLER_ERROR status."""

        async def failing_handler(envelope: ModelEventEnvelope[object]) -> None:
            raise ValueError("Something went wrong!")

        dispatch_engine.register_dispatcher(
            dispatcher_id="failing-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="failing-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None
        assert "Something went wrong" in result.error_message

    @pytest.mark.asyncio
    async def test_dispatch_partial_handler_failure(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test dispatch where some handlers succeed and some fail."""
        results: list[str] = []

        async def success_handler(envelope: ModelEventEnvelope[object]) -> str:
            results.append("success")
            return "success.output"

        async def failing_handler(envelope: ModelEventEnvelope[object]) -> None:
            results.append("failing")
            raise RuntimeError("Handler failed!")

        dispatch_engine.register_dispatcher(
            dispatcher_id="success-handler",
            dispatcher=success_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="failing-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )

        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="success-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="success-handler",
            )
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="failing-route",
                topic_pattern="dev.**",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="failing-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        # Both handlers should have been called
        assert len(results) == 2
        assert "success" in results
        assert "failing" in results

        # Status should be HANDLER_ERROR due to partial failure
        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None
        assert "Handler failed" in result.error_message

        # But we should still have the output from the successful handler
        assert result.outputs is not None
        assert "success.output" in result.outputs

    @pytest.mark.asyncio
    async def test_dispatch_disabled_route_not_matched(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that disabled routes are not matched."""

        async def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "handled"

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="disabled-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
                enabled=False,  # Disabled
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        # No handlers should match due to disabled route
        assert result.status == EnumDispatchStatus.NO_DISPATCHER

    # ---- Correlation ID Propagation in Error Results ----

    @pytest.mark.asyncio
    async def test_dispatch_no_dispatcher_preserves_correlation_id(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that NO_DISPATCHER error result preserves envelope correlation_id."""
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.NO_DISPATCHER
        assert result.correlation_id == event_envelope.correlation_id

    @pytest.mark.asyncio
    async def test_dispatch_handler_error_preserves_correlation_id(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that HANDLER_ERROR result preserves envelope correlation_id."""

        async def failing_handler(envelope: ModelEventEnvelope[object]) -> None:
            raise ValueError("Handler crashed!")

        dispatch_engine.register_dispatcher(
            dispatcher_id="failing-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="failing-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.correlation_id == event_envelope.correlation_id

    @pytest.mark.asyncio
    async def test_dispatch_invalid_message_preserves_correlation_id(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that INVALID_MESSAGE result preserves envelope correlation_id."""
        dispatch_engine.freeze()

        # Topic without valid category segment (no events/commands/intents)
        result = await dispatch_engine.dispatch("invalid.topic.here", event_envelope)

        assert result.status == EnumDispatchStatus.INVALID_MESSAGE
        assert result.correlation_id == event_envelope.correlation_id

    @pytest.mark.asyncio
    async def test_dispatch_partial_failure_preserves_correlation_id(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that partial handler failure result preserves envelope correlation_id."""

        async def success_handler(envelope: ModelEventEnvelope[object]) -> str:
            return "success"

        async def failing_handler(envelope: ModelEventEnvelope[object]) -> None:
            raise RuntimeError("Partial failure!")

        dispatch_engine.register_dispatcher(
            dispatcher_id="success-handler",
            dispatcher=success_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="failing-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )

        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="success-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="success-handler",
            )
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="failing-route",
                topic_pattern="dev.**",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="failing-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        # Partial failure should still result in HANDLER_ERROR
        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        # But correlation_id should still be preserved
        assert result.correlation_id == event_envelope.correlation_id


# ============================================================================
# Async Handler Tests
# ============================================================================


@pytest.mark.unit
class TestAsyncHandlers:
    """Tests for async handler functionality."""

    @pytest.mark.asyncio
    async def test_async_handler_with_await(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test async handler that uses await."""
        results: list[str] = []

        async def async_handler(envelope: ModelEventEnvelope[object]) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            results.append("async_complete")
            return "async.output"

        dispatch_engine.register_dispatcher(
            dispatcher_id="async-handler",
            dispatcher=async_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="async-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1
        assert results[0] == "async_complete"


# ============================================================================
# Metrics Tests
# ============================================================================


@pytest.mark.unit
class TestMetrics:
    """Tests for metrics collection."""

    def test_initial_metrics(self, dispatch_engine: MessageDispatchEngine) -> None:
        """Test initial metrics values."""
        metrics = dispatch_engine.get_structured_metrics()

        assert metrics.total_dispatches == 0
        assert metrics.successful_dispatches == 0
        assert metrics.failed_dispatches == 0
        assert metrics.total_latency_ms == 0.0
        assert metrics.dispatcher_execution_count == 0
        assert metrics.dispatcher_error_count == 0
        assert metrics.routes_matched_count == 0
        assert metrics.no_dispatcher_count == 0
        assert metrics.category_mismatch_count == 0

    @pytest.mark.asyncio
    async def test_metrics_updated_on_success(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test metrics are updated on successful dispatch."""

        async def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "test.output.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        dispatch_engine.freeze()

        await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == 1
        assert metrics.successful_dispatches == 1
        assert metrics.failed_dispatches == 0
        assert metrics.dispatcher_execution_count == 1
        assert metrics.total_latency_ms > 0
        assert metrics.routes_matched_count == 1

    @pytest.mark.asyncio
    async def test_metrics_updated_on_handler_error(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test metrics are updated on handler error."""

        async def failing_handler(envelope: ModelEventEnvelope[object]) -> None:
            raise ValueError("Failure!")

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        dispatch_engine.freeze()

        await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == 1
        assert metrics.failed_dispatches == 1
        assert metrics.dispatcher_execution_count == 1
        assert metrics.dispatcher_error_count == 1

    @pytest.mark.asyncio
    async def test_metrics_updated_on_no_dispatcher(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test metrics are updated when no dispatcher is found."""
        dispatch_engine.freeze()

        await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == 1
        assert metrics.failed_dispatches == 1
        assert metrics.no_dispatcher_count == 1

    @pytest.mark.skip(
        reason="TODO(OMN-934): Re-enable when ModelEventEnvelope.infer_category() is implemented in omnibase_core"
    )
    @pytest.mark.asyncio
    async def test_metrics_updated_on_category_mismatch(
        self,
        dispatch_engine: MessageDispatchEngine,
        command_envelope: ModelEventEnvelope[CreateUserCommand],
    ) -> None:
        """Test metrics are updated on category mismatch."""
        dispatch_engine.freeze()

        # Sending COMMAND envelope to events topic
        await dispatch_engine.dispatch("dev.user.events.v1", command_envelope)

        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == 1
        assert metrics.failed_dispatches == 1
        assert metrics.category_mismatch_count == 1

    @pytest.mark.asyncio
    async def test_metrics_accumulate_across_dispatches(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test metrics accumulate across multiple dispatches."""

        async def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "test.output.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        dispatch_engine.freeze()

        # Dispatch multiple times
        for _ in range(5):
            await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == 5
        assert metrics.successful_dispatches == 5
        assert metrics.dispatcher_execution_count == 5


# ============================================================================
# Deterministic Routing Tests
# ============================================================================


@pytest.mark.unit
class TestDeterministicRouting:
    """Tests for deterministic routing behavior (same input -> same handlers)."""

    @pytest.mark.asyncio
    async def test_same_input_same_handlers(
        self,
        dispatch_engine: MessageDispatchEngine,
    ) -> None:
        """Test that same input always produces same handler selection."""

        async def handler1(envelope: ModelEventEnvelope[object]) -> None:
            pass

        async def handler2(envelope: ModelEventEnvelope[object]) -> None:
            pass

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler-1",
            dispatcher=handler1,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="handler-2",
            dispatcher=handler2,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route-1",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler-1",
            )
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route-2",
                topic_pattern="dev.**",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler-2",
            )
        )
        dispatch_engine.freeze()

        # Dispatch multiple times with same input
        results: list[ModelDispatchResult] = []
        for _ in range(10):
            envelope = ModelEventEnvelope(
                payload=UserCreatedEvent(user_id="user-123", name="Test")
            )
            result = await dispatch_engine.dispatch("dev.user.events.v1", envelope)
            results.append(result)

        # All results should have the same dispatcher_id
        dispatcher_ids = [r.dispatcher_id for r in results]
        assert len(set(dispatcher_ids)) == 1  # All same
        assert all(r.status == EnumDispatchStatus.SUCCESS for r in results)

    @pytest.mark.asyncio
    async def test_different_topics_different_handlers(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that different topics route to different handlers."""

        async def user_handler(envelope: ModelEventEnvelope[object]) -> None:
            pass

        async def order_handler(envelope: ModelEventEnvelope[object]) -> None:
            pass

        dispatch_engine.register_dispatcher(
            dispatcher_id="user-handler",
            dispatcher=user_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="order-handler",
            dispatcher=order_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="user-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="user-handler",
            )
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="order-route",
                topic_pattern="*.order.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="order-handler",
            )
        )
        dispatch_engine.freeze()

        user_envelope = ModelEventEnvelope(
            payload=UserCreatedEvent(user_id="user-123", name="Test")
        )
        order_envelope = ModelEventEnvelope(payload=SomeGenericPayload(data="order"))

        user_result = await dispatch_engine.dispatch(
            "dev.user.events.v1", user_envelope
        )
        order_result = await dispatch_engine.dispatch(
            "dev.order.events.v1", order_envelope
        )

        assert user_result.dispatcher_id == "user-handler"
        assert order_result.dispatcher_id == "order-handler"


# ============================================================================
# Pure Routing Tests (No Workflow Inference)
# ============================================================================


@pytest.mark.unit
class TestPureRouting:
    """Tests verifying the engine performs pure routing without workflow inference."""

    @pytest.mark.asyncio
    async def test_no_workflow_inference_from_payload(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that routing is based on topic/category, not payload content.

        Note: With the strict JSON-safe dispatch contract (OMN-1518), payloads
        are serialized to dicts at the dispatch boundary. The original type
        information is not preserved - handlers should hydrate typed models
        locally if needed via model_validate().
        """
        handler_calls: list[dict[str, object]] = []

        async def handler(envelope: dict[str, object]) -> None:
            # Materialized envelope: payload is now a JSON-safe dict
            handler_calls.append(envelope["payload"])  # type: ignore[arg-type]

        dispatch_engine.register_dispatcher(
            dispatcher_id="generic-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="generic-handler",
            )
        )
        dispatch_engine.freeze()

        # Different payload types, same topic
        envelope1 = ModelEventEnvelope(
            payload=UserCreatedEvent(user_id="1", name="Alice")
        )
        envelope2 = ModelEventEnvelope(payload=SomeGenericPayload(data="test"))

        await dispatch_engine.dispatch("dev.user.events.v1", envelope1)
        await dispatch_engine.dispatch("dev.user.events.v1", envelope2)

        # Both should route to the same handler regardless of original payload type
        # Payloads are serialized to dicts (JSON-safe dispatch contract)
        assert len(handler_calls) == 2
        assert isinstance(handler_calls[0], dict)
        assert isinstance(handler_calls[1], dict)
        # Verify payload content is preserved after serialization
        assert handler_calls[0]["user_id"] == "1"
        assert handler_calls[0]["name"] == "Alice"
        assert handler_calls[1]["data"] == "test"

    @pytest.mark.asyncio
    async def test_outputs_are_publishing_only(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that outputs are collected for publishing, not interpreted."""

        # Handler returns various output formats
        async def handler(envelope: ModelEventEnvelope[object]) -> list[str]:
            return [
                "output.topic.v1",
                "another.output.v1",
                "third.output.commands.v1",  # Note: commands topic
            ]

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        # Outputs should be collected as-is for publishing
        assert result.status == EnumDispatchStatus.SUCCESS
        assert result.outputs is not None
        assert len(result.outputs) == 3
        # The engine doesn't interpret what these topics mean
        assert "output.topic.v1" in result.outputs
        assert "third.output.commands.v1" in result.outputs


# ============================================================================
# String Representation Tests
# ============================================================================


@pytest.mark.unit
class TestStringRepresentation:
    """Tests for __str__ and __repr__ methods."""

    def test_str_representation(self, dispatch_engine: MessageDispatchEngine) -> None:
        """Test __str__ method."""
        result = str(dispatch_engine)
        assert "MessageDispatchEngine" in result
        assert "routes=0" in result
        assert "dispatchers=0" in result
        assert "frozen=False" in result

    def test_str_representation_with_data(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test __str__ method with routes and handlers."""

        def handler(envelope: ModelEventEnvelope[object]) -> None:
            pass

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        dispatch_engine.freeze()

        result = str(dispatch_engine)
        assert "routes=1" in result
        assert "dispatchers=1" in result
        assert "frozen=True" in result

    def test_repr_representation(self, dispatch_engine: MessageDispatchEngine) -> None:
        """Test __repr__ method."""
        result = repr(dispatch_engine)
        assert "MessageDispatchEngine" in result
        assert "frozen=" in result


# ============================================================================
# Properties Tests
# ============================================================================


@pytest.mark.unit
class TestProperties:
    """Tests for engine properties."""

    def test_route_count(self, dispatch_engine: MessageDispatchEngine) -> None:
        """Test route_count property."""
        assert dispatch_engine.route_count == 0

        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route-1",
                topic_pattern="*.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        assert dispatch_engine.route_count == 1

        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route-2",
                topic_pattern="*.commands.*",
                message_category=EnumMessageCategory.COMMAND,
                dispatcher_id="handler",
            )
        )
        assert dispatch_engine.route_count == 2

    def test_dispatcher_count(self, dispatch_engine: MessageDispatchEngine) -> None:
        """Test dispatcher_count property."""

        def handler(envelope: ModelEventEnvelope[object]) -> None:
            pass

        assert dispatch_engine.dispatcher_count == 0

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler-1",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        assert dispatch_engine.dispatcher_count == 1

        dispatch_engine.register_dispatcher(
            dispatcher_id="handler-2",
            dispatcher=handler,
            category=EnumMessageCategory.COMMAND,
        )
        assert dispatch_engine.dispatcher_count == 2


# ============================================================================
# ModelDispatchResult Retry Scenario Tests
# ============================================================================


@pytest.mark.unit
class TestDispatchResultRetryScenarios:
    """Tests for ModelDispatchResult.requires_retry() method.

    These tests verify that the requires_retry() method correctly identifies
    which dispatch statuses should trigger retry attempts:
    - TIMEOUT: Transient failure, should retry
    - PUBLISH_FAILED: Transient failure, should retry
    - SUCCESS: Not an error, should NOT retry
    - HANDLER_ERROR: Permanent failure, should NOT retry
    - NO_DISPATCHER: Configuration error, should NOT retry
    - INVALID_MESSAGE: Validation error, should NOT retry
    - SKIPPED: Intentional skip, should NOT retry
    - ROUTED: Intermediate state, should NOT retry
    """

    def test_timeout_requires_retry(self) -> None:
        """Test that TIMEOUT status requires retry (transient failure)."""
        result = ModelDispatchResult(
            status=EnumDispatchStatus.TIMEOUT,
            topic="dev.user.events.v1",
            error_message="Handler execution timed out after 30s",
            started_at=datetime.now(UTC),
        )

        assert result.requires_retry() is True
        assert result.is_error() is True
        assert result.is_successful() is False

    def test_publish_failed_requires_retry(self) -> None:
        """Test that PUBLISH_FAILED status requires retry (transient failure)."""
        result = ModelDispatchResult(
            status=EnumDispatchStatus.PUBLISH_FAILED,
            topic="dev.user.events.v1",
            error_message="Failed to publish to output topic",
            started_at=datetime.now(UTC),
        )

        assert result.requires_retry() is True
        assert result.is_error() is True
        assert result.is_successful() is False

    def test_success_does_not_require_retry(self) -> None:
        """Test that SUCCESS status does NOT require retry."""
        result = ModelDispatchResult(
            status=EnumDispatchStatus.SUCCESS,
            topic="dev.user.events.v1",
            outputs=ModelDispatchOutputs(topics=["dev.notification.events.v1"]),
            started_at=datetime.now(UTC),
        )

        assert result.requires_retry() is False
        assert result.is_error() is False
        assert result.is_successful() is True

    def test_handler_error_does_not_require_retry(self) -> None:
        """Test that HANDLER_ERROR status does NOT require retry (permanent failure)."""
        result = ModelDispatchResult(
            status=EnumDispatchStatus.HANDLER_ERROR,
            topic="dev.user.events.v1",
            error_message="ValueError: Invalid user data",
            started_at=datetime.now(UTC),
        )

        assert result.requires_retry() is False
        assert result.is_error() is True
        assert result.is_successful() is False

    def test_no_dispatcher_does_not_require_retry(self) -> None:
        """Test that NO_DISPATCHER status does NOT require retry (configuration error)."""
        result = ModelDispatchResult(
            status=EnumDispatchStatus.NO_DISPATCHER,
            topic="dev.unknown.events.v1",
            error_message="No dispatcher registered for topic",
            started_at=datetime.now(UTC),
        )

        assert result.requires_retry() is False
        assert result.is_error() is True

    def test_invalid_message_does_not_require_retry(self) -> None:
        """Test that INVALID_MESSAGE status does NOT require retry (validation error)."""
        result = ModelDispatchResult(
            status=EnumDispatchStatus.INVALID_MESSAGE,
            topic="dev.user.events.v1",
            error_message="Message failed schema validation",
            started_at=datetime.now(UTC),
        )

        assert result.requires_retry() is False
        assert result.is_error() is True

    def test_skipped_does_not_require_retry(self) -> None:
        """Test that SKIPPED status does NOT require retry (intentional skip)."""
        result = ModelDispatchResult(
            status=EnumDispatchStatus.SKIPPED,
            topic="dev.user.events.v1",
            started_at=datetime.now(UTC),
        )

        assert result.requires_retry() is False
        assert result.is_error() is False  # SKIPPED is not an error
        assert result.is_successful() is False

    def test_routed_does_not_require_retry(self) -> None:
        """Test that ROUTED status does NOT require retry (intermediate state)."""
        result = ModelDispatchResult(
            status=EnumDispatchStatus.ROUTED,
            topic="dev.user.events.v1",
            route_id="user-events-route",
            started_at=datetime.now(UTC),
        )

        assert result.requires_retry() is False
        assert result.is_error() is False
        assert result.is_successful() is False
        assert result.is_terminal() is False  # ROUTED is not terminal


# ============================================================================
# Concurrency Tests
# ============================================================================


def _dispatch_in_thread_helper(
    dispatch_engine: MessageDispatchEngine,
    topic: str,
    envelope: ModelEventEnvelope[object],
) -> ModelDispatchResult:
    """Run async dispatch from a synchronous thread context.

    This helper function enables running async dispatch operations from
    synchronous threads in concurrent test scenarios. It creates a new
    event loop for each thread and properly cleans up afterward.

    Args:
        dispatch_engine: The dispatch engine to use for dispatching.
        topic: The topic to dispatch to.
        envelope: The envelope containing the message to dispatch.

    Returns:
        The dispatch result from the engine.

    Note:
        This is a shared helper to avoid code duplication in concurrency tests.
        Each thread gets its own event loop to ensure isolation.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(dispatch_engine.dispatch(topic, envelope))
        return result
    finally:
        loop.close()


def _create_envelope_with_category(
    payload: object, category: EnumMessageCategory
) -> ModelEventEnvelope[object]:
    """Create an envelope with infer_category method for testing.

    This helper adds the infer_category method that the MessageDispatchEngine
    expects but may not be present in all versions of omnibase_core.
    Uses object.__setattr__ to bypass Pydantic's strict attribute checking.
    """
    envelope = ModelEventEnvelope(
        payload=payload,
        correlation_id=uuid4(),
    )
    # Add infer_category method dynamically using object.__setattr__
    # to bypass Pydantic's validation
    object.__setattr__(envelope, "infer_category", lambda: category)
    return envelope


@pytest.mark.unit
class TestMessageDispatchEngineConcurrency:
    """Test thread safety of message dispatch engine."""

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_thread_safety(self) -> None:
        """Verify dispatch is thread-safe under concurrent load.

        This test validates the freeze-after-init pattern by:
        1. Setting up a dispatch engine with a handler
        2. Freezing the engine to enable thread-safe dispatch
        3. Dispatching from multiple threads concurrently
        4. Verifying all dispatches succeed and metrics are accurate
        """
        import concurrent.futures

        # Setup engine
        dispatch_engine = MessageDispatchEngine()
        dispatch_count = 20  # Number of concurrent dispatches
        results: list[str] = []
        results_lock = threading.Lock()  # Protect the results list

        async def handler(envelope: dict[str, object]) -> str:
            # Thread-safe append to results - materialized envelope is a dict
            payload = envelope["payload"]
            with results_lock:
                results.append(f"handled-{payload['user_id']}")
            return "output.topic.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="concurrent-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="concurrent-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="concurrent-handler",
            )
        )
        dispatch_engine.freeze()

        # Create envelopes for concurrent dispatch with infer_category method
        envelopes = [
            _create_envelope_with_category(
                UserCreatedEvent(user_id=f"user-{i}", name=f"User {i}"),
                EnumMessageCategory.EVENT,
            )
            for i in range(dispatch_count)
        ]

        # Execute concurrent dispatches using ThreadPoolExecutor
        # Uses shared helper _dispatch_in_thread_helper
        dispatch_results: list[ModelDispatchResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    _dispatch_in_thread_helper,
                    dispatch_engine,
                    "dev.user.events.v1",
                    envelope,
                )
                for envelope in envelopes
            ]
            for future in concurrent.futures.as_completed(futures):
                dispatch_results.append(future.result())

        # Verify all dispatches completed successfully
        assert len(dispatch_results) == dispatch_count
        for result in dispatch_results:
            assert result.status == EnumDispatchStatus.SUCCESS, (
                f"Dispatch failed: {result.error_message}"
            )

        # Verify handler was invoked for all dispatches
        assert len(results) == dispatch_count

        # Verify metrics match dispatch count
        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == dispatch_count
        assert metrics.successful_dispatches == dispatch_count
        assert metrics.failed_dispatches == 0
        assert metrics.dispatcher_execution_count == dispatch_count

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_with_multiple_handlers(self) -> None:
        """Test concurrent dispatch with fan-out to multiple handlers.

        Verifies thread safety when messages are routed to multiple handlers
        simultaneously from multiple threads.
        """
        import concurrent.futures

        dispatch_engine = MessageDispatchEngine()
        dispatch_count = 15
        handler1_results: list[str] = []
        handler2_results: list[str] = []
        lock = threading.Lock()

        async def handler1(envelope: dict[str, object]) -> str:
            payload = envelope["payload"]
            with lock:
                handler1_results.append(f"h1-{payload['user_id']}")
            return "output1.v1"

        async def handler2(envelope: dict[str, object]) -> str:
            payload = envelope["payload"]
            with lock:
                handler2_results.append(f"h2-{payload['user_id']}")
            return "output2.v1"

        # Register two handlers
        dispatch_engine.register_dispatcher(
            dispatcher_id="handler-1",
            dispatcher=handler1,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="handler-2",
            dispatcher=handler2,
            category=EnumMessageCategory.EVENT,
        )

        # Two routes matching the same topic pattern, different handlers
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route-1",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler-1",
            )
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="route-2",
                topic_pattern="dev.**",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler-2",
            )
        )
        dispatch_engine.freeze()

        # Create envelopes with infer_category method
        envelopes = [
            _create_envelope_with_category(
                UserCreatedEvent(user_id=f"user-{i}", name=f"User {i}"),
                EnumMessageCategory.EVENT,
            )
            for i in range(dispatch_count)
        ]

        # Execute concurrent dispatches using shared helper
        dispatch_results: list[ModelDispatchResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    _dispatch_in_thread_helper,
                    dispatch_engine,
                    "dev.user.events.v1",
                    envelope,
                )
                for envelope in envelopes
            ]
            for future in concurrent.futures.as_completed(futures):
                dispatch_results.append(future.result())

        # Verify all dispatches succeeded
        assert len(dispatch_results) == dispatch_count
        for result in dispatch_results:
            assert result.status == EnumDispatchStatus.SUCCESS

        # Both handlers should have been called for each dispatch (fan-out)
        assert len(handler1_results) == dispatch_count
        assert len(handler2_results) == dispatch_count

        # Verify metrics
        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == dispatch_count
        assert metrics.successful_dispatches == dispatch_count
        # Each dispatch invokes 2 handlers (fan-out)
        assert metrics.dispatcher_execution_count == dispatch_count * 2

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_with_failures(self) -> None:
        """Test concurrent dispatch where some handlers succeed and some fail.

        Verifies that the dispatch engine correctly handles mixed success/failure
        scenarios under concurrent load, including proper metrics tracking.
        """
        import concurrent.futures

        dispatch_engine = MessageDispatchEngine()
        dispatch_count = 20
        success_results: list[str] = []
        failure_results: list[str] = []
        lock = threading.Lock()

        async def success_handler(envelope: dict[str, object]) -> str:
            payload = envelope["payload"]
            with lock:
                success_results.append(f"success-{payload['user_id']}")
            return "success.output.v1"

        async def failing_handler(envelope: dict[str, object]) -> str:
            payload = envelope["payload"]
            with lock:
                failure_results.append(f"failed-{payload['user_id']}")
            raise RuntimeError("Simulated handler failure")

        # Register both handlers
        dispatch_engine.register_dispatcher(
            dispatcher_id="success-handler",
            dispatcher=success_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="failing-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )

        # Two routes, both matching the topic, one to each handler
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="success-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="success-handler",
            )
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="failing-route",
                topic_pattern="dev.**",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="failing-handler",
            )
        )
        dispatch_engine.freeze()

        # Create envelopes with infer_category method
        envelopes = [
            _create_envelope_with_category(
                UserCreatedEvent(user_id=f"user-{i}", name=f"User {i}"),
                EnumMessageCategory.EVENT,
            )
            for i in range(dispatch_count)
        ]

        # Execute concurrent dispatches using shared helper
        dispatch_results: list[ModelDispatchResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    _dispatch_in_thread_helper,
                    dispatch_engine,
                    "dev.user.events.v1",
                    envelope,
                )
                for envelope in envelopes
            ]
            for future in concurrent.futures.as_completed(futures):
                dispatch_results.append(future.result())

        # All dispatches should complete (with HANDLER_ERROR status due to partial failure)
        assert len(dispatch_results) == dispatch_count

        # Each dispatch should have HANDLER_ERROR status (partial failure)
        for result in dispatch_results:
            assert result.status == EnumDispatchStatus.HANDLER_ERROR, (
                f"Expected HANDLER_ERROR, got {result.status}"
            )
            # Should still have output from successful handler
            assert result.outputs is not None
            assert "success.output.v1" in result.outputs

        # Both handlers should have been called for each dispatch
        assert len(success_results) == dispatch_count
        assert len(failure_results) == dispatch_count

        # Verify metrics track both successes and failures
        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == dispatch_count
        # All dispatches are marked as errors (due to partial failure)
        assert metrics.failed_dispatches == dispatch_count
        # Each dispatch executes 2 handlers (1 success + 1 failure)
        assert metrics.dispatcher_execution_count == dispatch_count * 2
        # One handler fails per dispatch
        assert metrics.dispatcher_error_count == dispatch_count

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_metrics_accuracy(self) -> None:
        """Test that metrics remain accurate under concurrent load.

        Specifically tests that counter increments are not lost due to
        race conditions when multiple threads update metrics simultaneously.
        """
        import concurrent.futures

        dispatch_engine = MessageDispatchEngine()
        dispatch_count = 50  # Higher count to stress test metrics

        async def handler(envelope: ModelEventEnvelope[object]) -> str:
            return "output.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="metrics-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="metrics-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="metrics-handler",
            )
        )
        dispatch_engine.freeze()

        # Create envelopes with infer_category method
        envelopes = [
            _create_envelope_with_category(
                UserCreatedEvent(user_id=f"user-{i}", name=f"User {i}"),
                EnumMessageCategory.EVENT,
            )
            for i in range(dispatch_count)
        ]

        # Use more workers than dispatches to maximize concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(
                    _dispatch_in_thread_helper,
                    dispatch_engine,
                    "dev.user.events.v1",
                    envelope,
                )
                for envelope in envelopes
            ]
            concurrent.futures.wait(futures)

        # Verify metrics accuracy
        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == dispatch_count, (
            f"Expected {dispatch_count} dispatches, got {metrics.total_dispatches}"
        )
        assert metrics.successful_dispatches == dispatch_count, (
            f"Expected {dispatch_count} successes, got {metrics.successful_dispatches}"
        )
        assert metrics.dispatcher_execution_count == dispatch_count, (
            f"Expected {dispatch_count} dispatcher executions, "
            f"got {metrics.dispatcher_execution_count}"
        )
        assert metrics.routes_matched_count == dispatch_count, (
            f"Expected {dispatch_count} route matches, "
            f"got {metrics.routes_matched_count}"
        )


# ============================================================================
# Command and Intent Dispatch Tests
# ============================================================================


@pytest.mark.unit
class TestConcurrentDispatchAdvanced:
    """Advanced concurrency tests for thread safety validation.

    These tests focus on edge cases and stress scenarios:
    - Async handlers with varying delays (race conditions)
    - Extreme concurrency metrics lock validation
    - Message type filtering under concurrent load
    - Long-running stability tests
    - Async cancellation handling
    """

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_with_varying_delays(self) -> None:
        """Test concurrent dispatch with async handlers that have varying delays.

        This validates that handlers with different execution times complete
        correctly and don't interfere with each other under concurrent load.
        """
        import concurrent.futures
        import random

        dispatch_engine = MessageDispatchEngine()
        dispatch_count = 30
        execution_order: list[str] = []
        lock = threading.Lock()

        async def variable_delay_handler(envelope: dict[str, object]) -> str:
            """Handler with random delay to simulate varying workloads."""
            payload = envelope["payload"]
            user_id = payload["user_id"]
            # Random delay between 1-50ms
            delay = random.uniform(0.001, 0.05)
            await asyncio.sleep(delay)
            with lock:
                execution_order.append(f"{user_id}-delay-{delay:.3f}")
            return f"output.user.{user_id}"

        dispatch_engine.register_dispatcher(
            dispatcher_id="variable-handler",
            dispatcher=variable_delay_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="variable-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="variable-handler",
            )
        )
        dispatch_engine.freeze()

        # Create envelopes with infer_category method
        envelopes = [
            _create_envelope_with_category(
                UserCreatedEvent(user_id=f"user-{i}", name=f"User {i}"),
                EnumMessageCategory.EVENT,
            )
            for i in range(dispatch_count)
        ]

        # Execute concurrent dispatches using shared helper
        dispatch_results: list[ModelDispatchResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(
                    _dispatch_in_thread_helper,
                    dispatch_engine,
                    "dev.user.events.v1",
                    envelope,
                )
                for envelope in envelopes
            ]
            for future in concurrent.futures.as_completed(futures):
                dispatch_results.append(future.result())

        # Verify all dispatches completed successfully
        assert len(dispatch_results) == dispatch_count
        for result in dispatch_results:
            assert result.status == EnumDispatchStatus.SUCCESS, (
                f"Dispatch failed: {result.error_message}"
            )

        # Verify all handlers executed
        assert len(execution_order) == dispatch_count

        # Verify metrics
        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == dispatch_count
        assert metrics.successful_dispatches == dispatch_count

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_metrics_lock_stress(self) -> None:
        """Stress test the metrics lock under extreme concurrency.

        Uses high thread count and rapid dispatch to verify metrics lock
        correctly protects structured metrics from race conditions.
        """
        import concurrent.futures

        dispatch_engine = MessageDispatchEngine()
        dispatch_count = 100  # High count for stress testing
        thread_count = 30  # More threads than typical

        async def fast_handler(envelope: ModelEventEnvelope[object]) -> str:
            # Minimal work to maximize contention on metrics lock
            return "output.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="fast-handler",
            dispatcher=fast_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="fast-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="fast-handler",
            )
        )
        dispatch_engine.freeze()

        # Create envelopes
        envelopes = [
            _create_envelope_with_category(
                UserCreatedEvent(user_id=f"user-{i}", name=f"User {i}"),
                EnumMessageCategory.EVENT,
            )
            for i in range(dispatch_count)
        ]

        # Execute with high thread count using shared helper
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_count
        ) as executor:
            futures = [
                executor.submit(
                    _dispatch_in_thread_helper,
                    dispatch_engine,
                    "dev.user.events.v1",
                    envelope,
                )
                for envelope in envelopes
            ]
            concurrent.futures.wait(futures)

        # Verify metrics accuracy after high-concurrency stress
        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == dispatch_count, (
            f"Expected {dispatch_count}, got {metrics.total_dispatches}"
        )
        assert metrics.successful_dispatches == dispatch_count
        # dispatcher_execution_count is tracked per dispatcher call in the loop
        assert metrics.dispatcher_execution_count == dispatch_count

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_with_message_type_filtering(self) -> None:
        """Test concurrent dispatch with message type filtering.

        Verifies that message type filtering works correctly when multiple
        threads are dispatching different message types simultaneously.
        """
        import concurrent.futures

        dispatch_engine = MessageDispatchEngine()
        dispatch_count_per_type = 20
        created_results: list[str] = []
        updated_results: list[str] = []
        lock = threading.Lock()

        async def created_handler(envelope: dict[str, object]) -> str:
            payload = envelope["payload"]
            with lock:
                created_results.append(f"created-{payload['user_id']}")
            return "created.output"

        async def updated_handler(envelope: dict[str, object]) -> str:
            payload = envelope["payload"]
            with lock:
                updated_results.append(f"updated-{payload['data']}")
            return "updated.output"

        # Register handlers with specific message type filters
        dispatch_engine.register_dispatcher(
            dispatcher_id="created-handler",
            dispatcher=created_handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserCreatedEvent"},
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="updated-handler",
            dispatcher=updated_handler,
            category=EnumMessageCategory.EVENT,
            message_types={"SomeGenericPayload"},
        )

        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="created-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="created-handler",
            )
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="updated-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="updated-handler",
            )
        )
        dispatch_engine.freeze()

        # Create mixed envelopes (both types)
        created_envelopes = [
            _create_envelope_with_category(
                UserCreatedEvent(user_id=f"created-{i}", name=f"User {i}"),
                EnumMessageCategory.EVENT,
            )
            for i in range(dispatch_count_per_type)
        ]
        updated_envelopes = [
            _create_envelope_with_category(
                SomeGenericPayload(data=f"updated-{i}"),
                EnumMessageCategory.EVENT,
            )
            for i in range(dispatch_count_per_type)
        ]

        # Interleave envelopes for mixed concurrent access
        all_envelopes = []
        for i in range(dispatch_count_per_type):
            all_envelopes.append(created_envelopes[i])
            all_envelopes.append(updated_envelopes[i])

        # Execute concurrent dispatches using shared helper
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(
                    _dispatch_in_thread_helper,
                    dispatch_engine,
                    "dev.user.events.v1",
                    envelope,
                )
                for envelope in all_envelopes
            ]
            concurrent.futures.wait(futures)

        # Verify correct message type routing
        assert len(created_results) == dispatch_count_per_type, (
            f"Expected {dispatch_count_per_type} created, got {len(created_results)}"
        )
        assert len(updated_results) == dispatch_count_per_type, (
            f"Expected {dispatch_count_per_type} updated, got {len(updated_results)}"
        )

        # Verify no cross-routing (created handler should not receive updated events)
        for result in created_results:
            assert result.startswith("created-")
        for result in updated_results:
            assert result.startswith("updated-")

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_stability_extended(self) -> None:
        """Extended stability test for concurrent dispatch.

        Runs a larger number of dispatches over a longer period to validate
        stability under sustained concurrent load.
        """
        import concurrent.futures

        dispatch_engine = MessageDispatchEngine()
        dispatch_count = 200  # Extended count for stability
        batch_size = 50
        results_count = 0
        lock = threading.Lock()

        async def stable_handler(envelope: ModelEventEnvelope[object]) -> str:
            nonlocal results_count
            with lock:
                results_count += 1
            return "stable.output"

        dispatch_engine.register_dispatcher(
            dispatcher_id="stable-handler",
            dispatcher=stable_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="stable-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="stable-handler",
            )
        )
        dispatch_engine.freeze()

        # Process in batches to simulate sustained load using shared helper
        all_results: list[ModelDispatchResult] = []
        for batch_num in range(dispatch_count // batch_size):
            envelopes = [
                _create_envelope_with_category(
                    UserCreatedEvent(
                        user_id=f"user-{batch_num}-{i}", name=f"User {batch_num}-{i}"
                    ),
                    EnumMessageCategory.EVENT,
                )
                for i in range(batch_size)
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(
                        _dispatch_in_thread_helper,
                        dispatch_engine,
                        "dev.user.events.v1",
                        envelope,
                    )
                    for envelope in envelopes
                ]
                for future in concurrent.futures.as_completed(futures):
                    all_results.append(future.result())

        # Verify all dispatches completed successfully
        assert len(all_results) == dispatch_count
        success_count = sum(
            1 for r in all_results if r.status == EnumDispatchStatus.SUCCESS
        )
        assert success_count == dispatch_count, (
            f"Expected {dispatch_count} successes, got {success_count}"
        )

        # Verify handler execution count matches
        assert results_count == dispatch_count

        # Verify metrics after extended run
        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.total_dispatches == dispatch_count
        assert metrics.successful_dispatches == dispatch_count

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_with_sync_handlers(self) -> None:
        """Test concurrent dispatch with synchronous handlers.

        Verifies that sync handlers (run in executor) work correctly
        under concurrent load alongside async handlers.
        """
        import concurrent.futures

        dispatch_engine = MessageDispatchEngine()
        dispatch_count = 30
        sync_results: list[str] = []
        async_results: list[str] = []
        lock = threading.Lock()

        # Sync handler (will be run in executor)
        def sync_handler(envelope: dict[str, object]) -> str:
            payload = envelope["payload"]
            with lock:
                sync_results.append(f"sync-{payload['user_id']}")
            return "sync.output"

        # Async handler
        async def async_handler(envelope: dict[str, object]) -> str:
            payload = envelope["payload"]
            await asyncio.sleep(0.001)  # Small delay
            with lock:
                async_results.append(f"async-{payload['user_id']}")
            return "async.output"

        dispatch_engine.register_dispatcher(
            dispatcher_id="sync-handler",
            dispatcher=sync_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_dispatcher(
            dispatcher_id="async-handler",
            dispatcher=async_handler,
            category=EnumMessageCategory.EVENT,
        )

        # Both handlers match the same topic (fan-out)
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="sync-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="sync-handler",
            )
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="async-route",
                topic_pattern="dev.**",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="async-handler",
            )
        )
        dispatch_engine.freeze()

        envelopes = [
            _create_envelope_with_category(
                UserCreatedEvent(user_id=f"user-{i}", name=f"User {i}"),
                EnumMessageCategory.EVENT,
            )
            for i in range(dispatch_count)
        ]

        # Execute concurrent dispatches using shared helper
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    _dispatch_in_thread_helper,
                    dispatch_engine,
                    "dev.user.events.v1",
                    envelope,
                )
                for envelope in envelopes
            ]
            concurrent.futures.wait(futures)

        # Both handlers should have been called for each dispatch
        assert len(sync_results) == dispatch_count
        assert len(async_results) == dispatch_count

        # Verify metrics
        metrics = dispatch_engine.get_structured_metrics()
        assert metrics.dispatcher_execution_count == dispatch_count * 2

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_correlation_id_preservation(self) -> None:
        """Test that correlation IDs are preserved correctly under concurrent load.

        Verifies that each dispatch result contains the correct correlation ID
        from its originating envelope, even under concurrent dispatch.
        """
        import concurrent.futures
        from uuid import UUID

        dispatch_engine = MessageDispatchEngine()
        dispatch_count = 40
        received_correlation_ids: list[tuple[str, UUID]] = []
        lock = threading.Lock()

        async def tracking_handler(envelope: dict[str, object]) -> str:
            payload = envelope["payload"]
            # NOTE: __debug_trace is serialized snapshot, used here for test verification
            debug_trace = envelope["__debug_trace"]
            with lock:
                # correlation_id is now a serialized string in __debug_trace
                received_correlation_ids.append(
                    (payload["user_id"], UUID(debug_trace["correlation_id"]))  # type: ignore[index, arg-type]
                )
            return "test.output.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="tracking-handler",
            dispatcher=tracking_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="tracking-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="tracking-handler",
            )
        )
        dispatch_engine.freeze()

        # Create envelopes with unique correlation IDs
        envelopes_with_ids: list[tuple[ModelEventEnvelope[object], UUID]] = []
        for i in range(dispatch_count):
            correlation_id = uuid4()
            envelope = ModelEventEnvelope(
                payload=UserCreatedEvent(user_id=f"user-{i}", name=f"User {i}"),
                correlation_id=correlation_id,
            )
            object.__setattr__(
                envelope, "infer_category", lambda: EnumMessageCategory.EVENT
            )
            envelopes_with_ids.append((envelope, correlation_id))

        # Execute concurrent dispatches and collect results with original correlation IDs
        results_with_expected: list[tuple[ModelDispatchResult, UUID]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    _dispatch_in_thread_helper,
                    dispatch_engine,
                    "dev.user.events.v1",
                    envelope,
                ): expected_id
                for envelope, expected_id in envelopes_with_ids
            }
            for future in concurrent.futures.as_completed(futures):
                expected_id = futures[future]
                result = future.result()
                results_with_expected.append((result, expected_id))

        # Verify each result has the correct correlation ID
        for result, expected_id in results_with_expected:
            assert result.correlation_id == expected_id, (
                f"Correlation ID mismatch: expected {expected_id}, "
                f"got {result.correlation_id}"
            )

        # Verify all dispatches completed
        assert len(results_with_expected) == dispatch_count

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_no_data_corruption(self) -> None:
        """Test that concurrent dispatch does not corrupt handler data.

        Verifies that handlers receive the correct payload data even when
        multiple dispatches are happening simultaneously.
        """
        import concurrent.futures

        dispatch_engine = MessageDispatchEngine()
        dispatch_count = 50
        received_payloads: list[tuple[str, str]] = []
        lock = threading.Lock()

        async def verifying_handler(envelope: dict[str, object]) -> str:
            payload = envelope["payload"]
            user_id = payload["user_id"]
            name = payload["name"]
            with lock:
                received_payloads.append((user_id, name))
            return "test.output.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="verifying-handler",
            dispatcher=verifying_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="verifying-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="verifying-handler",
            )
        )
        dispatch_engine.freeze()

        # Create envelopes with unique, verifiable data
        expected_payloads = {
            f"user-{i}": f"Name-{i}-{i * 2}" for i in range(dispatch_count)
        }
        envelopes = [
            _create_envelope_with_category(
                UserCreatedEvent(user_id=f"user-{i}", name=f"Name-{i}-{i * 2}"),
                EnumMessageCategory.EVENT,
            )
            for i in range(dispatch_count)
        ]

        # Execute concurrent dispatches using shared helper
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(
                    _dispatch_in_thread_helper,
                    dispatch_engine,
                    "dev.user.events.v1",
                    envelope,
                )
                for envelope in envelopes
            ]
            concurrent.futures.wait(futures)

        # Verify all payloads were received correctly
        assert len(received_payloads) == dispatch_count

        # Verify no data corruption
        for user_id, name in received_payloads:
            assert user_id in expected_payloads, f"Unexpected user_id: {user_id}"
            assert name == expected_payloads[user_id], (
                f"Data corruption for {user_id}: expected {expected_payloads[user_id]}, "
                f"got {name}"
            )


# ============================================================================
# Command and Intent Dispatch Tests
# ============================================================================


@pytest.mark.unit
class TestCommandAndIntentDispatch:
    """Tests for dispatching commands and intents."""

    @pytest.mark.asyncio
    async def test_dispatch_command(
        self,
        dispatch_engine: MessageDispatchEngine,
        command_envelope: ModelEventEnvelope[CreateUserCommand],
    ) -> None:
        """Test successful command dispatch."""
        results: list[str] = []

        async def command_handler(envelope: ModelEventEnvelope[object]) -> str:
            results.append("command_handled")
            return "result.events.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="command-handler",
            dispatcher=command_handler,
            category=EnumMessageCategory.COMMAND,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="command-route",
                topic_pattern="*.user.commands.*",
                message_category=EnumMessageCategory.COMMAND,
                dispatcher_id="command-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch(
            "dev.user.commands.v1", command_envelope
        )

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1
        assert result.message_category == EnumMessageCategory.COMMAND

    @pytest.mark.asyncio
    async def test_dispatch_intent(
        self,
        dispatch_engine: MessageDispatchEngine,
        intent_envelope: ModelEventEnvelope[ProvisionUserIntent],
    ) -> None:
        """Test successful intent dispatch."""
        results: list[str] = []

        async def intent_handler(envelope: ModelEventEnvelope[object]) -> str:
            results.append("intent_handled")
            return "user.commands.v1"

        dispatch_engine.register_dispatcher(
            dispatcher_id="intent-handler",
            dispatcher=intent_handler,
            category=EnumMessageCategory.INTENT,
        )
        dispatch_engine.register_route(
            ModelDispatchRoute(
                route_id="intent-route",
                topic_pattern="*.user.intents.*",
                message_category=EnumMessageCategory.INTENT,
                dispatcher_id="intent-handler",
            )
        )
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.intents.v1", intent_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1
        assert result.message_category == EnumMessageCategory.INTENT


# ============================================================================
# Error Sanitization Tests
# ============================================================================


class TestErrorSanitization:
    """
    Tests for error message sanitization in dispatch results.

    Verifies that sensitive information (connection strings, passwords, API keys)
    is properly sanitized before being included in ModelDispatchResult.error_message
    and per-dispatcher metrics.

    Security requirements:
        - Connection strings with credentials must be redacted
        - Passwords and API keys must not leak in error messages
        - Protocol URLs (postgres://, mongodb://, etc.) must be sanitized
        - Long error messages should be truncated
    """

    @pytest.fixture
    def sanitization_engine(self) -> MessageDispatchEngine:
        """Create a fresh engine for sanitization tests."""
        return MessageDispatchEngine()

    @pytest.fixture
    def event_envelope(self) -> ModelEventEnvelope[UserCreatedEvent]:
        """Create a test event envelope."""
        return ModelEventEnvelope(
            correlation_id=uuid4(),
            payload=UserCreatedEvent(user_id="test-123", name="Test User"),
        )

    async def test_connection_string_password_is_sanitized(
        self,
        sanitization_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that connection strings with passwords are sanitized."""

        async def failing_handler(
            envelope: ModelEventEnvelope[object],
        ) -> None:
            # Simulate a database driver error that includes connection string
            raise ConnectionError(
                "Failed to connect to postgresql://user:supersecretpass@db.example.com:5432/mydb"
            )

        sanitization_engine.register_dispatcher(
            dispatcher_id="db-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )
        sanitization_engine.register_route(
            ModelDispatchRoute(
                route_id="db-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="db-handler",
            )
        )
        sanitization_engine.freeze()

        result = await sanitization_engine.dispatch(
            "dev.user.events.v1", event_envelope
        )

        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None
        # The connection string should be redacted
        assert "supersecretpass" not in result.error_message
        assert "postgresql://" not in result.error_message
        assert "[REDACTED" in result.error_message

    async def test_password_in_error_is_sanitized(
        self,
        sanitization_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that errors mentioning passwords are sanitized."""

        async def failing_handler(
            envelope: ModelEventEnvelope[object],
        ) -> None:
            raise ValueError("Authentication failed with password=secret123")

        sanitization_engine.register_dispatcher(
            dispatcher_id="auth-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )
        sanitization_engine.register_route(
            ModelDispatchRoute(
                route_id="auth-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="auth-handler",
            )
        )
        sanitization_engine.freeze()

        result = await sanitization_engine.dispatch(
            "dev.user.events.v1", event_envelope
        )

        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None
        # The password value should be redacted
        assert "secret123" not in result.error_message
        assert "password" not in result.error_message.lower()
        assert "[REDACTED" in result.error_message

    async def test_api_key_in_error_is_sanitized(
        self,
        sanitization_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that API keys in errors are sanitized."""

        async def failing_handler(
            envelope: ModelEventEnvelope[object],
        ) -> None:
            raise RuntimeError("API call failed with api_key=sk-1234567890abcdef")

        sanitization_engine.register_dispatcher(
            dispatcher_id="api-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )
        sanitization_engine.register_route(
            ModelDispatchRoute(
                route_id="api-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="api-handler",
            )
        )
        sanitization_engine.freeze()

        result = await sanitization_engine.dispatch(
            "dev.user.events.v1", event_envelope
        )

        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None
        # The API key should be redacted
        assert "sk-1234567890abcdef" not in result.error_message
        assert "api_key" not in result.error_message.lower()
        assert "[REDACTED" in result.error_message

    async def test_safe_error_is_not_redacted(
        self,
        sanitization_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that safe error messages are passed through (not redacted)."""

        async def failing_handler(
            envelope: ModelEventEnvelope[object],
        ) -> None:
            raise ValueError("User with ID 12345 not found in database")

        sanitization_engine.register_dispatcher(
            dispatcher_id="user-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )
        sanitization_engine.register_route(
            ModelDispatchRoute(
                route_id="user-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="user-handler",
            )
        )
        sanitization_engine.freeze()

        result = await sanitization_engine.dispatch(
            "dev.user.events.v1", event_envelope
        )

        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None
        # Safe error should not be redacted
        assert "[REDACTED" not in result.error_message
        assert "User with ID 12345 not found" in result.error_message

    async def test_long_error_message_is_truncated(
        self,
        sanitization_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that very long error messages are truncated."""

        async def failing_handler(
            envelope: ModelEventEnvelope[object],
        ) -> None:
            # Create a very long error message (over 500 chars)
            long_message = "Error: " + "x" * 600
            raise ValueError(long_message)

        sanitization_engine.register_dispatcher(
            dispatcher_id="long-error-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )
        sanitization_engine.register_route(
            ModelDispatchRoute(
                route_id="long-error-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="long-error-handler",
            )
        )
        sanitization_engine.freeze()

        result = await sanitization_engine.dispatch(
            "dev.user.events.v1", event_envelope
        )

        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None
        # Error should be truncated
        assert "[truncated]" in result.error_message
        # Original message was 607 chars, should be limited
        # The format is: "Dispatcher 'X' failed: ValueError: <truncated>"
        # So total could be longer, but the ValueError content is truncated
        assert len(result.error_message) < 700

    async def test_mongodb_connection_string_is_sanitized(
        self,
        sanitization_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that MongoDB connection strings are sanitized."""

        async def failing_handler(
            envelope: ModelEventEnvelope[object],
        ) -> None:
            raise ConnectionError(
                "Failed: mongodb://admin:password123@mongo.example.com:27017/db"
            )

        sanitization_engine.register_dispatcher(
            dispatcher_id="mongo-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )
        sanitization_engine.register_route(
            ModelDispatchRoute(
                route_id="mongo-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="mongo-handler",
            )
        )
        sanitization_engine.freeze()

        result = await sanitization_engine.dispatch(
            "dev.user.events.v1", event_envelope
        )

        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None
        assert "password123" not in result.error_message
        assert "mongodb://" not in result.error_message
        assert "[REDACTED" in result.error_message

    async def test_dispatcher_metrics_use_sanitized_error(
        self,
        sanitization_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that per-dispatcher metrics also use sanitized error messages."""

        async def failing_handler(
            envelope: ModelEventEnvelope[object],
        ) -> None:
            raise ConnectionError("redis://user:secret_token@redis.example.com:6379")

        sanitization_engine.register_dispatcher(
            dispatcher_id="redis-handler",
            dispatcher=failing_handler,
            category=EnumMessageCategory.EVENT,
        )
        sanitization_engine.register_route(
            ModelDispatchRoute(
                route_id="redis-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="redis-handler",
            )
        )
        sanitization_engine.freeze()

        await sanitization_engine.dispatch("dev.user.events.v1", event_envelope)

        # Check per-dispatcher metrics
        dispatcher_metrics = sanitization_engine.get_dispatcher_metrics("redis-handler")
        assert dispatcher_metrics is not None
        assert dispatcher_metrics.last_error_message is not None
        # The error message in metrics should also be sanitized
        assert "secret_token" not in dispatcher_metrics.last_error_message
        assert "redis://" not in dispatcher_metrics.last_error_message
        assert "[REDACTED" in dispatcher_metrics.last_error_message


# ============================================================================
# Context-Aware Dispatch Tests
# ============================================================================


@pytest.mark.unit
class TestContextAwareDispatch:
    """
    Tests for context-aware dispatch functionality.

    These tests verify the context creation and injection behavior for
    dispatchers registered with node_kind. Tests cover:
    - Error handling for None/invalid node_kind
    - Signature inspection edge cases
    - Correct context creation for each node type
    - Backwards compatibility for single-parameter dispatchers
    """

    @pytest.fixture
    def context_engine(self) -> MessageDispatchEngine:
        """Create a fresh engine for context-aware dispatch tests."""
        return MessageDispatchEngine()

    @pytest.fixture
    def event_envelope(self) -> ModelEventEnvelope[UserCreatedEvent]:
        """Create a test event envelope with correlation and trace IDs."""
        return ModelEventEnvelope(
            correlation_id=uuid4(),
            trace_id=uuid4(),
            payload=UserCreatedEvent(user_id="ctx-test-123", name="Context Test User"),
        )

    def test_create_context_for_entry_with_none_node_kind_raises_internal_error(
        self,
        context_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that _create_context_for_entry with None node_kind raises INTERNAL_ERROR.

        This tests the defensive check in _create_context_for_entry that validates
        node_kind is not None before proceeding with context creation. This branch
        should only be reached if there's a bug in the dispatch engine's internal
        logic (node_kind should be validated at registration time).
        """
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            DispatchEntryInternal,
        )

        # Create a DispatchEntryInternal with node_kind=None
        # This simulates an internal state that shouldn't occur in normal operation
        entry = DispatchEntryInternal(
            dispatcher_id="test-dispatcher",
            dispatcher=lambda e: None,
            category=EnumMessageCategory.EVENT,
            message_types=None,
            node_kind=None,  # Explicitly None - the error case we're testing
        )

        # Attempt to create context should raise ModelOnexError with INTERNAL_ERROR
        with pytest.raises(ModelOnexError) as exc_info:
            context_engine._create_context_for_entry(entry, event_envelope)

        assert exc_info.value.error_code == EnumCoreErrorCode.INTERNAL_ERROR
        assert "node_kind is None" in exc_info.value.message
        assert "test-dispatcher" in exc_info.value.message

    def test_dispatcher_accepts_context_returns_false_for_non_inspectable_callable(
        self,
        context_engine: MessageDispatchEngine,
    ) -> None:
        """Test _dispatcher_accepts_context() returns False when signature inspection fails.

        Test Scenario:
            A class with __call__ assigned to a built-in function (len).
            When inspect.signature() is called on this, it fails because
            len is a C extension with no introspectable signature.

        Why This Approach:
            Assigning a built-in to __call__ is a realistic way to create
            an uninspectable callable. This happens in practice when:
            - C extensions expose callable objects
            - Certain wrapper patterns delegate to built-ins
            - Performance-critical code uses built-in operations

        How inspect.signature() Fails:
            >>> import inspect
            >>> class C:
            ...     __call__ = len
            >>> inspect.signature(C())
            ValueError: no signature found for builtin <built-in function len>

        Expected Behavior:
            _dispatcher_accepts_context() catches the ValueError and returns False,
            allowing the dispatcher to be registered without context support.
        """

        # Create a mock callable that raises ValueError when inspected
        class NonInspectableCallable:
            """A callable that breaks signature inspection.

            By assigning len (a C built-in) to __call__, we create a callable
            instance where inspect.signature() raises ValueError. This mimics
            C extensions and certain wrapper patterns.

            Note: This is a real technique used in some Python libraries that
            wrap C functions, making it a valid test case.
            """

            # Use a builtin as __call__ - inspect.signature() fails on builtins
            __call__ = len  # type: ignore[assignment]

        non_inspectable = NonInspectableCallable()

        # The method should return False rather than raising an exception
        result = context_engine._dispatcher_accepts_context(non_inspectable)

        assert result is False

    def test_dispatcher_accepts_context_returns_false_when_signature_raises_value_error(
        self,
        context_engine: MessageDispatchEngine,
    ) -> None:
        """Test _dispatcher_accepts_context() handles built-in functions directly.

        Test Scenario:
            Passing a built-in function (len) directly to _dispatcher_accepts_context().
            This is the simplest way to trigger signature inspection failure.

        What Happens with Built-ins:
            >>> import inspect
            >>> inspect.signature(len)
            ValueError: no signature found for builtin <built-in function len>

            Built-in functions implemented in C don't have Python bytecode or
            a __code__ object, so inspect.signature() cannot determine their
            parameters.

        Expected Behavior:
            The method catches ValueError and returns False, logging a warning
            that explains the fallback behavior.

        Note:
            While passing len() as a dispatcher is unrealistic (wrong signature),
            this test verifies the exception handling path is robust.
        """
        # Use a builtin function directly - these often raise ValueError
        # when inspect.signature is called on them
        result = context_engine._dispatcher_accepts_context(len)  # type: ignore[arg-type]

        # Should return False, not raise an exception
        assert result is False

    def test_dispatcher_accepts_context_logs_warning_for_unconventional_parameter_name(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning logged for second param without context naming.

        When a dispatcher has 2+ parameters but the second parameter name doesn't
        contain 'context' or 'ctx', a warning should be logged to help developers
        identify potential signature mismatches.
        """
        import logging

        # Create engine with logger that captures warnings
        logger = logging.getLogger("test.dispatch.warning")
        engine = MessageDispatchEngine(logger=logger)

        # Dispatcher with unconventional second parameter name
        def dispatcher_with_unusual_param(
            envelope: ModelEventEnvelope[object],
            some_other_param: str,  # Doesn't contain 'context' or 'ctx'
        ) -> str:
            return "output"

        with caplog.at_level(logging.WARNING, logger="test.dispatch.warning"):
            result = engine._dispatcher_accepts_context(dispatcher_with_unusual_param)

        # Method should still return True (backwards compatible)
        assert result is True

        # Warning should have been logged
        assert len(caplog.records) == 1
        warning_record = caplog.records[0]
        assert warning_record.levelno == logging.WARNING
        assert "dispatcher_with_unusual_param" in warning_record.message
        assert "some_other_param" in warning_record.message
        assert "context naming convention" in warning_record.message

    def test_dispatcher_accepts_context_no_warning_for_context_parameter(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that no warning is logged when second parameter contains 'context'."""
        import logging

        logger = logging.getLogger("test.dispatch.no_warning")
        engine = MessageDispatchEngine(logger=logger)

        # Dispatcher with proper context parameter name
        def dispatcher_with_context(
            envelope: ModelEventEnvelope[object],
            dispatch_context: object,  # Contains 'context'
        ) -> str:
            return "output"

        with caplog.at_level(logging.WARNING, logger="test.dispatch.no_warning"):
            result = engine._dispatcher_accepts_context(dispatcher_with_context)

        # Method should return True
        assert result is True

        # No warning should be logged
        assert len(caplog.records) == 0

    def test_dispatcher_accepts_context_no_warning_for_ctx_parameter(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that no warning is logged when second parameter contains 'ctx'."""
        import logging

        logger = logging.getLogger("test.dispatch.ctx")
        engine = MessageDispatchEngine(logger=logger)

        # Dispatcher with 'ctx' abbreviation in parameter name
        def dispatcher_with_ctx(
            envelope: ModelEventEnvelope[object],
            dispatch_ctx: object,  # Contains 'ctx'
        ) -> str:
            return "output"

        with caplog.at_level(logging.WARNING, logger="test.dispatch.ctx"):
            result = engine._dispatcher_accepts_context(dispatcher_with_ctx)

        # Method should return True
        assert result is True

        # No warning should be logged
        assert len(caplog.records) == 0

    def test_dispatcher_accepts_context_case_insensitive_parameter_check(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that parameter name check is case-insensitive."""
        import logging

        logger = logging.getLogger("test.dispatch.case")
        engine = MessageDispatchEngine(logger=logger)

        # Dispatcher with CONTEXT in uppercase
        def dispatcher_with_uppercase_context(
            envelope: ModelEventEnvelope[object],
            DISPATCH_CONTEXT: object,
        ) -> str:
            return "output"

        with caplog.at_level(logging.WARNING, logger="test.dispatch.case"):
            result = engine._dispatcher_accepts_context(
                dispatcher_with_uppercase_context
            )

        # Method should return True
        assert result is True

        # No warning should be logged (case-insensitive check)
        assert len(caplog.records) == 0

    @pytest.mark.asyncio
    async def test_context_aware_dispatcher_with_reducer_gets_no_time_injection(
        self,
        context_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that REDUCER dispatchers receive context with now=None.

        REDUCER nodes are deterministic state aggregators and must never
        receive time injection per ONEX architecture rules.
        """
        from omnibase_core.enums.enum_node_kind import EnumNodeKind
        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        captured_context: list[ModelDispatchContext] = []

        async def reducer_dispatcher(
            envelope: ModelEventEnvelope[object],
            context: ModelDispatchContext,
        ) -> str:
            captured_context.append(context)
            return "reducer.output"

        # Register dispatcher with REDUCER node_kind
        context_engine.register_dispatcher(
            dispatcher_id="reducer-handler",
            dispatcher=reducer_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.REDUCER,
        )
        context_engine.register_route(
            ModelDispatchRoute(
                route_id="reducer-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="reducer-handler",
            )
        )
        context_engine.freeze()

        # Dispatch the message
        result = await context_engine.dispatch("dev.user.events.v1", event_envelope)

        # Verify dispatch succeeded
        assert result.status == EnumDispatchStatus.SUCCESS

        # Verify context was captured
        assert len(captured_context) == 1
        ctx = captured_context[0]

        # REDUCER should NOT have time injection
        assert ctx.now is None
        assert ctx.node_kind == EnumNodeKind.REDUCER
        assert ctx.has_time_injection is False

        # Correlation metadata should be propagated
        assert ctx.correlation_id == event_envelope.correlation_id
        assert ctx.trace_id == event_envelope.trace_id

    @pytest.mark.asyncio
    async def test_context_aware_dispatcher_with_compute_gets_no_time_injection(
        self,
        context_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that COMPUTE dispatchers receive context with now=None.

        COMPUTE nodes are pure transformation nodes and must never
        receive time injection per ONEX architecture rules.
        """
        from omnibase_core.enums.enum_node_kind import EnumNodeKind
        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        captured_context: list[ModelDispatchContext] = []

        async def compute_dispatcher(
            envelope: ModelEventEnvelope[object],
            context: ModelDispatchContext,
        ) -> str:
            captured_context.append(context)
            return "compute.output"

        context_engine.register_dispatcher(
            dispatcher_id="compute-handler",
            dispatcher=compute_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.COMPUTE,
        )
        context_engine.register_route(
            ModelDispatchRoute(
                route_id="compute-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="compute-handler",
            )
        )
        context_engine.freeze()

        result = await context_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(captured_context) == 1
        ctx = captured_context[0]

        # COMPUTE should NOT have time injection
        assert ctx.now is None
        assert ctx.node_kind == EnumNodeKind.COMPUTE
        assert ctx.has_time_injection is False

    @pytest.mark.asyncio
    async def test_context_aware_dispatcher_with_orchestrator_gets_time_injection(
        self,
        context_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that ORCHESTRATOR dispatchers receive context with now set.

        ORCHESTRATOR nodes coordinate workflows and can make time-dependent
        decisions. They MUST receive time injection.
        """
        from datetime import datetime

        from omnibase_core.enums.enum_node_kind import EnumNodeKind
        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        captured_context: list[ModelDispatchContext] = []
        dispatch_time = datetime.now(UTC)

        async def orchestrator_dispatcher(
            envelope: ModelEventEnvelope[object],
            context: ModelDispatchContext,
        ) -> str:
            captured_context.append(context)
            return "orchestrator.output"

        context_engine.register_dispatcher(
            dispatcher_id="orchestrator-handler",
            dispatcher=orchestrator_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.ORCHESTRATOR,
        )
        context_engine.register_route(
            ModelDispatchRoute(
                route_id="orchestrator-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="orchestrator-handler",
            )
        )
        context_engine.freeze()

        result = await context_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(captured_context) == 1
        ctx = captured_context[0]

        # ORCHESTRATOR MUST have time injection
        assert ctx.now is not None
        assert ctx.node_kind == EnumNodeKind.ORCHESTRATOR
        assert ctx.has_time_injection is True

        # Time should be close to dispatch time (within 1 second)
        assert abs((ctx.now - dispatch_time).total_seconds()) < 1.0

    @pytest.mark.asyncio
    async def test_context_aware_dispatcher_with_effect_gets_time_injection(
        self,
        context_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that EFFECT dispatchers receive context with now set.

        EFFECT nodes handle external I/O and can make time-dependent
        decisions (e.g., TTL calculations). They MUST receive time injection.
        """
        from datetime import datetime

        from omnibase_core.enums.enum_node_kind import EnumNodeKind
        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        captured_context: list[ModelDispatchContext] = []
        dispatch_time = datetime.now(UTC)

        async def effect_dispatcher(
            envelope: ModelEventEnvelope[object],
            context: ModelDispatchContext,
        ) -> str:
            captured_context.append(context)
            return "effect.output"

        context_engine.register_dispatcher(
            dispatcher_id="effect-handler",
            dispatcher=effect_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.EFFECT,
        )
        context_engine.register_route(
            ModelDispatchRoute(
                route_id="effect-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="effect-handler",
            )
        )
        context_engine.freeze()

        result = await context_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(captured_context) == 1
        ctx = captured_context[0]

        # EFFECT MUST have time injection
        assert ctx.now is not None
        assert ctx.node_kind == EnumNodeKind.EFFECT
        assert ctx.has_time_injection is True

        # Time should be close to dispatch time
        assert abs((ctx.now - dispatch_time).total_seconds()) < 1.0

    @pytest.mark.asyncio
    async def test_context_aware_dispatcher_with_runtime_host_gets_time_injection(
        self,
        context_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that RUNTIME_HOST dispatchers receive context with now set.

        RUNTIME_HOST nodes are infrastructure components that need time
        for operational decisions (health checks, scheduling). They MUST
        receive time injection.
        """
        from datetime import datetime

        from omnibase_core.enums.enum_node_kind import EnumNodeKind
        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        captured_context: list[ModelDispatchContext] = []
        dispatch_time = datetime.now(UTC)

        async def runtime_host_dispatcher(
            envelope: ModelEventEnvelope[object],
            context: ModelDispatchContext,
        ) -> str:
            captured_context.append(context)
            return "runtime_host.output"

        context_engine.register_dispatcher(
            dispatcher_id="runtime-host-handler",
            dispatcher=runtime_host_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.RUNTIME_HOST,
        )
        context_engine.register_route(
            ModelDispatchRoute(
                route_id="runtime-host-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="runtime-host-handler",
            )
        )
        context_engine.freeze()

        result = await context_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(captured_context) == 1
        ctx = captured_context[0]

        # RUNTIME_HOST MUST have time injection
        assert ctx.now is not None
        assert ctx.node_kind == EnumNodeKind.RUNTIME_HOST
        assert ctx.has_time_injection is True

        # Time should be close to dispatch time
        assert abs((ctx.now - dispatch_time).total_seconds()) < 1.0

    @pytest.mark.asyncio
    async def test_single_param_dispatcher_with_node_kind_works(
        self,
        context_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test backwards compatibility: single-param dispatcher with node_kind set.

        When a dispatcher is registered with node_kind but only accepts one
        parameter (envelope), the dispatch engine should detect this via
        signature inspection and skip context injection.
        """
        from omnibase_core.enums.enum_node_kind import EnumNodeKind

        call_count = [0]

        async def legacy_dispatcher(
            envelope: ModelEventEnvelope[object],
        ) -> str:
            """A dispatcher that doesn't accept context (backwards compatible)."""
            call_count[0] += 1
            return "legacy.output"

        # Register with node_kind even though dispatcher doesn't accept context
        context_engine.register_dispatcher(
            dispatcher_id="legacy-handler",
            dispatcher=legacy_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.EFFECT,  # node_kind set but dispatcher is single-param
        )
        context_engine.register_route(
            ModelDispatchRoute(
                route_id="legacy-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="legacy-handler",
            )
        )
        context_engine.freeze()

        # Dispatch should still work
        result = await context_engine.dispatch("dev.user.events.v1", event_envelope)

        # Verify dispatch succeeded
        assert result.status == EnumDispatchStatus.SUCCESS
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_sync_context_aware_dispatcher_works(
        self,
        context_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that synchronous context-aware dispatchers work correctly.

        The dispatch engine should handle sync dispatchers with context
        via run_in_executor.
        """
        from omnibase_core.enums.enum_node_kind import EnumNodeKind
        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        captured_context: list[ModelDispatchContext] = []

        def sync_effect_dispatcher(
            envelope: ModelEventEnvelope[object],
            context: ModelDispatchContext,
        ) -> str:
            """A synchronous dispatcher that accepts context."""
            captured_context.append(context)
            return "sync.effect.output"

        context_engine.register_dispatcher(
            dispatcher_id="sync-effect-handler",
            dispatcher=sync_effect_dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.EFFECT,
        )
        context_engine.register_route(
            ModelDispatchRoute(
                route_id="sync-effect-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="sync-effect-handler",
            )
        )
        context_engine.freeze()

        result = await context_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(captured_context) == 1
        assert captured_context[0].now is not None  # EFFECT gets time

    @pytest.mark.asyncio
    async def test_context_propagates_correlation_id_from_envelope(
        self,
        context_engine: MessageDispatchEngine,
    ) -> None:
        """Test that correlation_id is properly propagated from envelope to context."""
        from omnibase_core.enums.enum_node_kind import EnumNodeKind
        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        captured_context: list[ModelDispatchContext] = []
        specific_correlation_id = uuid4()
        specific_trace_id = uuid4()

        async def dispatcher(
            envelope: ModelEventEnvelope[object],
            context: ModelDispatchContext,
        ) -> str:
            captured_context.append(context)
            return "output"

        context_engine.register_dispatcher(
            dispatcher_id="corr-handler",
            dispatcher=dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.EFFECT,
        )
        context_engine.register_route(
            ModelDispatchRoute(
                route_id="corr-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="corr-handler",
            )
        )
        context_engine.freeze()

        # Create envelope with specific IDs
        envelope = ModelEventEnvelope(
            correlation_id=specific_correlation_id,
            trace_id=specific_trace_id,
            payload=UserCreatedEvent(user_id="corr-test", name="Correlation Test"),
        )

        await context_engine.dispatch("dev.user.events.v1", envelope)

        assert len(captured_context) == 1
        ctx = captured_context[0]

        # Verify correlation metadata is propagated
        assert ctx.correlation_id == specific_correlation_id
        assert ctx.trace_id == specific_trace_id

    @pytest.mark.asyncio
    async def test_context_generates_correlation_id_when_envelope_has_none(
        self,
        context_engine: MessageDispatchEngine,
    ) -> None:
        """Test that correlation_id is auto-generated when envelope has None."""
        from omnibase_core.enums.enum_node_kind import EnumNodeKind
        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        captured_context: list[ModelDispatchContext] = []

        async def dispatcher(
            envelope: ModelEventEnvelope[object],
            context: ModelDispatchContext,
        ) -> str:
            captured_context.append(context)
            return "output"

        context_engine.register_dispatcher(
            dispatcher_id="auto-corr-handler",
            dispatcher=dispatcher,
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.REDUCER,
        )
        context_engine.register_route(
            ModelDispatchRoute(
                route_id="auto-corr-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="auto-corr-handler",
            )
        )
        context_engine.freeze()

        # Create envelope without correlation_id
        envelope = ModelEventEnvelope(
            payload=UserCreatedEvent(user_id="auto-corr", name="Auto Correlation"),
            # correlation_id defaults to None
        )

        await context_engine.dispatch("dev.user.events.v1", envelope)

        assert len(captured_context) == 1
        ctx = captured_context[0]

        # Verify correlation_id was auto-generated (not None)
        assert ctx.correlation_id is not None


# ============================================================================
# Dispatcher Signature Inspection Edge Case Tests
# ============================================================================


@pytest.mark.unit
class TestDispatcherSignatureInspection:
    """
    Tests for _dispatcher_accepts_context edge cases.

    These tests specifically cover:
    - Dispatchers with 3+ parameters (verifies >= 2 logic)
    - Inspection failures when inspect.signature() raises exceptions
    - Warning logging for unconventional parameter naming

    These edge cases were identified during PR review to ensure the
    signature inspection logic is robust and handles all cases correctly.
    """

    @pytest.fixture
    def engine(self) -> MessageDispatchEngine:
        """Create a fresh engine for signature inspection tests."""
        return MessageDispatchEngine()

    def test_dispatcher_with_three_parameters_accepts_context(
        self,
        engine: MessageDispatchEngine,
    ) -> None:
        """Test that dispatcher with 3 parameters returns True for accepts_context.

        The implementation uses `len(params) >= 2` intentionally to support
        dispatchers with additional optional parameters beyond (envelope, context).
        This test verifies that behavior.
        """
        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        def dispatcher_with_three_params(
            envelope: ModelEventEnvelope[object],
            context: ModelDispatchContext,
            extra_arg: str | None = None,
        ) -> str:
            """Dispatcher with an optional third parameter."""
            return "output"

        result = engine._dispatcher_accepts_context(dispatcher_with_three_params)

        # Should return True because >= 2 parameters
        assert result is True

    def test_dispatcher_with_four_parameters_accepts_context(
        self,
        engine: MessageDispatchEngine,
    ) -> None:
        """Test that dispatcher with 4 parameters returns True for accepts_context.

        Further validates the >= 2 logic with even more parameters, which might
        be used for testing hooks, logging, or future extensibility.
        """
        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        def dispatcher_with_four_params(
            envelope: ModelEventEnvelope[object],
            context: ModelDispatchContext,
            debug_flag: bool = False,
            trace_callback: object | None = None,
        ) -> str:
            """Dispatcher with multiple optional parameters for testing/debugging."""
            return "output"

        result = engine._dispatcher_accepts_context(dispatcher_with_four_params)

        # Should return True because >= 2 parameters
        assert result is True

    def test_inspection_failure_returns_false(
        self,
        engine: MessageDispatchEngine,
    ) -> None:
        """Test _dispatcher_accepts_context() returns False when mocked signature fails.

        Test Scenario:
            Using unittest.mock to make inspect.signature() raise ValueError,
            simulating what happens with C extensions and built-in functions.

        Why Mock Instead of Real Uninspectable Callable:
            This test uses mocking to precisely control when and how
            inspect.signature() fails. It complements the other tests that
            use real uninspectable callables (like built-in functions).

        What This Tests:
            The exception handling path in _dispatcher_accepts_context():

            try:
                sig = inspect.signature(dispatcher)
                # ... parameter inspection
            except (ValueError, TypeError) as e:
                self._logger.warning(...)
                return False  # <-- This line

        Expected Behavior:
            ValueError is caught, warning is logged, False is returned.
            The exception is NOT propagated to the caller.
        """
        from unittest.mock import patch

        def valid_dispatcher(
            envelope: ModelEventEnvelope[object],
            context: object,
        ) -> str:
            return "output"

        # Mock inspect.signature to raise ValueError (simulating C extension behavior)
        with patch("inspect.signature") as mock_signature:
            mock_signature.side_effect = ValueError("no signature found")

            result = engine._dispatcher_accepts_context(valid_dispatcher)

        # Should return False when inspection fails
        assert result is False

    def test_inspection_failure_type_error_returns_false(
        self,
        engine: MessageDispatchEngine,
    ) -> None:
        """Test _dispatcher_accepts_context() handles TypeError from inspect.signature().

        Test Scenario:
            Using unittest.mock to make inspect.signature() raise TypeError.

        Why Test TypeError Separately:
            While ValueError is more common, TypeError can occur when:
            - The object isn't recognized as callable
            - The __signature__ attribute contains an invalid specification
            - Certain C extension edge cases

            The method should handle BOTH exception types identically:
            catch the exception, log warning, return False.

        Code Path Tested:
            except (ValueError, TypeError) as e:  # <-- Testing TypeError path
                self._logger.warning(...)
                return False
        """
        from unittest.mock import patch

        def valid_dispatcher(
            envelope: ModelEventEnvelope[object],
            context: object,
        ) -> str:
            return "output"

        # Mock inspect.signature to raise TypeError
        with patch("inspect.signature") as mock_signature:
            mock_signature.side_effect = TypeError(
                "callable is not a valid Python callable"
            )

            result = engine._dispatcher_accepts_context(valid_dispatcher)

        # Should return False when inspection fails
        assert result is False

    def test_inspection_failure_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that signature inspection failure logs a descriptive warning.

        Test Scenario:
            Mock inspect.signature() to fail, verify the warning message
            contains helpful information for debugging.

        Why Warnings Matter:
            When a dispatcher's signature cannot be inspected, it won't receive
            ModelDispatchContext. This might be unexpected behavior, so the
            warning helps developers understand:
            1. Why their dispatcher isn't getting context
            2. What caused the inspection failure (original exception message)
            3. How to work around it (wrap in inspectable function)

        Warning Message Content:
            - "Failed to inspect dispatcher signature" - explains what happened
            - Original exception message - helps identify the cause
            - "Uninspectable dispatchers" - provides context about fallback

        Note:
            This is a user-facing message that helps with debugging, so we
            verify its content is informative and actionable.
        """
        import logging
        from unittest.mock import patch

        logger = logging.getLogger("test.dispatch.inspection_failure")
        engine = MessageDispatchEngine(logger=logger)

        def valid_dispatcher(
            envelope: ModelEventEnvelope[object],
            context: object,
        ) -> str:
            return "output"

        with patch("inspect.signature") as mock_signature:
            mock_signature.side_effect = ValueError("no signature found for builtin")

            log_ctx = caplog.at_level(
                logging.WARNING, logger="test.dispatch.inspection_failure"
            )
            with log_ctx:
                result = engine._dispatcher_accepts_context(valid_dispatcher)

        # Method should return False
        assert result is False

        # Warning should have been logged
        assert len(caplog.records) == 1
        warning_record = caplog.records[0]
        assert warning_record.levelno == logging.WARNING
        assert "Failed to inspect dispatcher signature" in warning_record.message
        assert "no signature found for builtin" in warning_record.message
        assert "Uninspectable dispatchers" in warning_record.message

    def test_unconventional_param_name_logs_warning_but_returns_true(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning for 2+ params where second param lacks context naming.

        When a dispatcher has 2+ parameters but the second parameter name
        doesn't contain 'context' or 'ctx', a warning should be logged to
        help identify potential signature mismatches. The method should
        still return True for backwards compatibility.

        This is a more explicit test than the existing one, verifying both
        the warning content and the True return value together.
        """
        import logging

        logger = logging.getLogger("test.dispatch.unconventional_param")
        engine = MessageDispatchEngine(logger=logger)

        def dispatcher_with_data_param(
            envelope: ModelEventEnvelope[object],
            data: str,  # Unconventional - doesn't contain 'context' or 'ctx'
        ) -> str:
            """Dispatcher where second param is named 'data' not 'context'."""
            return "output"

        log_ctx = caplog.at_level(
            logging.WARNING, logger="test.dispatch.unconventional_param"
        )
        with log_ctx:
            result = engine._dispatcher_accepts_context(dispatcher_with_data_param)

        # Method should return True (backwards compatible)
        assert result is True

        # Warning should have been logged
        assert len(caplog.records) == 1
        warning_record = caplog.records[0]
        assert warning_record.levelno == logging.WARNING
        assert "dispatcher_with_data_param" in warning_record.message
        assert "data" in warning_record.message
        assert "context naming convention" in warning_record.message
        assert "ModelDispatchContext" in warning_record.message

    def test_three_params_with_unconventional_second_param_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that 3+ param dispatchers also get warnings for unconventional naming.

        Even with 3 or more parameters, if the second parameter doesn't follow
        the context naming convention, a warning should be logged. This ensures
        the warning logic applies regardless of total parameter count.
        """
        import logging

        logger = logging.getLogger("test.dispatch.three_param_warning")
        engine = MessageDispatchEngine(logger=logger)

        def dispatcher_three_params_bad_name(
            envelope: ModelEventEnvelope[object],
            metadata: dict[str, str],  # Unconventional second param name
            optional_flag: bool = False,
        ) -> str:
            """Three-param dispatcher with unconventional second param."""
            return "output"

        log_ctx = caplog.at_level(
            logging.WARNING, logger="test.dispatch.three_param_warning"
        )
        with log_ctx:
            result = engine._dispatcher_accepts_context(
                dispatcher_three_params_bad_name
            )

        # Method should return True (has 2+ params)
        assert result is True

        # Warning should have been logged for unconventional name
        assert len(caplog.records) == 1
        warning_record = caplog.records[0]
        assert warning_record.levelno == logging.WARNING
        assert "dispatcher_three_params_bad_name" in warning_record.message
        assert "metadata" in warning_record.message

    def test_three_params_with_context_naming_no_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that 3+ param dispatchers with proper naming don't get warnings.

        When the second parameter follows the context naming convention,
        no warning should be logged even with additional parameters.
        """
        import logging

        from omnibase_infra.models.dispatch.model_dispatch_context import (
            ModelDispatchContext,
        )

        logger = logging.getLogger("test.dispatch.three_param_no_warning")
        engine = MessageDispatchEngine(logger=logger)

        def dispatcher_three_params_good_name(
            envelope: ModelEventEnvelope[object],
            dispatch_context: ModelDispatchContext,  # Proper naming
            debug_mode: bool = False,
        ) -> str:
            """Three-param dispatcher with proper context naming."""
            return "output"

        log_ctx = caplog.at_level(
            logging.WARNING, logger="test.dispatch.three_param_no_warning"
        )
        with log_ctx:
            result = engine._dispatcher_accepts_context(
                dispatcher_three_params_good_name
            )

        # Method should return True
        assert result is True

        # No warning should be logged (proper naming)
        assert len(caplog.records) == 0


# ============================================================================
# Event Type Routing Tests (OMN-2037)
# ============================================================================


class EnvelopeWithEventType:
    """Envelope-like object with event_type attribute for testing.

    Mimics ModelEventEnvelope but includes an explicit event_type field
    for event_type-based routing. Used to test that dispatch() uses
    event_type as primary routing key when present.
    """

    def __init__(
        self,
        payload: object,
        event_type: str | None = None,
        correlation_id: UUID | None = None,
        trace_id: UUID | None = None,
        span_id: UUID | None = None,
    ) -> None:
        self.payload = payload
        self.event_type = event_type
        self.correlation_id = correlation_id or uuid4()
        self.trace_id = trace_id
        self.span_id = span_id


@pytest.mark.unit
class TestEventTypeRouting:
    """Tests for event_type-based routing in MessageDispatchEngine (OMN-2037).

    Verifies that:
    - envelope.event_type is used as primary routing key when present
    - Fallback to payload class name works when event_type is None
    - Existing message_type-based routing is unaffected
    """

    @pytest.mark.asyncio
    async def test_dispatch_uses_event_type_as_primary_routing_key(self) -> None:
        """When envelope has event_type, dispatchers registered for that type are called."""
        engine = MessageDispatchEngine()
        results: list[str] = []

        async def handler(envelope: object) -> str:
            results.append("event_type_handler")
            return "output.v1"

        # Register dispatcher for a specific event_type string
        engine.register_dispatcher(
            dispatcher_id="event-type-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types={"node.introspected.v1"},
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="event-type-route",
                topic_pattern="*.node.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="event-type-handler",
            )
        )
        engine.freeze()

        # Create envelope with event_type set
        envelope = EnvelopeWithEventType(
            payload=UserCreatedEvent(user_id="u1", name="Test"),
            event_type="node.introspected.v1",
        )

        result = await engine.dispatch(
            "dev.node.events.v1",
            envelope,  # type: ignore[arg-type]
        )

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1
        assert results[0] == "event_type_handler"

    @pytest.mark.asyncio
    async def test_dispatch_event_type_overrides_class_name(self) -> None:
        """When event_type is present, it takes priority over payload class name.

        A dispatcher registered for the class name should NOT be called
        when event_type is present and does not match the class name.
        """
        engine = MessageDispatchEngine()
        class_name_results: list[str] = []
        event_type_results: list[str] = []

        async def class_name_handler(envelope: object) -> str:
            class_name_results.append("class_name")
            return "class.output.v1"

        async def event_type_handler(envelope: object) -> str:
            event_type_results.append("event_type")
            return "event_type.output.v1"

        # Register one dispatcher for class name, one for event_type string
        engine.register_dispatcher(
            dispatcher_id="class-name-handler",
            dispatcher=class_name_handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserCreatedEvent"},
        )
        engine.register_dispatcher(
            dispatcher_id="event-type-handler",
            dispatcher=event_type_handler,
            category=EnumMessageCategory.EVENT,
            message_types={"user.created.v2"},
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="class-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="class-name-handler",
            )
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="event-type-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="event-type-handler",
            )
        )
        engine.freeze()

        # Envelope with event_type set - should route to event_type_handler
        envelope = EnvelopeWithEventType(
            payload=UserCreatedEvent(user_id="u1", name="Test"),
            event_type="user.created.v2",
        )

        result = await engine.dispatch(
            "dev.user.events.v1",
            envelope,  # type: ignore[arg-type]
        )

        assert result.status == EnumDispatchStatus.SUCCESS
        # Only event_type handler should have been called
        assert len(event_type_results) == 1
        assert len(class_name_results) == 0

    @pytest.mark.asyncio
    async def test_dispatch_falls_back_to_class_name_when_event_type_is_none(
        self,
    ) -> None:
        """When event_type is None, dispatch falls back to payload class name."""
        engine = MessageDispatchEngine()
        results: list[str] = []

        async def handler(envelope: object) -> str:
            results.append("class_name_handler")
            return "output.v1"

        engine.register_dispatcher(
            dispatcher_id="class-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserCreatedEvent"},
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="class-handler",
            )
        )
        engine.freeze()

        # Envelope with event_type=None (explicit)
        envelope = EnvelopeWithEventType(
            payload=UserCreatedEvent(user_id="u1", name="Test"),
            event_type=None,
        )

        result = await engine.dispatch(
            "dev.user.events.v1",
            envelope,  # type: ignore[arg-type]
        )

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dispatch_falls_back_when_event_type_missing(self) -> None:
        """When envelope has no event_type attribute, falls back to class name.

        This tests backwards compatibility with standard ModelEventEnvelope
        which does not have an event_type field.
        """
        engine = MessageDispatchEngine()
        results: list[str] = []

        async def handler(envelope: object) -> str:
            results.append("handled")
            return "output.v1"

        engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserCreatedEvent"},
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        engine.freeze()

        # Standard ModelEventEnvelope - no event_type attribute
        envelope = ModelEventEnvelope(
            payload=UserCreatedEvent(user_id="u1", name="Test"),
            correlation_id=uuid4(),
        )

        result = await engine.dispatch("dev.user.events.v1", envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dispatch_event_type_empty_string_falls_back(self) -> None:
        """When event_type is empty string, falls back to class name routing."""
        engine = MessageDispatchEngine()
        results: list[str] = []

        async def handler(envelope: object) -> str:
            results.append("handled")
            return "output.v1"

        engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserCreatedEvent"},
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        engine.freeze()

        # Envelope with empty event_type - should fall back
        envelope = EnvelopeWithEventType(
            payload=UserCreatedEvent(user_id="u1", name="Test"),
            event_type="",
        )

        result = await engine.dispatch(
            "dev.user.events.v1",
            envelope,  # type: ignore[arg-type]
        )

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dispatch_event_type_whitespace_falls_back(self) -> None:
        """When event_type is whitespace-only, falls back to class name routing."""
        engine = MessageDispatchEngine()
        results: list[str] = []

        async def handler(envelope: object) -> str:
            results.append("handled")
            return "output.v1"

        engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserCreatedEvent"},
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        engine.freeze()

        # Envelope with whitespace event_type - should fall back
        envelope = EnvelopeWithEventType(
            payload=UserCreatedEvent(user_id="u1", name="Test"),
            event_type="   ",
        )

        result = await engine.dispatch(
            "dev.user.events.v1",
            envelope,  # type: ignore[arg-type]
        )

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dispatch_event_type_no_match_returns_no_dispatcher(self) -> None:
        """When event_type does not match any registered dispatcher, NO_DISPATCHER is returned."""
        engine = MessageDispatchEngine()

        async def handler(envelope: object) -> str:
            return "output.v1"

        # Register for a specific event_type
        engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types={"user.created.v1"},
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        engine.freeze()

        # Envelope with different event_type that does not match
        envelope = EnvelopeWithEventType(
            payload=UserCreatedEvent(user_id="u1", name="Test"),
            event_type="order.placed.v1",
        )

        result = await engine.dispatch(
            "dev.user.events.v1",
            envelope,  # type: ignore[arg-type]
        )

        assert result.status == EnumDispatchStatus.NO_DISPATCHER

    @pytest.mark.asyncio
    async def test_dispatch_event_type_with_wildcard_dispatcher(self) -> None:
        """Dispatcher with message_types=None (wildcard) matches any event_type."""
        engine = MessageDispatchEngine()
        results: list[str] = []

        async def wildcard_handler(envelope: object) -> str:
            results.append("wildcard")
            return "output.v1"

        # Register wildcard dispatcher (no message_types filter)
        engine.register_dispatcher(
            dispatcher_id="wildcard-handler",
            dispatcher=wildcard_handler,
            category=EnumMessageCategory.EVENT,
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.node.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="wildcard-handler",
            )
        )
        engine.freeze()

        # Any event_type should match wildcard dispatcher
        envelope = EnvelopeWithEventType(
            payload=UserCreatedEvent(user_id="u1", name="Test"),
            event_type="node.introspected.v1",
        )

        result = await engine.dispatch(
            "dev.node.events.v1",
            envelope,  # type: ignore[arg-type]
        )

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dispatch_event_type_padded_whitespace_stripped(self) -> None:
        """Padded event_type is stripped before routing, matching the registered key."""
        engine = MessageDispatchEngine()
        results: list[str] = []

        async def handler(envelope: object) -> str:
            results.append("handled")
            return "output.v1"

        engine.register_dispatcher(
            dispatcher_id="handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types={"node.introspected.v1"},
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="route",
                topic_pattern="*.node.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="handler",
            )
        )
        engine.freeze()

        # Envelope with whitespace-padded event_type - should be stripped to match
        envelope = EnvelopeWithEventType(
            payload=UserCreatedEvent(user_id="u1", name="Test"),
            event_type="  node.introspected.v1  ",
        )

        result = await engine.dispatch(
            "dev.node.events.v1",
            envelope,  # type: ignore[arg-type]
        )

        assert result.status == EnumDispatchStatus.SUCCESS
        assert len(results) == 1
