# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
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

OMN-934: Message dispatch engine implementation
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.dispatch.enum_dispatch_status import EnumDispatchStatus

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

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

        def sync_handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "handled"

        dispatch_engine.register_handler(
            dispatcher_id="sync-handler",
            handler=sync_handler,
            category=EnumMessageCategory.EVENT,
        )

        assert dispatch_engine.handler_count == 1

    def test_register_handler_valid_async(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test successful async handler registration."""

        async def async_handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "handled"

        dispatch_engine.register_handler(
            dispatcher_id="async-handler",
            handler=async_handler,
            category=EnumMessageCategory.EVENT,
        )

        assert dispatch_engine.handler_count == 1

    def test_register_handler_with_message_types(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test handler registration with specific message types."""

        def handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "handled"

        dispatch_engine.register_handler(
            dispatcher_id="typed-handler",
            handler=handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserCreatedEvent", "UserUpdatedEvent"},
        )

        assert dispatch_engine.handler_count == 1

    def test_register_handler_multiple_categories(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test registering handlers for different categories."""

        def event_handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "event"

        def command_handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "command"

        def intent_handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "intent"

        dispatch_engine.register_handler(
            dispatcher_id="event-handler",
            handler=event_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_handler(
            dispatcher_id="command-handler",
            handler=command_handler,
            category=EnumMessageCategory.COMMAND,
        )
        dispatch_engine.register_handler(
            dispatcher_id="intent-handler",
            handler=intent_handler,
            category=EnumMessageCategory.INTENT,
        )

        assert dispatch_engine.handler_count == 3

    def test_register_handler_duplicate_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that duplicate dispatcher_id raises DUPLICATE_REGISTRATION error."""

        def handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "handled"

        dispatch_engine.register_handler(
            dispatcher_id="dup-handler",
            handler=handler,
            category=EnumMessageCategory.EVENT,
        )

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_handler(
                dispatcher_id="dup-handler",  # Same ID
                handler=handler,
                category=EnumMessageCategory.COMMAND,  # Different category
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.DUPLICATE_REGISTRATION
        assert "dup-handler" in exc_info.value.message

    def test_register_handler_empty_id_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that empty dispatcher_id raises INVALID_PARAMETER error."""

        def handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "handled"

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_handler(
                dispatcher_id="",
                handler=handler,
                category=EnumMessageCategory.EVENT,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_handler_whitespace_id_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that whitespace-only dispatcher_id raises INVALID_PARAMETER error."""

        def handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "handled"

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_handler(
                dispatcher_id="   ",
                handler=handler,
                category=EnumMessageCategory.EVENT,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_handler_none_callable_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that non-callable handler raises INVALID_PARAMETER error."""
        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_handler(
                dispatcher_id="bad-handler",
                handler=None,  # type: ignore[arg-type]
                category=EnumMessageCategory.EVENT,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_handler_non_callable_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that non-callable object raises INVALID_PARAMETER error."""
        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_handler(
                dispatcher_id="bad-handler",
                handler="not a function",  # type: ignore[arg-type]
                category=EnumMessageCategory.EVENT,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_handler_invalid_category_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that invalid category raises INVALID_PARAMETER error."""

        def handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "handled"

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_handler(
                dispatcher_id="handler",
                handler=handler,
                category="not_a_category",  # type: ignore[arg-type]
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER

    def test_register_handler_after_freeze_raises_error(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test that handler registration after freeze raises INVALID_STATE error."""
        dispatch_engine.freeze()

        def handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "handled"

        with pytest.raises(ModelOnexError) as exc_info:
            dispatch_engine.register_handler(
                dispatcher_id="late-handler",
                handler=handler,
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

        def handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "handled"

        dispatch_engine.register_handler(
            dispatcher_id="user-handler",
            handler=handler,
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
        assert dispatch_engine.handler_count == 0


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

        async def handler(envelope: ModelEventEnvelope[Any]) -> str:
            results.append("handled")
            return "output.topic.v1"

        dispatch_engine.register_handler(
            dispatcher_id="event-handler",
            handler=handler,
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

        def sync_handler(envelope: ModelEventEnvelope[Any]) -> str:
            results.append("sync_handled")
            return "sync.output.v1"

        dispatch_engine.register_handler(
            dispatcher_id="sync-handler",
            handler=sync_handler,
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

        async def handler1(envelope: ModelEventEnvelope[Any]) -> str:
            results.append("handler1")
            return "output1.v1"

        async def handler2(envelope: ModelEventEnvelope[Any]) -> str:
            results.append("handler2")
            return "output2.v1"

        dispatch_engine.register_handler(
            dispatcher_id="handler-1",
            handler=handler1,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_handler(
            dispatcher_id="handler-2",
            handler=handler2,
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

        async def handler(envelope: ModelEventEnvelope[Any]) -> list[str]:
            return ["output1.v1", "output2.v1", "output3.v1"]

        dispatch_engine.register_handler(
            dispatcher_id="multi-output-handler",
            handler=handler,
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

        async def handler(envelope: ModelEventEnvelope[Any]) -> None:
            pass  # No return value

        dispatch_engine.register_handler(
            dispatcher_id="void-handler",
            handler=handler,
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

        async def user_created_handler(envelope: ModelEventEnvelope[Any]) -> str:
            results.append("user_created")
            return "created.output"

        async def user_updated_handler(envelope: ModelEventEnvelope[Any]) -> str:
            results.append("user_updated")
            return "updated.output"

        dispatch_engine.register_handler(
            dispatcher_id="created-handler",
            handler=user_created_handler,
            category=EnumMessageCategory.EVENT,
            message_types={"UserCreatedEvent"},  # Only handles UserCreatedEvent
        )
        dispatch_engine.register_handler(
            dispatcher_id="updated-handler",
            handler=user_updated_handler,
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

        async def handler(envelope: ModelEventEnvelope[Any]) -> None:
            pass

        dispatch_engine.register_handler(
            dispatcher_id="handler",
            handler=handler,
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
    async def test_dispatch_no_handlers_returns_no_handler_status(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test dispatch with no matching handlers returns NO_HANDLER status."""
        dispatch_engine.freeze()

        result = await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        assert result.status == EnumDispatchStatus.NO_HANDLER
        assert result.error_message is not None
        assert "No handler" in result.error_message

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

        async def failing_handler(envelope: ModelEventEnvelope[Any]) -> None:
            raise ValueError("Something went wrong!")

        dispatch_engine.register_handler(
            dispatcher_id="failing-handler",
            handler=failing_handler,
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

        async def success_handler(envelope: ModelEventEnvelope[Any]) -> str:
            results.append("success")
            return "success.output"

        async def failing_handler(envelope: ModelEventEnvelope[Any]) -> None:
            results.append("failing")
            raise RuntimeError("Handler failed!")

        dispatch_engine.register_handler(
            dispatcher_id="success-handler",
            handler=success_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_handler(
            dispatcher_id="failing-handler",
            handler=failing_handler,
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

        async def handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "handled"

        dispatch_engine.register_handler(
            dispatcher_id="handler",
            handler=handler,
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
        assert result.status == EnumDispatchStatus.NO_HANDLER


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

        async def async_handler(envelope: ModelEventEnvelope[Any]) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            results.append("async_complete")
            return "async.output"

        dispatch_engine.register_handler(
            dispatcher_id="async-handler",
            handler=async_handler,
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
        metrics = dispatch_engine.get_metrics()

        assert metrics["dispatch_count"] == 0
        assert metrics["dispatch_success_count"] == 0
        assert metrics["dispatch_error_count"] == 0
        assert metrics["total_latency_ms"] == 0.0
        assert metrics["handler_execution_count"] == 0
        assert metrics["handler_error_count"] == 0
        assert metrics["routes_matched_count"] == 0
        assert metrics["no_handler_count"] == 0
        assert metrics["category_mismatch_count"] == 0

    @pytest.mark.asyncio
    async def test_metrics_updated_on_success(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test metrics are updated on successful dispatch."""

        async def handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "output"

        dispatch_engine.register_handler(
            dispatcher_id="handler",
            handler=handler,
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

        metrics = dispatch_engine.get_metrics()
        assert metrics["dispatch_count"] == 1
        assert metrics["dispatch_success_count"] == 1
        assert metrics["dispatch_error_count"] == 0
        assert metrics["handler_execution_count"] == 1
        assert metrics["total_latency_ms"] > 0
        assert metrics["routes_matched_count"] == 1

    @pytest.mark.asyncio
    async def test_metrics_updated_on_handler_error(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test metrics are updated on handler error."""

        async def failing_handler(envelope: ModelEventEnvelope[Any]) -> None:
            raise ValueError("Failure!")

        dispatch_engine.register_handler(
            dispatcher_id="handler",
            handler=failing_handler,
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

        metrics = dispatch_engine.get_metrics()
        assert metrics["dispatch_count"] == 1
        assert metrics["dispatch_error_count"] == 1
        assert metrics["handler_execution_count"] == 1
        assert metrics["handler_error_count"] == 1

    @pytest.mark.asyncio
    async def test_metrics_updated_on_no_handler(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test metrics are updated when no handler is found."""
        dispatch_engine.freeze()

        await dispatch_engine.dispatch("dev.user.events.v1", event_envelope)

        metrics = dispatch_engine.get_metrics()
        assert metrics["dispatch_count"] == 1
        assert metrics["dispatch_error_count"] == 1
        assert metrics["no_handler_count"] == 1

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

        metrics = dispatch_engine.get_metrics()
        assert metrics["dispatch_count"] == 1
        assert metrics["dispatch_error_count"] == 1
        assert metrics["category_mismatch_count"] == 1

    @pytest.mark.asyncio
    async def test_metrics_accumulate_across_dispatches(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test metrics accumulate across multiple dispatches."""

        async def handler(envelope: ModelEventEnvelope[Any]) -> str:
            return "output"

        dispatch_engine.register_handler(
            dispatcher_id="handler",
            handler=handler,
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

        metrics = dispatch_engine.get_metrics()
        assert metrics["dispatch_count"] == 5
        assert metrics["dispatch_success_count"] == 5
        assert metrics["handler_execution_count"] == 5


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
        handler_calls: list[list[str]] = []

        async def handler1(envelope: ModelEventEnvelope[Any]) -> None:
            pass

        async def handler2(envelope: ModelEventEnvelope[Any]) -> None:
            pass

        dispatch_engine.register_handler(
            dispatcher_id="handler-1",
            handler=handler1,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_handler(
            dispatcher_id="handler-2",
            handler=handler2,
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

        async def user_handler(envelope: ModelEventEnvelope[Any]) -> None:
            pass

        async def order_handler(envelope: ModelEventEnvelope[Any]) -> None:
            pass

        dispatch_engine.register_handler(
            dispatcher_id="user-handler",
            handler=user_handler,
            category=EnumMessageCategory.EVENT,
        )
        dispatch_engine.register_handler(
            dispatcher_id="order-handler",
            handler=order_handler,
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
        """Test that routing is based on topic/category, not payload content."""
        handler_calls: list[str] = []

        async def handler(envelope: ModelEventEnvelope[Any]) -> None:
            handler_calls.append(type(envelope.payload).__name__)

        dispatch_engine.register_handler(
            dispatcher_id="generic-handler",
            handler=handler,
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

        # Both should route to the same handler regardless of payload type
        assert len(handler_calls) == 2
        assert handler_calls[0] == "UserCreatedEvent"
        assert handler_calls[1] == "SomeGenericPayload"

    @pytest.mark.asyncio
    async def test_outputs_are_publishing_only(
        self,
        dispatch_engine: MessageDispatchEngine,
        event_envelope: ModelEventEnvelope[UserCreatedEvent],
    ) -> None:
        """Test that outputs are collected for publishing, not interpreted."""

        # Handler returns various output formats
        async def handler(envelope: ModelEventEnvelope[Any]) -> list[str]:
            return [
                "output.topic.v1",
                "another.output.v1",
                "third.output.commands.v1",  # Note: commands topic
            ]

        dispatch_engine.register_handler(
            dispatcher_id="handler",
            handler=handler,
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
        assert "handlers=0" in result
        assert "frozen=False" in result

    def test_str_representation_with_data(
        self, dispatch_engine: MessageDispatchEngine
    ) -> None:
        """Test __str__ method with routes and handlers."""

        def handler(envelope: ModelEventEnvelope[Any]) -> None:
            pass

        dispatch_engine.register_handler(
            dispatcher_id="handler",
            handler=handler,
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
        assert "handlers=1" in result
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

    def test_handler_count(self, dispatch_engine: MessageDispatchEngine) -> None:
        """Test handler_count property."""

        def handler(envelope: ModelEventEnvelope[Any]) -> None:
            pass

        assert dispatch_engine.handler_count == 0

        dispatch_engine.register_handler(
            dispatcher_id="handler-1",
            handler=handler,
            category=EnumMessageCategory.EVENT,
        )
        assert dispatch_engine.handler_count == 1

        dispatch_engine.register_handler(
            dispatcher_id="handler-2",
            handler=handler,
            category=EnumMessageCategory.COMMAND,
        )
        assert dispatch_engine.handler_count == 2


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

        async def command_handler(envelope: ModelEventEnvelope[Any]) -> str:
            results.append("command_handled")
            return "result.events.v1"

        dispatch_engine.register_handler(
            dispatcher_id="command-handler",
            handler=command_handler,
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

        async def intent_handler(envelope: ModelEventEnvelope[Any]) -> str:
            results.append("intent_handled")
            return "user.commands.v1"

        dispatch_engine.register_handler(
            dispatcher_id="intent-handler",
            handler=intent_handler,
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
