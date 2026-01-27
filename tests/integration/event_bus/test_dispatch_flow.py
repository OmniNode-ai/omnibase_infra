# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for end-to-end dispatch flow.

These tests validate the complete message dispatch flow including topic parsing,
category routing, pattern matching, and fan-out to multiple subscribers.

Test categories:
- End-to-End Dispatch Flow: Complete message journey from publish to handler
- Message Category Routing: Correct routing based on topic category
- Topic Pattern Matching: Glob pattern matching for route selection
- Fan-out to Multiple Subscribers: Multi-subscriber message delivery
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from tests.conftest import make_test_node_identity

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_infra.event_bus.models import ModelEventMessage
    from omnibase_infra.models.dispatch import ModelTopicParser

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def event_bus() -> AsyncGenerator[EventBusInmemory, None]:
    """Provide a started EventBusInmemory instance."""
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory

    bus = EventBusInmemory(environment="test", group="dispatch-flow")
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
def topic_parser() -> ModelTopicParser:
    """Provide a ModelTopicParser instance."""
    from omnibase_infra.models.dispatch import ModelTopicParser

    return ModelTopicParser()


# =============================================================================
# End-to-End Dispatch Flow Tests
# =============================================================================


class TestEndToEndDispatchFlow:
    """Tests for complete end-to-end dispatch flows."""

    @pytest.mark.asyncio
    async def test_simple_event_flow(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify simple event publish/subscribe flow works end-to-end."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        topic = f"test.events.{uuid4().hex[:8]}"
        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        identity = make_test_node_identity(uuid4().hex[:6])
        await event_bus.subscribe(topic, identity, handler)

        headers = ModelEventHeaders(
            source="test-publisher",
            event_type="user.created",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(topic, b"user-123", b'{"name": "John"}', headers)

        assert len(received) == 1
        assert received[0].topic == topic
        assert received[0].key == b"user-123"
        assert received[0].value == b'{"name": "John"}'
        assert received[0].headers.event_type == "user.created"

    @pytest.mark.asyncio
    async def test_command_flow(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify command publish/subscribe flow works end-to-end."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        topic = f"test.commands.{uuid4().hex[:8]}"
        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        identity = make_test_node_identity(uuid4().hex[:6])
        await event_bus.subscribe(topic, identity, handler)

        headers = ModelEventHeaders(
            source="test-publisher",
            event_type="create.user.command",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(topic, None, b'{"command": "create_user"}', headers)

        assert len(received) == 1
        assert received[0].topic == topic
        assert received[0].headers.event_type == "create.user.command"

    @pytest.mark.asyncio
    async def test_intent_flow(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify intent publish/subscribe flow works end-to-end."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        topic = f"test.intents.{uuid4().hex[:8]}"
        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        identity = make_test_node_identity(uuid4().hex[:6])
        await event_bus.subscribe(topic, identity, handler)

        headers = ModelEventHeaders(
            source="test-publisher",
            event_type="user.wants.checkout",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(topic, None, b'{"intent": "checkout"}', headers)

        assert len(received) == 1
        assert received[0].topic == topic
        assert received[0].headers.event_type == "user.wants.checkout"

    @pytest.mark.asyncio
    async def test_multiple_messages_ordering(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify multiple messages are received in order."""
        topic = f"test.ordering.{uuid4().hex[:8]}"
        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        identity = make_test_node_identity(uuid4().hex[:6])
        await event_bus.subscribe(topic, identity, handler)

        # Publish 10 messages
        for i in range(10):
            await event_bus.publish(topic, None, f"message-{i}".encode())

        assert len(received) == 10
        for i, msg in enumerate(received):
            assert msg.value == f"message-{i}".encode()

    @pytest.mark.asyncio
    async def test_envelope_publish_flow(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify envelope publishing works end-to-end."""
        import json

        topic = f"test.envelope.{uuid4().hex[:8]}"
        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        identity = make_test_node_identity(uuid4().hex[:6])
        await event_bus.subscribe(topic, identity, handler)

        envelope = {
            "event_type": "order.created",
            "payload": {"order_id": "ORD-123", "total": 99.99},
            "metadata": {"version": "1.0"},
        }
        await event_bus.publish_envelope(envelope, topic)

        assert len(received) == 1
        received_data = json.loads(received[0].value.decode("utf-8"))
        assert received_data["event_type"] == "order.created"
        assert received_data["payload"]["order_id"] == "ORD-123"


# =============================================================================
# Message Category Routing Tests
# =============================================================================


class TestMessageCategoryRouting:
    """Tests for message category-based routing."""

    def test_parse_event_topic(self, topic_parser) -> None:
        """Verify event topics are correctly parsed."""
        from omnibase_infra.enums import EnumMessageCategory

        result = topic_parser.parse("onex.registration.events")

        assert result.is_valid
        assert result.category == EnumMessageCategory.EVENT
        assert result.domain == "registration"

    def test_parse_command_topic(self, topic_parser) -> None:
        """Verify command topics are correctly parsed."""
        from omnibase_infra.enums import EnumMessageCategory

        result = topic_parser.parse("onex.order.commands")

        assert result.is_valid
        assert result.category == EnumMessageCategory.COMMAND
        assert result.domain == "order"

    def test_parse_intent_topic(self, topic_parser) -> None:
        """Verify intent topics are correctly parsed."""
        from omnibase_infra.enums import EnumMessageCategory

        result = topic_parser.parse("onex.checkout.intents")

        assert result.is_valid
        assert result.category == EnumMessageCategory.INTENT
        assert result.domain == "checkout"

    def test_parse_environment_aware_event_topic(self, topic_parser) -> None:
        """Verify environment-aware event topics are correctly parsed."""
        from omnibase_infra.enums import EnumMessageCategory
        from omnibase_infra.enums.enum_topic_standard import EnumTopicStandard

        result = topic_parser.parse("dev.user.events.v1")

        assert result.is_valid
        assert result.standard == EnumTopicStandard.ENVIRONMENT_AWARE
        assert result.category == EnumMessageCategory.EVENT
        assert result.domain == "user"
        assert result.environment == "dev"
        assert result.version == "v1"

    def test_parse_environment_aware_command_topic(self, topic_parser) -> None:
        """Verify environment-aware command topics are correctly parsed."""
        from omnibase_infra.enums import EnumMessageCategory
        from omnibase_infra.enums.enum_topic_standard import EnumTopicStandard

        result = topic_parser.parse("prod.order.commands.v2")

        assert result.is_valid
        assert result.standard == EnumTopicStandard.ENVIRONMENT_AWARE
        assert result.category == EnumMessageCategory.COMMAND
        assert result.domain == "order"
        assert result.environment == "prod"
        assert result.version == "v2"

    def test_get_category_from_topic(self, topic_parser) -> None:
        """Verify get_category extracts correct category."""
        from omnibase_infra.enums import EnumMessageCategory

        assert (
            topic_parser.get_category("onex.user.events") == EnumMessageCategory.EVENT
        )
        assert (
            topic_parser.get_category("onex.order.commands")
            == EnumMessageCategory.COMMAND
        )
        assert (
            topic_parser.get_category("onex.checkout.intents")
            == EnumMessageCategory.INTENT
        )
        assert (
            topic_parser.get_category("dev.user.events.v1") == EnumMessageCategory.EVENT
        )
        assert topic_parser.get_category("invalid.topic") is None

    def test_category_from_topic_class_method(self) -> None:
        """Verify EnumMessageCategory.from_topic class method works."""
        from omnibase_infra.enums import EnumMessageCategory

        assert (
            EnumMessageCategory.from_topic("onex.user.events")
            == EnumMessageCategory.EVENT
        )
        assert (
            EnumMessageCategory.from_topic("dev.order.commands.v1")
            == EnumMessageCategory.COMMAND
        )
        assert (
            EnumMessageCategory.from_topic("staging.checkout.intents.v2")
            == EnumMessageCategory.INTENT
        )
        assert EnumMessageCategory.from_topic("invalid") is None

    @pytest.mark.asyncio
    async def test_category_routing_with_event_bus(
        self,
        event_bus: EventBusInmemory,
        topic_parser,
    ) -> None:
        """Verify messages are routed correctly based on parsed category."""
        from omnibase_infra.enums import EnumMessageCategory
        from omnibase_infra.event_bus.models import ModelEventHeaders

        event_topic = "onex.user.events"
        command_topic = "onex.order.commands"

        event_messages: list[ModelEventMessage] = []
        command_messages: list[ModelEventMessage] = []

        async def event_handler(msg: ModelEventMessage) -> None:
            # Verify this handler only receives events
            category = topic_parser.get_category(msg.topic)
            assert category == EnumMessageCategory.EVENT
            event_messages.append(msg)

        async def command_handler(msg: ModelEventMessage) -> None:
            # Verify this handler only receives commands
            category = topic_parser.get_category(msg.topic)
            assert category == EnumMessageCategory.COMMAND
            command_messages.append(msg)

        event_identity = make_test_node_identity("event-group")
        command_identity = make_test_node_identity("command-group")
        await event_bus.subscribe(event_topic, event_identity, event_handler)
        await event_bus.subscribe(command_topic, command_identity, command_handler)

        # Publish to both topics
        await event_bus.publish(
            event_topic,
            None,
            b"event-payload",
            ModelEventHeaders(
                source="test",
                event_type="user.created",
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            ),
        )
        await event_bus.publish(
            command_topic,
            None,
            b"command-payload",
            ModelEventHeaders(
                source="test",
                event_type="create.user",
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            ),
        )

        assert len(event_messages) == 1
        assert len(command_messages) == 1
        assert event_messages[0].value == b"event-payload"
        assert command_messages[0].value == b"command-payload"


# =============================================================================
# Topic Pattern Matching Tests
# =============================================================================


class TestTopicPatternMatching:
    """Tests for topic pattern matching in dispatch routes."""

    def test_exact_match(self, topic_parser: ModelTopicParser) -> None:
        """Verify exact topic matching works."""
        assert topic_parser.matches_pattern(
            "onex.registration.events", "onex.registration.events"
        )
        assert not topic_parser.matches_pattern(
            "onex.registration.events", "onex.discovery.events"
        )

    def test_single_wildcard_match(self, topic_parser: ModelTopicParser) -> None:
        """Verify single wildcard (*) matches single segment."""
        assert topic_parser.matches_pattern("onex.*.events", "onex.registration.events")
        assert topic_parser.matches_pattern("onex.*.events", "onex.discovery.events")
        assert topic_parser.matches_pattern("onex.*.events", "onex.user.events")
        assert not topic_parser.matches_pattern("onex.*.events", "onex.user.commands")

    def test_double_wildcard_match(self, topic_parser: ModelTopicParser) -> None:
        """Verify double wildcard (**) matches multiple segments."""
        assert topic_parser.matches_pattern("dev.**", "dev.user.events.v1")
        assert topic_parser.matches_pattern("**.events", "onex.registration.events")
        assert topic_parser.matches_pattern("**.events.*", "dev.user.events.v1")
        assert topic_parser.matches_pattern("**.commands.*", "prod.order.commands.v2")

    def test_mixed_wildcards(self, topic_parser: ModelTopicParser) -> None:
        """Verify mixed wildcard patterns work correctly."""
        assert topic_parser.matches_pattern("*.*.events", "onex.user.events")
        assert topic_parser.matches_pattern("*.*.events.*", "dev.user.events.v1")
        assert not topic_parser.matches_pattern(
            "*.*.events", "dev.user.events.v1"
        )  # Extra segment

    def test_case_insensitive_matching(self, topic_parser: ModelTopicParser) -> None:
        """Verify pattern matching is case-insensitive."""
        assert topic_parser.matches_pattern("ONEX.*.EVENTS", "onex.registration.events")
        assert topic_parser.matches_pattern("onex.*.events", "ONEX.REGISTRATION.EVENTS")

    def test_empty_pattern_or_topic(self, topic_parser: ModelTopicParser) -> None:
        """Verify empty patterns or topics return False."""
        assert not topic_parser.matches_pattern("", "onex.user.events")
        assert not topic_parser.matches_pattern("onex.*.events", "")
        assert not topic_parser.matches_pattern("", "")

    def test_dispatch_route_pattern_matching(self) -> None:
        """Verify ModelDispatchRoute pattern matching works."""
        from omnibase_infra.enums import EnumMessageCategory
        from omnibase_infra.models.dispatch import ModelDispatchRoute

        route = ModelDispatchRoute(
            route_id="user-events-route",
            topic_pattern="*.user.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="user-event-dispatcher",
        )

        assert route.matches_topic("dev.user.events.v1")
        assert route.matches_topic("prod.user.events.v2")
        assert not route.matches_topic("dev.order.events.v1")
        assert not route.matches_topic("dev.user.commands.v1")

    def test_dispatch_route_full_match(self) -> None:
        """Verify ModelDispatchRoute.matches() with all criteria."""
        from omnibase_infra.enums import EnumMessageCategory
        from omnibase_infra.models.dispatch import ModelDispatchRoute

        route = ModelDispatchRoute(
            route_id="specific-event-route",
            topic_pattern="*.user.events.*",
            message_category=EnumMessageCategory.EVENT,
            message_type="UserCreatedEvent",
            dispatcher_id="user-created-dispatcher",
        )

        # Full match with message_type
        assert route.matches(
            "dev.user.events.v1", EnumMessageCategory.EVENT, "UserCreatedEvent"
        )

        # Wrong message_type
        assert not route.matches(
            "dev.user.events.v1", EnumMessageCategory.EVENT, "UserDeletedEvent"
        )

        # Wrong category
        assert not route.matches(
            "dev.user.events.v1", EnumMessageCategory.COMMAND, "UserCreatedEvent"
        )

    def test_dispatch_route_disabled(self) -> None:
        """Verify disabled routes don't match."""
        from omnibase_infra.enums import EnumMessageCategory
        from omnibase_infra.models.dispatch import ModelDispatchRoute

        route = ModelDispatchRoute(
            route_id="disabled-route",
            topic_pattern="*.user.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="disabled-dispatcher",
            enabled=False,
        )

        assert not route.matches_topic("dev.user.events.v1")
        assert not route.matches("dev.user.events.v1", EnumMessageCategory.EVENT, None)


# =============================================================================
# Fan-out to Multiple Subscribers Tests
# =============================================================================


class TestFanOutMultipleSubscribers:
    """Tests for fan-out message delivery to multiple subscribers."""

    @pytest.mark.asyncio
    async def test_same_topic_multiple_groups(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify message is delivered to all subscriber groups."""
        topic = f"test.fanout.{uuid4().hex[:8]}"

        group1_messages: list[ModelEventMessage] = []
        group2_messages: list[ModelEventMessage] = []
        group3_messages: list[ModelEventMessage] = []

        async def handler1(msg: ModelEventMessage) -> None:
            group1_messages.append(msg)

        async def handler2(msg: ModelEventMessage) -> None:
            group2_messages.append(msg)

        async def handler3(msg: ModelEventMessage) -> None:
            group3_messages.append(msg)

        identity1 = make_test_node_identity("group1")
        identity2 = make_test_node_identity("group2")
        identity3 = make_test_node_identity("group3")
        await event_bus.subscribe(topic, identity1, handler1)
        await event_bus.subscribe(topic, identity2, handler2)
        await event_bus.subscribe(topic, identity3, handler3)

        await event_bus.publish(topic, None, b"fanout-message")

        assert len(group1_messages) == 1
        assert len(group2_messages) == 1
        assert len(group3_messages) == 1
        assert group1_messages[0].value == b"fanout-message"
        assert group2_messages[0].value == b"fanout-message"
        assert group3_messages[0].value == b"fanout-message"

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_group(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify multiple handlers in same group all receive message."""
        topic = f"test.samegroup.{uuid4().hex[:8]}"
        shared_identity = make_test_node_identity(f"shared-group-{uuid4().hex[:6]}")

        handler1_messages: list[ModelEventMessage] = []
        handler2_messages: list[ModelEventMessage] = []

        async def handler1(msg: ModelEventMessage) -> None:
            handler1_messages.append(msg)

        async def handler2(msg: ModelEventMessage) -> None:
            handler2_messages.append(msg)

        # Both handlers use same group
        await event_bus.subscribe(topic, shared_identity, handler1)
        await event_bus.subscribe(topic, shared_identity, handler2)

        await event_bus.publish(topic, None, b"shared-group-message")

        # EventBusInmemory delivers to all handlers in same group
        assert len(handler1_messages) == 1
        assert len(handler2_messages) == 1

    @pytest.mark.asyncio
    async def test_fanout_with_different_topics(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify messages only go to subscribed topics."""
        topic1 = f"test.topic1.{uuid4().hex[:8]}"
        topic2 = f"test.topic2.{uuid4().hex[:8]}"
        topic3 = f"test.topic3.{uuid4().hex[:8]}"

        topic1_messages: list[ModelEventMessage] = []
        topic2_messages: list[ModelEventMessage] = []
        topic3_messages: list[ModelEventMessage] = []

        async def handler1(msg: ModelEventMessage) -> None:
            topic1_messages.append(msg)

        async def handler2(msg: ModelEventMessage) -> None:
            topic2_messages.append(msg)

        async def handler3(msg: ModelEventMessage) -> None:
            topic3_messages.append(msg)

        identity1 = make_test_node_identity("group1")
        identity2 = make_test_node_identity("group2")
        identity3 = make_test_node_identity("group3")
        await event_bus.subscribe(topic1, identity1, handler1)
        await event_bus.subscribe(topic2, identity2, handler2)
        await event_bus.subscribe(topic3, identity3, handler3)

        # Publish only to topic2
        await event_bus.publish(topic2, None, b"topic2-only")

        assert len(topic1_messages) == 0
        assert len(topic2_messages) == 1
        assert len(topic3_messages) == 0
        assert topic2_messages[0].value == b"topic2-only"

    @pytest.mark.asyncio
    async def test_broadcast_to_environment(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify broadcast_to_environment reaches subscribers."""
        import json

        broadcast_topic = "test.broadcast"  # Environment is "test"
        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        identity = make_test_node_identity(uuid4().hex[:6])
        await event_bus.subscribe(broadcast_topic, identity, handler)

        await event_bus.broadcast_to_environment(
            "reload_config",
            {"version": "2.0", "hot_reload": True},
        )

        assert len(received) == 1
        payload = json.loads(received[0].value.decode("utf-8"))
        assert payload["command"] == "reload_config"
        assert payload["payload"]["version"] == "2.0"
        assert payload["payload"]["hot_reload"] is True

    @pytest.mark.asyncio
    async def test_send_to_group(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify send_to_group reaches specific group subscribers."""
        import json

        target_group = f"target-{uuid4().hex[:8]}"
        group_topic = f"test.{target_group}"  # Environment.group pattern
        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        identity = make_test_node_identity(f"consumer-{uuid4().hex[:6]}")
        await event_bus.subscribe(group_topic, identity, handler)

        await event_bus.send_to_group(
            "process_batch",
            {"batch_id": "BATCH-123", "items": [1, 2, 3]},
            target_group,
        )

        assert len(received) == 1
        payload = json.loads(received[0].value.decode("utf-8"))
        assert payload["command"] == "process_batch"
        assert payload["payload"]["batch_id"] == "BATCH-123"
        assert payload["payload"]["items"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_high_volume_fanout(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify fanout handles high message volume."""
        topic = f"test.volume.{uuid4().hex[:8]}"
        message_count = 100
        subscriber_count = 5

        all_received: list[list[ModelEventMessage]] = [
            [] for _ in range(subscriber_count)
        ]

        def create_handler(
            idx: int,
        ) -> Callable[[ModelEventMessage], Awaitable[None]]:
            async def handler(msg: ModelEventMessage) -> None:
                all_received[idx].append(msg)

            return handler

        # Subscribe multiple handlers
        for i in range(subscriber_count):
            handler = create_handler(i)
            identity = make_test_node_identity(f"group-{i}")
            await event_bus.subscribe(topic, identity, handler)

        # Publish many messages
        for i in range(message_count):
            await event_bus.publish(topic, None, f"msg-{i}".encode())

        # All subscribers should receive all messages
        for i in range(subscriber_count):
            assert len(all_received[i]) == message_count

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(
        self,
        event_bus: EventBusInmemory,
    ) -> None:
        """Verify unsubscribe stops message delivery to handler."""
        topic = f"test.unsub.{uuid4().hex[:8]}"
        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        identity = make_test_node_identity(uuid4().hex[:6])
        unsubscribe = await event_bus.subscribe(topic, identity, handler)

        # First message should be received
        await event_bus.publish(topic, None, b"message-1")
        assert len(received) == 1

        # Unsubscribe
        await unsubscribe()

        # Second message should NOT be received
        await event_bus.publish(topic, None, b"message-2")
        assert len(received) == 1


# =============================================================================
# Dispatch Result Tests
# =============================================================================


class TestDispatchResult:
    """Tests for ModelDispatchResult in dispatch flows."""

    def test_dispatch_result_creation(self) -> None:
        """Verify ModelDispatchResult can be created correctly."""
        from omnibase_infra.models.dispatch import (
            EnumDispatchStatus,
            ModelDispatchResult,
        )

        result = ModelDispatchResult(
            status=EnumDispatchStatus.SUCCESS,
            topic="onex.user.events",
            route_id="user-route",
            dispatcher_id="user-dispatcher",
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert result.is_successful()
        assert not result.is_error()
        assert result.topic == "onex.user.events"
        assert result.route_id == "user-route"
        assert result.dispatcher_id == "user-dispatcher"

    def test_dispatch_result_error_status(self) -> None:
        """Verify dispatch result error states work correctly."""
        from omnibase_infra.models.dispatch import (
            EnumDispatchStatus,
            ModelDispatchResult,
        )

        result = ModelDispatchResult(
            status=EnumDispatchStatus.HANDLER_ERROR,
            topic="onex.user.events",
            error_message="Handler failed",
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert result.is_error()
        assert not result.is_successful()
        assert result.error_message == "Handler failed"

    def test_dispatch_result_with_error_transformation(self) -> None:
        """Verify with_error() creates new result with error info."""
        from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
        from omnibase_infra.models.dispatch import (
            EnumDispatchStatus,
            ModelDispatchResult,
        )

        initial = ModelDispatchResult(
            status=EnumDispatchStatus.ROUTED,
            topic="onex.user.events",
            route_id="user-route",
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )

        error_result = initial.with_error(
            status=EnumDispatchStatus.HANDLER_ERROR,
            message="Database connection failed",
            code=EnumCoreErrorCode.DATABASE_CONNECTION_ERROR,
        )

        assert error_result.is_error()
        assert error_result.error_message == "Database connection failed"
        assert error_result.error_code == EnumCoreErrorCode.DATABASE_CONNECTION_ERROR
        # Original fields preserved
        assert error_result.topic == "onex.user.events"
        assert error_result.route_id == "user-route"

    def test_dispatch_result_with_success_transformation(self) -> None:
        """Verify with_success() creates new successful result."""
        from omnibase_infra.models.dispatch import (
            EnumDispatchStatus,
            ModelDispatchResult,
        )

        initial = ModelDispatchResult(
            status=EnumDispatchStatus.ROUTED,
            topic="onex.user.events",
            route_id="user-route",
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )

        success_result = initial.with_success(
            outputs=["onex.notification.events"],
            output_count=1,
        )

        assert success_result.is_successful()
        assert success_result.outputs == ["onex.notification.events"]
        assert success_result.output_count == 1

    def test_dispatch_result_timeout_status(self) -> None:
        """Verify timeout status requires retry."""
        from omnibase_infra.models.dispatch import (
            EnumDispatchStatus,
            ModelDispatchResult,
        )

        result = ModelDispatchResult(
            status=EnumDispatchStatus.TIMEOUT,
            topic="onex.user.events",
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert result.is_error()
        assert result.requires_retry()

    def test_dispatch_result_terminal_status(self) -> None:
        """Verify terminal status detection."""
        from omnibase_infra.models.dispatch import (
            EnumDispatchStatus,
            ModelDispatchResult,
        )

        success_result = ModelDispatchResult(
            status=EnumDispatchStatus.SUCCESS,
            topic="onex.user.events",
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert success_result.is_terminal()

        error_result = ModelDispatchResult(
            status=EnumDispatchStatus.HANDLER_ERROR,
            topic="onex.user.events",
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert error_result.is_terminal()

        routed_result = ModelDispatchResult(
            status=EnumDispatchStatus.ROUTED,
            topic="onex.user.events",
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert not routed_result.is_terminal()
