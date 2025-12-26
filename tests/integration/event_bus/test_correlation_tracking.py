# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for correlation ID tracking across the event bus.

These tests validate that correlation IDs are properly propagated across
publish/subscribe operations, multi-hop message flows, and error scenarios.

Test categories:
- Correlation ID Propagation: Verify IDs flow from publisher to subscriber
- Correlation ID Context Management: Verify proper context handling
- Multi-hop Correlation Tracking: Verify IDs persist across multiple hops
- Error Scenario Correlation: Verify IDs are preserved in error flows
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

if TYPE_CHECKING:
    from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
    from omnibase_infra.event_bus.models import ModelEventMessage

# =============================================================================
# Test Configuration
# =============================================================================

pytestmark = [
    pytest.mark.integration,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def event_bus() -> AsyncGenerator[InMemoryEventBus, None]:
    """Provide a started InMemoryEventBus instance."""
    from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus

    bus = InMemoryEventBus(environment="test", group="correlation-test")
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
def unique_topic() -> str:
    """Generate unique topic name for test isolation."""
    return f"test.correlation.{uuid4().hex[:12]}"


@pytest.fixture
def unique_group() -> str:
    """Generate unique consumer group for test isolation."""
    return f"corr-group-{uuid4().hex[:8]}"


# =============================================================================
# Correlation ID Propagation Tests
# =============================================================================


class TestCorrelationIdPropagation:
    """Tests for correlation ID propagation across publish/subscribe."""

    @pytest.mark.asyncio
    async def test_correlation_id_preserved_in_publish_subscribe(
        self,
        event_bus: InMemoryEventBus,
        unique_topic: str,
        unique_group: str,
    ) -> None:
        """Verify correlation ID is preserved from publisher to subscriber."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        received_messages: list[ModelEventMessage] = []
        original_correlation_id = uuid4()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)

        await event_bus.subscribe(unique_topic, unique_group, handler)

        headers = ModelEventHeaders(
            source="test-publisher",
            event_type="test.correlation",
            correlation_id=original_correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(unique_topic, None, b"test-value", headers)

        assert len(received_messages) == 1
        assert received_messages[0].headers.correlation_id == original_correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_auto_generated_when_not_provided(
        self,
        event_bus: InMemoryEventBus,
        unique_topic: str,
        unique_group: str,
    ) -> None:
        """Verify correlation ID is auto-generated if not provided in headers."""
        received_messages: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)

        await event_bus.subscribe(unique_topic, unique_group, handler)

        # Publish without explicit headers - InMemoryEventBus creates defaults
        await event_bus.publish(unique_topic, None, b"test-value")

        assert len(received_messages) == 1
        assert received_messages[0].headers.correlation_id is not None
        assert isinstance(received_messages[0].headers.correlation_id, UUID)

    @pytest.mark.asyncio
    async def test_different_messages_have_different_correlation_ids(
        self,
        event_bus: InMemoryEventBus,
        unique_topic: str,
        unique_group: str,
    ) -> None:
        """Verify different messages get unique auto-generated correlation IDs."""
        received_messages: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)

        await event_bus.subscribe(unique_topic, unique_group, handler)

        # Publish multiple messages without explicit correlation IDs
        for i in range(5):
            await event_bus.publish(unique_topic, None, f"value-{i}".encode())

        assert len(received_messages) == 5

        # All correlation IDs should be unique
        correlation_ids = {msg.headers.correlation_id for msg in received_messages}
        assert len(correlation_ids) == 5

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_to_multiple_subscribers(
        self,
        event_bus: InMemoryEventBus,
        unique_topic: str,
    ) -> None:
        """Verify correlation ID is propagated to all subscribers."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        received_by_sub1: list[ModelEventMessage] = []
        received_by_sub2: list[ModelEventMessage] = []
        original_correlation_id = uuid4()

        async def handler1(msg: ModelEventMessage) -> None:
            received_by_sub1.append(msg)

        async def handler2(msg: ModelEventMessage) -> None:
            received_by_sub2.append(msg)

        group1 = f"group1-{uuid4().hex[:8]}"
        group2 = f"group2-{uuid4().hex[:8]}"

        await event_bus.subscribe(unique_topic, group1, handler1)
        await event_bus.subscribe(unique_topic, group2, handler2)

        headers = ModelEventHeaders(
            source="test-publisher",
            event_type="test.multi-sub",
            correlation_id=original_correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(unique_topic, None, b"test-value", headers)

        assert len(received_by_sub1) == 1
        assert len(received_by_sub2) == 1
        assert received_by_sub1[0].headers.correlation_id == original_correlation_id
        assert received_by_sub2[0].headers.correlation_id == original_correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_format_is_uuid(
        self,
        event_bus: InMemoryEventBus,
        unique_topic: str,
        unique_group: str,
    ) -> None:
        """Verify correlation ID is a valid UUID object."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        received_messages: list[ModelEventMessage] = []
        test_correlation_id = uuid4()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)

        await event_bus.subscribe(unique_topic, unique_group, handler)

        headers = ModelEventHeaders(
            source="test",
            event_type="test",
            correlation_id=test_correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(unique_topic, None, b"test", headers)

        assert len(received_messages) == 1
        correlation_id = received_messages[0].headers.correlation_id
        assert isinstance(correlation_id, UUID)
        assert str(correlation_id) == str(test_correlation_id)


# =============================================================================
# Correlation ID Context Management Tests
# =============================================================================


class TestCorrelationIdContextManagement:
    """Tests for correlation ID context management patterns."""

    @pytest.mark.asyncio
    async def test_correlation_id_survives_message_history(
        self,
        event_bus: InMemoryEventBus,
        unique_topic: str,
    ) -> None:
        """Verify correlation ID is preserved in event history."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        original_correlation_id = uuid4()

        headers = ModelEventHeaders(
            source="test-publisher",
            event_type="test.history",
            correlation_id=original_correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(unique_topic, None, b"test-value", headers)

        # Retrieve from history
        history = await event_bus.get_event_history(topic=unique_topic)

        assert len(history) == 1
        assert history[0].headers.correlation_id == original_correlation_id

    @pytest.mark.asyncio
    async def test_trace_id_preserved_alongside_correlation_id(
        self,
        event_bus: InMemoryEventBus,
        unique_topic: str,
        unique_group: str,
    ) -> None:
        """Verify trace_id is preserved alongside correlation_id."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        received_messages: list[ModelEventMessage] = []
        original_correlation_id = uuid4()
        original_trace_id = "trace-abc-123"
        original_span_id = "span-xyz-456"

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)

        await event_bus.subscribe(unique_topic, unique_group, handler)

        headers = ModelEventHeaders(
            source="test-publisher",
            event_type="test.trace",
            correlation_id=original_correlation_id,
            trace_id=original_trace_id,
            span_id=original_span_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(unique_topic, None, b"test-value", headers)

        assert len(received_messages) == 1
        msg = received_messages[0]
        assert msg.headers.correlation_id == original_correlation_id
        assert msg.headers.trace_id == original_trace_id
        assert msg.headers.span_id == original_span_id

    @pytest.mark.asyncio
    async def test_parent_span_id_propagation(
        self,
        event_bus: InMemoryEventBus,
        unique_topic: str,
        unique_group: str,
    ) -> None:
        """Verify parent_span_id is propagated for distributed tracing."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        received_messages: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)

        await event_bus.subscribe(unique_topic, unique_group, handler)

        headers = ModelEventHeaders(
            source="test-publisher",
            event_type="test.parent-span",
            trace_id="trace-123",
            span_id="span-child",
            parent_span_id="span-parent",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(unique_topic, None, b"test-value", headers)

        assert len(received_messages) == 1
        msg = received_messages[0]
        assert msg.headers.parent_span_id == "span-parent"
        assert msg.headers.span_id == "span-child"


# =============================================================================
# Multi-hop Correlation Tracking Tests
# =============================================================================


class TestMultiHopCorrelationTracking:
    """Tests for correlation ID tracking across multiple message hops."""

    @pytest.mark.asyncio
    async def test_correlation_id_preserved_across_republish(
        self,
        event_bus: InMemoryEventBus,
    ) -> None:
        """Verify correlation ID is preserved when a subscriber republishes."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        topic1 = f"test.hop1.{uuid4().hex[:8]}"
        topic2 = f"test.hop2.{uuid4().hex[:8]}"
        final_messages: list[ModelEventMessage] = []
        original_correlation_id = uuid4()

        async def hop1_handler(msg: ModelEventMessage) -> None:
            """First hop: receive and republish with same correlation ID."""
            # Republish to second topic preserving correlation ID
            headers = ModelEventHeaders(
                source="hop1-processor",
                event_type="test.hop2",
                correlation_id=msg.headers.correlation_id,
                trace_id=msg.headers.trace_id,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            )
            await event_bus.publish(topic2, None, b"processed", headers)

        async def hop2_handler(msg: ModelEventMessage) -> None:
            """Second hop: final destination."""
            final_messages.append(msg)

        group1 = f"group1-{uuid4().hex[:8]}"
        group2 = f"group2-{uuid4().hex[:8]}"

        await event_bus.subscribe(topic1, group1, hop1_handler)
        await event_bus.subscribe(topic2, group2, hop2_handler)

        # Start the chain
        headers = ModelEventHeaders(
            source="originator",
            event_type="test.hop1",
            correlation_id=original_correlation_id,
            trace_id="trace-multi-hop",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(topic1, None, b"initial", headers)

        # Wait for async processing
        await asyncio.sleep(0.1)

        assert len(final_messages) == 1
        assert final_messages[0].headers.correlation_id == original_correlation_id
        assert final_messages[0].headers.trace_id == "trace-multi-hop"

    @pytest.mark.asyncio
    async def test_three_hop_correlation_chain(
        self,
        event_bus: InMemoryEventBus,
    ) -> None:
        """Verify correlation ID survives a three-hop message chain."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        topic1 = f"test.chain1.{uuid4().hex[:8]}"
        topic2 = f"test.chain2.{uuid4().hex[:8]}"
        topic3 = f"test.chain3.{uuid4().hex[:8]}"
        final_messages: list[ModelEventMessage] = []
        original_correlation_id = uuid4()

        async def chain1_handler(msg: ModelEventMessage) -> None:
            headers = ModelEventHeaders(
                source="chain1",
                event_type="chain2.event",
                correlation_id=msg.headers.correlation_id,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            )
            await event_bus.publish(topic2, None, b"hop2", headers)

        async def chain2_handler(msg: ModelEventMessage) -> None:
            headers = ModelEventHeaders(
                source="chain2",
                event_type="chain3.event",
                correlation_id=msg.headers.correlation_id,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            )
            await event_bus.publish(topic3, None, b"hop3", headers)

        async def chain3_handler(msg: ModelEventMessage) -> None:
            final_messages.append(msg)

        group1 = f"g1-{uuid4().hex[:6]}"
        group2 = f"g2-{uuid4().hex[:6]}"
        group3 = f"g3-{uuid4().hex[:6]}"

        await event_bus.subscribe(topic1, group1, chain1_handler)
        await event_bus.subscribe(topic2, group2, chain2_handler)
        await event_bus.subscribe(topic3, group3, chain3_handler)

        headers = ModelEventHeaders(
            source="originator",
            event_type="chain1.event",
            correlation_id=original_correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(topic1, None, b"start", headers)

        await asyncio.sleep(0.1)

        assert len(final_messages) == 1
        assert final_messages[0].headers.correlation_id == original_correlation_id

    @pytest.mark.asyncio
    async def test_fan_out_preserves_correlation_id(
        self,
        event_bus: InMemoryEventBus,
    ) -> None:
        """Verify correlation ID is preserved in fan-out scenarios."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        source_topic = f"test.fanout.source.{uuid4().hex[:8]}"
        target_topic1 = f"test.fanout.target1.{uuid4().hex[:8]}"
        target_topic2 = f"test.fanout.target2.{uuid4().hex[:8]}"
        target_topic3 = f"test.fanout.target3.{uuid4().hex[:8]}"

        target1_messages: list[ModelEventMessage] = []
        target2_messages: list[ModelEventMessage] = []
        target3_messages: list[ModelEventMessage] = []
        original_correlation_id = uuid4()

        async def fanout_handler(msg: ModelEventMessage) -> None:
            """Fan out to three target topics."""
            for target_topic in [target_topic1, target_topic2, target_topic3]:
                headers = ModelEventHeaders(
                    source="fanout-processor",
                    event_type="fanout.target",
                    correlation_id=msg.headers.correlation_id,
                    timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                )
                await event_bus.publish(target_topic, None, b"fanned", headers)

        async def target1_handler(msg: ModelEventMessage) -> None:
            target1_messages.append(msg)

        async def target2_handler(msg: ModelEventMessage) -> None:
            target2_messages.append(msg)

        async def target3_handler(msg: ModelEventMessage) -> None:
            target3_messages.append(msg)

        await event_bus.subscribe(source_topic, "source-group", fanout_handler)
        await event_bus.subscribe(target_topic1, "t1-group", target1_handler)
        await event_bus.subscribe(target_topic2, "t2-group", target2_handler)
        await event_bus.subscribe(target_topic3, "t3-group", target3_handler)

        headers = ModelEventHeaders(
            source="originator",
            event_type="fanout.source",
            correlation_id=original_correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(source_topic, None, b"start", headers)

        await asyncio.sleep(0.1)

        # All targets should receive messages with same correlation ID
        assert len(target1_messages) == 1
        assert len(target2_messages) == 1
        assert len(target3_messages) == 1
        assert target1_messages[0].headers.correlation_id == original_correlation_id
        assert target2_messages[0].headers.correlation_id == original_correlation_id
        assert target3_messages[0].headers.correlation_id == original_correlation_id


# =============================================================================
# Error Scenario Correlation Tests
# =============================================================================


class TestErrorScenarioCorrelation:
    """Tests for correlation ID handling in error scenarios."""

    @pytest.mark.asyncio
    async def test_correlation_id_preserved_when_handler_fails(
        self,
        event_bus: InMemoryEventBus,
        unique_topic: str,
    ) -> None:
        """Verify correlation ID is preserved even when handler fails."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        successful_messages: list[ModelEventMessage] = []
        original_correlation_id = uuid4()

        async def failing_handler(msg: ModelEventMessage) -> None:
            raise ValueError("Intentional test failure")

        async def successful_handler(msg: ModelEventMessage) -> None:
            successful_messages.append(msg)

        fail_group = f"fail-{uuid4().hex[:8]}"
        success_group = f"success-{uuid4().hex[:8]}"

        await event_bus.subscribe(unique_topic, fail_group, failing_handler)
        await event_bus.subscribe(unique_topic, success_group, successful_handler)

        headers = ModelEventHeaders(
            source="test-publisher",
            event_type="test.error",
            correlation_id=original_correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(unique_topic, None, b"test-value", headers)

        # Successful handler should still receive message with correlation ID
        assert len(successful_messages) == 1
        assert successful_messages[0].headers.correlation_id == original_correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_in_event_history_after_error(
        self,
        event_bus: InMemoryEventBus,
        unique_topic: str,
    ) -> None:
        """Verify correlation ID is preserved in history even after handler error."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        original_correlation_id = uuid4()

        async def failing_handler(msg: ModelEventMessage) -> None:
            raise ValueError("Intentional failure")

        await event_bus.subscribe(
            unique_topic, f"fail-{uuid4().hex[:8]}", failing_handler
        )

        headers = ModelEventHeaders(
            source="test-publisher",
            event_type="test.error-history",
            correlation_id=original_correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await event_bus.publish(unique_topic, None, b"test-value", headers)

        # History should have the message with correlation ID
        history = await event_bus.get_event_history(topic=unique_topic)
        assert len(history) == 1
        assert history[0].headers.correlation_id == original_correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_tracked_with_circuit_breaker_open(
        self,
        event_bus: InMemoryEventBus,
    ) -> None:
        """Verify correlation tracking works even when circuit breaker opens."""
        from omnibase_infra.event_bus.models import ModelEventHeaders

        unique_topic = f"test.cb.{uuid4().hex[:8]}"
        fail_group = f"fail-{uuid4().hex[:8]}"
        success_group = f"success-{uuid4().hex[:8]}"

        successful_messages: list[ModelEventMessage] = []
        correlation_ids_received: list[UUID] = []

        async def always_failing_handler(msg: ModelEventMessage) -> None:
            raise ValueError("Always fails")

        async def successful_handler(msg: ModelEventMessage) -> None:
            successful_messages.append(msg)
            correlation_ids_received.append(msg.headers.correlation_id)

        await event_bus.subscribe(unique_topic, fail_group, always_failing_handler)
        await event_bus.subscribe(unique_topic, success_group, successful_handler)

        # Publish enough messages to trigger circuit breaker (threshold=5)
        for i in range(7):
            headers = ModelEventHeaders(
                source="test-publisher",
                event_type="test.cb",
                correlation_id=uuid4(),  # Each message gets unique correlation ID
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            )
            await event_bus.publish(unique_topic, None, f"msg-{i}".encode(), headers)

        # All successful messages should have valid correlation IDs
        assert len(successful_messages) == 7
        assert len(correlation_ids_received) == 7
        assert all(isinstance(cid, UUID) for cid in correlation_ids_received)
        # All correlation IDs should be unique
        assert len(set(correlation_ids_received)) == 7


# =============================================================================
# Dispatch Context Correlation Tests
# =============================================================================


class TestDispatchContextCorrelation:
    """Tests for correlation ID in dispatch context."""

    @pytest.mark.asyncio
    async def test_dispatch_context_includes_correlation_id(self) -> None:
        """Verify ModelDispatchContext properly handles correlation IDs."""
        from omnibase_infra.models.dispatch import ModelDispatchContext

        correlation_id = uuid4()
        trace_id = uuid4()

        # Create context for different node types
        reducer_ctx = ModelDispatchContext.for_reducer(
            correlation_id=correlation_id,
            trace_id=trace_id,
        )

        assert reducer_ctx.correlation_id == correlation_id
        assert reducer_ctx.trace_id == trace_id

    @pytest.mark.asyncio
    async def test_dispatch_context_correlation_for_all_node_types(self) -> None:
        """Verify correlation ID works for all node type contexts."""
        from omnibase_infra.models.dispatch import ModelDispatchContext

        correlation_id = uuid4()
        trace_id = uuid4()
        now = datetime.now(UTC)

        # Reducer (no time injection)
        reducer_ctx = ModelDispatchContext.for_reducer(
            correlation_id=correlation_id,
            trace_id=trace_id,
        )
        assert reducer_ctx.correlation_id == correlation_id

        # Compute (no time injection)
        compute_ctx = ModelDispatchContext.for_compute(
            correlation_id=correlation_id,
            trace_id=trace_id,
        )
        assert compute_ctx.correlation_id == correlation_id

        # Orchestrator (with time injection)
        orchestrator_ctx = ModelDispatchContext.for_orchestrator(
            correlation_id=correlation_id,
            now=now,
            trace_id=trace_id,
        )
        assert orchestrator_ctx.correlation_id == correlation_id

        # Effect (with time injection)
        effect_ctx = ModelDispatchContext.for_effect(
            correlation_id=correlation_id,
            now=now,
            trace_id=trace_id,
        )
        assert effect_ctx.correlation_id == correlation_id

        # Runtime host (with time injection)
        runtime_ctx = ModelDispatchContext.for_runtime_host(
            correlation_id=correlation_id,
            now=now,
            trace_id=trace_id,
        )
        assert runtime_ctx.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_dispatch_result_includes_correlation_id(self) -> None:
        """Verify ModelDispatchResult properly stores correlation IDs."""
        from omnibase_infra.models.dispatch import (
            EnumDispatchStatus,
            ModelDispatchResult,
        )

        correlation_id = uuid4()
        trace_id = uuid4()

        result = ModelDispatchResult(
            status=EnumDispatchStatus.SUCCESS,
            topic="test.dispatch",
            route_id="test-route",
            dispatcher_id="test-dispatcher",
            correlation_id=correlation_id,
            trace_id=trace_id,
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert result.correlation_id == correlation_id
        assert result.trace_id == trace_id

    @pytest.mark.asyncio
    async def test_dispatch_result_error_preserves_correlation_id(self) -> None:
        """Verify correlation ID is preserved when dispatch result has error."""
        from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode

        from omnibase_infra.models.dispatch import (
            EnumDispatchStatus,
            ModelDispatchResult,
        )

        correlation_id = uuid4()

        result = ModelDispatchResult(
            status=EnumDispatchStatus.ROUTED,
            topic="test.dispatch",
            correlation_id=correlation_id,
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )

        error_result = result.with_error(
            status=EnumDispatchStatus.HANDLER_ERROR,
            message="Test error",
            code=EnumCoreErrorCode.HANDLER_EXECUTION_ERROR,
        )

        # Correlation ID should be preserved through error transformation
        assert error_result.correlation_id == correlation_id
        assert error_result.error_message == "Test error"
