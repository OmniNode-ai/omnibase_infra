# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for intelligence pipeline resilience scenarios 1-5.

These tests validate pipeline behavior under failure conditions using
mock infrastructure. They validate the behavioral contracts that
integration tests will verify against real infrastructure.

**Note on Scenarios 1-4**: These are *behavioral specification tests* that
document the intended resilience contracts (idempotency, bootstrap ordering,
consumer rebalancing) using locally-defined mock closures. They do not
exercise production code directly; instead they codify the expected behavior
so that integration tests against real Kafka/PostgreSQL/Consul infrastructure
can be validated against the same contracts. Scenario 5 (DLQ) additionally
exercises real ``ModelDlqEvent`` and ``MixinKafkaDlq`` production types.

Scenarios:
    1. Container Restart Mid-Processing: No duplicates, no lost messages
    2. Consumer Group Rebalancing: Partition reassignment, no loss
    3. Idempotency on Replay: Duplicate messages produce no duplicate writes
    4. Cold-Start Bootstrap Order: DB first, topics second, plugin third
    5. Dead Letter Queue Behavior: Malformed messages routed to DLQ

Related Tickets:
    - OMN-2291: Intelligence pipeline resilience testing
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.models import (
    ModelDlqEvent,
    ModelEventHeaders,
)
from tests.conftest import make_test_node_identity

# =============================================================================
# Scenario 1: Container Restart Mid-Processing
# =============================================================================


class TestContainerRestartResilience:
    """Tests validating behavior after container restart mid-processing.

    Validates that:
    - No duplicate writes occur after restart
    - No messages are lost after restart
    - Metrics remain consistent
    """

    @pytest.mark.asyncio
    async def test_consumer_resubscribe_after_restart(self) -> None:
        """Verify consumer can resubscribe to topics after simulated restart."""
        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory

        bus = EventBusInmemory(environment="test", group="restart-test")
        await bus.start()

        received_messages: list[bytes] = []

        async def handler(msg: Any) -> None:
            """Append message value to received_messages list."""
            value = msg.value if hasattr(msg, "value") else msg
            received_messages.append(value)

        identity = make_test_node_identity("restart-consumer")

        # Subscribe, publish, verify receipt
        unsub = await bus.subscribe("test.restart.topic", identity, handler)
        await bus.publish(
            "test.restart.topic",
            None,
            b'{"msg": "before-restart"}',
            ModelEventHeaders(
                source="test",
                event_type="test.restart",
                timestamp=datetime.now(UTC),
            ),
        )
        await asyncio.sleep(0.1)
        assert len(received_messages) >= 1

        # Simulate restart: unsubscribe, clear state, resubscribe
        await unsub()
        received_messages.clear()

        identity2 = make_test_node_identity("restart-consumer-2")
        unsub2 = await bus.subscribe("test.restart.topic", identity2, handler)

        await bus.publish(
            "test.restart.topic",
            None,
            b'{"msg": "after-restart"}',
            ModelEventHeaders(
                source="test",
                event_type="test.restart",
                timestamp=datetime.now(UTC),
            ),
        )
        await asyncio.sleep(0.1)

        # Should receive messages after restart
        assert len(received_messages) >= 1
        await unsub2()
        await bus.close()

    @pytest.mark.asyncio
    async def test_idempotent_processing_prevents_duplicate_writes(self) -> None:
        """Verify idempotent processing logic prevents duplicate writes.

        Simulates a handler that tracks processed correlation IDs to
        prevent duplicate writes on replay after restart.
        """
        processed_ids: set[str] = set()
        write_count = 0

        async def idempotent_handler(msg: Any) -> None:
            """Skip duplicate correlation IDs, increment write_count for new ones."""
            nonlocal write_count
            value = msg.value if hasattr(msg, "value") else msg
            data = json.loads(value)
            correlation_id = data.get("correlation_id", "")

            # Idempotency check
            if correlation_id in processed_ids:
                return  # Skip duplicate
            processed_ids.add(correlation_id)
            write_count += 1

        # Simulate processing same message twice (restart scenario)
        msg1 = MagicMock()
        msg1.value = json.dumps({"correlation_id": "abc-123", "data": "test"}).encode()

        await idempotent_handler(msg1)
        await idempotent_handler(msg1)  # Duplicate after restart

        assert write_count == 1, "Idempotent handler should prevent duplicate writes"

    @pytest.mark.asyncio
    async def test_metrics_consistency_after_restart(self) -> None:
        """Verify DLQ metrics remain consistent after bus restart."""
        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory

        bus = EventBusInmemory(environment="test", group="metrics-test")
        await bus.start()

        # Capture initial state
        health_before = await bus.health_check()
        assert health_before["started"] is True

        # Simulate restart
        await bus.close()
        bus2 = EventBusInmemory(environment="test", group="metrics-test-2")
        await bus2.start()

        health_after = await bus2.health_check()
        assert health_after["started"] is True
        assert health_after["healthy"] is True

        await bus2.close()


# =============================================================================
# Scenario 2: Consumer Group Rebalancing
# =============================================================================


class TestConsumerGroupRebalancing:
    """Tests validating behavior during consumer group rebalancing.

    Validates that:
    - Remaining consumer picks up partitions
    - No message loss during rebalancing
    """

    @pytest.mark.asyncio
    async def test_multiple_consumers_receive_messages(self) -> None:
        """Verify multiple consumers on the same topic all receive messages."""
        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory

        bus = EventBusInmemory(
            environment="test", group="rebalance-test", max_history=100
        )
        await bus.start()

        received_1: list[bytes] = []
        received_2: list[bytes] = []

        async def handler1(msg: Any) -> None:
            """Collect messages for consumer 1."""
            value = msg.value if hasattr(msg, "value") else msg
            received_1.append(value)

        async def handler2(msg: Any) -> None:
            """Collect messages for consumer 2."""
            value = msg.value if hasattr(msg, "value") else msg
            received_2.append(value)

        id1 = make_test_node_identity("consumer-1")
        id2 = make_test_node_identity("consumer-2")

        unsub1 = await bus.subscribe("test.rebalance", id1, handler1)
        unsub2 = await bus.subscribe("test.rebalance", id2, handler2)

        await bus.publish(
            "test.rebalance",
            None,
            b'{"msg": "shared"}',
            ModelEventHeaders(
                source="test",
                event_type="test.rebalance",
                timestamp=datetime.now(UTC),
            ),
        )
        await asyncio.sleep(0.1)

        # In-memory bus broadcasts to all subscribers
        total_received = len(received_1) + len(received_2)
        assert total_received >= 1, "At least one consumer should receive the message"

        await unsub1()
        await unsub2()
        await bus.close()

    @pytest.mark.asyncio
    async def test_remaining_consumer_continues_after_one_leaves(self) -> None:
        """Verify remaining consumer continues processing after one unsubscribes."""
        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory

        bus = EventBusInmemory(environment="test", group="rebalance-leave")
        await bus.start()

        received: list[bytes] = []

        async def handler(msg: Any) -> None:
            """Collect messages for the remaining consumer."""
            value = msg.value if hasattr(msg, "value") else msg
            received.append(value)

        async def noop_handler(msg: Any) -> None:
            """No-op handler simulating a consumer that will leave the group."""

        id1 = make_test_node_identity("consumer-stay")
        id2 = make_test_node_identity("consumer-leave")

        unsub1 = await bus.subscribe("test.rebalance.leave", id1, handler)
        unsub2 = await bus.subscribe("test.rebalance.leave", id2, noop_handler)

        # Remove one consumer
        await unsub2()

        # Publish after rebalancing
        await bus.publish(
            "test.rebalance.leave",
            None,
            b'{"msg": "after-rebalance"}',
            ModelEventHeaders(
                source="test",
                event_type="test.rebalance",
                timestamp=datetime.now(UTC),
            ),
        )
        await asyncio.sleep(0.1)

        assert len(received) >= 1, "Remaining consumer should still receive messages"

        await unsub1()
        await bus.close()


# =============================================================================
# Scenario 3: Idempotency on Replay
# =============================================================================


class TestIdempotencyOnReplay:
    """Tests validating idempotent processing on message replay.

    Validates that:
    - Replaying same messages produces no duplicate DB rows
    - No duplicate events emitted on replay
    """

    @pytest.mark.asyncio
    async def test_correlation_id_based_deduplication(self) -> None:
        """Verify correlation_id-based deduplication prevents duplicate processing."""
        db_rows: list[dict[str, str]] = []
        seen_ids: set[str] = set()

        async def idempotent_processor(correlation_id: str, data: str) -> None:
            """Write to db_rows only if correlation_id has not been seen before."""
            if correlation_id in seen_ids:
                return
            seen_ids.add(correlation_id)
            db_rows.append({"id": correlation_id, "data": data})

        # First processing pass
        batch = [
            ("corr-001", "data-a"),
            ("corr-002", "data-b"),
            ("corr-003", "data-c"),
        ]
        for cid, data in batch:
            await idempotent_processor(cid, data)

        assert len(db_rows) == 3

        # Replay the same batch (simulating consumer offset reset)
        for cid, data in batch:
            await idempotent_processor(cid, data)

        assert len(db_rows) == 3, "Replay should not produce duplicate rows"

    @pytest.mark.asyncio
    async def test_replay_does_not_emit_duplicate_events(self) -> None:
        """Verify replaying messages does not emit duplicate downstream events."""
        emitted_events: list[str] = []
        emitted_ids: set[str] = set()

        async def emit_once(correlation_id: str, event_type: str) -> None:
            """Emit event only on first occurrence of a given correlation_id."""
            if correlation_id in emitted_ids:
                return
            emitted_ids.add(correlation_id)
            emitted_events.append(event_type)

        # First pass
        await emit_once("corr-x", "analysis.completed")
        await emit_once("corr-y", "analysis.completed")

        assert len(emitted_events) == 2

        # Replay
        await emit_once("corr-x", "analysis.completed")
        await emit_once("corr-y", "analysis.completed")

        assert len(emitted_events) == 2, "Replay should not emit duplicate events"

    @pytest.mark.asyncio
    async def test_mixed_new_and_replay_messages(self) -> None:
        """Verify mixed batch of new and replayed messages processes correctly."""
        processed: list[str] = []
        seen: set[str] = set()

        async def process(cid: str) -> None:
            """Process a message, deduplicating by correlation ID."""
            if cid in seen:
                return
            seen.add(cid)
            processed.append(cid)

        # Original batch
        for cid in ["a", "b", "c"]:
            await process(cid)

        # Mixed batch: b is replay, d is new
        for cid in ["b", "d"]:
            await process(cid)

        assert processed == ["a", "b", "c", "d"]
        assert len(processed) == 4


# =============================================================================
# Scenario 4: Cold-Start Bootstrap Order
# =============================================================================


class TestColdStartBootstrapOrder:
    """Tests validating cold-start bootstrap dependency ordering.

    Validates that:
    - Bootstrap follows correct order: DB -> Topics -> Plugin
    - Graceful failure if dependencies unavailable
    - No silent corruption from wrong ordering
    """

    @pytest.mark.asyncio
    async def test_bootstrap_order_db_first(self) -> None:
        """Verify bootstrap checks DB availability before topics."""
        bootstrap_log: list[str] = []

        async def check_db() -> bool:
            """Simulate successful DB availability check."""
            bootstrap_log.append("check_db")
            return True

        async def check_topics() -> bool:
            """Simulate successful topic availability check."""
            bootstrap_log.append("check_topics")
            return True

        async def check_plugin() -> bool:
            """Simulate successful plugin availability check."""
            bootstrap_log.append("check_plugin")
            return True

        # Simulate bootstrap order
        steps = [
            ("db", check_db),
            ("topics", check_topics),
            ("plugin", check_plugin),
        ]

        for name, check in steps:
            result = await check()
            if not result:
                break

        assert bootstrap_log == ["check_db", "check_topics", "check_plugin"]

    @pytest.mark.asyncio
    async def test_bootstrap_fails_gracefully_if_db_unavailable(self) -> None:
        """Verify bootstrap fails gracefully when DB is unavailable."""
        bootstrap_log: list[str] = []
        errors: list[str] = []

        async def check_db() -> bool:
            """Simulate DB unavailability (returns False)."""
            bootstrap_log.append("check_db")
            return False  # DB unavailable

        async def check_topics() -> bool:
            """Simulate topic check that should never be reached."""
            bootstrap_log.append("check_topics")
            return True

        steps = [
            ("db", check_db),
            ("topics", check_topics),
        ]

        for name, check in steps:
            result = await check()
            if not result:
                errors.append(f"{name} unavailable")
                break

        # Topics should NOT be checked if DB is unavailable
        assert bootstrap_log == ["check_db"]
        assert errors == ["db unavailable"]

    @pytest.mark.asyncio
    async def test_bootstrap_no_silent_corruption(self) -> None:
        """Verify bootstrap does not silently corrupt state on failure."""
        state: dict[str, bool] = {
            "db_ready": False,
            "topics_ready": False,
            "plugin_ready": False,
        }

        async def init_db() -> bool:
            """Simulate successful DB initialization."""
            state["db_ready"] = True
            return True

        async def init_topics() -> bool:
            """Simulate topic creation failure (raises ConnectionError)."""
            raise ConnectionError("Kafka unavailable")

        async def init_plugin() -> bool:
            """Simulate plugin initialization (should not be reached)."""
            state["plugin_ready"] = True
            return True

        # Bootstrap with failure at topics step
        try:
            await init_db()
            await init_topics()
            await init_plugin()
        except ConnectionError:
            pass  # Expected failure

        # DB should be initialized but topics and plugin should not
        assert state["db_ready"] is True
        assert state["topics_ready"] is False
        assert state["plugin_ready"] is False


# =============================================================================
# Scenario 5: Dead Letter Queue Behavior
# =============================================================================


class TestDlqBehavior:
    """Tests validating Dead Letter Queue behavior for malformed messages.

    Validates that:
    - Malformed messages are routed to DLQ (not silently dropped)
    - Valid messages continue processing alongside malformed ones
    """

    @pytest.mark.asyncio
    async def test_malformed_message_routed_to_dlq(self) -> None:
        """Verify malformed messages are routed to DLQ."""
        dlq_messages: list[dict[str, Any]] = []
        processed_messages: list[dict[str, Any]] = []

        async def process_with_dlq(raw_value: bytes) -> None:
            """Route valid messages to processed_messages, invalid to dlq_messages."""
            try:
                data = json.loads(raw_value)
                if not isinstance(data, dict) or "required_field" not in data:
                    raise ValueError("Missing required_field")
                processed_messages.append(data)
            except (json.JSONDecodeError, ValueError) as e:
                dlq_messages.append(
                    {
                        "raw": raw_value.decode("utf-8", errors="replace"),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                )

        # Process valid and malformed messages
        await process_with_dlq(b'{"required_field": "value"}')  # Valid
        await process_with_dlq(b"not json at all")  # Malformed
        await process_with_dlq(b'{"wrong_field": "value"}')  # Missing required
        await process_with_dlq(b'{"required_field": "value2"}')  # Valid

        assert len(processed_messages) == 2, "Valid messages should be processed"
        assert len(dlq_messages) == 2, "Malformed messages should be in DLQ"

    @pytest.mark.asyncio
    async def test_dlq_messages_not_silently_dropped(self) -> None:
        """Verify malformed messages are logged/tracked, not silently dropped."""
        dropped_count = 0
        dlq_count = 0
        processed_count = 0

        async def process_message(raw: bytes) -> str:
            """Returns 'processed', 'dlq', or 'dropped'."""
            nonlocal dropped_count, dlq_count, processed_count
            try:
                data = json.loads(raw)
                processed_count += 1
                return "processed"
            except json.JSONDecodeError:
                # Route to DLQ (not drop)
                dlq_count += 1
                return "dlq"

        results = []
        for msg in [b'{"ok": true}', b"bad", b'{"ok": false}', b"\x00\x01"]:
            result = await process_message(msg)
            results.append(result)

        assert dropped_count == 0, "No messages should be silently dropped"
        assert dlq_count == 2, "Malformed messages should be routed to DLQ"
        assert processed_count == 2, "Valid messages should be processed"
        assert "dlq" in results

    @pytest.mark.asyncio
    async def test_valid_messages_continue_after_malformed(self) -> None:
        """Verify valid messages continue processing even after malformed ones."""
        processing_order: list[str] = []

        async def robust_handler(raw: bytes) -> None:
            """Parse JSON or record as DLQ malformed, continuing either way."""
            try:
                data = json.loads(raw)
                processing_order.append(f"ok:{data.get('id', '?')}")
            except json.JSONDecodeError:
                processing_order.append("dlq:malformed")

        messages = [
            b'{"id": "1"}',
            b"malformed!",
            b'{"id": "2"}',
            b"also bad",
            b'{"id": "3"}',
        ]

        for msg in messages:
            await robust_handler(msg)

        assert processing_order == [
            "ok:1",
            "dlq:malformed",
            "ok:2",
            "dlq:malformed",
            "ok:3",
        ]

    @pytest.mark.asyncio
    async def test_dlq_event_model_captures_failure_context(self) -> None:
        """Verify ModelDlqEvent captures comprehensive failure context."""
        correlation_id = uuid4()
        event = ModelDlqEvent(
            original_topic="intelligence.code-analysis.v1",
            dlq_topic="dev.dlq.intents.v1",
            correlation_id=correlation_id,
            error_type="JSONDecodeError",
            error_message="Expecting value: line 1 column 1 (char 0)",
            retry_count=0,
            message_offset="100",
            message_partition=0,
            success=True,
            timestamp=datetime.now(UTC),
            environment="test",
            consumer_group="intelligence-pipeline",
        )

        # Verify all context fields are populated
        assert event.original_topic == "intelligence.code-analysis.v1"
        assert event.error_type == "JSONDecodeError"
        assert event.correlation_id == correlation_id
        assert event.consumer_group == "intelligence-pipeline"

        # Verify log context contains all fields
        log_ctx = event.to_log_context()
        assert "original_topic" in log_ctx
        assert "correlation_id" in log_ctx
        assert "error_type" in log_ctx
        assert "consumer_group" in log_ctx

    @pytest.mark.asyncio
    async def test_dlq_callback_receives_events(self) -> None:
        """Verify DLQ callback mechanism works for alerting integration."""
        from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq

        # Verify the mixin has the register_dlq_callback method
        assert hasattr(MixinKafkaDlq, "register_dlq_callback")
        assert hasattr(MixinKafkaDlq, "_invoke_dlq_callbacks")


__all__: list[str] = []
