# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for Event Ledger end-to-end pipeline.

These tests verify the complete flow:
1. HandlerLedgerProjection transforms ModelEventMessage to ModelIntent
2. Extract ModelPayloadLedgerAppend from the intent
3. HandlerLedgerAppend persists to PostgreSQL
4. Verify data appears correctly in event_ledger table

This validates the full pipeline without requiring Kafka infrastructure.

Note:
    HandlerLedgerProjection.project() is synchronous (pure compute) - no await needed.
    ModelEventMessage.headers is required - use empty ModelEventHeaders for headerless events.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_append import (
        HandlerLedgerAppend,
    )
    from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_query import (
        HandlerLedgerQuery,
    )


class TestLedgerE2EPipeline:
    """Test the complete ledger pipeline from event to database."""

    @pytest.fixture
    def mock_container(self) -> MagicMock:
        """Create a mock container for handlers."""
        container = MagicMock()
        container.get_config.return_value = {}
        return container

    @pytest.fixture
    def projection_handler(self, mock_container: MagicMock):
        """Create a HandlerLedgerProjection instance."""
        from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
            HandlerLedgerProjection,
        )

        return HandlerLedgerProjection(mock_container)

    @pytest.mark.asyncio
    async def test_event_message_to_database_flow(
        self,
        projection_handler,
        ledger_append_handler: HandlerLedgerAppend,
        postgres_pool: asyncpg.Pool,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Full E2E: EventMessage → Projection → Append → Database."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        # Create a realistic event message (simulating Kafka consumer output)
        correlation_id = uuid4()
        event_timestamp = datetime.now(UTC)
        unique_offset = int(uuid4().int % (2**62))

        headers = ModelEventHeaders(
            correlation_id=correlation_id,
            message_id=uuid4(),
            event_type="NodeRegistered",
            source="registration-orchestrator",
            timestamp=event_timestamp,
        )

        event_payload = {
            "node_id": "test-node-123",
            "node_type": "EFFECT_GENERIC",
            "registration_timestamp": event_timestamp.isoformat(),
        }

        message = ModelEventMessage(
            topic="onex.evt.platform.node-registration.v1",
            key=b"test-node-123",
            value=json.dumps(event_payload).encode("utf-8"),
            headers=headers,
            partition=0,
            offset=str(unique_offset),
        )

        # Step 1: Projection handler transforms message to intent
        # Note: project() is synchronous (pure compute) - no await
        intent = projection_handler.project(message)

        assert intent.intent_type == "extension"
        assert intent.payload.intent_type == "ledger.append"

        # Step 2: Append handler persists to database
        result = await ledger_append_handler.append(intent.payload)

        assert result.success is True
        assert result.duplicate is False
        assert result.ledger_entry_id is not None
        cleanup_event_ledger.append(result.ledger_entry_id)

        # Step 3: Verify data in database
        async with postgres_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    ledger_entry_id,
                    topic,
                    partition,
                    kafka_offset,
                    event_key,
                    event_value,
                    correlation_id,
                    event_type,
                    source,
                    onex_headers,
                    ledger_written_at
                FROM event_ledger
                WHERE ledger_entry_id = $1
                """,
                result.ledger_entry_id,
            )

        assert row is not None
        assert row["topic"] == "onex.evt.platform.node-registration.v1"
        assert row["partition"] == 0
        assert row["kafka_offset"] == unique_offset
        assert row["correlation_id"] == correlation_id
        assert row["event_type"] == "NodeRegistered"
        assert row["source"] == "registration-orchestrator"

        # Verify event_value roundtrip (BYTEA in DB)
        decoded_value = json.loads(row["event_value"].decode("utf-8"))
        assert decoded_value["node_id"] == "test-node-123"
        assert decoded_value["node_type"] == "EFFECT_GENERIC"

    @pytest.mark.asyncio
    async def test_event_without_headers_still_persists(
        self,
        projection_handler,
        ledger_append_handler: HandlerLedgerAppend,
        postgres_pool: asyncpg.Pool,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Events without ONEX metadata should still be captured (best-effort metadata)."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        unique_offset = int(uuid4().int % (2**62))

        # Create message with minimal headers but no correlation_id (simulating external event)
        # Note: timestamp, source, event_type are required, but correlation_id is optional
        minimal_headers = ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="external-system",
            event_type="ExternalEvent",
            # No correlation_id - this is the key test case
        )

        message = ModelEventMessage(
            topic="external.events.v1",
            key=b"external-key",
            value=b'{"external": "data"}',
            headers=minimal_headers,  # Headers without correlation_id
            partition=3,
            offset=str(unique_offset),
        )

        # Should still be processed (never drop events)
        intent = projection_handler.project(message)
        result = await ledger_append_handler.append(intent.payload)

        assert result.success is True
        cleanup_event_ledger.append(result.ledger_entry_id)

        # Verify in database - event was persisted even without correlation_id in original headers
        # Note: The handler extracts correlation_id from ModelEventHeaders.correlation_id,
        # which is None here. But the handler may auto-generate one, so we check
        # that event_type and source from headers are preserved.
        async with postgres_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT event_type, source
                FROM event_ledger
                WHERE ledger_entry_id = $1
                """,
                result.ledger_entry_id,
            )

        # Verify metadata from headers was extracted
        assert row["event_type"] == "ExternalEvent"
        assert row["source"] == "external-system"

    @pytest.mark.asyncio
    async def test_binary_event_value_preserved(
        self,
        projection_handler,
        ledger_append_handler: HandlerLedgerAppend,
        postgres_pool: asyncpg.Pool,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Binary (non-JSON) event values should be preserved exactly."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        unique_offset = int(uuid4().int % (2**62))

        # Create binary payload (not valid JSON)
        binary_payload = bytes(range(256))  # All byte values 0-255

        # Minimal headers for binary event
        headers = ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="binary-producer",
            event_type="BinaryEvent",
        )

        message = ModelEventMessage(
            topic="binary.events.v1",
            key=b"binary-key",
            value=binary_payload,
            headers=headers,
            partition=0,
            offset=str(unique_offset),
        )

        intent = projection_handler.project(message)
        result = await ledger_append_handler.append(intent.payload)

        assert result.success is True
        cleanup_event_ledger.append(result.ledger_entry_id)

        # Verify binary data is preserved exactly
        async with postgres_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT event_value FROM event_ledger WHERE ledger_entry_id = $1",
                result.ledger_entry_id,
            )

        assert row["event_value"] == binary_payload

    @pytest.mark.asyncio
    async def test_query_after_append_finds_event(
        self,
        projection_handler,
        ledger_append_handler: HandlerLedgerAppend,
        ledger_query_handler: HandlerLedgerQuery,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Events can be queried immediately after append."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        correlation_id = uuid4()
        unique_offset = int(uuid4().int % (2**62))

        headers = ModelEventHeaders(
            correlation_id=correlation_id,
            message_id=uuid4(),
            event_type="TestQueryEvent",
            source="e2e-test",
            timestamp=datetime.now(UTC),
        )

        message = ModelEventMessage(
            topic="test.query.after.append.v1",
            key=b"query-test",
            value=b'{"query": "test"}',
            headers=headers,
            partition=0,
            offset=str(unique_offset),
        )

        # Append
        intent = projection_handler.project(message)
        result = await ledger_append_handler.append(intent.payload)
        cleanup_event_ledger.append(result.ledger_entry_id)

        # Query immediately
        entries = await ledger_query_handler.query_by_correlation_id(correlation_id)

        assert len(entries) == 1
        assert entries[0].ledger_entry_id == result.ledger_entry_id
        assert entries[0].event_type == "TestQueryEvent"

    @pytest.mark.asyncio
    async def test_multiple_events_same_correlation_id(
        self,
        projection_handler,
        ledger_append_handler: HandlerLedgerAppend,
        ledger_query_handler: HandlerLedgerQuery,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Multiple events with same correlation_id are all captured and queryable."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        correlation_id = uuid4()

        # Simulate a workflow with multiple events sharing correlation_id
        event_types = ["WorkflowStarted", "StepCompleted", "WorkflowCompleted"]

        for i, event_type in enumerate(event_types):
            unique_offset = int(uuid4().int % (2**62))

            headers = ModelEventHeaders(
                correlation_id=correlation_id,
                message_id=uuid4(),
                event_type=event_type,
                source="workflow-orchestrator",
                timestamp=datetime.now(UTC),
            )

            message = ModelEventMessage(
                topic="workflow.events.v1",
                key=f"workflow-{correlation_id}".encode(),
                value=json.dumps({"step": i, "event_type": event_type}).encode(),
                headers=headers,
                partition=0,
                offset=str(unique_offset),
            )

            intent = projection_handler.project(message)
            result = await ledger_append_handler.append(intent.payload)
            cleanup_event_ledger.append(result.ledger_entry_id)

        # Query all events for the correlation_id
        entries = await ledger_query_handler.query_by_correlation_id(correlation_id)

        assert len(entries) == 3

        # Verify all event types are present
        found_types = {e.event_type for e in entries}
        assert found_types == {"WorkflowStarted", "StepCompleted", "WorkflowCompleted"}


class TestLedgerProjectionHandler:
    """Test HandlerLedgerProjection in isolation."""

    @pytest.fixture
    def mock_container(self) -> MagicMock:
        """Create a mock container."""
        container = MagicMock()
        container.get_config.return_value = {}
        return container

    @pytest.fixture
    def handler(self, mock_container: MagicMock):
        """Create a HandlerLedgerProjection instance."""
        from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
            HandlerLedgerProjection,
        )

        return HandlerLedgerProjection(mock_container)

    def test_projection_extracts_metadata_correctly(
        self,
        handler,
    ) -> None:
        """Projection should extract all metadata from headers."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        correlation_id = uuid4()
        envelope_id = uuid4()
        event_timestamp = datetime.now(UTC)

        headers = ModelEventHeaders(
            correlation_id=correlation_id,
            message_id=envelope_id,
            event_type="TestEvent",
            source="test-source",
            timestamp=event_timestamp,
        )

        message = ModelEventMessage(
            topic="test.topic.v1",
            key=b"test-key",
            value=b'{"data": "test"}',
            headers=headers,
            partition=5,
            offset="12345",
        )

        # Note: project() is synchronous
        intent = handler.project(message)

        payload = intent.payload
        assert payload.topic == "test.topic.v1"
        assert payload.partition == 5
        assert payload.kafka_offset == 12345
        assert payload.correlation_id == correlation_id
        assert payload.envelope_id == envelope_id
        assert payload.event_type == "TestEvent"
        assert payload.source == "test-source"

    def test_projection_base64_encodes_bytes(
        self,
        handler,
    ) -> None:
        """Projection should base64-encode event_key and event_value."""
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        raw_key = b"raw-key-bytes"
        raw_value = b"raw-value-bytes"

        # Minimal headers
        headers = ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="test-source",
            event_type="TestEvent",
        )

        message = ModelEventMessage(
            topic="test.b64.v1",
            key=raw_key,
            value=raw_value,
            headers=headers,
            partition=0,
            offset="100",
        )

        # Note: project() is synchronous
        intent = handler.project(message)

        # Verify base64 encoding
        assert intent.payload.event_key == base64.b64encode(raw_key).decode("ascii")
        assert intent.payload.event_value == base64.b64encode(raw_value).decode("ascii")

        # Verify roundtrip
        decoded_key = base64.b64decode(intent.payload.event_key)
        decoded_value = base64.b64decode(intent.payload.event_value)
        assert decoded_key == raw_key
        assert decoded_value == raw_value

    def test_projection_returns_model_intent(
        self,
        handler,
    ) -> None:
        """Projection should return a properly structured ModelIntent."""
        from omnibase_core.models.reducer.model_intent import ModelIntent
        from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

        # Minimal headers
        headers = ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="test-source",
            event_type="TestEvent",
        )

        message = ModelEventMessage(
            topic="test.intent.v1",
            key=b"key",
            value=b"value",
            headers=headers,
            partition=0,
            offset="100",
        )

        # Note: project() is synchronous
        intent = handler.project(message)

        assert isinstance(intent, ModelIntent)
        assert intent.intent_type == "extension"
        assert intent.payload.intent_type == "ledger.append"
        # Target includes topic/partition/offset for idempotency routing
        assert intent.target == "postgres://event_ledger/test.intent.v1/0/100"
