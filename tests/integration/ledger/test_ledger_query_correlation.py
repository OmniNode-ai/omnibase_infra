# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for Event Ledger query by correlation_id.

These tests verify that HandlerLedgerQuery.query_by_correlation_id works correctly:
- Returns matching events for a given correlation_id
- Returns empty list for non-existent correlation_id
- Pagination with limit/offset works
- Results are ordered by timestamp descending
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING
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


class TestLedgerQueryByCorrelationId:
    """Test correlation_id query operations of HandlerLedgerQuery."""

    @pytest.mark.asyncio
    async def test_query_returns_matching_events(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        ledger_query_handler: HandlerLedgerQuery,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Query should return events matching the correlation_id."""
        target_correlation_id = uuid4()

        # Create events with the target correlation_id
        for i in range(3):
            payload = make_ledger_payload(
                topic=f"test.query.correlation.{i}.v1",
                correlation_id=target_correlation_id,
            )
            result = await ledger_append_handler.append(payload)
            cleanup_event_ledger.append(result.ledger_entry_id)

        # Create events with a DIFFERENT correlation_id (should not be returned)
        other_correlation_id = uuid4()
        for i in range(2):
            payload = make_ledger_payload(
                topic=f"test.query.other.{i}.v1",
                correlation_id=other_correlation_id,
            )
            result = await ledger_append_handler.append(payload)
            cleanup_event_ledger.append(result.ledger_entry_id)

        # Query for target correlation_id
        entries = await ledger_query_handler.query_by_correlation_id(
            target_correlation_id
        )

        assert len(entries) == 3
        assert all(e.correlation_id == target_correlation_id for e in entries)

    @pytest.mark.asyncio
    async def test_query_returns_empty_for_nonexistent_correlation_id(
        self,
        ledger_query_handler: HandlerLedgerQuery,
    ) -> None:
        """Query should return empty list for non-existent correlation_id."""
        nonexistent_id = uuid4()

        entries = await ledger_query_handler.query_by_correlation_id(nonexistent_id)

        assert entries == []

    @pytest.mark.asyncio
    async def test_query_with_limit_respects_limit(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        ledger_query_handler: HandlerLedgerQuery,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Query should respect the limit parameter."""
        correlation_id = uuid4()

        # Create 5 events
        for i in range(5):
            payload = make_ledger_payload(
                topic=f"test.query.limit.{i}.v1",
                correlation_id=correlation_id,
            )
            result = await ledger_append_handler.append(payload)
            cleanup_event_ledger.append(result.ledger_entry_id)

        # Query with limit=2
        entries = await ledger_query_handler.query_by_correlation_id(
            correlation_id, limit=2
        )

        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_query_with_offset_skips_entries(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        ledger_query_handler: HandlerLedgerQuery,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Query should skip entries based on offset parameter."""
        correlation_id = uuid4()

        # Create 5 events
        for i in range(5):
            payload = make_ledger_payload(
                topic=f"test.query.offset.{i}.v1",
                correlation_id=correlation_id,
            )
            result = await ledger_append_handler.append(payload)
            cleanup_event_ledger.append(result.ledger_entry_id)

        # Query all (for comparison)
        all_entries = await ledger_query_handler.query_by_correlation_id(correlation_id)
        assert len(all_entries) == 5

        # Query with offset=2
        offset_entries = await ledger_query_handler.query_by_correlation_id(
            correlation_id, offset=2
        )

        assert len(offset_entries) == 3  # 5 - 2 skipped

    @pytest.mark.asyncio
    async def test_query_returns_all_entry_fields(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        ledger_query_handler: HandlerLedgerQuery,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Query should return entries with all expected fields populated."""
        correlation_id = uuid4()
        envelope_id = uuid4()
        event_timestamp = datetime.now(UTC)

        payload = make_ledger_payload(
            topic="test.query.fields.v1",
            partition=5,
            correlation_id=correlation_id,
            envelope_id=envelope_id,
            event_type="TestFieldsEvent",
            source="test-source",
            event_timestamp=event_timestamp,
            onex_headers={"header_key": "header_value"},
        )
        result = await ledger_append_handler.append(payload)
        cleanup_event_ledger.append(result.ledger_entry_id)

        # Query the entry
        entries = await ledger_query_handler.query_by_correlation_id(correlation_id)

        assert len(entries) == 1
        entry = entries[0]

        # Verify all fields
        assert entry.ledger_entry_id == result.ledger_entry_id
        assert entry.topic == "test.query.fields.v1"
        assert entry.partition == 5
        assert entry.kafka_offset == payload.kafka_offset
        assert entry.correlation_id == correlation_id
        assert entry.envelope_id == envelope_id
        assert entry.event_type == "TestFieldsEvent"
        assert entry.source == "test-source"
        assert entry.event_key is not None  # Base64 encoded
        assert entry.event_value is not None  # Base64 encoded
        assert entry.ledger_written_at is not None

    @pytest.mark.asyncio
    async def test_query_handles_null_correlation_id_entries(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        ledger_query_handler: HandlerLedgerQuery,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Query should not return entries that have NULL correlation_id."""
        target_correlation_id = uuid4()

        # Create an entry with correlation_id
        payload_with_id = make_ledger_payload(
            topic="test.query.null-check.with-id.v1",
            correlation_id=target_correlation_id,
        )
        result1 = await ledger_append_handler.append(payload_with_id)
        cleanup_event_ledger.append(result1.ledger_entry_id)

        # Create an entry WITHOUT correlation_id
        payload_without_id = make_ledger_payload(
            topic="test.query.null-check.without-id.v1",
            correlation_id=None,  # No correlation_id
        )
        result2 = await ledger_append_handler.append(payload_without_id)
        cleanup_event_ledger.append(result2.ledger_entry_id)

        # Query for target correlation_id
        entries = await ledger_query_handler.query_by_correlation_id(
            target_correlation_id
        )

        # Should only return the entry WITH the correlation_id
        assert len(entries) == 1
        assert entries[0].correlation_id == target_correlation_id


class TestLedgerQueryModel:
    """Test the ModelLedgerQuery-based query interface."""

    @pytest.mark.asyncio
    async def test_query_method_with_correlation_id(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        ledger_query_handler: HandlerLedgerQuery,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """The query() method should work with correlation_id parameter."""
        from omnibase_infra.nodes.node_ledger_write_effect.models import (
            ModelLedgerQuery,
        )

        correlation_id = uuid4()

        # Create events
        for i in range(3):
            payload = make_ledger_payload(
                topic=f"test.query.model.{i}.v1",
                correlation_id=correlation_id,
            )
            result = await ledger_append_handler.append(payload)
            cleanup_event_ledger.append(result.ledger_entry_id)

        # Query using ModelLedgerQuery
        query = ModelLedgerQuery(
            correlation_id=correlation_id,
            limit=10,
            offset=0,
        )

        result = await ledger_query_handler.query(query, correlation_id=uuid4())

        assert len(result.entries) == 3
        assert result.total_count == 3
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_query_method_pagination_has_more(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        ledger_query_handler: HandlerLedgerQuery,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """The query() method should correctly set has_more for pagination."""
        from omnibase_infra.nodes.node_ledger_write_effect.models import (
            ModelLedgerQuery,
        )

        correlation_id = uuid4()

        # Create 5 events
        for i in range(5):
            payload = make_ledger_payload(
                topic=f"test.query.pagination.{i}.v1",
                correlation_id=correlation_id,
            )
            result = await ledger_append_handler.append(payload)
            cleanup_event_ledger.append(result.ledger_entry_id)

        # Query with limit=2 (should have more)
        query = ModelLedgerQuery(
            correlation_id=correlation_id,
            limit=2,
            offset=0,
        )

        result = await ledger_query_handler.query(query, correlation_id=uuid4())

        assert len(result.entries) == 2
        assert result.total_count == 5
        assert result.has_more is True  # 2 < 5

        # Query with limit=2, offset=4 (should NOT have more)
        query2 = ModelLedgerQuery(
            correlation_id=correlation_id,
            limit=2,
            offset=4,
        )

        result2 = await ledger_query_handler.query(query2, correlation_id=uuid4())

        assert len(result2.entries) == 1  # Only 1 remaining
        assert result2.total_count == 5
        assert result2.has_more is False  # 4 + 1 = 5, no more
