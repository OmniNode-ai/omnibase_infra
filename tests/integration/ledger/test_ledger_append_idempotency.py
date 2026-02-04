# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for Event Ledger append idempotency.

These tests verify that the ledger's idempotent write behavior works correctly:
- Same event (topic, partition, offset) twice → second returns duplicate=True
- Only one row exists in database after duplicate append
- Different offsets create separate entries

Idempotency Key: (topic, partition, kafka_offset)
Implementation: INSERT ... ON CONFLICT DO NOTHING RETURNING
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_append import (
        HandlerLedgerAppend,
    )
    from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
        ModelPayloadLedgerAppend,
    )


class TestLedgerAppendIdempotency:
    """Test idempotent append behavior of HandlerLedgerAppend."""

    @pytest.mark.asyncio
    async def test_first_append_succeeds_with_entry_id(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        sample_ledger_payload: ModelPayloadLedgerAppend,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """First append should succeed and return a ledger_entry_id."""
        result = await ledger_append_handler.append(sample_ledger_payload)

        assert result.success is True
        assert result.duplicate is False
        assert result.ledger_entry_id is not None
        assert isinstance(result.ledger_entry_id, UUID)
        assert result.topic == sample_ledger_payload.topic
        assert result.partition == sample_ledger_payload.partition
        assert result.kafka_offset == sample_ledger_payload.kafka_offset

        # Track for cleanup
        cleanup_event_ledger.append(result.ledger_entry_id)

    @pytest.mark.asyncio
    async def test_duplicate_append_returns_duplicate_true(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Second append with same (topic, partition, offset) returns duplicate=True."""
        # Create payload with fixed Kafka position
        fixed_offset = int(uuid4().int % (2**62))
        payload = make_ledger_payload(
            topic="test.idempotency.topic.v1",
            partition=0,
            kafka_offset=fixed_offset,
        )

        # First append - should succeed
        result1 = await ledger_append_handler.append(payload)
        assert result1.success is True
        assert result1.duplicate is False
        assert result1.ledger_entry_id is not None
        cleanup_event_ledger.append(result1.ledger_entry_id)

        # Second append with SAME (topic, partition, offset) - should be duplicate
        result2 = await ledger_append_handler.append(payload)
        assert result2.success is True
        assert result2.duplicate is True
        assert result2.ledger_entry_id is None  # No new entry created

    @pytest.mark.asyncio
    async def test_duplicate_append_does_not_create_second_row(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        postgres_pool: asyncpg.Pool,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Verify only one row exists after duplicate append."""
        # Create payload with fixed Kafka position
        fixed_offset = int(uuid4().int % (2**62))
        topic = "test.idempotency.single-row.v1"
        payload = make_ledger_payload(
            topic=topic,
            partition=0,
            kafka_offset=fixed_offset,
        )

        # First append
        result1 = await ledger_append_handler.append(payload)
        cleanup_event_ledger.append(result1.ledger_entry_id)

        # Second append (duplicate)
        await ledger_append_handler.append(payload)

        # Third append (duplicate)
        await ledger_append_handler.append(payload)

        # Verify only ONE row exists in database
        async with postgres_pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM event_ledger
                WHERE topic = $1 AND partition = $2 AND kafka_offset = $3
                """,
                topic,
                0,
                fixed_offset,
            )

        assert count == 1, f"Expected 1 row but found {count}"

    @pytest.mark.asyncio
    async def test_different_offsets_create_separate_entries(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Different kafka_offsets should create separate ledger entries."""
        topic = "test.idempotency.multi-offset.v1"

        # Create three payloads with same topic/partition but different offsets
        payloads = [
            make_ledger_payload(topic=topic, partition=0, kafka_offset=100),
            make_ledger_payload(topic=topic, partition=0, kafka_offset=101),
            make_ledger_payload(topic=topic, partition=0, kafka_offset=102),
        ]

        results = []
        for payload in payloads:
            result = await ledger_append_handler.append(payload)
            results.append(result)
            if result.ledger_entry_id:
                cleanup_event_ledger.append(result.ledger_entry_id)

        # All should succeed with unique entry IDs
        assert all(r.success for r in results)
        assert all(r.duplicate is False for r in results)

        entry_ids = [r.ledger_entry_id for r in results]
        assert len(set(entry_ids)) == 3, "All entries should have unique IDs"

    @pytest.mark.asyncio
    async def test_different_partitions_create_separate_entries(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Same offset on different partitions should create separate entries."""
        topic = "test.idempotency.multi-partition.v1"
        fixed_offset = int(uuid4().int % (2**62))

        # Same topic, same offset, but different partitions
        payloads = [
            make_ledger_payload(topic=topic, partition=0, kafka_offset=fixed_offset),
            make_ledger_payload(topic=topic, partition=1, kafka_offset=fixed_offset),
            make_ledger_payload(topic=topic, partition=2, kafka_offset=fixed_offset),
        ]

        results = []
        for payload in payloads:
            result = await ledger_append_handler.append(payload)
            results.append(result)
            if result.ledger_entry_id:
                cleanup_event_ledger.append(result.ledger_entry_id)

        # All should succeed - different partitions = different idempotency keys
        assert all(r.success for r in results)
        assert all(r.duplicate is False for r in results)

    @pytest.mark.asyncio
    async def test_different_topics_create_separate_entries(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Same offset/partition on different topics should create separate entries."""
        fixed_offset = int(uuid4().int % (2**62))

        # Same partition, same offset, but different topics
        payloads = [
            make_ledger_payload(
                topic="test.idempotency.topic-a.v1",
                partition=0,
                kafka_offset=fixed_offset,
            ),
            make_ledger_payload(
                topic="test.idempotency.topic-b.v1",
                partition=0,
                kafka_offset=fixed_offset,
            ),
            make_ledger_payload(
                topic="test.idempotency.topic-c.v1",
                partition=0,
                kafka_offset=fixed_offset,
            ),
        ]

        results = []
        for payload in payloads:
            result = await ledger_append_handler.append(payload)
            results.append(result)
            if result.ledger_entry_id:
                cleanup_event_ledger.append(result.ledger_entry_id)

        # All should succeed - different topics = different idempotency keys
        assert all(r.success for r in results)
        assert all(r.duplicate is False for r in results)

    @pytest.mark.asyncio
    async def test_idempotency_preserves_original_data(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        postgres_pool: asyncpg.Pool,
        make_ledger_payload,
        cleanup_event_ledger: list[UUID],
    ) -> None:
        """Duplicate append should not modify the original entry's data."""
        fixed_offset = int(uuid4().int % (2**62))
        topic = "test.idempotency.preserve-data.v1"
        original_correlation_id = uuid4()

        # First append with specific correlation_id
        payload1 = make_ledger_payload(
            topic=topic,
            partition=0,
            kafka_offset=fixed_offset,
            correlation_id=original_correlation_id,
            event_type="OriginalEvent",
        )
        result1 = await ledger_append_handler.append(payload1)
        cleanup_event_ledger.append(result1.ledger_entry_id)

        # Second append with DIFFERENT correlation_id but same position
        payload2 = make_ledger_payload(
            topic=topic,
            partition=0,
            kafka_offset=fixed_offset,
            correlation_id=uuid4(),  # Different correlation_id
            event_type="DifferentEvent",  # Different event type
        )
        await ledger_append_handler.append(payload2)

        # Verify the stored data is from the ORIGINAL entry
        async with postgres_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT correlation_id, event_type
                FROM event_ledger
                WHERE topic = $1 AND partition = $2 AND kafka_offset = $3
                """,
                topic,
                0,
                fixed_offset,
            )

        assert row is not None
        assert row["correlation_id"] == original_correlation_id
        assert row["event_type"] == "OriginalEvent"
