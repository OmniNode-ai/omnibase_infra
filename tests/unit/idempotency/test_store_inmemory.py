# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for InMemoryIdempotencyStore.

These tests verify the in-memory idempotency store implementation
conforms to ProtocolIdempotencyStore and provides correct behavior
for testing scenarios.

Ticket: OMN-945
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.idempotency import (
    InMemoryIdempotencyStore,
    ModelIdempotencyRecord,
)


class TestInMemoryIdempotencyStoreProtocol:
    """Test protocol conformance.

    Per ONEX conventions, protocol conformance is verified via duck typing
    by checking for required method presence and callability, rather than
    using isinstance checks with Protocol types.
    """

    def test_conforms_to_protocol(self) -> None:
        """InMemoryIdempotencyStore should conform to ProtocolIdempotencyStore.

        Verifies protocol conformance via duck typing by checking that all
        required methods exist and are callable.
        """
        store = InMemoryIdempotencyStore()

        # Verify all ProtocolIdempotencyStore methods via duck typing
        # Required methods: check_and_record, is_processed, mark_processed, cleanup_expired
        required_methods = [
            "check_and_record",
            "is_processed",
            "mark_processed",
            "cleanup_expired",
        ]
        for method_name in required_methods:
            assert hasattr(
                store, method_name
            ), f"Store must have '{method_name}' method"
            assert callable(
                getattr(store, method_name)
            ), f"'{method_name}' must be callable"


class TestCheckAndRecord:
    """Tests for check_and_record method."""

    @pytest.mark.asyncio
    async def test_first_call_returns_true(self) -> None:
        """First call for a message_id should return True."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()

        result = await store.check_and_record(message_id, domain="test")

        assert result is True

    @pytest.mark.asyncio
    async def test_duplicate_returns_false(self) -> None:
        """Second call for same message_id should return False."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()

        await store.check_and_record(message_id, domain="test")
        result = await store.check_and_record(message_id, domain="test")

        assert result is False

    @pytest.mark.asyncio
    async def test_different_domains_are_isolated(self) -> None:
        """Same message_id in different domains should be independent."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()

        result1 = await store.check_and_record(message_id, domain="domain1")
        result2 = await store.check_and_record(message_id, domain="domain2")
        result3 = await store.check_and_record(message_id, domain="domain1")

        assert result1 is True  # First in domain1
        assert result2 is True  # First in domain2
        assert result3 is False  # Duplicate in domain1

    @pytest.mark.asyncio
    async def test_none_domain_works(self) -> None:
        """None domain should work correctly."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()

        result1 = await store.check_and_record(message_id)  # None domain
        result2 = await store.check_and_record(message_id)  # Same None domain

        assert result1 is True
        assert result2 is False

    @pytest.mark.asyncio
    async def test_stores_correlation_id(self) -> None:
        """Correlation ID should be stored with the record."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()
        correlation_id = uuid4()

        await store.check_and_record(
            message_id, domain="test", correlation_id=correlation_id
        )
        record = await store.get_record(message_id, domain="test")

        assert record is not None
        assert record.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_concurrent_access_atomic(self) -> None:
        """Concurrent calls for same message_id should be atomic."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()

        # Simulate concurrent access
        results = await asyncio.gather(
            store.check_and_record(message_id, domain="test"),
            store.check_and_record(message_id, domain="test"),
            store.check_and_record(message_id, domain="test"),
        )

        # Exactly one should return True
        assert sum(results) == 1


class TestIsProcessed:
    """Tests for is_processed method."""

    @pytest.mark.asyncio
    async def test_returns_true_for_processed(self) -> None:
        """Should return True for processed messages."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()

        await store.check_and_record(message_id, domain="test")
        result = await store.is_processed(message_id, domain="test")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_unprocessed(self) -> None:
        """Should return False for unprocessed messages."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()

        result = await store.is_processed(message_id, domain="test")

        assert result is False

    @pytest.mark.asyncio
    async def test_respects_domain(self) -> None:
        """Should respect domain when checking."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()

        await store.check_and_record(message_id, domain="domain1")

        assert await store.is_processed(message_id, domain="domain1") is True
        assert await store.is_processed(message_id, domain="domain2") is False


class TestMarkProcessed:
    """Tests for mark_processed method."""

    @pytest.mark.asyncio
    async def test_marks_message_as_processed(self) -> None:
        """Should mark message as processed."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()

        await store.mark_processed(message_id, domain="test")

        assert await store.is_processed(message_id, domain="test") is True

    @pytest.mark.asyncio
    async def test_idempotent_operation(self) -> None:
        """Calling mark_processed multiple times should be safe."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()

        await store.mark_processed(message_id, domain="test")
        await store.mark_processed(message_id, domain="test")

        assert await store.get_record_count() == 1

    @pytest.mark.asyncio
    async def test_uses_provided_timestamp(self) -> None:
        """Should use provided processed_at timestamp."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()
        custom_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        await store.mark_processed(message_id, domain="test", processed_at=custom_time)
        record = await store.get_record(message_id, domain="test")

        assert record is not None
        assert record.processed_at == custom_time

    @pytest.mark.asyncio
    async def test_uses_current_time_when_not_provided(self) -> None:
        """Should use current time when processed_at not provided."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()
        before = datetime.now(UTC)

        await store.mark_processed(message_id, domain="test")
        record = await store.get_record(message_id, domain="test")

        after = datetime.now(UTC)
        assert record is not None
        assert before <= record.processed_at <= after


class TestCleanupExpired:
    """Tests for cleanup_expired method."""

    @pytest.mark.asyncio
    async def test_removes_old_entries(self) -> None:
        """Should remove entries older than TTL."""
        store = InMemoryIdempotencyStore()
        old_time = datetime.now(UTC) - timedelta(seconds=100)
        recent_time = datetime.now(UTC) - timedelta(seconds=10)

        old_id = uuid4()
        recent_id = uuid4()

        await store.mark_processed(old_id, domain="test", processed_at=old_time)
        await store.mark_processed(recent_id, domain="test", processed_at=recent_time)

        removed = await store.cleanup_expired(ttl_seconds=50)

        assert removed == 1
        assert await store.is_processed(old_id, domain="test") is False
        assert await store.is_processed(recent_id, domain="test") is True

    @pytest.mark.asyncio
    async def test_returns_count_of_removed(self) -> None:
        """Should return count of removed entries."""
        store = InMemoryIdempotencyStore()
        old_time = datetime.now(UTC) - timedelta(seconds=100)

        for _ in range(5):
            await store.mark_processed(uuid4(), domain="test", processed_at=old_time)

        removed = await store.cleanup_expired(ttl_seconds=50)

        assert removed == 5

    @pytest.mark.asyncio
    async def test_no_removal_when_nothing_expired(self) -> None:
        """Should return 0 when nothing to clean up."""
        store = InMemoryIdempotencyStore()
        recent_time = datetime.now(UTC) - timedelta(seconds=10)

        await store.mark_processed(uuid4(), domain="test", processed_at=recent_time)
        await store.mark_processed(uuid4(), domain="test", processed_at=recent_time)

        removed = await store.cleanup_expired(ttl_seconds=3600)

        assert removed == 0
        assert await store.get_record_count() == 2


class TestUtilityMethods:
    """Tests for test utility methods."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_records(self) -> None:
        """clear() should remove all records."""
        store = InMemoryIdempotencyStore()

        await store.check_and_record(uuid4(), domain="test1")
        await store.check_and_record(uuid4(), domain="test2")
        await store.check_and_record(uuid4(), domain="test3")

        await store.clear()

        assert await store.get_record_count() == 0

    @pytest.mark.asyncio
    async def test_get_record_count_returns_correct_count(self) -> None:
        """get_record_count() should return correct count."""
        store = InMemoryIdempotencyStore()

        assert await store.get_record_count() == 0

        await store.check_and_record(uuid4())
        assert await store.get_record_count() == 1

        await store.check_and_record(uuid4())
        assert await store.get_record_count() == 2

    @pytest.mark.asyncio
    async def test_get_all_records_returns_all_records(self) -> None:
        """get_all_records() should return all records."""
        store = InMemoryIdempotencyStore()
        ids = [uuid4(), uuid4(), uuid4()]

        for msg_id in ids:
            await store.check_and_record(msg_id, domain="test")

        records = await store.get_all_records()

        assert len(records) == 3
        record_ids = {r.message_id for r in records}
        assert record_ids == set(ids)

    @pytest.mark.asyncio
    async def test_get_record_returns_specific_record(self) -> None:
        """get_record() should return specific record by key."""
        store = InMemoryIdempotencyStore()
        message_id = uuid4()
        correlation_id = uuid4()

        await store.check_and_record(
            message_id, domain="test", correlation_id=correlation_id
        )

        record = await store.get_record(message_id, domain="test")

        assert record is not None
        assert isinstance(record, ModelIdempotencyRecord)
        assert record.message_id == message_id
        assert record.domain == "test"
        assert record.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_get_record_returns_none_for_missing(self) -> None:
        """get_record() should return None for non-existent records."""
        store = InMemoryIdempotencyStore()

        record = await store.get_record(uuid4(), domain="test")

        assert record is None
