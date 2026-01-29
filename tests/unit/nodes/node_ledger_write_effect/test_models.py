# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for Event Ledger Effect Node Models.

This module validates the models used by NodeLedgerWriteEffect:
- ModelLedgerEntry: Single event ledger entry (row in event_ledger table)
- ModelLedgerAppendResult: Result of a ledger append operation
- ModelLedgerQuery: Query parameters for ledger searches
- ModelLedgerQueryResult: Result of a ledger query operation

Test Coverage:
    - Model construction and validation
    - Immutability (frozen=True)
    - Default values
    - Validation constraints (min_length, ge, le, etc.)
    - Serialization and round-trip
    - Extra fields forbidden (extra="forbid")

Related:
    - OMN-1646: Event Ledger Schema and Models
    - PR #208: Add event ledger schema and models
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_ledger_write_effect.models import (
    ModelLedgerAppendResult,
    ModelLedgerEntry,
    ModelLedgerQuery,
    ModelLedgerQueryResult,
)

# =============================================================================
# ModelLedgerEntry Tests
# =============================================================================


class TestModelLedgerEntryConstruction:
    """Tests for ModelLedgerEntry construction and validation."""

    def test_create_minimal_entry(self) -> None:
        """Create ledger entry with only required fields."""
        ledger_entry_id = uuid4()
        now = datetime.now(UTC)

        entry = ModelLedgerEntry(
            ledger_entry_id=ledger_entry_id,
            topic="test.events",
            partition=0,
            kafka_offset=100,
            event_value="SGVsbG8gV29ybGQ=",  # base64 for "Hello World"
            ledger_written_at=now,
        )

        assert entry.ledger_entry_id == ledger_entry_id
        assert entry.topic == "test.events"
        assert entry.partition == 0
        assert entry.kafka_offset == 100
        assert entry.event_value == "SGVsbG8gV29ybGQ="
        assert entry.ledger_written_at == now

    def test_create_full_entry(self) -> None:
        """Create ledger entry with all fields populated."""
        ledger_entry_id = uuid4()
        envelope_id = uuid4()
        correlation_id = uuid4()
        event_timestamp = datetime.now(UTC)
        ledger_written_at = datetime.now(UTC)

        entry = ModelLedgerEntry(
            ledger_entry_id=ledger_entry_id,
            topic="domain.service.event.v1",
            partition=5,
            kafka_offset=99999,
            event_key="dXNlci0xMjM=",  # base64 for "user-123"
            event_value="eyJldmVudCI6ICJkYXRhIn0=",  # base64 for JSON
            onex_headers={"x-onex-trace-id": "abc123"},
            envelope_id=envelope_id,
            correlation_id=correlation_id,
            event_type="user.created",
            source="user-service",
            event_timestamp=event_timestamp,
            ledger_written_at=ledger_written_at,
        )

        assert entry.ledger_entry_id == ledger_entry_id
        assert entry.topic == "domain.service.event.v1"
        assert entry.partition == 5
        assert entry.kafka_offset == 99999
        assert entry.event_key == "dXNlci0xMjM="
        assert entry.event_value == "eyJldmVudCI6ICJkYXRhIn0="
        assert entry.onex_headers == {"x-onex-trace-id": "abc123"}
        assert entry.envelope_id == envelope_id
        assert entry.correlation_id == correlation_id
        assert entry.event_type == "user.created"
        assert entry.source == "user-service"
        assert entry.event_timestamp == event_timestamp
        assert entry.ledger_written_at == ledger_written_at

    def test_optional_fields_default_correctly(self) -> None:
        """Optional fields have correct default values."""
        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            ledger_written_at=datetime.now(UTC),
        )

        # All nullable fields should default to None
        assert entry.event_key is None
        assert entry.envelope_id is None
        assert entry.correlation_id is None
        assert entry.event_type is None
        assert entry.source is None
        assert entry.event_timestamp is None
        # Dict field defaults to empty dict
        assert entry.onex_headers == {}


class TestModelLedgerEntryValidation:
    """Tests for ModelLedgerEntry validation constraints."""

    def test_partition_cannot_be_negative(self) -> None:
        """partition must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLedgerEntry(
                ledger_entry_id=uuid4(),
                topic="test",
                partition=-1,  # Invalid
                kafka_offset=0,
                event_value="dGVzdA==",
                ledger_written_at=datetime.now(UTC),
            )

        errors = exc_info.value.errors()
        assert any("partition" in str(e) for e in errors)

    def test_kafka_offset_cannot_be_negative(self) -> None:
        """kafka_offset must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLedgerEntry(
                ledger_entry_id=uuid4(),
                topic="test",
                partition=0,
                kafka_offset=-1,  # Invalid
                event_value="dGVzdA==",
                ledger_written_at=datetime.now(UTC),
            )

        errors = exc_info.value.errors()
        assert any("kafka_offset" in str(e) for e in errors)

    def test_partition_at_zero_boundary(self) -> None:
        """partition accepts zero (boundary value)."""
        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            ledger_written_at=datetime.now(UTC),
        )

        assert entry.partition == 0

    def test_kafka_offset_at_zero_boundary(self) -> None:
        """kafka_offset accepts zero (boundary value)."""
        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            ledger_written_at=datetime.now(UTC),
        )

        assert entry.kafka_offset == 0

    def test_large_partition_and_offset_values(self) -> None:
        """Large partition and offset values are accepted."""
        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test",
            partition=999999,
            kafka_offset=9999999999999,
            event_value="dGVzdA==",
            ledger_written_at=datetime.now(UTC),
        )

        assert entry.partition == 999999
        assert entry.kafka_offset == 9999999999999


class TestModelLedgerEntryImmutability:
    """Tests for ModelLedgerEntry immutability (frozen=True)."""

    def test_model_is_frozen(self) -> None:
        """ModelLedgerEntry is immutable - assignment raises error."""
        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            ledger_written_at=datetime.now(UTC),
        )

        with pytest.raises((TypeError, ValidationError)):
            entry.topic = "new-topic"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed (extra='forbid')."""
        with pytest.raises(ValidationError):
            ModelLedgerEntry(
                ledger_entry_id=uuid4(),
                topic="test",
                partition=0,
                kafka_offset=0,
                event_value="dGVzdA==",
                ledger_written_at=datetime.now(UTC),
                unknown_field="value",  # type: ignore[call-arg]
            )


# =============================================================================
# ModelLedgerAppendResult Tests
# =============================================================================


class TestModelLedgerAppendResultConstruction:
    """Tests for ModelLedgerAppendResult construction and validation."""

    def test_create_success_result(self) -> None:
        """Create successful append result with new entry."""
        ledger_entry_id = uuid4()

        result = ModelLedgerAppendResult(
            success=True,
            ledger_entry_id=ledger_entry_id,
            duplicate=False,
            topic="test.events",
            partition=3,
            kafka_offset=42,
        )

        assert result.success is True
        assert result.ledger_entry_id == ledger_entry_id
        assert result.duplicate is False
        assert result.topic == "test.events"
        assert result.partition == 3
        assert result.kafka_offset == 42

    def test_create_duplicate_result(self) -> None:
        """Create result for duplicate event (ON CONFLICT DO NOTHING)."""
        result = ModelLedgerAppendResult(
            success=True,
            ledger_entry_id=None,  # No ID for duplicate
            duplicate=True,
            topic="test.events",
            partition=3,
            kafka_offset=42,
        )

        assert result.success is True
        assert result.ledger_entry_id is None
        assert result.duplicate is True
        assert result.topic == "test.events"
        assert result.partition == 3
        assert result.kafka_offset == 42

    def test_create_failure_result(self) -> None:
        """Create result for failed append operation."""
        result = ModelLedgerAppendResult(
            success=False,
            ledger_entry_id=None,
            duplicate=False,
            topic="test.events",
            partition=0,
            kafka_offset=100,
        )

        assert result.success is False
        assert result.ledger_entry_id is None
        assert result.duplicate is False

    def test_default_values(self) -> None:
        """Optional fields have correct default values."""
        result = ModelLedgerAppendResult(
            success=True,
            topic="test",
            partition=0,
            kafka_offset=0,
        )

        assert result.ledger_entry_id is None
        assert result.duplicate is False


class TestModelLedgerAppendResultValidation:
    """Tests for ModelLedgerAppendResult validation constraints."""

    def test_partition_cannot_be_negative(self) -> None:
        """partition must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLedgerAppendResult(
                success=True,
                topic="test",
                partition=-1,  # Invalid
                kafka_offset=0,
            )

        errors = exc_info.value.errors()
        assert any("partition" in str(e) for e in errors)

    def test_kafka_offset_cannot_be_negative(self) -> None:
        """kafka_offset must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLedgerAppendResult(
                success=True,
                topic="test",
                partition=0,
                kafka_offset=-1,  # Invalid
            )

        errors = exc_info.value.errors()
        assert any("kafka_offset" in str(e) for e in errors)

    def test_partition_at_boundary(self) -> None:
        """partition accepts zero (boundary value)."""
        result = ModelLedgerAppendResult(
            success=True,
            topic="test",
            partition=0,
            kafka_offset=0,
        )

        assert result.partition == 0

    def test_kafka_offset_at_boundary(self) -> None:
        """kafka_offset accepts zero (boundary value)."""
        result = ModelLedgerAppendResult(
            success=True,
            topic="test",
            partition=0,
            kafka_offset=0,
        )

        assert result.kafka_offset == 0


class TestModelLedgerAppendResultImmutability:
    """Tests for ModelLedgerAppendResult immutability (frozen=True)."""

    def test_model_is_frozen(self) -> None:
        """ModelLedgerAppendResult is immutable - assignment raises error."""
        result = ModelLedgerAppendResult(
            success=True,
            topic="test",
            partition=0,
            kafka_offset=0,
        )

        with pytest.raises((TypeError, ValidationError)):
            result.success = False  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed (extra='forbid')."""
        with pytest.raises(ValidationError):
            ModelLedgerAppendResult(
                success=True,
                topic="test",
                partition=0,
                kafka_offset=0,
                unknown_field="value",  # type: ignore[call-arg]
            )


# =============================================================================
# ModelLedgerQuery Tests
# =============================================================================


class TestModelLedgerQueryConstruction:
    """Tests for ModelLedgerQuery construction and validation."""

    def test_create_empty_query(self) -> None:
        """Create query with all defaults (match all entries)."""
        query = ModelLedgerQuery()

        assert query.correlation_id is None
        assert query.event_type is None
        assert query.topic is None
        assert query.start_time is None
        assert query.end_time is None
        assert query.limit == 100
        assert query.offset == 0

    def test_create_query_by_correlation_id(self) -> None:
        """Create query filtering by correlation ID."""
        correlation_id = uuid4()
        query = ModelLedgerQuery(correlation_id=correlation_id)

        assert query.correlation_id == correlation_id

    def test_create_query_by_event_type(self) -> None:
        """Create query filtering by event type."""
        query = ModelLedgerQuery(event_type="user.created")

        assert query.event_type == "user.created"

    def test_create_query_by_topic(self) -> None:
        """Create query filtering by Kafka topic."""
        query = ModelLedgerQuery(topic="domain.service.event.v1")

        assert query.topic == "domain.service.event.v1"

    def test_create_query_with_time_range(self) -> None:
        """Create query filtering by time range."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)

        query = ModelLedgerQuery(start_time=start, end_time=end)

        assert query.start_time == start
        assert query.end_time == end

    def test_create_query_with_pagination(self) -> None:
        """Create query with custom pagination."""
        query = ModelLedgerQuery(limit=50, offset=100)

        assert query.limit == 50
        assert query.offset == 100

    def test_create_combined_query(self) -> None:
        """Create query with multiple filters."""
        correlation_id = uuid4()
        start = datetime(2024, 1, 1, tzinfo=UTC)

        query = ModelLedgerQuery(
            correlation_id=correlation_id,
            event_type="order.completed",
            topic="orders.events",
            start_time=start,
            limit=200,
            offset=50,
        )

        assert query.correlation_id == correlation_id
        assert query.event_type == "order.completed"
        assert query.topic == "orders.events"
        assert query.start_time == start
        assert query.limit == 200
        assert query.offset == 50


class TestModelLedgerQueryValidation:
    """Tests for ModelLedgerQuery validation constraints."""

    def test_limit_minimum_is_one(self) -> None:
        """limit must be at least 1."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLedgerQuery(limit=0)

        errors = exc_info.value.errors()
        assert any("limit" in str(e) for e in errors)

    def test_limit_maximum_is_10000(self) -> None:
        """limit must be at most 10000."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLedgerQuery(limit=10001)

        errors = exc_info.value.errors()
        assert any("limit" in str(e) for e in errors)

    def test_limit_at_boundaries(self) -> None:
        """limit accepts boundary values."""
        query_min = ModelLedgerQuery(limit=1)
        query_max = ModelLedgerQuery(limit=10000)

        assert query_min.limit == 1
        assert query_max.limit == 10000

    def test_offset_cannot_be_negative(self) -> None:
        """offset must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLedgerQuery(offset=-1)

        errors = exc_info.value.errors()
        assert any("offset" in str(e) for e in errors)

    def test_offset_at_zero_boundary(self) -> None:
        """offset accepts zero (boundary value)."""
        query = ModelLedgerQuery(offset=0)

        assert query.offset == 0

    def test_large_offset_value(self) -> None:
        """Large offset values are accepted."""
        query = ModelLedgerQuery(offset=999999999)

        assert query.offset == 999999999


class TestModelLedgerQueryImmutability:
    """Tests for ModelLedgerQuery immutability (frozen=True)."""

    def test_model_is_frozen(self) -> None:
        """ModelLedgerQuery is immutable - assignment raises error."""
        query = ModelLedgerQuery()

        with pytest.raises((TypeError, ValidationError)):
            query.limit = 50  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed (extra='forbid')."""
        with pytest.raises(ValidationError):
            ModelLedgerQuery(
                unknown_field="value",  # type: ignore[call-arg]
            )


# =============================================================================
# ModelLedgerQueryResult Tests
# =============================================================================


class TestModelLedgerQueryResultConstruction:
    """Tests for ModelLedgerQueryResult construction and validation."""

    def test_create_empty_result(self) -> None:
        """Create query result with no matching entries."""
        query = ModelLedgerQuery()

        result = ModelLedgerQueryResult(
            entries=[],
            total_count=0,
            has_more=False,
            query=query,
        )

        assert result.entries == []
        assert result.total_count == 0
        assert result.has_more is False
        assert result.query == query

    def test_create_result_with_entries(self) -> None:
        """Create query result with matching entries."""
        query = ModelLedgerQuery(limit=10)
        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            ledger_written_at=datetime.now(UTC),
        )

        result = ModelLedgerQueryResult(
            entries=[entry],
            total_count=1,
            has_more=False,
            query=query,
        )

        assert len(result.entries) == 1
        assert result.entries[0] == entry
        assert result.total_count == 1
        assert result.has_more is False

    def test_create_result_with_pagination(self) -> None:
        """Create paginated query result with has_more=True."""
        query = ModelLedgerQuery(limit=10, offset=0)
        entries = [
            ModelLedgerEntry(
                ledger_entry_id=uuid4(),
                topic="test",
                partition=0,
                kafka_offset=i,
                event_value="dGVzdA==",
                ledger_written_at=datetime.now(UTC),
            )
            for i in range(10)
        ]

        result = ModelLedgerQueryResult(
            entries=entries,
            total_count=100,  # More than returned
            has_more=True,
            query=query,
        )

        assert len(result.entries) == 10
        assert result.total_count == 100
        assert result.has_more is True


class TestModelLedgerQueryResultValidation:
    """Tests for ModelLedgerQueryResult validation constraints."""

    def test_total_count_cannot_be_negative(self) -> None:
        """total_count must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLedgerQueryResult(
                entries=[],
                total_count=-1,  # Invalid
                has_more=False,
                query=ModelLedgerQuery(),
            )

        errors = exc_info.value.errors()
        assert any("total_count" in str(e) for e in errors)

    def test_total_count_at_zero_boundary(self) -> None:
        """total_count accepts zero (boundary value)."""
        result = ModelLedgerQueryResult(
            entries=[],
            total_count=0,
            has_more=False,
            query=ModelLedgerQuery(),
        )

        assert result.total_count == 0


class TestModelLedgerQueryResultImmutability:
    """Tests for ModelLedgerQueryResult immutability (frozen=True)."""

    def test_model_is_frozen(self) -> None:
        """ModelLedgerQueryResult is immutable - assignment raises error."""
        result = ModelLedgerQueryResult(
            entries=[],
            total_count=0,
            has_more=False,
            query=ModelLedgerQuery(),
        )

        with pytest.raises((TypeError, ValidationError)):
            result.total_count = 10  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed (extra='forbid')."""
        with pytest.raises(ValidationError):
            ModelLedgerQueryResult(
                entries=[],
                total_count=0,
                has_more=False,
                query=ModelLedgerQuery(),
                unknown_field="value",  # type: ignore[call-arg]
            )


# =============================================================================
# Serialization Tests
# =============================================================================


class TestLedgerModelSerialization:
    """Tests for ledger model serialization."""

    def test_ledger_entry_to_dict(self) -> None:
        """ModelLedgerEntry serializes to dict correctly."""
        ledger_entry_id = uuid4()
        now = datetime.now(UTC)

        entry = ModelLedgerEntry(
            ledger_entry_id=ledger_entry_id,
            topic="test.events",
            partition=0,
            kafka_offset=100,
            event_value="dGVzdA==",
            ledger_written_at=now,
        )

        data = entry.model_dump()

        assert data["ledger_entry_id"] == ledger_entry_id
        assert data["topic"] == "test.events"
        assert data["partition"] == 0
        assert data["kafka_offset"] == 100
        assert data["event_value"] == "dGVzdA=="
        assert data["ledger_written_at"] == now

    def test_ledger_entry_json_round_trip(self) -> None:
        """ModelLedgerEntry survives JSON round-trip."""
        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test.events",
            partition=5,
            kafka_offset=999,
            event_key="a2V5",
            event_value="dmFsdWU=",
            onex_headers={"trace": "123"},
            envelope_id=uuid4(),
            correlation_id=uuid4(),
            event_type="test.event",
            source="test-service",
            event_timestamp=datetime.now(UTC),
            ledger_written_at=datetime.now(UTC),
        )

        json_str = entry.model_dump_json()
        restored = ModelLedgerEntry.model_validate_json(json_str)

        assert restored.ledger_entry_id == entry.ledger_entry_id
        assert restored.topic == entry.topic
        assert restored.partition == entry.partition
        assert restored.kafka_offset == entry.kafka_offset
        assert restored.event_key == entry.event_key
        assert restored.event_value == entry.event_value
        assert restored.envelope_id == entry.envelope_id
        assert restored.correlation_id == entry.correlation_id
        assert restored.event_type == entry.event_type
        assert restored.source == entry.source

    def test_append_result_json_round_trip(self) -> None:
        """ModelLedgerAppendResult survives JSON round-trip."""
        result = ModelLedgerAppendResult(
            success=True,
            ledger_entry_id=uuid4(),
            duplicate=False,
            topic="test",
            partition=3,
            kafka_offset=42,
        )

        json_str = result.model_dump_json()
        restored = ModelLedgerAppendResult.model_validate_json(json_str)

        assert restored.success == result.success
        assert restored.ledger_entry_id == result.ledger_entry_id
        assert restored.duplicate == result.duplicate
        assert restored.topic == result.topic
        assert restored.partition == result.partition
        assert restored.kafka_offset == result.kafka_offset

    def test_query_json_round_trip(self) -> None:
        """ModelLedgerQuery survives JSON round-trip."""
        query = ModelLedgerQuery(
            correlation_id=uuid4(),
            event_type="test.event",
            topic="test",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            limit=50,
            offset=10,
        )

        json_str = query.model_dump_json()
        restored = ModelLedgerQuery.model_validate_json(json_str)

        assert restored.correlation_id == query.correlation_id
        assert restored.event_type == query.event_type
        assert restored.topic == query.topic
        assert restored.limit == query.limit
        assert restored.offset == query.offset

    def test_query_result_json_round_trip(self) -> None:
        """ModelLedgerQueryResult survives JSON round-trip."""
        query = ModelLedgerQuery()
        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            ledger_written_at=datetime.now(UTC),
        )
        result = ModelLedgerQueryResult(
            entries=[entry],
            total_count=1,
            has_more=False,
            query=query,
        )

        json_str = result.model_dump_json()
        restored = ModelLedgerQueryResult.model_validate_json(json_str)

        assert restored.total_count == result.total_count
        assert restored.has_more == result.has_more
        assert len(restored.entries) == 1
        assert restored.entries[0].ledger_entry_id == entry.ledger_entry_id


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestLedgerModelEdgeCases:
    """Edge case tests for ledger models."""

    def test_entry_with_empty_onex_headers(self) -> None:
        """Empty onex_headers dict is valid."""
        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            onex_headers={},
            ledger_written_at=datetime.now(UTC),
        )

        assert entry.onex_headers == {}

    def test_entry_with_complex_onex_headers(self) -> None:
        """Complex nested onex_headers are accepted."""
        headers = {
            "x-trace-id": "abc123",
            "x-correlation-ids": ["id1", "id2"],
            "x-metadata": {"region": "us-east", "version": 2},
        }

        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            onex_headers=headers,
            ledger_written_at=datetime.now(UTC),
        )

        assert entry.onex_headers == headers

    def test_query_result_with_many_entries(self) -> None:
        """Query result with many entries is valid."""
        entries = [
            ModelLedgerEntry(
                ledger_entry_id=uuid4(),
                topic="test",
                partition=0,
                kafka_offset=i,
                event_value="dGVzdA==",
                ledger_written_at=datetime.now(UTC),
            )
            for i in range(100)
        ]

        result = ModelLedgerQueryResult(
            entries=entries,
            total_count=1000,
            has_more=True,
            query=ModelLedgerQuery(limit=100),
        )

        assert len(result.entries) == 100
        assert result.total_count == 1000

    def test_entry_with_very_long_topic(self) -> None:
        """Long topic names are accepted."""
        long_topic = "domain.subdomain.service.event.namespace.version.v1"

        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic=long_topic,
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
            ledger_written_at=datetime.now(UTC),
        )

        assert entry.topic == long_topic

    def test_entry_with_large_event_value(self) -> None:
        """Large event_value strings are accepted."""
        # Simulate a large base64 encoded payload
        large_value = "A" * 10000

        entry = ModelLedgerEntry(
            ledger_entry_id=uuid4(),
            topic="test",
            partition=0,
            kafka_offset=0,
            event_value=large_value,
            ledger_written_at=datetime.now(UTC),
        )

        assert len(entry.event_value) == 10000


__all__: list[str] = [
    "TestModelLedgerEntryConstruction",
    "TestModelLedgerEntryValidation",
    "TestModelLedgerEntryImmutability",
    "TestModelLedgerAppendResultConstruction",
    "TestModelLedgerAppendResultValidation",
    "TestModelLedgerAppendResultImmutability",
    "TestModelLedgerQueryConstruction",
    "TestModelLedgerQueryValidation",
    "TestModelLedgerQueryImmutability",
    "TestModelLedgerQueryResultConstruction",
    "TestModelLedgerQueryResultValidation",
    "TestModelLedgerQueryResultImmutability",
    "TestLedgerModelSerialization",
    "TestLedgerModelEdgeCases",
]
