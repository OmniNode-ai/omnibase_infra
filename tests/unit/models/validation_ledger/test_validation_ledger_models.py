# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for validation ledger domain models.

Tests cover:
    - ModelValidationLedgerEntry: field validation, immutability, serialization
    - ModelValidationLedgerQuery: filter combinations, pagination defaults
    - ModelValidationLedgerReplayBatch: structure, has_more logic
    - ModelValidationLedgerAppendResult: success/duplicate states

Ticket: OMN-1908
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.models.validation_ledger import (
    ModelValidationLedgerAppendResult,
    ModelValidationLedgerEntry,
    ModelValidationLedgerQuery,
    ModelValidationLedgerReplayBatch,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_entry() -> ModelValidationLedgerEntry:
    """Create a sample validation ledger entry."""
    return ModelValidationLedgerEntry(
        id=uuid4(),
        run_id=uuid4(),
        repo_id="omnibase_core",
        event_type="onex.validation.cross_repo.run.started.v1",
        event_version="v1",
        occurred_at=datetime.now(UTC),
        kafka_topic="onex.validation.cross_repo.run.started.v1",
        kafka_partition=0,
        kafka_offset=42,
        envelope_bytes="eyJ0ZXN0IjogImRhdGEifQ==",  # base64 of {"test": "data"}
        envelope_hash="a1b2c3d4e5f6",
        created_at=datetime.now(UTC),
    )


# =============================================================================
# TestModelValidationLedgerEntry
# =============================================================================


class TestModelValidationLedgerEntry:
    """Tests for ModelValidationLedgerEntry."""

    def test_create_valid_entry(self, sample_entry: ModelValidationLedgerEntry) -> None:
        """Valid entry creates successfully."""
        assert sample_entry.repo_id == "omnibase_core"
        assert sample_entry.kafka_partition == 0
        assert sample_entry.kafka_offset == 42

    def test_entry_is_frozen(self, sample_entry: ModelValidationLedgerEntry) -> None:
        """Entry is immutable (frozen=True)."""
        with pytest.raises((TypeError, ValidationError)):
            sample_entry.repo_id = "new_repo"  # type: ignore[misc]

    def test_entry_forbids_extra_fields(self) -> None:
        """Extra fields are rejected (extra='forbid')."""
        with pytest.raises(ValidationError):
            ModelValidationLedgerEntry(
                id=uuid4(),
                run_id=uuid4(),
                repo_id="test",
                event_type="test.v1",
                event_version="v1",
                occurred_at=datetime.now(UTC),
                kafka_topic="test.topic",
                kafka_partition=0,
                kafka_offset=0,
                envelope_bytes="dGVzdA==",
                envelope_hash="abc",
                created_at=datetime.now(UTC),
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

    def test_entry_requires_non_empty_repo_id(self) -> None:
        """repo_id must be non-empty."""
        with pytest.raises(ValidationError):
            ModelValidationLedgerEntry(
                id=uuid4(),
                run_id=uuid4(),
                repo_id="",
                event_type="test.v1",
                event_version="v1",
                occurred_at=datetime.now(UTC),
                kafka_topic="test.topic",
                kafka_partition=0,
                kafka_offset=0,
                envelope_bytes="dGVzdA==",
                envelope_hash="abc",
                created_at=datetime.now(UTC),
            )

    def test_entry_requires_non_negative_partition(self) -> None:
        """kafka_partition must be >= 0."""
        with pytest.raises(ValidationError):
            ModelValidationLedgerEntry(
                id=uuid4(),
                run_id=uuid4(),
                repo_id="test",
                event_type="test.v1",
                event_version="v1",
                occurred_at=datetime.now(UTC),
                kafka_topic="test.topic",
                kafka_partition=-1,
                kafka_offset=0,
                envelope_bytes="dGVzdA==",
                envelope_hash="abc",
                created_at=datetime.now(UTC),
            )

    def test_entry_requires_non_negative_offset(self) -> None:
        """kafka_offset must be >= 0."""
        with pytest.raises(ValidationError):
            ModelValidationLedgerEntry(
                id=uuid4(),
                run_id=uuid4(),
                repo_id="test",
                event_type="test.v1",
                event_version="v1",
                occurred_at=datetime.now(UTC),
                kafka_topic="test.topic",
                kafka_partition=0,
                kafka_offset=-1,
                envelope_bytes="dGVzdA==",
                envelope_hash="abc",
                created_at=datetime.now(UTC),
            )

    def test_entry_serialization_roundtrip(
        self, sample_entry: ModelValidationLedgerEntry
    ) -> None:
        """Entry survives JSON serialization roundtrip."""
        data = sample_entry.model_dump(mode="json")
        restored = ModelValidationLedgerEntry.model_validate(data)
        assert restored == sample_entry

    def test_entry_from_attributes(self) -> None:
        """Entry can be created from ORM-like attribute access."""
        # Simulates from_attributes=True behavior
        data = {
            "id": uuid4(),
            "run_id": uuid4(),
            "repo_id": "test_repo",
            "event_type": "test.event.v1",
            "event_version": "v1",
            "occurred_at": datetime.now(UTC),
            "kafka_topic": "test.topic",
            "kafka_partition": 0,
            "kafka_offset": 100,
            "envelope_bytes": "dGVzdA==",
            "envelope_hash": "hash123",
            "created_at": datetime.now(UTC),
        }
        entry = ModelValidationLedgerEntry.model_validate(data)
        assert entry.repo_id == "test_repo"


# =============================================================================
# TestModelValidationLedgerQuery
# =============================================================================


class TestModelValidationLedgerQuery:
    """Tests for ModelValidationLedgerQuery."""

    def test_default_query_has_no_filters(self) -> None:
        """Default query has all filters as None."""
        query = ModelValidationLedgerQuery()
        assert query.run_id is None
        assert query.repo_id is None
        assert query.event_type is None
        assert query.start_time is None
        assert query.end_time is None

    def test_default_pagination_values(self) -> None:
        """Default pagination: limit=100, offset=0."""
        query = ModelValidationLedgerQuery()
        assert query.limit == 100
        assert query.offset == 0

    def test_query_with_run_id_filter(self) -> None:
        """Query with run_id filter."""
        run_id = uuid4()
        query = ModelValidationLedgerQuery(run_id=run_id)
        assert query.run_id == run_id

    def test_query_with_all_filters(self) -> None:
        """Query with all filters set."""
        now = datetime.now(UTC)
        query = ModelValidationLedgerQuery(
            run_id=uuid4(),
            repo_id="omnibase_core",
            event_type="onex.validation.cross_repo.run.started.v1",
            start_time=now,
            end_time=now,
            limit=50,
            offset=10,
        )
        assert query.repo_id == "omnibase_core"
        assert query.limit == 50
        assert query.offset == 10

    def test_query_limit_minimum_is_1(self) -> None:
        """Limit must be >= 1."""
        with pytest.raises(ValidationError):
            ModelValidationLedgerQuery(limit=0)

    def test_query_limit_maximum_is_10000(self) -> None:
        """Limit must be <= 10000."""
        with pytest.raises(ValidationError):
            ModelValidationLedgerQuery(limit=10001)

    def test_query_offset_minimum_is_0(self) -> None:
        """Offset must be >= 0."""
        with pytest.raises(ValidationError):
            ModelValidationLedgerQuery(offset=-1)

    def test_query_is_frozen(self) -> None:
        """Query is immutable."""
        query = ModelValidationLedgerQuery()
        with pytest.raises((TypeError, ValidationError)):
            query.limit = 50  # type: ignore[misc]


# =============================================================================
# TestModelValidationLedgerReplayBatch
# =============================================================================


class TestModelValidationLedgerReplayBatch:
    """Tests for ModelValidationLedgerReplayBatch."""

    def test_create_empty_batch(self) -> None:
        """Empty batch with zero results."""
        query = ModelValidationLedgerQuery()
        batch = ModelValidationLedgerReplayBatch(
            entries=[],
            total_count=0,
            has_more=False,
            query=query,
        )
        assert len(batch.entries) == 0
        assert batch.total_count == 0
        assert batch.has_more is False

    def test_create_batch_with_entries(
        self, sample_entry: ModelValidationLedgerEntry
    ) -> None:
        """Batch with entries."""
        query = ModelValidationLedgerQuery()
        batch = ModelValidationLedgerReplayBatch(
            entries=[sample_entry],
            total_count=1,
            has_more=False,
            query=query,
        )
        assert len(batch.entries) == 1
        assert batch.entries[0] == sample_entry

    def test_batch_has_more_flag(
        self, sample_entry: ModelValidationLedgerEntry
    ) -> None:
        """has_more indicates more results available."""
        query = ModelValidationLedgerQuery(limit=1)
        batch = ModelValidationLedgerReplayBatch(
            entries=[sample_entry],
            total_count=10,
            has_more=True,
            query=query,
        )
        assert batch.has_more is True
        assert batch.total_count == 10

    def test_batch_preserves_query(self) -> None:
        """Original query is preserved in batch."""
        run_id = uuid4()
        query = ModelValidationLedgerQuery(run_id=run_id, limit=50)
        batch = ModelValidationLedgerReplayBatch(
            entries=[],
            total_count=0,
            has_more=False,
            query=query,
        )
        assert batch.query.run_id == run_id
        assert batch.query.limit == 50

    def test_batch_is_frozen(self, sample_entry: ModelValidationLedgerEntry) -> None:
        """Batch is immutable."""
        query = ModelValidationLedgerQuery()
        batch = ModelValidationLedgerReplayBatch(
            entries=[sample_entry],
            total_count=1,
            has_more=False,
            query=query,
        )
        with pytest.raises((TypeError, ValidationError)):
            batch.total_count = 99  # type: ignore[misc]


# =============================================================================
# TestModelValidationLedgerAppendResult
# =============================================================================


class TestModelValidationLedgerAppendResult:
    """Tests for ModelValidationLedgerAppendResult."""

    def test_successful_append(self) -> None:
        """Successful append with entry_id."""
        result = ModelValidationLedgerAppendResult(
            success=True,
            entry_id=uuid4(),
            duplicate=False,
            kafka_topic="onex.validation.cross_repo.run.started.v1",
            kafka_partition=0,
            kafka_offset=42,
        )
        assert result.success is True
        assert result.entry_id is not None
        assert result.duplicate is False

    def test_duplicate_append(self) -> None:
        """Duplicate append with no entry_id."""
        result = ModelValidationLedgerAppendResult(
            success=True,
            entry_id=None,
            duplicate=True,
            kafka_topic="onex.validation.cross_repo.run.started.v1",
            kafka_partition=0,
            kafka_offset=42,
        )
        assert result.success is True
        assert result.entry_id is None
        assert result.duplicate is True

    def test_result_is_frozen(self) -> None:
        """Result is immutable."""
        result = ModelValidationLedgerAppendResult(
            success=True,
            duplicate=False,
            kafka_topic="test.topic",
            kafka_partition=0,
            kafka_offset=0,
        )
        with pytest.raises((TypeError, ValidationError)):
            result.success = False  # type: ignore[misc]
