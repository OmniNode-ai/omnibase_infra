# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for validation ledger Pydantic models.

Tests validate:
- ModelValidationLedgerEntry: required fields, frozen, extra=forbid, validation
- ModelValidationLedgerQuery: defaults, optional filters, bounds
- ModelValidationLedgerReplayBatch: construction, pagination metadata
- ModelValidationLedgerAppendResult: success/duplicate patterns
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

pytestmark = [pytest.mark.unit]

from omnibase_infra.models.validation_ledger import (
    ModelValidationLedgerAppendResult,
    ModelValidationLedgerEntry,
    ModelValidationLedgerQuery,
    ModelValidationLedgerReplayBatch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(**overrides: object) -> ModelValidationLedgerEntry:
    """Create a valid ModelValidationLedgerEntry with sensible defaults.

    Any keyword argument overrides the corresponding default.
    """
    defaults: dict[str, object] = {
        "id": uuid4(),
        "run_id": uuid4(),
        "repo_id": "omnibase_core",
        "event_type": "onex.evt.validation.cross-repo-run-started.v1",
        "event_version": "v1",
        "occurred_at": datetime.now(UTC),
        "kafka_topic": "validation.events",
        "kafka_partition": 0,
        "kafka_offset": 42,
        "envelope_bytes": "dGVzdA==",
        "envelope_hash": "abc123def456",
        "created_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return ModelValidationLedgerEntry(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# ModelValidationLedgerEntry
# ===========================================================================


class TestModelValidationLedgerEntryValid:
    """Tests for valid construction of ModelValidationLedgerEntry."""

    def test_valid_construction_all_required_fields(self) -> None:
        """Test creating entry with all required fields succeeds."""
        entry = _make_entry()
        assert entry.repo_id == "omnibase_core"
        assert entry.kafka_partition == 0
        assert entry.kafka_offset == 42
        assert entry.envelope_hash == "abc123def456"

    def test_uuid_fields_accept_valid_uuids(self) -> None:
        """Test that id and run_id accept valid UUID values."""
        entry_id = uuid4()
        run_id = uuid4()
        entry = _make_entry(id=entry_id, run_id=run_id)
        assert entry.id == entry_id
        assert entry.run_id == run_id

    def test_datetime_fields_accept_timezone_aware(self) -> None:
        """Test that datetime fields accept timezone-aware datetimes."""
        now = datetime.now(UTC)
        entry = _make_entry(occurred_at=now, created_at=now)
        assert entry.occurred_at == now
        assert entry.created_at == now


class TestModelValidationLedgerEntryImmutability:
    """Tests for frozen=True on ModelValidationLedgerEntry."""

    def test_frozen_raises_on_field_assignment(self) -> None:
        """Test that assigning to any field on a frozen model raises TypeError."""
        entry = _make_entry()
        with pytest.raises(ValidationError):
            entry.repo_id = "another_repo"  # type: ignore[misc]

    def test_frozen_raises_on_kafka_partition_assignment(self) -> None:
        """Test that assigning kafka_partition raises on frozen model."""
        entry = _make_entry()
        with pytest.raises(ValidationError):
            entry.kafka_partition = 99  # type: ignore[misc]


class TestModelValidationLedgerEntryExtraForbid:
    """Tests for extra=forbid on ModelValidationLedgerEntry."""

    def test_extra_field_rejected(self) -> None:
        """Test that an unknown extra field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _make_entry(extra_field="not_allowed")
        assert "extra_field" in str(exc_info.value)


class TestModelValidationLedgerEntryValidation:
    """Tests for field-level validation on ModelValidationLedgerEntry."""

    def test_repo_id_min_length_rejects_empty(self) -> None:
        """Test that repo_id with min_length=1 rejects empty string."""
        with pytest.raises(ValidationError) as exc_info:
            _make_entry(repo_id="")
        assert "repo_id" in str(exc_info.value)

    def test_event_type_min_length_rejects_empty(self) -> None:
        """Test that event_type with min_length=1 rejects empty string."""
        with pytest.raises(ValidationError) as exc_info:
            _make_entry(event_type="")
        assert "event_type" in str(exc_info.value)

    def test_event_version_min_length_rejects_empty(self) -> None:
        """Test that event_version with min_length=1 rejects empty string."""
        with pytest.raises(ValidationError) as exc_info:
            _make_entry(event_version="")
        assert "event_version" in str(exc_info.value)

    def test_kafka_topic_min_length_rejects_empty(self) -> None:
        """Test that kafka_topic with min_length=1 rejects empty string."""
        with pytest.raises(ValidationError) as exc_info:
            _make_entry(kafka_topic="")
        assert "kafka_topic" in str(exc_info.value)

    def test_envelope_hash_min_length_rejects_empty(self) -> None:
        """Test that envelope_hash with min_length=1 rejects empty string."""
        with pytest.raises(ValidationError) as exc_info:
            _make_entry(envelope_hash="")
        assert "envelope_hash" in str(exc_info.value)

    def test_kafka_partition_ge_zero_rejects_negative(self) -> None:
        """Test that kafka_partition with ge=0 rejects negative values."""
        with pytest.raises(ValidationError) as exc_info:
            _make_entry(kafka_partition=-1)
        assert "kafka_partition" in str(exc_info.value)

    def test_kafka_offset_ge_zero_rejects_negative(self) -> None:
        """Test that kafka_offset with ge=0 rejects negative values."""
        with pytest.raises(ValidationError) as exc_info:
            _make_entry(kafka_offset=-1)
        assert "kafka_offset" in str(exc_info.value)

    def test_kafka_partition_zero_allowed(self) -> None:
        """Test that kafka_partition=0 is allowed (boundary)."""
        entry = _make_entry(kafka_partition=0)
        assert entry.kafka_partition == 0

    def test_kafka_offset_zero_allowed(self) -> None:
        """Test that kafka_offset=0 is allowed (boundary)."""
        entry = _make_entry(kafka_offset=0)
        assert entry.kafka_offset == 0

    def test_missing_required_field_raises(self) -> None:
        """Test that omitting a required field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelValidationLedgerEntry(
                id=uuid4(),
                run_id=uuid4(),
                # repo_id omitted
                event_type="evt",
                event_version="v1",
                occurred_at=datetime.now(UTC),
                kafka_topic="t",
                kafka_partition=0,
                kafka_offset=0,
                envelope_bytes="dGVzdA==",
                envelope_hash="h",
                created_at=datetime.now(UTC),
            )  # type: ignore[call-arg]
        assert "repo_id" in str(exc_info.value)


class TestModelValidationLedgerEntryFromAttributes:
    """Tests for from_attributes=True ORM support."""

    def test_from_dict_like_object(self) -> None:
        """Test that model can be created from an object with attributes."""
        now = datetime.now(UTC)
        entry_id = uuid4()
        run_id = uuid4()

        class Row:
            def __init__(self) -> None:
                self.id = entry_id
                self.run_id = run_id
                self.repo_id = "omnibase_core"
                self.event_type = "started"
                self.event_version = "v1"
                self.occurred_at = now
                self.kafka_topic = "t"
                self.kafka_partition = 0
                self.kafka_offset = 0
                self.envelope_bytes = "dGVzdA=="
                self.envelope_hash = "h"
                self.created_at = now

        entry = ModelValidationLedgerEntry.model_validate(Row())
        assert entry.id == entry_id
        assert entry.run_id == run_id


# ===========================================================================
# ModelValidationLedgerQuery
# ===========================================================================


class TestModelValidationLedgerQueryDefaults:
    """Tests for ModelValidationLedgerQuery default values."""

    def test_all_defaults(self) -> None:
        """Test that default construction sets all filters to None with pagination defaults."""
        query = ModelValidationLedgerQuery()
        assert query.run_id is None
        assert query.repo_id is None
        assert query.event_type is None
        assert query.start_time is None
        assert query.end_time is None
        assert query.limit == 100
        assert query.offset == 0


class TestModelValidationLedgerQueryFilters:
    """Tests for setting individual filter fields on ModelValidationLedgerQuery."""

    def test_set_run_id(self) -> None:
        """Test that run_id filter can be set independently."""
        run_id = uuid4()
        query = ModelValidationLedgerQuery(run_id=run_id)
        assert query.run_id == run_id
        assert query.repo_id is None

    def test_set_repo_id(self) -> None:
        """Test that repo_id filter can be set independently."""
        query = ModelValidationLedgerQuery(repo_id="omnibase_core")
        assert query.repo_id == "omnibase_core"
        assert query.run_id is None

    def test_set_event_type(self) -> None:
        """Test that event_type filter can be set independently."""
        query = ModelValidationLedgerQuery(event_type="run.started")
        assert query.event_type == "run.started"

    def test_set_time_range(self) -> None:
        """Test that start_time and end_time can be set."""
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 2, 1, tzinfo=UTC)
        query = ModelValidationLedgerQuery(start_time=start, end_time=end)
        assert query.start_time == start
        assert query.end_time == end


class TestModelValidationLedgerQueryValidation:
    """Tests for validation on ModelValidationLedgerQuery."""

    def test_frozen_raises_on_assignment(self) -> None:
        """Test that query is frozen."""
        query = ModelValidationLedgerQuery()
        with pytest.raises(ValidationError):
            query.limit = 50  # type: ignore[misc]

    def test_extra_field_rejected(self) -> None:
        """Test that unknown extra fields are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelValidationLedgerQuery(extra_field="nope")  # type: ignore[call-arg]
        assert "extra_field" in str(exc_info.value)

    def test_limit_ge_1_rejects_zero(self) -> None:
        """Test that limit with ge=1 rejects zero."""
        with pytest.raises(ValidationError) as exc_info:
            ModelValidationLedgerQuery(limit=0)
        assert "limit" in str(exc_info.value)

    def test_limit_le_10000_rejects_too_large(self) -> None:
        """Test that limit with le=10000 rejects values over 10000."""
        with pytest.raises(ValidationError) as exc_info:
            ModelValidationLedgerQuery(limit=10001)
        assert "limit" in str(exc_info.value)

    def test_offset_ge_0_rejects_negative(self) -> None:
        """Test that offset with ge=0 rejects negative values."""
        with pytest.raises(ValidationError) as exc_info:
            ModelValidationLedgerQuery(offset=-1)
        assert "offset" in str(exc_info.value)

    def test_limit_boundary_values_accepted(self) -> None:
        """Test that limit=1 and limit=10000 are accepted."""
        q1 = ModelValidationLedgerQuery(limit=1)
        assert q1.limit == 1
        q2 = ModelValidationLedgerQuery(limit=10000)
        assert q2.limit == 10000

    def test_offset_zero_accepted(self) -> None:
        """Test that offset=0 is accepted (boundary)."""
        query = ModelValidationLedgerQuery(offset=0)
        assert query.offset == 0


# ===========================================================================
# ModelValidationLedgerReplayBatch
# ===========================================================================


class TestModelValidationLedgerReplayBatch:
    """Tests for ModelValidationLedgerReplayBatch construction and fields."""

    def test_construction_with_entries(self) -> None:
        """Test constructing replay batch with entries."""
        entry = _make_entry()
        query = ModelValidationLedgerQuery(run_id=entry.run_id)
        batch = ModelValidationLedgerReplayBatch(
            entries=[entry],
            total_count=1,
            has_more=False,
            query=query,
        )
        assert len(batch.entries) == 1
        assert batch.total_count == 1
        assert batch.has_more is False
        assert batch.query == query

    def test_construction_with_empty_entries(self) -> None:
        """Test that empty entries list is valid."""
        query = ModelValidationLedgerQuery()
        batch = ModelValidationLedgerReplayBatch(
            entries=[],
            total_count=0,
            has_more=False,
            query=query,
        )
        assert len(batch.entries) == 0
        assert batch.total_count == 0

    def test_has_more_true(self) -> None:
        """Test that has_more=True is valid when more entries exist."""
        query = ModelValidationLedgerQuery(limit=10)
        batch = ModelValidationLedgerReplayBatch(
            entries=[],
            total_count=100,
            has_more=True,
            query=query,
        )
        assert batch.has_more is True

    def test_frozen(self) -> None:
        """Test that replay batch is frozen."""
        query = ModelValidationLedgerQuery()
        batch = ModelValidationLedgerReplayBatch(
            entries=[],
            total_count=0,
            has_more=False,
            query=query,
        )
        with pytest.raises(ValidationError):
            batch.total_count = 99  # type: ignore[misc]

    def test_extra_field_rejected(self) -> None:
        """Test that extra fields are rejected."""
        query = ModelValidationLedgerQuery()
        with pytest.raises(ValidationError):
            ModelValidationLedgerReplayBatch(
                entries=[],
                total_count=0,
                has_more=False,
                query=query,
                extra="nope",  # type: ignore[call-arg]
            )

    def test_total_count_ge_zero_rejects_negative(self) -> None:
        """Test that total_count with ge=0 rejects negative values."""
        query = ModelValidationLedgerQuery()
        with pytest.raises(ValidationError) as exc_info:
            ModelValidationLedgerReplayBatch(
                entries=[],
                total_count=-1,
                has_more=False,
                query=query,
            )
        assert "total_count" in str(exc_info.value)

    def test_default_entries_is_empty_tuple(self) -> None:
        """Test that entries defaults to an empty tuple."""
        query = ModelValidationLedgerQuery()
        batch = ModelValidationLedgerReplayBatch(
            total_count=0,
            has_more=False,
            query=query,
        )
        assert batch.entries == ()
        assert isinstance(batch.entries, tuple)


# ===========================================================================
# ModelValidationLedgerAppendResult
# ===========================================================================


class TestModelValidationLedgerAppendResult:
    """Tests for ModelValidationLedgerAppendResult construction."""

    def test_successful_append(self) -> None:
        """Test successful append result with entry ID."""
        entry_id = uuid4()
        result = ModelValidationLedgerAppendResult(
            success=True,
            ledger_entry_id=entry_id,
            duplicate=False,
            kafka_topic="validation.events",
            kafka_partition=0,
            kafka_offset=42,
        )
        assert result.success is True
        assert result.ledger_entry_id == entry_id
        assert result.duplicate is False
        assert result.kafka_topic == "validation.events"
        assert result.kafka_partition == 0
        assert result.kafka_offset == 42

    def test_duplicate_append(self) -> None:
        """Test duplicate append result with None entry ID."""
        result = ModelValidationLedgerAppendResult(
            success=True,
            ledger_entry_id=None,
            duplicate=True,
            kafka_topic="validation.events",
            kafka_partition=0,
            kafka_offset=42,
        )
        assert result.success is True
        assert result.ledger_entry_id is None
        assert result.duplicate is True

    def test_frozen(self) -> None:
        """Test that append result is frozen."""
        result = ModelValidationLedgerAppendResult(
            success=True,
            kafka_topic="t",
            kafka_partition=0,
            kafka_offset=0,
        )
        with pytest.raises(ValidationError):
            result.success = False  # type: ignore[misc]

    def test_extra_field_rejected(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelValidationLedgerAppendResult(
                success=True,
                kafka_topic="t",
                kafka_partition=0,
                kafka_offset=0,
                extra="nope",  # type: ignore[call-arg]
            )

    def test_kafka_partition_ge_zero_rejects_negative(self) -> None:
        """Test that kafka_partition with ge=0 rejects negative."""
        with pytest.raises(ValidationError) as exc_info:
            ModelValidationLedgerAppendResult(
                success=True,
                kafka_topic="t",
                kafka_partition=-1,
                kafka_offset=0,
            )
        assert "kafka_partition" in str(exc_info.value)

    def test_kafka_offset_ge_zero_rejects_negative(self) -> None:
        """Test that kafka_offset with ge=0 rejects negative."""
        with pytest.raises(ValidationError) as exc_info:
            ModelValidationLedgerAppendResult(
                success=True,
                kafka_topic="t",
                kafka_partition=0,
                kafka_offset=-1,
            )
        assert "kafka_offset" in str(exc_info.value)

    def test_default_duplicate_is_false(self) -> None:
        """Test that duplicate defaults to False."""
        result = ModelValidationLedgerAppendResult(
            success=True,
            kafka_topic="t",
            kafka_partition=0,
            kafka_offset=0,
        )
        assert result.duplicate is False

    def test_default_ledger_entry_id_is_none(self) -> None:
        """Test that ledger_entry_id defaults to None."""
        result = ModelValidationLedgerAppendResult(
            success=True,
            kafka_topic="t",
            kafka_partition=0,
            kafka_offset=0,
        )
        assert result.ledger_entry_id is None
