# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for PostgresValidationLedgerRepository.

Tests use mocked asyncpg pool to verify:
    - Idempotent append with duplicate detection
    - Query by run_id with correct ordering
    - Flexible query building from optional filters
    - Retention cleanup with min runs per repo floor
    - Error handling with RepositoryExecutionError

Ticket: OMN-1908
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.errors.repository.errors_repository import (
    RepositoryExecutionError,
)
from omnibase_infra.models.validation_ledger import (
    ModelValidationLedgerEntry,
    ModelValidationLedgerQuery,
)
from omnibase_infra.runtime.db.postgres_validation_ledger_repository import (
    PostgresValidationLedgerRepository,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool() -> AsyncMock:
    """Create a mock asyncpg pool."""
    pool = AsyncMock()
    return pool


@pytest.fixture
def repo(mock_pool: AsyncMock) -> PostgresValidationLedgerRepository:
    """Create repository with mock pool."""
    return PostgresValidationLedgerRepository(mock_pool)


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
        envelope_bytes=base64.b64encode(b'{"test": "data"}').decode("ascii"),
        envelope_hash="abc123",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_db_row() -> dict:
    """Create a sample database row as returned by asyncpg."""
    return {
        "id": uuid4(),
        "run_id": uuid4(),
        "repo_id": "omnibase_core",
        "event_type": "onex.validation.cross_repo.run.started.v1",
        "event_version": "v1",
        "occurred_at": datetime.now(UTC),
        "kafka_topic": "onex.validation.cross_repo.run.started.v1",
        "kafka_partition": 0,
        "kafka_offset": 42,
        "envelope_bytes": base64.b64encode(b'{"test": "data"}').decode("ascii"),
        "envelope_hash": "abc123",
        "created_at": datetime.now(UTC),
    }


# =============================================================================
# TestAppend - Idempotent write tests
# =============================================================================


class TestAppend:
    """Test idempotent append operation."""

    @pytest.mark.asyncio
    async def test_append_new_entry(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
        sample_entry: ModelValidationLedgerEntry,
    ) -> None:
        """New entry returns success with entry_id."""
        new_id = uuid4()
        mock_pool.fetchrow.return_value = {"id": str(new_id)}

        result = await repo.append(sample_entry)

        assert result.success is True
        assert result.entry_id == new_id
        assert result.duplicate is False
        mock_pool.fetchrow.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_append_duplicate_entry(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
        sample_entry: ModelValidationLedgerEntry,
    ) -> None:
        """Duplicate entry returns success with duplicate=True."""
        mock_pool.fetchrow.return_value = None  # ON CONFLICT DO NOTHING

        result = await repo.append(sample_entry)

        assert result.success is True
        assert result.entry_id is None
        assert result.duplicate is True

    @pytest.mark.asyncio
    async def test_append_preserves_kafka_position(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
        sample_entry: ModelValidationLedgerEntry,
    ) -> None:
        """Kafka position is reflected in result."""
        mock_pool.fetchrow.return_value = {"id": str(uuid4())}

        result = await repo.append(sample_entry)

        assert result.kafka_topic == sample_entry.kafka_topic
        assert result.kafka_partition == sample_entry.kafka_partition
        assert result.kafka_offset == sample_entry.kafka_offset

    @pytest.mark.asyncio
    async def test_append_decodes_base64_envelope(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
        sample_entry: ModelValidationLedgerEntry,
    ) -> None:
        """Base64 envelope_bytes is decoded to BYTEA for storage."""
        mock_pool.fetchrow.return_value = {"id": str(uuid4())}

        await repo.append(sample_entry)

        # Verify the 9th argument ($9) is bytes, not base64 string
        call_args = mock_pool.fetchrow.call_args
        actual_bytes = call_args[0][9]  # 9th positional arg after SQL
        assert isinstance(actual_bytes, bytes)
        assert actual_bytes == base64.b64decode(sample_entry.envelope_bytes)

    @pytest.mark.asyncio
    async def test_append_invalid_base64_raises_error(
        self,
        repo: PostgresValidationLedgerRepository,
    ) -> None:
        """Invalid base64 in envelope_bytes raises RepositoryExecutionError."""
        entry = ModelValidationLedgerEntry(
            id=uuid4(),
            run_id=uuid4(),
            repo_id="test",
            event_type="test.v1",
            event_version="v1",
            occurred_at=datetime.now(UTC),
            kafka_topic="test.topic",
            kafka_partition=0,
            kafka_offset=0,
            envelope_bytes="!!!not-valid-base64!!!",
            envelope_hash="abc",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(RepositoryExecutionError, match="decode base64"):
            await repo.append(entry)

    @pytest.mark.asyncio
    async def test_append_db_error_raises_repository_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
        sample_entry: ModelValidationLedgerEntry,
    ) -> None:
        """Database error raises RepositoryExecutionError."""
        mock_pool.fetchrow.side_effect = Exception("Connection refused")

        with pytest.raises(RepositoryExecutionError, match="Failed to append"):
            await repo.append(sample_entry)


# =============================================================================
# TestQueryByRunId - Run ID queries
# =============================================================================


class TestQueryByRunId:
    """Test query by run_id."""

    @pytest.mark.asyncio
    async def test_query_returns_entries(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        """Query returns list of entries."""
        mock_pool.fetch.return_value = [sample_db_row]
        run_id = sample_db_row["run_id"]

        entries = await repo.query_by_run_id(run_id)

        assert len(entries) == 1
        assert isinstance(entries[0], ModelValidationLedgerEntry)
        assert entries[0].run_id == run_id

    @pytest.mark.asyncio
    async def test_query_empty_results(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
    ) -> None:
        """Query with no matches returns empty list."""
        mock_pool.fetch.return_value = []

        entries = await repo.query_by_run_id(uuid4())

        assert entries == []

    @pytest.mark.asyncio
    async def test_query_respects_limit(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
    ) -> None:
        """Limit parameter is passed to SQL query."""
        mock_pool.fetch.return_value = []

        await repo.query_by_run_id(uuid4(), limit=50)

        call_args = mock_pool.fetch.call_args
        assert call_args[0][2] == 50  # limit param

    @pytest.mark.asyncio
    async def test_query_normalizes_limit(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
    ) -> None:
        """Limit above max is capped."""
        mock_pool.fetch.return_value = []

        await repo.query_by_run_id(uuid4(), limit=99999)

        call_args = mock_pool.fetch.call_args
        assert call_args[0][2] == 10000  # capped to max

    @pytest.mark.asyncio
    async def test_query_db_error_raises_repository_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
    ) -> None:
        """Database error raises RepositoryExecutionError."""
        mock_pool.fetch.side_effect = Exception("Connection timeout")

        with pytest.raises(RepositoryExecutionError, match="Failed to query"):
            await repo.query_by_run_id(uuid4())


# =============================================================================
# TestFlexibleQuery - Dynamic query building
# =============================================================================


class TestFlexibleQuery:
    """Test flexible query with optional filters."""

    @pytest.mark.asyncio
    async def test_query_with_run_id_filter(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
    ) -> None:
        """Query with run_id filter builds correct SQL."""
        mock_pool.fetch.return_value = []
        mock_pool.fetchrow.return_value = {"total": 0}

        run_id = uuid4()
        query = ModelValidationLedgerQuery(run_id=run_id)
        result = await repo.query(query)

        assert result.total_count == 0
        assert result.entries == []
        assert result.has_more is False
        assert result.query == query

    @pytest.mark.asyncio
    async def test_query_with_repo_and_time_filters(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
    ) -> None:
        """Query with multiple filters."""
        mock_pool.fetch.return_value = []
        mock_pool.fetchrow.return_value = {"total": 0}

        query = ModelValidationLedgerQuery(
            repo_id="omnibase_core",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        result = await repo.query(query)

        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_query_has_more_flag(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        """has_more is True when more results exist."""
        mock_pool.fetch.return_value = [sample_db_row]
        mock_pool.fetchrow.return_value = {"total": 10}

        query = ModelValidationLedgerQuery(limit=1)
        result = await repo.query(query)

        assert result.has_more is True
        assert result.total_count == 10


# =============================================================================
# TestCleanupExpired - Retention cleanup
# =============================================================================


class TestCleanupExpired:
    """Test retention cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_returns_deletion_count(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
    ) -> None:
        """Cleanup returns number of deleted rows."""
        mock_pool.execute.return_value = "DELETE 5"

        deleted = await repo.cleanup_expired(
            retention_days=30,
            min_runs_per_repo=25,
        )

        assert deleted == 5

    @pytest.mark.asyncio
    async def test_cleanup_with_no_deletions(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
    ) -> None:
        """Cleanup with nothing to delete returns 0."""
        mock_pool.execute.return_value = "DELETE 0"

        deleted = await repo.cleanup_expired()

        assert deleted == 0

    @pytest.mark.asyncio
    async def test_cleanup_passes_correct_params(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
    ) -> None:
        """Cleanup passes min_runs_per_repo and max_deletions."""
        mock_pool.execute.return_value = "DELETE 0"

        await repo.cleanup_expired(
            retention_days=7,
            min_runs_per_repo=10,
            max_deletions=500,
        )

        call_args = mock_pool.execute.call_args
        assert call_args[0][1] == 10  # min_runs_per_repo
        assert call_args[0][3] == 500  # max_deletions

    @pytest.mark.asyncio
    async def test_cleanup_db_error_raises_repository_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_pool: AsyncMock,
    ) -> None:
        """Database error raises RepositoryExecutionError."""
        mock_pool.execute.side_effect = Exception("Connection lost")

        with pytest.raises(RepositoryExecutionError, match="Failed to cleanup"):
            await repo.cleanup_expired()


# =============================================================================
# TestNormalizeLimit - Limit normalization
# =============================================================================


class TestNormalizeLimit:
    """Test limit normalization helper."""

    def test_normalize_valid_limit(
        self, repo: PostgresValidationLedgerRepository
    ) -> None:
        """Valid limit passes through."""
        assert repo._normalize_limit(50) == 50

    def test_normalize_zero_limit(
        self, repo: PostgresValidationLedgerRepository
    ) -> None:
        """Zero limit returns default."""
        assert repo._normalize_limit(0) == 100

    def test_normalize_negative_limit(
        self, repo: PostgresValidationLedgerRepository
    ) -> None:
        """Negative limit returns default."""
        assert repo._normalize_limit(-1) == 100

    def test_normalize_above_max_limit(
        self, repo: PostgresValidationLedgerRepository
    ) -> None:
        """Above max returns max."""
        assert repo._normalize_limit(99999) == 10000
