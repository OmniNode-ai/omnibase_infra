# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for PostgresValidationLedgerRepository.

Tests validate repository operations with MOCKED asyncpg:
- append: successful insert, duplicate detection, error handling
- query_by_run_id: entry conversion, ordering
- query: dynamic WHERE building, pagination, has_more logic
- cleanup_expired: batched deletion, protected runs
- Error mapping: PostgresError -> RepositoryExecutionError,
  QueryCanceledError -> RepositoryTimeoutError
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import asyncpg
import pytest

from omnibase_infra.errors.repository import (
    RepositoryExecutionError,
    RepositoryTimeoutError,
)
from omnibase_infra.models.validation_ledger import (
    ModelValidationLedgerEntry,
    ModelValidationLedgerQuery,
)
from omnibase_infra.runtime.db.postgres_validation_ledger_repository import (
    PostgresValidationLedgerRepository,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg pool with context-manager acquire."""
    pool = MagicMock(spec=asyncpg.Pool)
    connection = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


@pytest.fixture
def mock_connection(mock_pool: MagicMock) -> AsyncMock:
    """Return the mock connection obtained from pool.acquire()."""
    return mock_pool.acquire.return_value.__aenter__.return_value


@pytest.fixture
def repo(mock_pool: MagicMock) -> PostgresValidationLedgerRepository:
    """Create a repository instance with the mock pool."""
    return PostgresValidationLedgerRepository(mock_pool)


def _make_row(**overrides: object) -> dict[str, object]:
    """Create a dict mimicking an asyncpg Record for a ledger row."""
    defaults: dict[str, object] = {
        "id": uuid4(),
        "run_id": uuid4(),
        "repo_id": "omnibase_core",
        "event_type": "onex.validation.cross_repo.run.started.v1",
        "event_version": "v1",
        "occurred_at": datetime.now(UTC),
        "kafka_topic": "validation.events",
        "kafka_partition": 0,
        "kafka_offset": 42,
        "envelope_bytes": "dGVzdA==",
        "envelope_hash": "abc123",
        "created_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return defaults


def _append_kwargs(**overrides: object) -> dict[str, object]:
    """Create keyword arguments for repo.append()."""
    defaults: dict[str, object] = {
        "run_id": uuid4(),
        "repo_id": "omnibase_core",
        "event_type": "run.started",
        "event_version": "v1",
        "occurred_at": datetime.now(UTC),
        "kafka_topic": "validation.events",
        "kafka_partition": 0,
        "kafka_offset": 42,
        "envelope_bytes": b"raw-bytes",
        "envelope_hash": "sha256:abc",
    }
    defaults.update(overrides)
    return defaults


# ===========================================================================
# append
# ===========================================================================


class TestAppend:
    """Tests for PostgresValidationLedgerRepository.append()."""

    @pytest.mark.asyncio
    async def test_successful_insert(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that a successful INSERT returns success=True, duplicate=False."""
        entry_id = uuid4()
        mock_connection.fetchrow = AsyncMock(return_value={"id": entry_id})

        result = await repo.append(**_append_kwargs())

        assert result.success is True
        assert result.duplicate is False
        assert result.ledger_entry_id == entry_id
        mock_connection.fetchrow.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duplicate_returns_none_entry_id(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that ON CONFLICT (duplicate) returns success=True, duplicate=True, id=None."""
        mock_connection.fetchrow = AsyncMock(return_value=None)

        result = await repo.append(**_append_kwargs())

        assert result.success is True
        assert result.duplicate is True
        assert result.ledger_entry_id is None

    @pytest.mark.asyncio
    async def test_append_passes_kafka_fields_to_result(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that kafka_topic, kafka_partition, kafka_offset are in the result."""
        mock_connection.fetchrow = AsyncMock(return_value={"id": uuid4()})

        kwargs = _append_kwargs(
            kafka_topic="my.topic",
            kafka_partition=3,
            kafka_offset=999,
        )
        result = await repo.append(**kwargs)

        assert result.kafka_topic == "my.topic"
        assert result.kafka_partition == 3
        assert result.kafka_offset == 999

    @pytest.mark.asyncio
    async def test_postgres_error_raises_repository_execution_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that asyncpg.PostgresError maps to RepositoryExecutionError."""
        mock_connection.fetchrow = AsyncMock(
            side_effect=asyncpg.PostgresError("db error")
        )

        with pytest.raises(RepositoryExecutionError):
            await repo.append(**_append_kwargs())

    @pytest.mark.asyncio
    async def test_query_canceled_raises_repository_timeout_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that asyncpg.QueryCanceledError maps to RepositoryTimeoutError."""
        mock_connection.fetchrow = AsyncMock(
            side_effect=asyncpg.QueryCanceledError("timeout")
        )

        with pytest.raises(RepositoryTimeoutError):
            await repo.append(**_append_kwargs())


# ===========================================================================
# query_by_run_id
# ===========================================================================


class TestQueryByRunId:
    """Tests for PostgresValidationLedgerRepository.query_by_run_id()."""

    @pytest.mark.asyncio
    async def test_returns_entries(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that rows are converted to ModelValidationLedgerEntry instances."""
        row1 = _make_row(kafka_offset=0)
        row2 = _make_row(kafka_offset=1)
        mock_connection.fetch = AsyncMock(return_value=[row1, row2])

        run_id = uuid4()
        entries = await repo.query_by_run_id(run_id)

        assert len(entries) == 2
        assert all(isinstance(e, ModelValidationLedgerEntry) for e in entries)

    @pytest.mark.asyncio
    async def test_empty_result(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that empty result set returns empty list."""
        mock_connection.fetch = AsyncMock(return_value=[])

        entries = await repo.query_by_run_id(uuid4())

        assert entries == []

    @pytest.mark.asyncio
    async def test_passes_limit_and_offset(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that limit and offset are passed to the query."""
        mock_connection.fetch = AsyncMock(return_value=[])

        run_id = uuid4()
        await repo.query_by_run_id(run_id, limit=50, offset=10)

        # Verify the call included run_id, limit, and offset args
        call_args = mock_connection.fetch.call_args
        positional = call_args[0]
        # SQL is first arg, then run_id, limit, offset
        assert positional[1] == run_id
        assert positional[2] == 50
        assert positional[3] == 10

    @pytest.mark.asyncio
    async def test_postgres_error_raises_execution_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that PostgresError maps to RepositoryExecutionError."""
        mock_connection.fetch = AsyncMock(side_effect=asyncpg.PostgresError("fail"))

        with pytest.raises(RepositoryExecutionError):
            await repo.query_by_run_id(uuid4())

    @pytest.mark.asyncio
    async def test_query_canceled_raises_timeout_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that QueryCanceledError maps to RepositoryTimeoutError."""
        mock_connection.fetch = AsyncMock(
            side_effect=asyncpg.QueryCanceledError("timeout")
        )

        with pytest.raises(RepositoryTimeoutError):
            await repo.query_by_run_id(uuid4())


# ===========================================================================
# query (flexible filters)
# ===========================================================================


class TestQueryFlexible:
    """Tests for PostgresValidationLedgerRepository.query()."""

    @pytest.mark.asyncio
    async def test_query_with_run_id_filter(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that run_id filter produces a WHERE clause."""
        mock_connection.fetchrow = AsyncMock(return_value={"total": 0})
        mock_connection.fetch = AsyncMock(return_value=[])

        run_id = uuid4()
        q = ModelValidationLedgerQuery(run_id=run_id)
        batch = await repo.query(q)

        assert batch.total_count == 0
        assert batch.has_more is False
        assert batch.query == q
        # Verify fetchrow (count) was called with the run_id parameter
        count_call = mock_connection.fetchrow.call_args
        assert run_id in count_call[0]

    @pytest.mark.asyncio
    async def test_query_with_multiple_filters(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that multiple filters are combined with AND."""
        mock_connection.fetchrow = AsyncMock(return_value={"total": 5})
        mock_connection.fetch = AsyncMock(return_value=[])

        run_id = uuid4()
        q = ModelValidationLedgerQuery(
            run_id=run_id,
            repo_id="omnibase_core",
            event_type="run.started",
        )
        batch = await repo.query(q)

        assert batch.total_count == 5
        # SQL should contain run_id, repo_id, event_type as parameters
        count_call = mock_connection.fetchrow.call_args
        assert run_id in count_call[0]
        assert "omnibase_core" in count_call[0]
        assert "run.started" in count_call[0]

    @pytest.mark.asyncio
    async def test_query_pagination_has_more_true(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that has_more is True when offset + limit < total_count."""
        mock_connection.fetchrow = AsyncMock(return_value={"total": 150})
        mock_connection.fetch = AsyncMock(return_value=[])

        q = ModelValidationLedgerQuery(limit=100, offset=0)
        batch = await repo.query(q)

        assert batch.total_count == 150
        assert batch.has_more is True

    @pytest.mark.asyncio
    async def test_query_pagination_has_more_false(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that has_more is False when offset + limit >= total_count."""
        mock_connection.fetchrow = AsyncMock(return_value={"total": 150})
        mock_connection.fetch = AsyncMock(return_value=[])

        q = ModelValidationLedgerQuery(limit=100, offset=100)
        batch = await repo.query(q)

        assert batch.total_count == 150
        assert batch.has_more is False

    @pytest.mark.asyncio
    async def test_query_converts_rows_to_entries(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that returned rows are converted to entry models."""
        row = _make_row()
        mock_connection.fetchrow = AsyncMock(return_value={"total": 1})
        mock_connection.fetch = AsyncMock(return_value=[row])

        q = ModelValidationLedgerQuery()
        batch = await repo.query(q)

        assert len(batch.entries) == 1
        assert isinstance(batch.entries[0], ModelValidationLedgerEntry)

    @pytest.mark.asyncio
    async def test_query_postgres_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that PostgresError maps to RepositoryExecutionError."""
        mock_connection.fetchrow = AsyncMock(side_effect=asyncpg.PostgresError("fail"))

        with pytest.raises(RepositoryExecutionError):
            await repo.query(ModelValidationLedgerQuery())

    @pytest.mark.asyncio
    async def test_query_canceled_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that QueryCanceledError maps to RepositoryTimeoutError."""
        mock_connection.fetchrow = AsyncMock(
            side_effect=asyncpg.QueryCanceledError("timeout")
        )

        with pytest.raises(RepositoryTimeoutError):
            await repo.query(ModelValidationLedgerQuery())

    @pytest.mark.asyncio
    async def test_query_no_filters_produces_no_where(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that an empty query produces no WHERE parameters for count."""
        mock_connection.fetchrow = AsyncMock(return_value={"total": 0})
        mock_connection.fetch = AsyncMock(return_value=[])

        q = ModelValidationLedgerQuery()
        await repo.query(q)

        # Count query should only have the SQL string, no WHERE params
        count_call = mock_connection.fetchrow.call_args
        # First positional arg is the SQL
        assert "validation_event_ledger" in count_call[0][0]
        # No extra positional args besides the SQL (no filter params)
        assert len(count_call[0]) == 1


# ===========================================================================
# cleanup_expired
# ===========================================================================


class TestCleanupExpired:
    """Tests for PostgresValidationLedgerRepository.cleanup_expired()."""

    @pytest.mark.asyncio
    async def test_cleanup_returns_total_deleted(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that cleanup returns the total number of deleted entries."""
        # Phase 1: protected runs query
        mock_connection.fetch = AsyncMock(return_value=[{"run_id": uuid4()}])
        # Phase 2: batched delete (returns fewer than batch_size => done)
        mock_connection.execute = AsyncMock(return_value="DELETE 50")

        total = await repo.cleanup_expired(
            retention_days=30,
            min_runs_per_repo=5,
            batch_size=1000,
        )

        assert total == 50

    @pytest.mark.asyncio
    async def test_cleanup_multiple_batches(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that cleanup loops when batch_size records are deleted."""
        mock_connection.fetch = AsyncMock(return_value=[])
        # First batch deletes batch_size, second deletes fewer => stop
        mock_connection.execute = AsyncMock(side_effect=["DELETE 100", "DELETE 30"])

        total = await repo.cleanup_expired(
            retention_days=7,
            min_runs_per_repo=3,
            batch_size=100,
        )

        assert total == 130
        assert mock_connection.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_no_rows_to_delete(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that cleanup returns 0 when no rows match criteria."""
        mock_connection.fetch = AsyncMock(return_value=[])
        mock_connection.execute = AsyncMock(return_value="DELETE 0")

        total = await repo.cleanup_expired()

        assert total == 0

    @pytest.mark.asyncio
    async def test_cleanup_postgres_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that PostgresError in cleanup maps to RepositoryExecutionError."""
        mock_connection.fetch = AsyncMock(side_effect=asyncpg.PostgresError("fail"))

        with pytest.raises(RepositoryExecutionError):
            await repo.cleanup_expired()

    @pytest.mark.asyncio
    async def test_cleanup_timeout_error(
        self,
        repo: PostgresValidationLedgerRepository,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that QueryCanceledError in cleanup maps to RepositoryTimeoutError."""
        mock_connection.fetch = AsyncMock(
            side_effect=asyncpg.QueryCanceledError("timeout")
        )

        with pytest.raises(RepositoryTimeoutError):
            await repo.cleanup_expired()


# ===========================================================================
# _row_to_entry
# ===========================================================================


class TestRowToEntry:
    """Tests for the internal _row_to_entry conversion."""

    def test_converts_row_dict_to_model(
        self,
        repo: PostgresValidationLedgerRepository,
    ) -> None:
        """Test that a dict-like row is converted to ModelValidationLedgerEntry."""
        row = _make_row()
        entry = repo._row_to_entry(row)

        assert isinstance(entry, ModelValidationLedgerEntry)
        assert entry.id == row["id"]
        assert entry.run_id == row["run_id"]
        assert entry.repo_id == row["repo_id"]
        assert entry.event_type == row["event_type"]
        assert entry.kafka_partition == row["kafka_partition"]
        assert entry.kafka_offset == row["kafka_offset"]
