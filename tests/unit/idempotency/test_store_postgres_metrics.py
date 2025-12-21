# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for PostgresIdempotencyStore metrics and observability.

Tests cover:
- Initial metrics state
- Metrics tracking for new messages, duplicates, and errors
- Cleanup metrics tracking
- Rate calculations (duplicate_rate, error_rate, success_rate)
- Metrics isolation (get_metrics returns a copy)

Uses mocked asyncpg connections to enable fast unit testing without
requiring an actual PostgreSQL database.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import asyncpg
import pytest

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
)
from omnibase_infra.idempotency import (
    ModelPostgresIdempotencyStoreConfig,
    PostgresIdempotencyStore,
)


class TestPostgresIdempotencyStoreMetrics:
    """Test suite for store metrics and observability."""

    @pytest.fixture
    def config(self) -> ModelPostgresIdempotencyStoreConfig:
        """Create configuration fixture."""
        return ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
        )

    @pytest.fixture
    def store(
        self, config: ModelPostgresIdempotencyStoreConfig
    ) -> PostgresIdempotencyStore:
        """Create uninitialized store fixture."""
        return PostgresIdempotencyStore(config)

    @pytest.fixture
    async def initialized_store(
        self, config: ModelPostgresIdempotencyStoreConfig
    ) -> PostgresIdempotencyStore:
        """Create and initialize store fixture with mocked pool."""
        store = PostgresIdempotencyStore(config)
        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await store.initialize()

        yield store
        await store.shutdown()

    def test_initial_metrics_are_zero(
        self, store: PostgresIdempotencyStore
    ) -> None:
        """Test that metrics start at zero for new store."""
        metrics = store.get_metrics()

        assert metrics.total_checks == 0
        assert metrics.duplicate_count == 0
        assert metrics.error_count == 0
        assert metrics.total_cleanup_deleted == 0
        assert metrics.last_cleanup_deleted == 0
        assert metrics.last_cleanup_at is None
        assert metrics.duplicate_rate == 0.0
        assert metrics.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_metrics_track_new_message(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test metrics increment for new message."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        await initialized_store.check_and_record(uuid4())

        metrics = initialized_store.get_metrics()
        assert metrics.total_checks == 1
        assert metrics.duplicate_count == 0
        assert metrics.error_count == 0
        assert metrics.duplicate_rate == 0.0

    @pytest.mark.asyncio
    async def test_metrics_track_duplicate(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test metrics increment for duplicate message."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 0")
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        await initialized_store.check_and_record(uuid4())

        metrics = initialized_store.get_metrics()
        assert metrics.total_checks == 1
        assert metrics.duplicate_count == 1
        assert metrics.error_count == 0
        assert metrics.duplicate_rate == 1.0

    @pytest.mark.asyncio
    async def test_metrics_track_error_on_timeout(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test metrics increment for timeout error."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=asyncpg.QueryCanceledError("timeout"))
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        with pytest.raises(InfraTimeoutError):
            await initialized_store.check_and_record(uuid4())

        metrics = initialized_store.get_metrics()
        assert metrics.total_checks == 1
        assert metrics.duplicate_count == 0
        assert metrics.error_count == 1
        assert metrics.error_rate == 1.0

    @pytest.mark.asyncio
    async def test_metrics_track_error_on_connection_loss(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test metrics increment for connection error."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(
            side_effect=asyncpg.PostgresConnectionError("connection lost")
        )
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        with pytest.raises(InfraConnectionError):
            await initialized_store.check_and_record(uuid4())

        metrics = initialized_store.get_metrics()
        assert metrics.total_checks == 1
        assert metrics.error_count == 1
        assert metrics.error_rate == 1.0

    @pytest.mark.asyncio
    async def test_metrics_track_cleanup(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test metrics track cleanup operations."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 42")
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        await initialized_store.cleanup_expired(ttl_seconds=86400)

        metrics = initialized_store.get_metrics()
        assert metrics.total_cleanup_deleted == 42
        assert metrics.last_cleanup_deleted == 42
        assert metrics.last_cleanup_at is not None

    @pytest.mark.asyncio
    async def test_metrics_accumulate_over_multiple_cleanups(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test metrics accumulate cleanup totals correctly."""
        mock_conn = AsyncMock()
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        # First cleanup
        mock_conn.execute = AsyncMock(return_value="DELETE 10")
        await initialized_store.cleanup_expired(ttl_seconds=86400)

        # Second cleanup
        mock_conn.execute = AsyncMock(return_value="DELETE 20")
        await initialized_store.cleanup_expired(ttl_seconds=86400)

        metrics = initialized_store.get_metrics()
        assert metrics.total_cleanup_deleted == 30  # 10 + 20
        assert metrics.last_cleanup_deleted == 20  # Most recent

    @pytest.mark.asyncio
    async def test_metrics_calculate_rates(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test metrics calculate duplicate and error rates correctly."""
        mock_conn = AsyncMock()
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        # 2 new messages
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        await initialized_store.check_and_record(uuid4())
        await initialized_store.check_and_record(uuid4())

        # 1 duplicate
        mock_conn.execute = AsyncMock(return_value="INSERT 0 0")
        await initialized_store.check_and_record(uuid4())

        # 1 error
        mock_conn.execute = AsyncMock(side_effect=asyncpg.QueryCanceledError("timeout"))
        with pytest.raises(InfraTimeoutError):
            await initialized_store.check_and_record(uuid4())

        metrics = initialized_store.get_metrics()
        assert metrics.total_checks == 4
        assert metrics.duplicate_count == 1
        assert metrics.error_count == 1
        assert metrics.duplicate_rate == 0.25  # 1/4
        assert metrics.error_rate == 0.25  # 1/4
        assert metrics.success_count == 2  # 4 - 1 - 1
        assert metrics.success_rate == 0.5  # 2/4

    def test_get_metrics_returns_copy(
        self, store: PostgresIdempotencyStore
    ) -> None:
        """Test that get_metrics returns a copy to prevent external mutation."""
        metrics1 = store.get_metrics()
        metrics1.total_checks = 100  # Mutate the copy

        metrics2 = store.get_metrics()
        assert metrics2.total_checks == 0  # Original unchanged
