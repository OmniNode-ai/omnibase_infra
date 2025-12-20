# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for PostgresIdempotencyStore.

Comprehensive test suite covering initialization, atomic check-and-record,
idempotency verification, cleanup, error handling, and lifecycle management.

Uses mocked asyncpg connections to enable fast unit testing without
requiring an actual PostgreSQL database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import asyncpg
import pytest

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    RuntimeHostError,
)
from omnibase_infra.idempotency import (
    ModelPostgresIdempotencyStoreConfig,
    PostgresIdempotencyStore,
)


class TestPostgresIdempotencyStoreInitialization:
    """Test suite for PostgresIdempotencyStore initialization."""

    @pytest.fixture
    def config(self) -> ModelPostgresIdempotencyStoreConfig:
        """Create configuration fixture."""
        return ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
            table_name="idempotency_records",
            pool_min_size=1,
            pool_max_size=5,
            command_timeout=30.0,
        )

    @pytest.fixture
    def store(
        self, config: ModelPostgresIdempotencyStoreConfig
    ) -> PostgresIdempotencyStore:
        """Create store fixture."""
        return PostgresIdempotencyStore(config)

    def test_store_init_default_state(
        self, store: PostgresIdempotencyStore
    ) -> None:
        """Test store initializes in uninitialized state."""
        assert store.is_initialized is False
        assert store._pool is None

    @pytest.mark.asyncio
    async def test_initialize_creates_pool_and_table(
        self, store: PostgresIdempotencyStore
    ) -> None:
        """Test initialize creates asyncpg pool and ensures table exists."""
        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await store.initialize()

            assert store.is_initialized is True
            assert store._pool is mock_pool
            mock_create.assert_called_once()
            # Verify table creation SQL was executed
            assert mock_conn.execute.call_count >= 2  # CREATE TABLE and CREATE INDEX

            await store.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(
        self, store: PostgresIdempotencyStore
    ) -> None:
        """Test calling initialize multiple times is safe."""
        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            await store.initialize()
            await store.initialize()  # Second call should be no-op

            # Should only create pool once
            assert mock_create.call_count == 1

            await store.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_auth_error_raises_infra_connection_error(
        self, store: PostgresIdempotencyStore
    ) -> None:
        """Test initialize with auth failure raises InfraConnectionError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = asyncpg.InvalidPasswordError("")

            with pytest.raises(InfraConnectionError) as exc_info:
                await store.initialize()

            assert "authentication" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_database_not_found_raises_error(
        self, store: PostgresIdempotencyStore
    ) -> None:
        """Test initialize with invalid database raises InfraConnectionError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = asyncpg.InvalidCatalogNameError("")

            with pytest.raises(InfraConnectionError) as exc_info:
                await store.initialize()

            assert "database" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_connection_error_raises_infra_connection_error(
        self, store: PostgresIdempotencyStore
    ) -> None:
        """Test initialize with network error raises InfraConnectionError."""
        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = OSError("Connection refused")

            with pytest.raises(InfraConnectionError) as exc_info:
                await store.initialize()

            assert "host" in str(exc_info.value).lower() or "connect" in str(exc_info.value).lower()


class TestPostgresIdempotencyStoreCheckAndRecord:
    """Test suite for check_and_record atomic operation."""

    @pytest.fixture
    def config(self) -> ModelPostgresIdempotencyStoreConfig:
        """Create configuration fixture."""
        return ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
        )

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

    @pytest.mark.asyncio
    async def test_check_and_record_new_message_returns_true(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record returns True for new message."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        message_id = uuid4()
        result = await initialized_store.check_and_record(
            message_id=message_id,
            domain="test",
            correlation_id=uuid4(),
        )

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_and_record_duplicate_returns_false(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record returns False for duplicate message."""
        mock_conn = AsyncMock()
        # "INSERT 0 0" indicates conflict (no rows inserted)
        mock_conn.execute = AsyncMock(return_value="INSERT 0 0")
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        message_id = uuid4()
        result = await initialized_store.check_and_record(
            message_id=message_id,
            domain="test",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_check_and_record_without_domain(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record works with None domain."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        message_id = uuid4()
        result = await initialized_store.check_and_record(
            message_id=message_id,
            domain=None,
        )

        assert result is True
        # Verify SQL includes NULL for domain
        call_args = mock_conn.execute.call_args
        assert call_args[0][2] is None  # domain parameter

    @pytest.mark.asyncio
    async def test_check_and_record_not_initialized_raises_error(
        self, config: ModelPostgresIdempotencyStoreConfig
    ) -> None:
        """Test check_and_record raises error if not initialized."""
        store = PostgresIdempotencyStore(config)

        with pytest.raises(RuntimeHostError) as exc_info:
            await store.check_and_record(uuid4())

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_check_and_record_timeout_raises_infra_timeout_error(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record raises InfraTimeoutError on timeout."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=asyncpg.QueryCanceledError("timeout"))
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        with pytest.raises(InfraTimeoutError) as exc_info:
            await initialized_store.check_and_record(uuid4())

        assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_check_and_record_connection_lost_raises_error(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record raises InfraConnectionError on connection loss."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(
            side_effect=asyncpg.PostgresConnectionError("connection lost")
        )
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await initialized_store.check_and_record(uuid4())

        assert "connection" in str(exc_info.value).lower()


class TestPostgresIdempotencyStoreIsProcessed:
    """Test suite for is_processed read-only query."""

    @pytest.fixture
    def config(self) -> ModelPostgresIdempotencyStoreConfig:
        """Create configuration fixture."""
        return ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
        )

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

    @pytest.mark.asyncio
    async def test_is_processed_returns_true_when_exists(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test is_processed returns True when record exists."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"1": 1})  # Row exists
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_store.is_processed(uuid4(), domain="test")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_processed_returns_false_when_not_exists(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test is_processed returns False when record does not exist."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)  # No row
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_store.is_processed(uuid4(), domain="test")

        assert result is False


class TestPostgresIdempotencyStoreMarkProcessed:
    """Test suite for mark_processed upsert operation."""

    @pytest.fixture
    def config(self) -> ModelPostgresIdempotencyStoreConfig:
        """Create configuration fixture."""
        return ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
        )

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

    @pytest.mark.asyncio
    async def test_mark_processed_inserts_record(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test mark_processed inserts new record."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        await initialized_store.mark_processed(
            message_id=uuid4(),
            domain="test",
            correlation_id=uuid4(),
            processed_at=datetime.now(UTC),
        )

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_processed_with_naive_datetime_warns(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test mark_processed handles naive datetime with warning."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        naive_dt = datetime.now()  # No timezone
        await initialized_store.mark_processed(
            message_id=uuid4(),
            processed_at=naive_dt,
        )

        # Should not raise, just log warning
        mock_conn.execute.assert_called_once()


class TestPostgresIdempotencyStoreCleanupExpired:
    """Test suite for cleanup_expired TTL operation."""

    @pytest.fixture
    def config(self) -> ModelPostgresIdempotencyStoreConfig:
        """Create configuration fixture."""
        return ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
        )

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

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_old_records(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test cleanup_expired removes records older than TTL."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 42")
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_store.cleanup_expired(ttl_seconds=86400)

        assert result == 42
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_returns_zero_when_nothing_to_delete(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test cleanup_expired returns 0 when no records match."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 0")
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_store.cleanup_expired(ttl_seconds=86400)

        assert result == 0


class TestPostgresIdempotencyStoreHealthCheck:
    """Test suite for health_check operation."""

    @pytest.fixture
    def config(self) -> ModelPostgresIdempotencyStoreConfig:
        """Create configuration fixture."""
        return ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
        )

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

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_healthy(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test health_check returns True when database is reachable."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_store.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_not_initialized(
        self, config: ModelPostgresIdempotencyStoreConfig
    ) -> None:
        """Test health_check returns False when not initialized."""
        store = PostgresIdempotencyStore(config)

        result = await store.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_connection_error(
        self, initialized_store: PostgresIdempotencyStore
    ) -> None:
        """Test health_check returns False when database is unreachable."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("connection error"))
        initialized_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_store.health_check()

        assert result is False


class TestPostgresIdempotencyStoreLifecycle:
    """Test suite for store lifecycle (shutdown)."""

    @pytest.fixture
    def config(self) -> ModelPostgresIdempotencyStoreConfig:
        """Create configuration fixture."""
        return ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
        )

    @pytest.mark.asyncio
    async def test_shutdown_closes_pool(
        self, config: ModelPostgresIdempotencyStoreConfig
    ) -> None:
        """Test shutdown closes the connection pool."""
        store = PostgresIdempotencyStore(config)
        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_pool.close = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await store.initialize()

        assert store.is_initialized is True

        await store.shutdown()

        assert store.is_initialized is False
        assert store._pool is None
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(
        self, config: ModelPostgresIdempotencyStoreConfig
    ) -> None:
        """Test shutdown can be called multiple times safely."""
        store = PostgresIdempotencyStore(config)

        # Shutdown without initialization should be safe
        await store.shutdown()
        await store.shutdown()

        assert store.is_initialized is False
