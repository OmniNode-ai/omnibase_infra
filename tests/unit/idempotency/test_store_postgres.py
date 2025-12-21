# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for PostgresIdempotencyStore.

Comprehensive test suite covering initialization, atomic check-and-record,
idempotency verification, cleanup, error handling, and lifecycle management.

Uses mocked asyncpg connections to enable fast unit testing without
requiring an actual PostgreSQL database.

Shared fixtures are defined in conftest.py:
- postgres_config: Minimal configuration for most tests
- postgres_config_extended: Extended configuration with pool settings
- postgres_store: Uninitialized store instance
- postgres_store_extended: Uninitialized store with extended config
- initialized_postgres_store: Initialized store with mocked pool
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import asyncpg
import pytest
from pydantic import ValidationError

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.idempotency import (
    ModelPostgresIdempotencyStoreConfig,
    PostgresIdempotencyStore,
)


class TestPostgresIdempotencyStoreInitialization:
    """Test suite for PostgresIdempotencyStore initialization."""

    def test_store_init_default_state(
        self, postgres_store_extended: PostgresIdempotencyStore
    ) -> None:
        """Test store initializes in uninitialized state."""
        assert postgres_store_extended.is_initialized is False
        assert postgres_store_extended._pool is None

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
            # Verify table and index creation SQL was executed:
            # 1. CREATE TABLE
            # 2. CREATE INDEX on processed_at
            # 3. CREATE INDEX on domain
            # 4. CREATE INDEX on correlation_id (partial)
            assert mock_conn.execute.call_count == 4

            await store.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_creates_domain_index(
        self, store: PostgresIdempotencyStore
    ) -> None:
        """Test initialize creates index on domain column for query performance."""
        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        executed_sql: list[str] = []

        async def capture_execute(sql: str, *args: object) -> str:
            executed_sql.append(sql)
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=capture_execute)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await store.initialize()

        # Verify domain index creation SQL was executed
        domain_index_sql = [sql for sql in executed_sql if "_domain" in sql]
        assert len(domain_index_sql) == 1
        assert "CREATE INDEX IF NOT EXISTS" in domain_index_sql[0]
        assert "ON" in domain_index_sql[0]
        assert "(domain)" in domain_index_sql[0]

        await store.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, store: PostgresIdempotencyStore) -> None:
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

            assert (
                "host" in str(exc_info.value).lower()
                or "connect" in str(exc_info.value).lower()
            )


class TestPostgresIdempotencyStoreTableNameValidation:
    """Test suite for defense-in-depth table name validation."""

    def test_valid_table_name_accepted(self) -> None:
        """Test valid table names are accepted without error."""
        valid_names = [
            "idempotency_records",
            "_private_table",
            "Table123",
            "a",
            "_",
            "test_table_name",
        ]
        for name in valid_names:
            config = ModelPostgresIdempotencyStoreConfig(
                dsn="postgresql://user:pass@localhost:5432/db",
                table_name=name,
            )
            # Should not raise - constructor succeeds
            store = PostgresIdempotencyStore(config)
            assert store._config.table_name == name

    def test_invalid_table_name_raises_protocol_configuration_error(self) -> None:
        """Test invalid table names are rejected at construction time."""
        invalid_names = [
            "1_starts_with_digit",
            "has-dash",
            "has.dot",
            "has space",
            "has;semicolon",
            "DROP TABLE users;--",
            "",
        ]
        for name in invalid_names:
            # Pydantic validation may catch some of these first,
            # but if it doesn't, runtime validation should catch it
            with pytest.raises((ProtocolConfigurationError, ValueError, ValidationError)):
                config = ModelPostgresIdempotencyStoreConfig(
                    dsn="postgresql://user:pass@localhost:5432/db",
                    table_name=name,
                )
                PostgresIdempotencyStore(config)

    def test_sql_injection_attempt_rejected(self) -> None:
        """Test SQL injection attempts are rejected by table name validation."""
        injection_attempts = [
            "users; DROP TABLE users;--",
            "test'); DROP TABLE idempotency_records;--",
            "a OR 1=1",
        ]
        for name in injection_attempts:
            with pytest.raises((ProtocolConfigurationError, ValueError, ValidationError)):
                config = ModelPostgresIdempotencyStoreConfig(
                    dsn="postgresql://user:pass@localhost:5432/db",
                    table_name=name,
                )
                PostgresIdempotencyStore(config)


class TestPostgresIdempotencyStoreCheckAndRecord:
    """Test suite for check_and_record atomic operation."""

    @pytest.mark.asyncio
    async def test_check_and_record_new_message_returns_true(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record returns True for new message."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        message_id = uuid4()
        result = await initialized_postgres_store.check_and_record(
            message_id=message_id,
            domain="test",
            correlation_id=uuid4(),
        )

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_and_record_duplicate_returns_false(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record returns False for duplicate message."""
        mock_conn = AsyncMock()
        # "INSERT 0 0" indicates conflict (no rows inserted)
        mock_conn.execute = AsyncMock(return_value="INSERT 0 0")
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        message_id = uuid4()
        result = await initialized_postgres_store.check_and_record(
            message_id=message_id,
            domain="test",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_check_and_record_without_domain(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record works with None domain."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        message_id = uuid4()
        result = await initialized_postgres_store.check_and_record(
            message_id=message_id,
            domain=None,
        )

        assert result is True
        # Verify SQL includes NULL for domain
        call_args = mock_conn.execute.call_args
        assert call_args[0][2] is None  # domain parameter

    @pytest.mark.asyncio
    async def test_check_and_record_not_initialized_raises_error(
        self, postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record raises error if not initialized."""
        with pytest.raises(RuntimeHostError) as exc_info:
            await postgres_store.check_and_record(uuid4())

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_check_and_record_timeout_raises_infra_timeout_error(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record raises InfraTimeoutError on timeout."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=asyncpg.QueryCanceledError("timeout"))
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        with pytest.raises(InfraTimeoutError) as exc_info:
            await initialized_postgres_store.check_and_record(uuid4())

        assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_check_and_record_connection_lost_raises_error(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test check_and_record raises InfraConnectionError on connection loss."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(
            side_effect=asyncpg.PostgresConnectionError("connection lost")
        )
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await initialized_postgres_store.check_and_record(uuid4())

        assert "connection" in str(exc_info.value).lower()


class TestPostgresIdempotencyStoreIsProcessed:
    """Test suite for is_processed read-only query."""

    @pytest.mark.asyncio
    async def test_is_processed_returns_true_when_exists(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test is_processed returns True when record exists."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"1": 1})  # Row exists
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_postgres_store.is_processed(uuid4(), domain="test")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_processed_returns_false_when_not_exists(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test is_processed returns False when record does not exist."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)  # No row
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_postgres_store.is_processed(uuid4(), domain="test")

        assert result is False


class TestPostgresIdempotencyStoreMarkProcessed:
    """Test suite for mark_processed upsert operation."""

    @pytest.mark.asyncio
    async def test_mark_processed_inserts_record(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test mark_processed inserts new record."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        await initialized_postgres_store.mark_processed(
            message_id=uuid4(),
            domain="test",
            correlation_id=uuid4(),
            processed_at=datetime.now(UTC),
        )

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_processed_with_naive_datetime_raises_error(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test mark_processed raises RuntimeHostError for naive datetime.

        Naive datetimes can cause subtle data correctness issues due to
        clock skew assumptions, so we fail fast rather than silently converting.
        """
        naive_dt = datetime.now()  # No timezone

        with pytest.raises(RuntimeHostError) as exc_info:
            await initialized_postgres_store.mark_processed(
                message_id=uuid4(),
                processed_at=naive_dt,
            )

        assert "timezone-aware" in str(exc_info.value).lower()
        assert "naive" in str(exc_info.value).lower()


class TestPostgresIdempotencyStoreCleanupExpired:
    """Test suite for cleanup_expired TTL operation."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_old_records(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test cleanup_expired removes records older than TTL."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 42")
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_postgres_store.cleanup_expired(ttl_seconds=86400)

        assert result == 42
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_returns_zero_when_nothing_to_delete(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test cleanup_expired returns 0 when no records match."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 0")
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_postgres_store.cleanup_expired(ttl_seconds=86400)

        assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_applies_clock_skew_tolerance(self) -> None:
        """Test cleanup_expired adds clock_skew_tolerance_seconds to TTL.

        This prevents premature deletion of records in distributed systems
        where nodes may have slightly different system clocks.

        Example scenario:
            - ttl_seconds = 86400 (24 hours)
            - clock_skew_tolerance_seconds = 120 (2 minutes)
            - effective_ttl = 86520 (24 hours + 2 minutes)

        The SQL DELETE should use the effective_ttl, not the raw ttl_seconds.
        """
        config = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
            clock_skew_tolerance_seconds=120,  # 2 minutes tolerance
        )
        store = PostgresIdempotencyStore(config)

        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value="DELETE 10")

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await store.initialize()

        try:
            ttl_seconds = 86400  # 24 hours
            expected_effective_ttl = ttl_seconds + 120  # 86520

            # Reset mock after initialization (which calls execute for table creation)
            mock_conn.execute.reset_mock()

            result = await store.cleanup_expired(ttl_seconds=ttl_seconds)

            assert result == 10
            mock_conn.execute.assert_called_once()

            # Verify the SQL was called with effective_ttl, not raw ttl_seconds
            call_args = mock_conn.execute.call_args
            # The second argument to execute() is the effective TTL
            actual_ttl_used = call_args[0][1]
            assert actual_ttl_used == expected_effective_ttl, (
                f"Expected effective_ttl {expected_effective_ttl}, "
                f"but cleanup used {actual_ttl_used}"
            )
        finally:
            await store.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_expired_with_zero_clock_skew_tolerance(self) -> None:
        """Test cleanup_expired with zero tolerance uses raw TTL.

        When clock_skew_tolerance_seconds is 0, the effective TTL should
        equal the raw ttl_seconds (no buffer added).
        """
        config = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
            clock_skew_tolerance_seconds=0,  # No tolerance
        )
        store = PostgresIdempotencyStore(config)

        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value="DELETE 5")

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await store.initialize()

        try:
            ttl_seconds = 3600  # 1 hour

            # Reset mock after initialization (which calls execute for table creation)
            mock_conn.execute.reset_mock()

            await store.cleanup_expired(ttl_seconds=ttl_seconds)

            call_args = mock_conn.execute.call_args
            actual_ttl_used = call_args[0][1]
            assert actual_ttl_used == ttl_seconds, (
                f"With zero tolerance, expected raw ttl {ttl_seconds}, "
                f"but cleanup used {actual_ttl_used}"
            )
        finally:
            await store.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_expired_batched_deletion(self) -> None:
        """Test cleanup_expired uses batched deletion with multiple iterations.

        When there are more expired records than batch_size, cleanup should
        iterate multiple times until all records are deleted.
        """
        config = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
            clock_skew_tolerance_seconds=0,
            cleanup_batch_size=100,  # Small batch size for testing
            cleanup_max_iterations=10,
        )
        store = PostgresIdempotencyStore(config)

        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # Use default return for initialization (table + indexes)
        mock_conn.execute = AsyncMock(return_value="OK")

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await store.initialize()

        try:
            # Reset mock after initialization and set up batched delete results
            mock_conn.execute.reset_mock()
            # Simulate 3 batches: 100, 100, 50 (total 250 records)
            delete_results = ["DELETE 100", "DELETE 100", "DELETE 50"]
            mock_conn.execute.side_effect = delete_results

            result = await store.cleanup_expired(ttl_seconds=86400)

            # Total deleted should be 250
            assert result == 250
            # Should have made 3 delete calls
            assert mock_conn.execute.call_count == 3
        finally:
            await store.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_expired_stops_when_batch_incomplete(self) -> None:
        """Test cleanup_expired stops iterating when a batch is incomplete.

        If a batch returns fewer than batch_size records, cleanup should
        stop iterating as there are no more records to delete.
        """
        config = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
            clock_skew_tolerance_seconds=0,
            cleanup_batch_size=100,
            cleanup_max_iterations=10,
        )
        store = PostgresIdempotencyStore(config)

        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # First batch returns 42 records (less than 100)
        mock_conn.execute = AsyncMock(return_value="DELETE 42")

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await store.initialize()

        try:
            mock_conn.execute.reset_mock()
            mock_conn.execute.return_value = "DELETE 42"

            result = await store.cleanup_expired(ttl_seconds=86400)

            assert result == 42
            # Should only call once since first batch was incomplete
            assert mock_conn.execute.call_count == 1
        finally:
            await store.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_expired_respects_max_iterations(self) -> None:
        """Test cleanup_expired stops at max_iterations even if records remain.

        This prevents runaway cleanup loops in extreme cases.
        """
        config = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
            clock_skew_tolerance_seconds=0,
            cleanup_batch_size=100,
            cleanup_max_iterations=3,  # Limit to 3 iterations
        )
        store = PostgresIdempotencyStore(config)

        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # Always return full batch (simulating infinite records)
        mock_conn.execute = AsyncMock(return_value="DELETE 100")

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await store.initialize()

        try:
            mock_conn.execute.reset_mock()
            mock_conn.execute.return_value = "DELETE 100"

            result = await store.cleanup_expired(ttl_seconds=86400)

            # Should delete 3 * 100 = 300 records
            assert result == 300
            # Should stop at max_iterations (3)
            assert mock_conn.execute.call_count == 3
        finally:
            await store.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_expired_custom_batch_size(self) -> None:
        """Test cleanup_expired accepts custom batch_size parameter."""
        config = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
            clock_skew_tolerance_seconds=0,
            cleanup_batch_size=10000,  # Default
            cleanup_max_iterations=100,
        )
        store = PostgresIdempotencyStore(config)

        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value="DELETE 50")

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await store.initialize()

        try:
            mock_conn.execute.reset_mock()
            mock_conn.execute.return_value = "DELETE 50"

            # Pass custom batch_size (smaller than config default)
            result = await store.cleanup_expired(
                ttl_seconds=86400,
                batch_size=500,  # Override config default
            )

            assert result == 50
            # Verify the SQL used the custom batch_size
            call_args = mock_conn.execute.call_args
            batch_size_param = call_args[0][2]  # Third argument is batch_size
            assert batch_size_param == 500
        finally:
            await store.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_expired_custom_max_iterations(self) -> None:
        """Test cleanup_expired accepts custom max_iterations parameter."""
        config = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
            clock_skew_tolerance_seconds=0,
            cleanup_batch_size=100,
            cleanup_max_iterations=100,  # Default high
        )
        store = PostgresIdempotencyStore(config)

        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value="DELETE 100")  # Always full batch

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await store.initialize()

        try:
            mock_conn.execute.reset_mock()
            mock_conn.execute.return_value = "DELETE 100"

            # Pass custom max_iterations (lower than config default)
            result = await store.cleanup_expired(
                ttl_seconds=86400,
                max_iterations=2,  # Override config default
            )

            # Should delete 2 * 100 = 200 records
            assert result == 200
            # Should stop at custom max_iterations (2)
            assert mock_conn.execute.call_count == 2
        finally:
            await store.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_expired_batched_sql_uses_limit(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test cleanup_expired uses LIMIT clause for batched deletion.

        The batched delete SQL should use a subquery with LIMIT to
        select only batch_size records per iteration.
        """
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 10")
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        await initialized_postgres_store.cleanup_expired(
            ttl_seconds=86400, batch_size=1000
        )

        # Check that the SQL contains LIMIT and subquery pattern
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "LIMIT" in sql
        assert "SELECT id FROM" in sql
        assert "WHERE id IN" in sql


class TestPostgresIdempotencyStoreHealthCheck:
    """Test suite for health_check operation.

    The health_check method tests read and table existence:
    1. Read check via SELECT 1
    2. Table existence check via information_schema query
    """

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_read_and_table_check_succeed(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test health_check returns True when read and table check both work.

        The health check verifies:
        1. Read access (SELECT 1)
        2. Table existence (information_schema query)
        """
        mock_conn = AsyncMock()
        # First fetchval call: SELECT 1 returns 1
        # Second fetchval call: table existence check returns 1
        mock_conn.fetchval = AsyncMock(side_effect=[1, 1])

        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_postgres_store.health_check()

        assert result is True
        # Verify both fetchval calls were made
        assert mock_conn.fetchval.call_count == 2

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_table_not_found(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test health_check returns False when table does not exist."""
        mock_conn = AsyncMock()
        # First fetchval call: SELECT 1 returns 1 (read succeeds)
        # Second fetchval call: table check returns None (table not found)
        mock_conn.fetchval = AsyncMock(side_effect=[1, None])

        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_postgres_store.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_not_initialized(
        self, postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test health_check returns False when not initialized."""
        result = await postgres_store.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_read_error(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test health_check returns False when read check fails."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("connection error"))
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_postgres_store.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_table_check_error(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test health_check returns False when table check fails.

        Even if read succeeds, if table check fails the store is unhealthy.
        """
        mock_conn = AsyncMock()
        # First fetchval succeeds (SELECT 1), second raises exception
        mock_conn.fetchval = AsyncMock(
            side_effect=[1, Exception("permission denied")]
        )

        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )

        result = await initialized_postgres_store.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_acquire_error(
        self, initialized_postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test health_check returns False when acquiring connection fails."""
        initialized_postgres_store._pool.acquire.return_value.__aenter__ = AsyncMock(
            side_effect=asyncpg.PostgresConnectionError("pool exhausted")
        )

        result = await initialized_postgres_store.health_check()

        assert result is False


class TestPostgresIdempotencyStoreLifecycle:
    """Test suite for store lifecycle (shutdown)."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_pool(
        self, postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test shutdown closes the connection pool."""
        mock_pool = MagicMock(spec=asyncpg.Pool)
        mock_pool.close = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool
            await postgres_store.initialize()

        assert postgres_store.is_initialized is True

        await postgres_store.shutdown()

        assert postgres_store.is_initialized is False
        assert postgres_store._pool is None
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(
        self, postgres_store: PostgresIdempotencyStore
    ) -> None:
        """Test shutdown can be called multiple times safely."""
        # Shutdown without initialization should be safe
        await postgres_store.shutdown()
        await postgres_store.shutdown()

        assert postgres_store.is_initialized is False
