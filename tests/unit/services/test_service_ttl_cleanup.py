# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ServiceTTLCleanup.

This module tests:
    - Single cleanup pass (cleanup_once)
    - Per-table batch deletion logic
    - Empty table handling (no rows to delete)
    - Circuit breaker state and error handling
    - Configuration validation
    - Graceful shutdown signaling
    - Health status reporting
    - Table/column allowlist enforcement
    - ModelTTLCleanupResult metrics

All tests mock asyncpg pool - no real database required.

Related Tickets:
    - OMN-1759: Implement 30-day TTL cleanup for observability tables
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ProtocolConfigurationError,
)
from omnibase_infra.services.observability.agent_actions.config_ttl_cleanup import (
    DEFAULT_TABLE_TTL_COLUMNS,
    ConfigTTLCleanup,
)
from omnibase_infra.services.observability.agent_actions.models.model_ttl_cleanup_result import (
    ModelTTLCleanupResult,
)
from omnibase_infra.services.observability.agent_actions.service_ttl_cleanup import (
    ALLOWED_TABLES,
    ALLOWED_TTL_COLUMNS,
    ServiceTTLCleanup,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    conn = AsyncMock()
    # Default: DELETE returns "DELETE 0" (no rows deleted)
    conn.execute = AsyncMock(return_value="DELETE 0")

    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    return pool


@pytest.fixture
def mock_conn(mock_pool: MagicMock) -> AsyncMock:
    """Get the mock connection from the pool."""
    conn: AsyncMock = mock_pool.acquire.return_value.__aenter__.return_value
    return conn


@pytest.fixture
def config() -> ConfigTTLCleanup:
    """Create a test configuration."""
    return ConfigTTLCleanup(
        postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
        retention_days=30,
        batch_size=1000,
        interval_seconds=60,
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout=30.0,
    )


@pytest.fixture
def service(mock_pool: MagicMock, config: ConfigTTLCleanup) -> ServiceTTLCleanup:
    """Create a service with mocked pool."""
    return ServiceTTLCleanup(pool=mock_pool, config=config)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfigTTLCleanup:
    """Test ConfigTTLCleanup validation and defaults."""

    def test_default_retention_days(self) -> None:
        """Default retention should be 30 days."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
        )
        assert config.retention_days == 30

    def test_default_batch_size(self) -> None:
        """Default batch size should be 1000."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
        )
        assert config.batch_size == 1000

    def test_default_interval_seconds(self) -> None:
        """Default interval should be 600 seconds (10 minutes)."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
        )
        assert config.interval_seconds == 600

    def test_default_table_ttl_columns(self) -> None:
        """Default table_ttl_columns should match DEFAULT_TABLE_TTL_COLUMNS."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
        )
        assert config.table_ttl_columns == DEFAULT_TABLE_TTL_COLUMNS

    def test_agent_execution_logs_uses_updated_at(self) -> None:
        """agent_execution_logs should use updated_at for TTL."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
        )
        assert config.table_ttl_columns["agent_execution_logs"] == "updated_at"

    def test_custom_retention_days(self) -> None:
        """Custom retention days should be accepted."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            retention_days=7,
        )
        assert config.retention_days == 7

    def test_empty_table_config_raises_error(self) -> None:
        """Empty table_ttl_columns should raise ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError):
            ConfigTTLCleanup(
                postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
                table_ttl_columns={},
            )

    def test_invalid_ttl_column_raises_error(self) -> None:
        """Invalid TTL column name should raise ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError):
            ConfigTTLCleanup(
                postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
                table_ttl_columns={"agent_actions": "invalid_column"},
            )


# =============================================================================
# Service Initialization Tests
# =============================================================================


class TestServiceInitialization:
    """Test ServiceTTLCleanup initialization."""

    def test_service_initializes_successfully(
        self,
        mock_pool: MagicMock,
        config: ConfigTTLCleanup,
    ) -> None:
        """Service should initialize with valid config."""
        service = ServiceTTLCleanup(pool=mock_pool, config=config)
        assert service.last_result is None

    def test_invalid_table_name_raises_error(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Invalid table name should raise ProtocolConfigurationError."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            table_ttl_columns={
                "agent_actions": "created_at",
                "malicious_table; DROP TABLE users": "created_at",
            },
        )
        with pytest.raises(ProtocolConfigurationError, match="Invalid table names"):
            ServiceTTLCleanup(pool=mock_pool, config=config)


# =============================================================================
# Single Cleanup Tests - Empty Tables
# =============================================================================


class TestCleanupOnceEmpty:
    """Test cleanup_once when tables have no expired rows."""

    @pytest.mark.asyncio
    async def test_cleanup_empty_tables_returns_zero(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Cleanup with no expired rows should return 0 total deleted."""
        mock_conn.execute.return_value = "DELETE 0"

        result = await service.cleanup_once()

        assert result.total_rows_deleted == 0
        assert all(count == 0 for _, count in result.tables_cleaned)

    @pytest.mark.asyncio
    async def test_cleanup_empty_returns_all_tables(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Result should include entries for all configured tables."""
        mock_conn.execute.return_value = "DELETE 0"

        result = await service.cleanup_once()

        assert set(dict(result.tables_cleaned).keys()) == set(
            DEFAULT_TABLE_TTL_COLUMNS.keys()
        )

    @pytest.mark.asyncio
    async def test_cleanup_empty_result_is_falsy(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Result with 0 deletions should be falsy."""
        mock_conn.execute.return_value = "DELETE 0"

        result = await service.cleanup_once()

        assert not result

    @pytest.mark.asyncio
    async def test_cleanup_empty_has_timing_metrics(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Result should have timing metrics even with no deletions."""
        mock_conn.execute.return_value = "DELETE 0"

        result = await service.cleanup_once()

        assert result.duration_ms >= 0
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at


# =============================================================================
# Single Cleanup Tests - With Rows
# =============================================================================


class TestCleanupOnceWithRows:
    """Test cleanup_once when tables have expired rows."""

    @pytest.mark.asyncio
    async def test_cleanup_single_batch(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Should delete rows in a single batch when count < batch_size."""
        # First call returns 500 (< 1000 batch_size), so loop exits
        mock_conn.execute.return_value = "DELETE 500"

        result = await service.cleanup_once()

        # 500 rows * 7 tables = 3500 total
        assert result.total_rows_deleted == 500 * len(DEFAULT_TABLE_TTL_COLUMNS)
        assert result

    @pytest.mark.asyncio
    async def test_cleanup_multiple_batches(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """Should loop through batches until fewer than batch_size deleted."""
        # Configure with small batch_size for easier testing
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            batch_size=100,
            table_ttl_columns={"agent_actions": "created_at"},
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        # First batch: 100 (= batch_size, so continue)
        # Second batch: 50 (< batch_size, so stop)
        mock_conn.execute.side_effect = ["DELETE 100", "DELETE 50"]

        result = await service.cleanup_once()

        assert dict(result.tables_cleaned)["agent_actions"] == 150
        assert result.total_rows_deleted == 150
        assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_result_is_truthy_with_deletions(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Result with deletions should be truthy."""
        mock_conn.execute.return_value = "DELETE 10"

        result = await service.cleanup_once()

        assert result

    @pytest.mark.asyncio
    async def test_cleanup_stores_last_result(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """cleanup_once should store result as last_result."""
        mock_conn.execute.return_value = "DELETE 42"

        result = await service.cleanup_once()

        assert service.last_result is result

    @pytest.mark.asyncio
    async def test_cleanup_uses_correct_cutoff(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """DELETE should use correct cutoff timestamp based on retention_days."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            retention_days=7,
            table_ttl_columns={"agent_actions": "created_at"},
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        mock_conn.execute.return_value = "DELETE 0"

        before = datetime.now(UTC) - timedelta(days=7)
        await service.cleanup_once()
        after = datetime.now(UTC) - timedelta(days=7)

        # Verify the cutoff passed to execute is within expected range
        call_args = mock_conn.execute.call_args
        cutoff_arg = call_args[0][1]  # Second positional arg (after SQL)
        assert before <= cutoff_arg <= after


# =============================================================================
# SQL Query Tests
# =============================================================================


class TestSQLQueries:
    """Test that correct SQL queries are generated."""

    @pytest.mark.asyncio
    async def test_sql_contains_table_name(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """SQL should reference the correct table name."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            table_ttl_columns={"agent_actions": "created_at"},
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        mock_conn.execute.return_value = "DELETE 0"
        await service.cleanup_once()

        sql = mock_conn.execute.call_args[0][0]
        assert "agent_actions" in sql

    @pytest.mark.asyncio
    async def test_sql_uses_created_at_for_standard_tables(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """Standard tables should use created_at in the WHERE clause."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            table_ttl_columns={"agent_actions": "created_at"},
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        mock_conn.execute.return_value = "DELETE 0"
        await service.cleanup_once()

        sql = mock_conn.execute.call_args[0][0]
        assert "created_at" in sql

    @pytest.mark.asyncio
    async def test_sql_uses_updated_at_for_execution_logs(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """agent_execution_logs should use updated_at in the WHERE clause."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            table_ttl_columns={"agent_execution_logs": "updated_at"},
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        mock_conn.execute.return_value = "DELETE 0"
        await service.cleanup_once()

        sql = mock_conn.execute.call_args[0][0]
        assert "updated_at" in sql

    @pytest.mark.asyncio
    async def test_sql_uses_batch_size_limit(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """SQL should pass batch_size as LIMIT parameter."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            batch_size=500,
            table_ttl_columns={"agent_actions": "created_at"},
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        mock_conn.execute.return_value = "DELETE 0"
        await service.cleanup_once()

        # batch_size is the third positional arg ($2)
        call_args = mock_conn.execute.call_args
        batch_size_arg = call_args[0][2]
        assert batch_size_arg == 500

    @pytest.mark.asyncio
    async def test_sql_uses_ctid_subquery(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """SQL should use ctid-based subquery for efficient batched deletes."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            table_ttl_columns={"agent_actions": "created_at"},
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        mock_conn.execute.return_value = "DELETE 0"
        await service.cleanup_once()

        sql = mock_conn.execute.call_args[0][0]
        assert "ctid" in sql
        assert "LIMIT" in sql


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Test circuit breaker state and behavior."""

    def test_circuit_breaker_initially_closed(
        self,
        service: ServiceTTLCleanup,
    ) -> None:
        """Circuit breaker should start in closed state."""
        health = service.get_health_status()
        circuit_breaker = health["circuit_breaker"]
        assert isinstance(circuit_breaker, dict)
        assert circuit_breaker["state"] == "closed"

    @pytest.mark.asyncio
    async def test_connection_error_raises_infra_connection_error(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """Connection errors should raise InfraConnectionError."""
        import asyncpg

        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            table_ttl_columns={"agent_actions": "created_at"},
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        mock_conn.execute.side_effect = asyncpg.PostgresConnectionError(
            "Connection refused"
        )

        result = await service.cleanup_once()

        # cleanup_once catches exceptions and records them in errors
        errors_dict = dict(result.errors)
        assert "agent_actions" in errors_dict
        assert "InfraConnectionError" in errors_dict["agent_actions"]

    @pytest.mark.asyncio
    async def test_timeout_error_raises_infra_timeout_error(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """Query timeout should raise InfraTimeoutError."""
        import asyncpg

        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            table_ttl_columns={"agent_actions": "created_at"},
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        mock_conn.execute.side_effect = asyncpg.QueryCanceledError(
            "canceling statement due to statement timeout"
        )

        result = await service.cleanup_once()

        errors_dict = dict(result.errors)
        assert "agent_actions" in errors_dict
        assert "InfraTimeoutError" in errors_dict["agent_actions"]

    @pytest.mark.asyncio
    async def test_repeated_failures_open_circuit(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """Repeated failures should open the circuit breaker."""
        import asyncpg

        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            table_ttl_columns={"agent_actions": "created_at"},
            circuit_breaker_threshold=2,
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        mock_conn.execute.side_effect = asyncpg.PostgresConnectionError(
            "Connection refused"
        )

        # Run cleanup twice to trigger circuit breaker threshold
        for _ in range(2):
            await service.cleanup_once()

        # Third call should hit circuit breaker
        result = await service.cleanup_once()
        errors_dict = dict(result.errors)
        assert "agent_actions" in errors_dict
        assert "InfraUnavailableError" in errors_dict["agent_actions"]


# =============================================================================
# Execution Tests
# =============================================================================


class TestExecution:
    """Test runtime execution behavior of cleanup_once."""

    pytestmark = pytest.mark.unit

    @pytest.mark.asyncio
    async def test_execute_passes_query_timeout(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """DELETE execution should use the configured query_timeout_seconds."""
        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            query_timeout_seconds=45.0,
            table_ttl_columns={"agent_actions": "created_at"},
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        mock_conn.execute.return_value = "DELETE 0"
        await service.cleanup_once()

        call_kwargs = mock_conn.execute.call_args.kwargs
        assert call_kwargs.get("timeout") == 45.0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling during cleanup."""

    @pytest.mark.asyncio
    async def test_partial_failure_continues_other_tables(
        self,
        mock_pool: MagicMock,
        mock_conn: AsyncMock,
    ) -> None:
        """Failure on one table should not prevent cleanup of other tables."""
        import asyncpg

        config = ConfigTTLCleanup(
            postgres_dsn="postgresql://postgres:test@localhost:5432/test_db",
            table_ttl_columns={
                "agent_actions": "created_at",
                "agent_routing_decisions": "created_at",
            },
        )
        service = ServiceTTLCleanup(pool=mock_pool, config=config)

        # First table fails, second succeeds
        call_count = 0

        async def side_effect(*args: object, **kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncpg.PostgresConnectionError("Connection refused")
            return "DELETE 10"

        mock_conn.execute.side_effect = side_effect

        result = await service.cleanup_once()

        # One table should have errors, the other should succeed
        assert len(result.errors) == 1
        assert result.total_rows_deleted == 10

    @pytest.mark.asyncio
    async def test_correlation_id_propagated(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Provided correlation_id should be used in the result."""
        mock_conn.execute.return_value = "DELETE 0"
        cid = uuid4()

        result = await service.cleanup_once(correlation_id=cid)

        assert result.correlation_id == cid

    @pytest.mark.asyncio
    async def test_auto_correlation_id_when_none(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Auto-generated correlation_id when none provided."""
        mock_conn.execute.return_value = "DELETE 0"

        result = await service.cleanup_once()

        assert result.correlation_id is not None


# =============================================================================
# Graceful Shutdown Tests
# =============================================================================


class TestGracefulShutdown:
    """Test graceful shutdown signaling."""

    @pytest.mark.asyncio
    async def test_stop_signals_shutdown(
        self,
        service: ServiceTTLCleanup,
    ) -> None:
        """stop() should signal the shutdown event."""
        service.stop()
        assert service._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_run_exits_on_stop(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """run() should exit when stop() is called."""
        mock_conn.execute.return_value = "DELETE 0"

        # Stop immediately after starting
        async def stop_after_delay() -> None:
            await asyncio.sleep(0.1)
            service.stop()

        stop_task = asyncio.create_task(stop_after_delay())

        # run() should exit within a reasonable time
        await asyncio.wait_for(service.run(), timeout=5.0)
        await stop_task


# =============================================================================
# Health Status Tests
# =============================================================================


class TestHealthStatus:
    """Test health status reporting."""

    def test_health_status_initial(
        self,
        service: ServiceTTLCleanup,
    ) -> None:
        """Initial health status should have no last_result."""
        health = service.get_health_status()

        assert health["service"] == "ttl-cleanup"
        assert health["last_result"] is None
        config = health["config"]
        assert isinstance(config, dict)
        assert config["retention_days"] == 30
        assert config["batch_size"] == 1000

    @pytest.mark.asyncio
    async def test_health_status_after_cleanup(
        self,
        service: ServiceTTLCleanup,
        mock_conn: AsyncMock,
    ) -> None:
        """Health status should include last_result after cleanup."""
        mock_conn.execute.return_value = "DELETE 5"

        await service.cleanup_once()

        health = service.get_health_status()
        assert health["last_result"] is not None
        last_result = health["last_result"]
        assert isinstance(last_result, dict)
        rows_deleted = last_result["total_rows_deleted"]
        assert isinstance(rows_deleted, (int, float))
        assert rows_deleted > 0


# =============================================================================
# Allowlist Tests
# =============================================================================


class TestAllowlists:
    """Test table and column name allowlists."""

    def test_allowed_tables_contains_all_observability_tables(self) -> None:
        """ALLOWED_TABLES should contain all default observability tables."""
        for table in DEFAULT_TABLE_TTL_COLUMNS:
            assert table in ALLOWED_TABLES

    def test_allowed_ttl_columns_contains_valid_columns(self) -> None:
        """ALLOWED_TTL_COLUMNS should contain created_at and updated_at."""
        assert "created_at" in ALLOWED_TTL_COLUMNS
        assert "updated_at" in ALLOWED_TTL_COLUMNS

    def test_default_ttl_columns_use_allowed_columns(self) -> None:
        """All default TTL columns should be in the allowlist."""
        for col in DEFAULT_TABLE_TTL_COLUMNS.values():
            assert col in ALLOWED_TTL_COLUMNS


# =============================================================================
# Result Model Tests
# =============================================================================


class TestModelTTLCleanupResult:
    """Test ModelTTLCleanupResult model."""

    def test_result_with_deletions_is_truthy(self) -> None:
        """Result with total_rows_deleted > 0 should be truthy."""
        result = ModelTTLCleanupResult(
            correlation_id=uuid4(),
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            tables_cleaned=(("agent_actions", 100),),
            total_rows_deleted=100,
            duration_ms=500,
        )
        assert result

    def test_result_without_deletions_is_falsy(self) -> None:
        """Result with total_rows_deleted == 0 should be falsy."""
        result = ModelTTLCleanupResult(
            correlation_id=uuid4(),
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            tables_cleaned=(),
            total_rows_deleted=0,
            duration_ms=100,
        )
        assert not result

    def test_result_is_frozen(self) -> None:
        """Result model should be immutable."""
        result = ModelTTLCleanupResult(
            correlation_id=uuid4(),
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        with pytest.raises(Exception):
            result.total_rows_deleted = 42  # type: ignore[misc]

    def test_result_forbids_extra_fields(self) -> None:
        """Result model should reject extra fields."""
        with pytest.raises(Exception):
            ModelTTLCleanupResult(
                correlation_id=uuid4(),
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                unknown_field="should fail",  # type: ignore[call-arg]
            )

    def test_result_default_errors_empty(self) -> None:
        """Default errors should be an empty tuple."""
        result = ModelTTLCleanupResult(
            correlation_id=uuid4(),
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        assert result.errors == ()


__all__ = [
    "TestConfigTTLCleanup",
    "TestServiceInitialization",
    "TestCleanupOnceEmpty",
    "TestCleanupOnceWithRows",
    "TestSQLQueries",
    "TestCircuitBreaker",
    "TestExecution",
    "TestErrorHandling",
    "TestGracefulShutdown",
    "TestHealthStatus",
    "TestAllowlists",
    "TestModelTTLCleanupResult",
]
