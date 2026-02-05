# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for PostgresRepositoryRuntime ledger emission."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.models.ledger import (
    ModelDbQueryFailed,
    ModelDbQueryRequested,
    ModelDbQuerySucceeded,
)
from omnibase_infra.runtime.db import (
    ModelDbOperation,
    ModelDbParam,
    ModelDbRepositoryContract,
    ModelDbReturn,
    ModelRepositoryRuntimeConfig,
    PostgresRepositoryRuntime,
)
from omnibase_infra.sinks import InMemoryLedgerSink


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    connection = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


@pytest.fixture
def test_contract() -> ModelDbRepositoryContract:
    """Create a test repository contract."""
    return ModelDbRepositoryContract(
        name="test_users",
        database_ref="primary",
        tables=["users"],
        ops={
            "find_by_id": ModelDbOperation(
                mode="read",
                sql="SELECT * FROM users WHERE id = $1",
                params={"user_id": ModelDbParam(name="user_id", param_type="integer")},
                returns=ModelDbReturn(many=False),
            ),
            "find_all": ModelDbOperation(
                mode="read",
                sql="SELECT * FROM users ORDER BY id",
                params={},
                returns=ModelDbReturn(many=True),
            ),
        },
    )


@pytest.fixture
def test_config() -> ModelRepositoryRuntimeConfig:
    """Create a test runtime config."""
    return ModelRepositoryRuntimeConfig(
        max_row_limit=100,
        timeout_ms=5000,
        primary_key_column="id",
    )


@pytest.fixture
def ledger_sink() -> InMemoryLedgerSink:
    """Create an in-memory ledger sink for testing."""
    return InMemoryLedgerSink()


@pytest.mark.unit
class TestPostgresRepositoryRuntimeLedger:
    """Tests for PostgresRepositoryRuntime ledger emission."""

    @pytest.mark.asyncio
    async def test_emits_requested_and_succeeded_on_success(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
        ledger_sink: InMemoryLedgerSink,
    ) -> None:
        """Test that successful call emits requested and succeeded events."""
        # Setup mock to return a result
        connection = mock_pool.acquire.return_value.__aenter__.return_value
        connection.fetchrow = AsyncMock(return_value={"id": 1, "name": "test"})

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,
            contract=test_contract,
            config=test_config,
            ledger_sink=ledger_sink,
        )

        result = await runtime.call("find_by_id", 1)

        assert result == {"id": 1, "name": "test"}
        assert ledger_sink.pending_count == 2

        # Check requested event
        requested = ledger_sink.events[0]
        assert isinstance(requested, ModelDbQueryRequested)
        assert requested.event_type == "db.query.requested"
        assert requested.operation_name == "find_by_id"
        assert requested.contract_id == "test_users"
        assert requested.contract_fingerprint.startswith("sha256:")

        # Check succeeded event
        succeeded = ledger_sink.events[1]
        assert isinstance(succeeded, ModelDbQuerySucceeded)
        assert succeeded.event_type == "db.query.succeeded"
        assert succeeded.operation_name == "find_by_id"
        assert succeeded.rows_returned == 1
        assert succeeded.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_emits_failed_on_exception(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
        ledger_sink: InMemoryLedgerSink,
    ) -> None:
        """Test that failed call emits requested and failed events."""
        # Setup mock to raise exception
        connection = mock_pool.acquire.return_value.__aenter__.return_value
        connection.fetchrow = AsyncMock(side_effect=Exception("Database error"))

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,
            contract=test_contract,
            config=test_config,
            ledger_sink=ledger_sink,
        )

        with pytest.raises(Exception, match="Database error"):
            await runtime.call("find_by_id", 1)

        assert ledger_sink.pending_count == 2

        # Check requested event
        requested = ledger_sink.events[0]
        assert isinstance(requested, ModelDbQueryRequested)

        # Check failed event
        failed = ledger_sink.events[1]
        assert isinstance(failed, ModelDbQueryFailed)
        assert failed.event_type == "db.query.failed"
        assert failed.operation_name == "find_by_id"
        assert "RepositoryExecutionError" in failed.error_type
        assert failed.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_correlation_id_propagates(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
        ledger_sink: InMemoryLedgerSink,
    ) -> None:
        """Test that correlation_id propagates through all events."""
        connection = mock_pool.acquire.return_value.__aenter__.return_value
        connection.fetchrow = AsyncMock(return_value={"id": 1})

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,
            contract=test_contract,
            config=test_config,
            ledger_sink=ledger_sink,
        )

        correlation_id = uuid4()
        await runtime.call("find_by_id", 1, correlation_id=correlation_id)

        # Both events should have same correlation_id
        assert ledger_sink.events[0].correlation_id == correlation_id
        assert ledger_sink.events[1].correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_auto_generates_correlation_id(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
        ledger_sink: InMemoryLedgerSink,
    ) -> None:
        """Test that correlation_id is auto-generated if not provided."""
        connection = mock_pool.acquire.return_value.__aenter__.return_value
        connection.fetchrow = AsyncMock(return_value={"id": 1})

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,
            contract=test_contract,
            config=test_config,
            ledger_sink=ledger_sink,
        )

        await runtime.call("find_by_id", 1)

        # Both events should have a UUID (auto-generated)
        assert isinstance(ledger_sink.events[0].correlation_id, UUID)
        assert (
            ledger_sink.events[0].correlation_id == ledger_sink.events[1].correlation_id
        )

    @pytest.mark.asyncio
    async def test_no_raw_sql_in_events(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
        ledger_sink: InMemoryLedgerSink,
    ) -> None:
        """Test that raw SQL does not appear in ledger events."""
        connection = mock_pool.acquire.return_value.__aenter__.return_value
        connection.fetchrow = AsyncMock(return_value={"id": 1})

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,
            contract=test_contract,
            config=test_config,
            ledger_sink=ledger_sink,
        )

        await runtime.call("find_by_id", 1)

        # Serialize events to JSON and check no SQL
        for event in ledger_sink.events:
            event_json = event.model_dump_json()
            assert "SELECT" not in event_json
            assert "FROM" not in event_json
            assert "WHERE" not in event_json

    @pytest.mark.asyncio
    async def test_idempotency_key_format(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
        ledger_sink: InMemoryLedgerSink,
    ) -> None:
        """Test that idempotency_key has correct format."""
        connection = mock_pool.acquire.return_value.__aenter__.return_value
        connection.fetchrow = AsyncMock(return_value={"id": 1})

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,
            contract=test_contract,
            config=test_config,
            ledger_sink=ledger_sink,
        )

        correlation_id = uuid4()
        await runtime.call("find_by_id", 1, correlation_id=correlation_id)

        requested = ledger_sink.events[0]
        expected_key = f"{correlation_id}:find_by_id:db.query.requested"
        assert requested.idempotency_key == expected_key

    @pytest.mark.asyncio
    async def test_no_events_without_sink(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test that no events are emitted without ledger_sink."""
        connection = mock_pool.acquire.return_value.__aenter__.return_value
        connection.fetchrow = AsyncMock(return_value={"id": 1})

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,
            contract=test_contract,
            config=test_config,
            ledger_sink=None,  # No sink
        )

        result = await runtime.call("find_by_id", 1)

        assert result == {"id": 1}
        # No way to check events without sink, but should not raise

    @pytest.mark.asyncio
    async def test_contract_fingerprint_is_stable(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test that contract fingerprint is computed once and stable."""
        runtime1 = PostgresRepositoryRuntime(
            pool=mock_pool, contract=test_contract, config=test_config
        )
        runtime2 = PostgresRepositoryRuntime(
            pool=mock_pool, contract=test_contract, config=test_config
        )

        # Same contract should produce same fingerprint
        assert runtime1.contract_fingerprint == runtime2.contract_fingerprint
        assert runtime1.contract_fingerprint.startswith("sha256:")

    @pytest.mark.asyncio
    async def test_multi_row_query_counts_rows(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
        ledger_sink: InMemoryLedgerSink,
    ) -> None:
        """Test that multi-row queries report correct row count."""
        connection = mock_pool.acquire.return_value.__aenter__.return_value
        connection.fetch = AsyncMock(return_value=[{"id": 1}, {"id": 2}, {"id": 3}])

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,
            contract=test_contract,
            config=test_config,
            ledger_sink=ledger_sink,
        )

        result = await runtime.call("find_all")

        assert len(result) == 3

        succeeded = ledger_sink.events[1]
        assert isinstance(succeeded, ModelDbQuerySucceeded)
        assert succeeded.rows_returned == 3

    @pytest.mark.asyncio
    async def test_emits_failed_on_timeout(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
        ledger_sink: InMemoryLedgerSink,
    ) -> None:
        """Test that timeout emits requested and failed events with retriable=True."""
        # Setup mock to raise TimeoutError (simulating asyncio.wait_for timeout)
        connection = mock_pool.acquire.return_value.__aenter__.return_value
        connection.fetchrow = AsyncMock(side_effect=TimeoutError("Query timed out"))

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,
            contract=test_contract,
            config=test_config,
            ledger_sink=ledger_sink,
        )

        from omnibase_infra.errors.repository import RepositoryTimeoutError

        with pytest.raises(RepositoryTimeoutError):
            await runtime.call("find_by_id", 1)

        assert ledger_sink.pending_count == 2

        # Check requested event
        requested = ledger_sink.events[0]
        assert isinstance(requested, ModelDbQueryRequested)

        # Check failed event
        failed = ledger_sink.events[1]
        assert isinstance(failed, ModelDbQueryFailed)
        assert failed.event_type == "db.query.failed"
        assert failed.error_type == "RepositoryTimeoutError"
        assert failed.retriable is True  # Timeout is retriable
        assert failed.duration_ms >= 0


@pytest.mark.unit
class TestRetriableErrorClassification:
    """Tests for _is_retriable_error() error classification."""

    @pytest.fixture
    def runtime(
        self,
        mock_pool: MagicMock,
        test_contract: ModelDbRepositoryContract,
        test_config: ModelRepositoryRuntimeConfig,
    ) -> PostgresRepositoryRuntime:
        """Create a runtime instance for testing."""
        return PostgresRepositoryRuntime(
            pool=mock_pool, contract=test_contract, config=test_config
        )

    def test_timeout_error_is_retriable(
        self, runtime: PostgresRepositoryRuntime
    ) -> None:
        """Test that TimeoutError is classified as retriable."""
        assert runtime._is_retriable_error(TimeoutError("operation timed out")) is True

    def test_connection_refused_is_retriable(
        self, runtime: PostgresRepositoryRuntime
    ) -> None:
        """Test that ConnectionRefusedError is classified as retriable."""
        assert runtime._is_retriable_error(ConnectionRefusedError()) is True

    def test_connection_reset_is_retriable(
        self, runtime: PostgresRepositoryRuntime
    ) -> None:
        """Test that ConnectionResetError is classified as retriable."""
        assert runtime._is_retriable_error(ConnectionResetError()) is True

    def test_broken_pipe_is_retriable(self, runtime: PostgresRepositoryRuntime) -> None:
        """Test that BrokenPipeError is classified as retriable."""
        assert runtime._is_retriable_error(BrokenPipeError()) is True

    def test_os_error_is_retriable(self, runtime: PostgresRepositoryRuntime) -> None:
        """Test that OSError is classified as retriable."""
        assert runtime._is_retriable_error(OSError("network unreachable")) is True

    def test_value_error_not_retriable(
        self, runtime: PostgresRepositoryRuntime
    ) -> None:
        """Test that ValueError is NOT classified as retriable."""
        assert runtime._is_retriable_error(ValueError("invalid value")) is False

    def test_key_error_not_retriable(self, runtime: PostgresRepositoryRuntime) -> None:
        """Test that KeyError is NOT classified as retriable."""
        assert runtime._is_retriable_error(KeyError("missing key")) is False

    def test_runtime_error_not_retriable(
        self, runtime: PostgresRepositoryRuntime
    ) -> None:
        """Test that RuntimeError is NOT classified as retriable."""
        assert runtime._is_retriable_error(RuntimeError("general error")) is False

    def test_connection_keyword_in_message_is_retriable(
        self, runtime: PostgresRepositoryRuntime
    ) -> None:
        """Test that errors with 'connection' in message are retriable."""
        assert (
            runtime._is_retriable_error(Exception("Connection to database lost"))
            is True
        )

    def test_timeout_keyword_in_message_is_retriable(
        self, runtime: PostgresRepositoryRuntime
    ) -> None:
        """Test that errors with 'timeout' in message are retriable."""
        assert (
            runtime._is_retriable_error(Exception("Request timeout exceeded")) is True
        )

    def test_constraint_violation_not_retriable(
        self, runtime: PostgresRepositoryRuntime
    ) -> None:
        """Test that constraint violations are NOT retriable."""
        assert (
            runtime._is_retriable_error(
                Exception("unique constraint violated for key 'email'")
            )
            is False
        )

    def test_syntax_error_not_retriable(
        self, runtime: PostgresRepositoryRuntime
    ) -> None:
        """Test that SQL syntax errors are NOT retriable."""
        assert (
            runtime._is_retriable_error(Exception("syntax error at or near 'SELECT'"))
            is False
        )
