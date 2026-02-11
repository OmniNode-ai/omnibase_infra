# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for util_db_transaction module.

This test suite verifies the transaction_context async context manager
for asyncpg database transactions. Tests use mocked asyncpg objects
to validate behavior without requiring a real database.

Test Organization:
    - TestTransactionContextBasic: Core transaction context functionality
    - TestTransactionContextIsolation: Isolation level parameter handling
    - TestTransactionContextReadonly: Readonly and deferrable flag handling
    - TestTransactionContextTimeout: Statement timeout handling
    - TestTransactionContextLogging: Correlation ID logging behavior
    - TestTransactionContextExceptionHandling: Rollback on exception

Coverage Goals:
    - transaction_context function: All code paths
    - Isolation level propagation
    - Timeout setting via SET LOCAL
    - Logging with correlation ID
    - Exception handling and automatic rollback
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.utils.util_db_transaction import transaction_context

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class MockTransaction:
    """Mock asyncpg transaction context manager."""

    def __init__(
        self,
        isolation: str = "read_committed",
        readonly: bool = False,
        deferrable: bool = False,
    ) -> None:
        """Store transaction parameters for verification."""
        self.isolation = isolation
        self.readonly = readonly
        self.deferrable = deferrable
        self._entered = False
        self._exited = False

    async def __aenter__(self) -> MockTransaction:
        """Enter transaction context."""
        self._entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit transaction context."""
        self._exited = True


class MockConnection:
    """Mock asyncpg connection."""

    def __init__(self) -> None:
        """Initialize mock connection."""
        self.execute = AsyncMock()
        self.fetch = AsyncMock(return_value=[])
        self.fetchval = AsyncMock(return_value=None)
        self._last_transaction_params: dict[str, object] = {}
        self._transactions: list[MockTransaction] = []

    def transaction(
        self,
        isolation: str = "read_committed",
        readonly: bool = False,
        deferrable: bool = False,
    ) -> MockTransaction:
        """Create mock transaction context manager."""
        self._last_transaction_params = {
            "isolation": isolation,
            "readonly": readonly,
            "deferrable": deferrable,
        }
        txn = MockTransaction(
            isolation=isolation, readonly=readonly, deferrable=deferrable
        )
        self._transactions.append(txn)
        return txn


class MockPool:
    """Mock asyncpg connection pool."""

    def __init__(self, connection: MockConnection | None = None) -> None:
        """Initialize mock pool with optional connection."""
        self._connection = connection or MockConnection()
        self._acquired = False

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[MockConnection]:
        """Acquire connection from pool."""
        self._acquired = True
        yield self._connection


@pytest.mark.unit
class TestTransactionContextBasic:
    """Test suite for basic transaction_context functionality.

    Tests verify:
    - Connection is acquired from pool
    - Transaction is started
    - Connection is yielded correctly
    - Transaction commits on successful exit
    """

    @pytest.mark.asyncio
    async def test_acquires_connection_from_pool(self) -> None:
        """Test transaction_context acquires connection from pool."""
        pool = MockPool()

        async with transaction_context(pool) as conn:  # type: ignore[arg-type]
            # Connection should be the mock connection
            assert conn is pool._connection

        # Pool should have been acquired
        assert pool._acquired is True

    @pytest.mark.asyncio
    async def test_starts_transaction(self) -> None:
        """Test transaction_context starts a transaction."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool) as conn:  # type: ignore[arg-type]
            pass

        # Transaction should have been created and entered
        assert len(connection._transactions) == 1
        txn = connection._transactions[0]
        assert txn._entered is True
        assert txn._exited is True

    @pytest.mark.asyncio
    async def test_yields_connection_for_queries(self) -> None:
        """Test yielded connection can execute queries."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool) as conn:  # type: ignore[arg-type]
            await conn.execute("SELECT 1")

        connection.execute.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_default_isolation_is_read_committed(self) -> None:
        """Test default isolation level is read_committed."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool):  # type: ignore[arg-type]
            pass

        assert connection._last_transaction_params["isolation"] == "read_committed"


@pytest.mark.unit
class TestTransactionContextIsolation:
    """Test suite for isolation level parameter handling.

    Tests verify:
    - read_committed is passed correctly
    - repeatable_read is passed correctly
    - serializable is passed correctly
    """

    @pytest.mark.asyncio
    async def test_read_committed_isolation(self) -> None:
        """Test read_committed isolation level is passed to transaction."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool, isolation="read_committed"):  # type: ignore[arg-type]
            pass

        assert connection._last_transaction_params["isolation"] == "read_committed"

    @pytest.mark.asyncio
    async def test_repeatable_read_isolation(self) -> None:
        """Test repeatable_read isolation level is passed to transaction."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool, isolation="repeatable_read"):  # type: ignore[arg-type]
            pass

        assert connection._last_transaction_params["isolation"] == "repeatable_read"

    @pytest.mark.asyncio
    async def test_serializable_isolation(self) -> None:
        """Test serializable isolation level is passed to transaction."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool, isolation="serializable"):  # type: ignore[arg-type]
            pass

        assert connection._last_transaction_params["isolation"] == "serializable"


@pytest.mark.unit
class TestTransactionContextReadonly:
    """Test suite for readonly and deferrable flag handling.

    Tests verify:
    - readonly=False is the default
    - readonly=True is passed correctly
    - deferrable=False is the default
    - deferrable=True is passed correctly
    """

    @pytest.mark.asyncio
    async def test_default_readonly_is_false(self) -> None:
        """Test default readonly is False."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool):  # type: ignore[arg-type]
            pass

        assert connection._last_transaction_params["readonly"] is False

    @pytest.mark.asyncio
    async def test_readonly_true_passed(self) -> None:
        """Test readonly=True is passed to transaction."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool, readonly=True):  # type: ignore[arg-type]
            pass

        assert connection._last_transaction_params["readonly"] is True

    @pytest.mark.asyncio
    async def test_default_deferrable_is_false(self) -> None:
        """Test default deferrable is False."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool):  # type: ignore[arg-type]
            pass

        assert connection._last_transaction_params["deferrable"] is False

    @pytest.mark.asyncio
    async def test_deferrable_true_passed(self) -> None:
        """Test deferrable=True is passed to transaction."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool, deferrable=True):  # type: ignore[arg-type]
            pass

        assert connection._last_transaction_params["deferrable"] is True

    @pytest.mark.asyncio
    async def test_serializable_readonly_deferrable_combination(self) -> None:
        """Test combination of serializable, readonly, and deferrable flags."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(
            pool,  # type: ignore[arg-type]
            isolation="serializable",
            readonly=True,
            deferrable=True,
        ):
            pass

        assert connection._last_transaction_params["isolation"] == "serializable"
        assert connection._last_transaction_params["readonly"] is True
        assert connection._last_transaction_params["deferrable"] is True


@pytest.mark.unit
class TestTransactionContextTimeout:
    """Test suite for statement timeout handling.

    Tests verify:
    - No timeout set when timeout=None (default)
    - SET LOCAL statement_timeout executed when timeout provided
    - Timeout converted to milliseconds correctly
    """

    @pytest.mark.asyncio
    async def test_no_timeout_by_default(self) -> None:
        """Test no SET LOCAL is called when timeout is None."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool):  # type: ignore[arg-type]
            pass

        # execute should not be called for SET LOCAL
        connection.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_sets_statement_timeout(self) -> None:
        """Test timeout parameter sets statement_timeout via SET LOCAL."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool, timeout=5.0):  # type: ignore[arg-type]
            pass

        # SET LOCAL should be called with timeout in milliseconds
        connection.execute.assert_called_once_with(
            "SET LOCAL statement_timeout = '5000'"
        )

    @pytest.mark.asyncio
    async def test_timeout_converted_to_milliseconds(self) -> None:
        """Test timeout is correctly converted to milliseconds."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool, timeout=1.5):  # type: ignore[arg-type]
            pass

        # 1.5 seconds = 1500 milliseconds
        connection.execute.assert_called_once_with(
            "SET LOCAL statement_timeout = '1500'"
        )

    @pytest.mark.asyncio
    async def test_fractional_timeout_truncated(self) -> None:
        """Test fractional milliseconds are truncated."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool, timeout=0.0015):  # type: ignore[arg-type]
            pass

        # 0.0015 seconds = 1.5 milliseconds, truncated to 1
        connection.execute.assert_called_once_with("SET LOCAL statement_timeout = '1'")

    @pytest.mark.asyncio
    async def test_timeout_zero(self) -> None:
        """Test timeout=0 sets statement_timeout to 0 (no timeout)."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool, timeout=0.0):  # type: ignore[arg-type]
            pass

        connection.execute.assert_called_once_with("SET LOCAL statement_timeout = '0'")


@pytest.mark.unit
class TestTransactionContextLogging:
    """Test suite for correlation ID logging behavior.

    Tests verify:
    - No logging when correlation_id is None
    - Transaction start logged with correlation_id
    - Transaction commit logged with correlation_id
    - Transaction rollback logged with correlation_id
    """

    @pytest.mark.asyncio
    async def test_no_logging_without_correlation_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test no debug logging when correlation_id is None."""
        connection = MockConnection()
        pool = MockPool(connection)

        with caplog.at_level(logging.DEBUG):
            async with transaction_context(pool):  # type: ignore[arg-type]
                pass

        # Should not have any transaction logging
        assert "Starting database transaction" not in caplog.text
        assert "Database transaction committed" not in caplog.text

    @pytest.mark.asyncio
    async def test_transaction_start_logged_with_correlation_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test transaction start is logged when correlation_id provided."""
        connection = MockConnection()
        pool = MockPool(connection)
        corr_id = uuid4()

        with caplog.at_level(logging.DEBUG):
            async with transaction_context(pool, correlation_id=corr_id):  # type: ignore[arg-type]
                pass

        assert "Starting database transaction" in caplog.text

    @pytest.mark.asyncio
    async def test_transaction_commit_logged_with_correlation_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test transaction commit is logged when correlation_id provided."""
        connection = MockConnection()
        pool = MockPool(connection)
        corr_id = uuid4()

        with caplog.at_level(logging.DEBUG):
            async with transaction_context(pool, correlation_id=corr_id):  # type: ignore[arg-type]
                pass

        assert "Database transaction committed" in caplog.text

    @pytest.mark.asyncio
    async def test_correlation_id_in_log_extra(self) -> None:
        """Test correlation_id is included in log extra data."""
        connection = MockConnection()
        pool = MockPool(connection)
        corr_id = uuid4()

        with patch("omnibase_infra.utils.util_db_transaction.logger") as mock_logger:
            async with transaction_context(pool, correlation_id=corr_id):  # type: ignore[arg-type]
                pass

            # Check start log call
            start_call = mock_logger.debug.call_args_list[0]
            assert str(corr_id) in str(start_call)

    @pytest.mark.asyncio
    async def test_isolation_level_in_start_log(self) -> None:
        """Test isolation level is included in start log."""
        connection = MockConnection()
        pool = MockPool(connection)
        corr_id = uuid4()

        with patch("omnibase_infra.utils.util_db_transaction.logger") as mock_logger:
            async with transaction_context(
                pool,  # type: ignore[arg-type]
                isolation="serializable",
                correlation_id=corr_id,
            ):
                pass

            start_call = mock_logger.debug.call_args_list[0]
            assert "extra" in start_call.kwargs
            assert start_call.kwargs["extra"]["isolation"] == "serializable"


@pytest.mark.unit
class TestTransactionContextExceptionHandling:
    """Test suite for exception handling and rollback.

    Tests verify:
    - Exceptions propagate out of context manager
    - Transaction is rolled back on exception
    - Rollback is logged when correlation_id provided
    """

    @pytest.mark.asyncio
    async def test_exception_propagates(self) -> None:
        """Test exceptions raised within context propagate out."""
        connection = MockConnection()
        pool = MockPool(connection)

        with pytest.raises(ValueError, match="test error"):
            async with transaction_context(pool):  # type: ignore[arg-type]
                raise ValueError("test error")

    @pytest.mark.asyncio
    async def test_transaction_exited_on_exception(self) -> None:
        """Test transaction __aexit__ is called even on exception."""
        connection = MockConnection()
        pool = MockPool(connection)

        with pytest.raises(RuntimeError):
            async with transaction_context(pool):  # type: ignore[arg-type]
                raise RuntimeError("boom")

        # Transaction should have exited (which triggers rollback in asyncpg)
        assert len(connection._transactions) == 1
        assert connection._transactions[0]._exited is True

    @pytest.mark.asyncio
    async def test_rollback_logged_on_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test transaction rollback is logged when correlation_id provided."""
        connection = MockConnection()
        pool = MockPool(connection)
        corr_id = uuid4()

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError):
                async with transaction_context(pool, correlation_id=corr_id):  # type: ignore[arg-type]
                    raise ValueError("test")

        assert "Database transaction rolled back" in caplog.text

    @pytest.mark.asyncio
    async def test_no_commit_log_on_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test commit is not logged when exception occurs."""
        connection = MockConnection()
        pool = MockPool(connection)
        corr_id = uuid4()

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError):
                async with transaction_context(pool, correlation_id=corr_id):  # type: ignore[arg-type]
                    raise ValueError("test")

        # Should see rollback but not commit
        assert "Database transaction rolled back" in caplog.text
        assert "Database transaction committed" not in caplog.text


@pytest.mark.unit
class TestTransactionContextCombinations:
    """Test suite for various parameter combinations.

    Tests verify common real-world parameter combinations work correctly.
    """

    @pytest.mark.asyncio
    async def test_full_parameter_combination(self) -> None:
        """Test all parameters provided together."""
        connection = MockConnection()
        pool = MockPool(connection)
        corr_id = uuid4()

        async with transaction_context(
            pool,  # type: ignore[arg-type]
            isolation="serializable",
            readonly=True,
            deferrable=True,
            timeout=30.0,
            correlation_id=corr_id,
        ) as conn:
            # Use connection
            await conn.fetchval("SELECT 1")

        # Verify all parameters were passed
        assert connection._last_transaction_params["isolation"] == "serializable"
        assert connection._last_transaction_params["readonly"] is True
        assert connection._last_transaction_params["deferrable"] is True

        # Verify timeout was set
        connection.execute.assert_called_with("SET LOCAL statement_timeout = '30000'")

        # Verify fetchval was called
        connection.fetchval.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_multiple_queries_in_transaction(self) -> None:
        """Test multiple queries can be executed in same transaction."""
        connection = MockConnection()
        pool = MockPool(connection)

        async with transaction_context(pool) as conn:  # type: ignore[arg-type]
            await conn.execute("INSERT INTO t1 VALUES ($1)", 1)
            await conn.execute("INSERT INTO t2 VALUES ($1)", 2)
            await conn.fetch("SELECT * FROM t1")

        assert connection.execute.call_count == 2
        connection.fetch.assert_called_once()
