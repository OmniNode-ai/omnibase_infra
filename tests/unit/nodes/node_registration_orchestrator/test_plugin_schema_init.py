# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceRegistration._initialize_schema (OMN-3567).

Verifies the deadlock-prevention logic introduced to serialize concurrent
schema initialization via pg_advisory_xact_lock.

R1: Schema migration runs at most once concurrently — advisory lock acquired
    before schema SQL executes, serializing concurrent callers.
R1: Idempotent — DuplicateTableError and DuplicateObjectError are silently
    handled; no exception propagates to the caller.
R2: Workers do not block silently — unexpected errors are logged as WARNING
    but do not propagate (plugin startup completes, avoids silent hang).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

_PLUGIN_MOD = "omnibase_infra.nodes.node_registration_orchestrator.plugin"


# =============================================================================
# Helpers
# =============================================================================


def _make_conn_mock_with_tracking() -> tuple[AsyncMock, list[str]]:
    """Return a (conn_mock, call_order) pair.

    The conn_mock records each execute() call as either 'advisory_lock' or
    'schema_sql' so tests can assert execution order.
    """
    call_order: list[str] = []
    conn = AsyncMock()

    async def _execute(sql: str) -> None:
        if "pg_advisory_xact_lock" in sql:
            call_order.append("advisory_lock")
        else:
            call_order.append("schema_sql")

    conn.execute = AsyncMock(side_effect=_execute)

    @asynccontextmanager
    async def _transaction() -> AsyncIterator[None]:
        yield

    conn.transaction = MagicMock(side_effect=_transaction)
    return conn, call_order


def _make_conn_mock_raising(*, on_call: int, error: Exception) -> AsyncMock:
    """Return a conn_mock whose execute() raises *error* on call number *on_call* (1-indexed)."""
    conn = AsyncMock()
    call_count = 0

    async def _execute(sql: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == on_call:
            raise error

    conn.execute = AsyncMock(side_effect=_execute)

    @asynccontextmanager
    async def _transaction() -> AsyncIterator[None]:
        yield

    conn.transaction = MagicMock(side_effect=_transaction)
    return conn


def _make_conn_mock_simple() -> AsyncMock:
    """Return a conn_mock whose execute() always succeeds."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)

    @asynccontextmanager
    async def _transaction() -> AsyncIterator[None]:
        yield

    conn.transaction = MagicMock(side_effect=_transaction)
    return conn


def _make_pool_mock(conn: AsyncMock) -> MagicMock:
    """Wrap *conn* in a mock asyncpg Pool."""
    pool = MagicMock()

    @asynccontextmanager
    async def _acquire() -> AsyncIterator[AsyncMock]:
        yield conn

    pool.acquire = MagicMock(side_effect=_acquire)
    return pool


def _make_config() -> MagicMock:
    """Build a minimal ModelDomainPluginConfig-like mock."""
    config = MagicMock()
    config.correlation_id = uuid4()
    return config


def _make_plugin_with_pool(pool: MagicMock) -> MagicMock:
    """Construct a ServiceRegistration with an injected pool.

    Returns the plugin as MagicMock to avoid a forward-reference import at
    module level; the object is a real ServiceRegistration at runtime.
    """
    from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
        ServiceRegistration,
    )

    plugin = ServiceRegistration()
    plugin._pool = pool  # type: ignore[attr-defined]
    return plugin  # type: ignore[return-value]


# =============================================================================
# TestSchemaInitAdvisoryLock
# =============================================================================


class TestSchemaInitAdvisoryLock:
    """Advisory lock is acquired before schema SQL executes (R1)."""

    @pytest.mark.unit
    async def test_advisory_lock_inside_transaction(self) -> None:
        """Advisory lock must execute while conn.transaction() is active.

        The entire deadlock fix (OMN-3567) relies on pg_advisory_xact_lock
        being called *inside* ``async with conn.transaction()`` so that the
        lock auto-releases on commit/rollback.  The other ordering tests
        (``test_advisory_lock_called_before_schema_sql``, etc.) would still
        pass if the advisory lock call were moved *outside* the transaction
        block, because their mock ``_transaction()`` is a no-op passthrough.

        This test uses a transaction-tracking context manager that flips a
        flag while the ``async with conn.transaction()`` block is active.
        The ``execute()`` side-effect captures the flag value at call time,
        so the assertion proves the advisory lock runs while the transaction
        is open.
        """
        transaction_active = False
        advisory_called_inside: list[bool] = []
        schema_called_inside: list[bool] = []

        @asynccontextmanager
        async def _tracking_transaction() -> AsyncIterator[None]:
            nonlocal transaction_active
            transaction_active = True
            try:
                yield
            finally:
                transaction_active = False

        conn = AsyncMock()

        async def _execute(sql: str) -> None:
            if "pg_advisory_xact_lock" in sql:
                advisory_called_inside.append(transaction_active)
            else:
                schema_called_inside.append(transaction_active)

        conn.execute = AsyncMock(side_effect=_execute)
        conn.transaction = MagicMock(side_effect=_tracking_transaction)

        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.read_text",
                return_value="CREATE TABLE IF NOT EXISTS t (id serial);",
            ),
        ):
            await plugin._initialize_schema(config)  # type: ignore[attr-defined]

        assert advisory_called_inside == [True], (
            "pg_advisory_xact_lock must execute while conn.transaction() is "
            "active so the lock auto-releases on commit/rollback. "
            f"transaction_active at call time: {advisory_called_inside}"
        )
        assert schema_called_inside == [True], (
            "Schema SQL must also execute inside the transaction block. "
            f"transaction_active at call time: {schema_called_inside}"
        )

    @pytest.mark.unit
    async def test_advisory_lock_called_before_schema_sql(self) -> None:
        """pg_advisory_xact_lock execute precedes schema SQL execute (R1).

        The implementation must call:
            1. execute(SELECT pg_advisory_xact_lock(...))
            2. execute(<schema SQL>)

        in that order inside a transaction.
        """
        conn, call_order = _make_conn_mock_with_tracking()
        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.read_text",
                return_value="CREATE TABLE IF NOT EXISTS t (id serial);",
            ),
        ):
            await plugin._initialize_schema(config)  # type: ignore[attr-defined]

        assert call_order == ["advisory_lock", "schema_sql"], (
            "Expected pg_advisory_xact_lock to be called before schema SQL. "
            f"Actual order: {call_order}"
        )

    @pytest.mark.unit
    async def test_advisory_lock_key_contains_registration(self) -> None:
        """Advisory lock key is deterministic and contains 'registration' (R1)."""
        conn = AsyncMock()
        advisory_sql_calls: list[str] = []

        async def _execute(sql: str) -> None:
            if "pg_advisory_xact_lock" in sql:
                advisory_sql_calls.append(sql)

        conn.execute = AsyncMock(side_effect=_execute)

        @asynccontextmanager
        async def _transaction() -> AsyncIterator[None]:
            yield

        conn.transaction = MagicMock(side_effect=_transaction)

        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="-- schema"),
        ):
            await plugin._initialize_schema(config)  # type: ignore[attr-defined]

        assert len(advisory_sql_calls) == 1
        assert "registration_projection_schema_init" in advisory_sql_calls[0], (
            f"Advisory lock key should contain 'registration_projection_schema_init'. "
            f"Got: {advisory_sql_calls[0]!r}"
        )

    @pytest.mark.unit
    async def test_execute_called_twice_advisory_then_schema(self) -> None:
        """execute() is called exactly twice: advisory lock then schema SQL."""
        conn = _make_conn_mock_simple()
        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        schema_sql_text = "CREATE TABLE IF NOT EXISTS t (id serial);"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value=schema_sql_text),
        ):
            await plugin._initialize_schema(config)  # type: ignore[attr-defined]

        assert conn.execute.call_count == 2, (
            f"Expected 2 execute() calls, got {conn.execute.call_count}"
        )
        first_call_sql: str = conn.execute.call_args_list[0][0][0]
        second_call_sql: str = conn.execute.call_args_list[1][0][0]

        assert "pg_advisory_xact_lock" in first_call_sql, (
            f"First execute() should be advisory lock, got: {first_call_sql!r}"
        )
        assert second_call_sql == schema_sql_text, (
            f"Second execute() should be schema SQL, got: {second_call_sql!r}"
        )


# =============================================================================
# TestSchemaInitIdempotency
# =============================================================================


class TestSchemaInitIdempotency:
    """Duplicate object/table errors are silently handled (R1 idempotency)."""

    @pytest.mark.unit
    async def test_duplicate_table_error_is_silent(self) -> None:
        """DuplicateTableError is caught and not re-raised (R1)."""
        import asyncpg.exceptions

        duplicate_error = asyncpg.exceptions.DuplicateTableError(
            "relation already exists"
        )
        # Second call (schema SQL) raises the duplicate error
        conn = _make_conn_mock_raising(on_call=2, error=duplicate_error)
        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.read_text",
                return_value="CREATE TABLE IF NOT EXISTS registration_projections (id uuid);",
            ),
        ):
            # Must NOT raise — idempotent
            await plugin._initialize_schema(config)  # type: ignore[attr-defined]

    @pytest.mark.unit
    async def test_duplicate_object_error_is_silent(self) -> None:
        """DuplicateObjectError (index already exists) is caught at DEBUG, not re-raised (R1)."""
        import asyncpg.exceptions

        duplicate_error = asyncpg.exceptions.DuplicateObjectError(
            "object already exists"
        )
        conn = _make_conn_mock_raising(on_call=2, error=duplicate_error)
        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.read_text",
                return_value="CREATE INDEX IF NOT EXISTS idx ON t (c);",
            ),
        ):
            # Must NOT raise
            await plugin._initialize_schema(config)  # type: ignore[attr-defined]

    @pytest.mark.unit
    async def test_duplicate_table_error_logged_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """DuplicateTableError is logged at DEBUG level (not WARNING/ERROR)."""
        import asyncpg.exceptions

        duplicate_error = asyncpg.exceptions.DuplicateTableError("table exists")
        conn = _make_conn_mock_raising(on_call=2, error=duplicate_error)
        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        plugin_logger = f"{_PLUGIN_MOD}"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="-- schema"),
        ):
            with caplog.at_level(logging.DEBUG, logger=plugin_logger):
                await plugin._initialize_schema(config)  # type: ignore[attr-defined]

        debug_msgs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("idempotent" in r.message.lower() for r in debug_msgs), (
            "Expected debug log containing 'idempotent' for DuplicateTableError. "
            f"Got DEBUG records: {[r.message for r in debug_msgs]}"
        )

        # Must NOT have WARNING-level log for this specific duplicate error
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("duplicate" in r.message.lower() for r in warning_msgs), (
            f"Unexpected WARNING for duplicate table error: {[r.message for r in warning_msgs]}"
        )


# =============================================================================
# TestSchemaInitErrorHandling
# =============================================================================


class TestSchemaInitErrorHandling:
    """Unexpected errors are logged as WARNING; startup does not hang (R2)."""

    @pytest.mark.unit
    async def test_unexpected_error_logged_as_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unexpected RuntimeError is logged at WARNING level and not re-raised (R2)."""
        unexpected_error = RuntimeError("database connection reset")
        conn = _make_conn_mock_raising(on_call=2, error=unexpected_error)
        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        plugin_logger = _PLUGIN_MOD

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="-- schema"),
        ):
            with caplog.at_level(logging.WARNING, logger=plugin_logger):
                # Must NOT raise (R2: clear error, not silent hang)
                await plugin._initialize_schema(config)  # type: ignore[attr-defined]

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "schema" in r.message.lower() or "initialize" in r.message.lower()
            for r in warning_msgs
        ), (
            "WARNING must reference schema initialization, not be a spurious log. "
            f"Got: {[r.message for r in warning_msgs]}"
        )

    @pytest.mark.unit
    async def test_unexpected_error_does_not_propagate(self) -> None:
        """Unexpected Exception during schema exec does not propagate (R2: no silent hang)."""
        conn = _make_conn_mock_raising(on_call=2, error=OSError("filesystem error"))
        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="-- schema"),
        ):
            # Must not raise — workers should start even if schema init fails
            await plugin._initialize_schema(config)  # type: ignore[attr-defined]

    @pytest.mark.unit
    async def test_no_pool_returns_early_without_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """pool is None -> method returns early, logs WARNING but does not raise (R2)."""
        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            ServiceRegistration,
        )

        plugin = ServiceRegistration()
        plugin._pool = None  # type: ignore[attr-defined]
        config = _make_config()

        plugin_logger = _PLUGIN_MOD

        with (
            patch("pathlib.Path.exists", return_value=True),
            caplog.at_level(logging.WARNING, logger=plugin_logger),
        ):
            await plugin._initialize_schema(config)  # type: ignore[attr-defined]

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("pool" in r.message.lower() for r in warning_msgs), (
            f"Expected WARNING about pool being None. Got: {[r.message for r in warning_msgs]}"
        )

    @pytest.mark.unit
    async def test_missing_schema_file_returns_early_without_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Schema file not found -> method returns early, logs WARNING but does not raise (R2)."""
        pool = MagicMock()
        pool.acquire = MagicMock()  # Should NOT be called

        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        plugin_logger = _PLUGIN_MOD

        with (
            patch("pathlib.Path.exists", return_value=False),
            caplog.at_level(logging.WARNING, logger=plugin_logger),
        ):
            await plugin._initialize_schema(config)  # type: ignore[attr-defined]

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("schema file" in r.message.lower() for r in warning_msgs), (
            f"Expected WARNING about missing schema file. Got: {[r.message for r in warning_msgs]}"
        )

        # Pool should NOT have been touched (early return before pool access)
        pool.acquire.assert_not_called()

    @pytest.mark.unit
    async def test_advisory_lock_failure_does_not_propagate(self) -> None:
        """Advisory lock failure (on_call=1) does not propagate out of _initialize_schema (R2).

        If pg_advisory_xact_lock raises (e.g., lock wait timeout, connection drop,
        role permission error), the exception falls into the generic except branch
        and must be swallowed. A refactor that accidentally re-raised on advisory
        lock failure would be caught by this test.
        """
        conn = _make_conn_mock_raising(
            on_call=1, error=RuntimeError("lock wait timeout")
        )
        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="-- schema"),
        ):
            # Must NOT raise — advisory lock failure must be swallowed (R2)
            await plugin._initialize_schema(config)  # type: ignore[attr-defined]

    @pytest.mark.unit
    async def test_advisory_lock_failure_logged_as_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Advisory lock failure (on_call=1) is logged as WARNING, not ERROR (R2).

        Covers the untested branch: when pg_advisory_xact_lock itself raises,
        the code falls into the generic except handler and emits WARNING.
        No ERROR or exception must propagate to the caller.
        """
        conn = _make_conn_mock_raising(
            on_call=1, error=RuntimeError("lock wait timeout")
        )
        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        plugin_logger = _PLUGIN_MOD

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="-- schema"),
        ):
            with caplog.at_level(logging.WARNING, logger=plugin_logger):
                await plugin._initialize_schema(config)  # type: ignore[attr-defined]

        assert not any(r.levelno >= logging.ERROR for r in caplog.records), (
            "No ERROR should propagate — advisory lock failure must be swallowed. "
            f"Got records: {[(r.levelname, r.message) for r in caplog.records if r.levelno >= logging.ERROR]}"
        )
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "schema" in r.message.lower() or "initialize" in r.message.lower()
            for r in warning_msgs
        ), (
            "WARNING must reference schema initialization, not be a spurious log. "
            f"Got: {[r.message for r in warning_msgs]}"
        )


# =============================================================================
# TestSchemaInitSuccessPath
# =============================================================================


class TestSchemaInitSuccessPath:
    """Happy-path: schema SQL executes and success is logged at INFO."""

    @pytest.mark.unit
    async def test_success_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Successful schema init logs at INFO level."""
        conn = _make_conn_mock_simple()
        pool = _make_pool_mock(conn)
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        plugin_logger = _PLUGIN_MOD

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="-- schema"),
        ):
            with caplog.at_level(logging.INFO, logger=plugin_logger):
                await plugin._initialize_schema(config)  # type: ignore[attr-defined]

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("schema initialized" in r.message.lower() for r in info_msgs), (
            f"Expected INFO log with 'schema initialized'. "
            f"Got: {[r.message for r in info_msgs]}"
        )
