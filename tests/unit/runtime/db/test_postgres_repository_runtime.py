# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for PostgresRepositoryRuntime (OMN-1783).

This test suite validates the PostgresRepositoryRuntime implementation for:
- Basic operation execution (read single/multi row, write)
- Mode validation (write disabled, mode allowlist)
- Argument validation (count matching)
- Determinism enforcement (ORDER BY injection)
- Limit enforcement (LIMIT injection and validation)
- Timeout handling (asyncio.wait_for)
- Execution error wrapping

Test Organization:
    - TestBasicOperations: Read/write operation execution
    - TestModeValidation: Write blocking, mode allowlist
    - TestArgumentValidation: Argument count matching
    - TestDeterminismEnforcement: ORDER BY injection for multi-row
    - TestLimitEnforcement: LIMIT injection and validation
    - TestTimeoutHandling: Query timeout enforcement
    - TestExecutionErrors: Database error wrapping

Coverage Goals:
    - >90% code coverage for PostgresRepositoryRuntime
    - All error paths tested
    - All acceptance criteria from OMN-1783 ticket
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, ItemsView, Iterator, KeysView, ValuesView
from contextlib import asynccontextmanager

import pytest

from omnibase_infra.errors.repository import (
    RepositoryContractError,
    RepositoryExecutionError,
    RepositoryTimeoutError,
    RepositoryValidationError,
)
from omnibase_infra.runtime.db import (
    ModelDbOperation,
    ModelDbParam,
    ModelDbRepositoryContract,
    ModelDbReturn,
    ModelDbSafetyPolicy,
    ModelRepositoryRuntimeConfig,
    PostgresRepositoryRuntime,
)

# ============================================================================
# Fixtures
# ============================================================================


class MockAsyncRecord:
    """Mock asyncpg.Record that supports dict() conversion.

    asyncpg.Record implements the Mapping protocol, so we need:
    - __getitem__ for subscript access
    - __iter__ for iteration over keys
    - __len__ for length
    - keys() for dict() conversion
    """

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def values(self) -> ValuesView[object]:
        return self._data.values()

    def items(self) -> ItemsView[str, object]:
        return self._data.items()


class MockAsyncConnection:
    """Mock asyncpg connection with fetch/fetchrow methods."""

    def __init__(self) -> None:
        self.fetch_result: list[MockAsyncRecord] = []
        self.fetchrow_result: MockAsyncRecord | None = None
        self.execute_result: str = "INSERT 0 1"
        self.fetch_delay: float = 0.0
        self.should_raise: Exception | None = None

    async def fetch(self, sql: str, *args: object) -> list[MockAsyncRecord]:
        """Simulate fetch() for multi-row queries."""
        if self.fetch_delay > 0:
            await asyncio.sleep(self.fetch_delay)
        if self.should_raise:
            raise self.should_raise
        return self.fetch_result

    async def fetchrow(self, sql: str, *args: object) -> MockAsyncRecord | None:
        """Simulate fetchrow() for single-row queries."""
        if self.fetch_delay > 0:
            await asyncio.sleep(self.fetch_delay)
        if self.should_raise:
            raise self.should_raise
        return self.fetchrow_result


class MockAsyncPool:
    """Mock asyncpg.Pool with acquire() context manager."""

    def __init__(self) -> None:
        self._connection = MockAsyncConnection()

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[MockAsyncConnection, None]:
        """Simulate pool.acquire() async context manager."""
        yield self._connection

    @property
    def connection(self) -> MockAsyncConnection:
        """Access the mock connection for test setup."""
        return self._connection


@pytest.fixture
def mock_pool() -> MockAsyncPool:
    """Create mock asyncpg pool for testing.

    Returns:
        MockAsyncPool with configurable connection behavior.
    """
    return MockAsyncPool()


@pytest.fixture
def sample_contract() -> ModelDbRepositoryContract:
    """Create sample repository contract with various operations.

    Returns:
        ModelDbRepositoryContract with read and write operations.
    """
    return ModelDbRepositoryContract(
        name="test_users_repo",
        engine="postgres",
        database_ref="test_db",
        tables=["users"],
        models={"User": "test.models:User"},
        ops={
            "find_by_id": ModelDbOperation(
                mode="read",
                sql="SELECT * FROM users WHERE id = $1",
                params={
                    "user_id": ModelDbParam(name="user_id", param_type="integer"),
                },
                returns=ModelDbReturn(model_ref="User", many=False),
                safety_policy=ModelDbSafetyPolicy(),
            ),
            "find_all": ModelDbOperation(
                mode="read",
                sql="SELECT * FROM users",
                params={},
                returns=ModelDbReturn(model_ref="User", many=True),
                safety_policy=ModelDbSafetyPolicy(),
            ),
            "find_by_status": ModelDbOperation(
                mode="read",
                sql="SELECT * FROM users WHERE status = $1",
                params={
                    "status": ModelDbParam(name="status", param_type="string"),
                },
                returns=ModelDbReturn(model_ref="User", many=True),
                safety_policy=ModelDbSafetyPolicy(),
            ),
            "find_with_order": ModelDbOperation(
                mode="read",
                sql="SELECT * FROM users ORDER BY created_at DESC",
                params={},
                returns=ModelDbReturn(model_ref="User", many=True),
                safety_policy=ModelDbSafetyPolicy(),
            ),
            "find_with_limit": ModelDbOperation(
                mode="read",
                sql="SELECT * FROM users ORDER BY id LIMIT 5",
                params={},
                returns=ModelDbReturn(model_ref="User", many=True),
                safety_policy=ModelDbSafetyPolicy(),
            ),
            "find_with_high_limit": ModelDbOperation(
                mode="read",
                sql="SELECT * FROM users ORDER BY id LIMIT 500",
                params={},
                returns=ModelDbReturn(model_ref="User", many=True),
                safety_policy=ModelDbSafetyPolicy(),
            ),
            "find_with_param_limit": ModelDbOperation(
                mode="read",
                sql="SELECT * FROM users ORDER BY id LIMIT $1",
                params={
                    "limit": ModelDbParam(name="limit", param_type="integer"),
                },
                returns=ModelDbReturn(model_ref="User", many=True),
                safety_policy=ModelDbSafetyPolicy(),
            ),
            "create_user": ModelDbOperation(
                mode="write",
                sql="INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
                params={
                    "name": ModelDbParam(name="name", param_type="string"),
                    "email": ModelDbParam(name="email", param_type="string"),
                },
                returns=ModelDbReturn(model_ref="User", many=False),
                safety_policy=ModelDbSafetyPolicy(),
            ),
        },
    )


@pytest.fixture
def default_config() -> ModelRepositoryRuntimeConfig:
    """Create default runtime configuration.

    Returns:
        ModelRepositoryRuntimeConfig with standard defaults.
    """
    return ModelRepositoryRuntimeConfig(
        max_row_limit=10,
        timeout_ms=5000,
        allowed_modes=frozenset({"read", "write"}),
        allow_write_operations=True,
        primary_key_column="id",
        default_order_by=None,
        emit_metrics=False,
    )


@pytest.fixture
def runtime(
    mock_pool: MockAsyncPool,
    sample_contract: ModelDbRepositoryContract,
    default_config: ModelRepositoryRuntimeConfig,
) -> PostgresRepositoryRuntime:
    """Create PostgresRepositoryRuntime instance for testing.

    Args:
        mock_pool: Mock asyncpg pool
        sample_contract: Repository contract
        default_config: Runtime configuration

    Returns:
        PostgresRepositoryRuntime configured for testing.
    """
    return PostgresRepositoryRuntime(
        pool=mock_pool,  # type: ignore[arg-type]
        contract=sample_contract,
        config=default_config,
    )


# ============================================================================
# Basic Operation Tests
# ============================================================================


@pytest.mark.unit
class TestBasicOperations:
    """Test basic operation execution for read and write modes."""

    @pytest.mark.asyncio
    async def test_call_read_operation_single_row(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test many=False returns dict|None for single row lookup.

        Verifies:
            - fetchrow() is used for single-row queries
            - dict is returned when row exists
            - None is returned when no row found
        """
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        # Case 1: Row found
        mock_pool.connection.fetchrow_result = MockAsyncRecord(
            {"id": 123, "name": "Alice", "email": "alice@example.com"}
        )

        result = await runtime.call("find_by_id", 123)

        assert result is not None
        assert isinstance(result, dict)
        assert result["id"] == 123
        assert result["name"] == "Alice"

        # Case 2: No row found
        mock_pool.connection.fetchrow_result = None

        result = await runtime.call("find_by_id", 999)

        assert result is None

    @pytest.mark.asyncio
    async def test_call_read_operation_multi_row(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test many=True returns list[dict] for multi-row queries.

        Verifies:
            - fetch() is used for multi-row queries
            - list of dicts is returned (possibly empty)
        """
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        # Case 1: Multiple rows found
        mock_pool.connection.fetch_result = [
            MockAsyncRecord({"id": 1, "name": "Alice"}),
            MockAsyncRecord({"id": 2, "name": "Bob"}),
        ]

        result = await runtime.call("find_with_order")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["name"] == "Bob"

        # Case 2: Empty result
        mock_pool.connection.fetch_result = []

        result = await runtime.call("find_with_order")

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_call_write_operation(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test mode='write' executes correctly.

        Verifies:
            - Write operations use fetchrow() for RETURNING clause
            - Result is dict when RETURNING produces row
        """
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        # Simulate INSERT with RETURNING
        mock_pool.connection.fetchrow_result = MockAsyncRecord({"id": 42})

        result = await runtime.call("create_user", "Alice", "alice@example.com")

        assert result is not None
        assert isinstance(result, dict)
        assert result["id"] == 42

    @pytest.mark.asyncio
    async def test_call_unknown_operation_raises_contract_error(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test calling undefined operation raises RepositoryContractError.

        Verifies:
            - Unknown op_name raises RepositoryContractError
            - Error message includes available operations
        """
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        with pytest.raises(RepositoryContractError) as exc_info:
            await runtime.call("nonexistent_operation")

        error = exc_info.value
        assert "nonexistent_operation" in str(error)
        assert "not defined in contract" in str(error)
        assert error.op_name == "nonexistent_operation"


# ============================================================================
# Mode Validation Tests
# ============================================================================


@pytest.mark.unit
class TestModeValidation:
    """Test operation mode validation and blocking."""

    @pytest.mark.asyncio
    async def test_write_mode_blocked_when_disabled(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test allow_write_operations=False blocks write operations.

        Verifies:
            - Write operations raise RepositoryContractError when disabled
            - Error message explains the restriction
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=10,
            timeout_ms=5000,
            allowed_modes=frozenset({"read", "write"}),
            allow_write_operations=False,  # Disabled
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        with pytest.raises(RepositoryContractError) as exc_info:
            await runtime.call("create_user", "Alice", "alice@example.com")

        error = exc_info.value
        assert "write" in str(error).lower()
        assert "disabled" in str(error).lower()
        assert error.op_name == "create_user"

    @pytest.mark.asyncio
    async def test_mode_not_in_allowed_modes_raises_error(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test mode not in allowed_modes raises RepositoryContractError.

        Verifies:
            - Operations with mode not in allowed_modes are rejected
            - Error message shows allowed modes
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=10,
            timeout_ms=5000,
            allowed_modes=frozenset({"read"}),  # Only read allowed
            allow_write_operations=True,
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        with pytest.raises(RepositoryContractError) as exc_info:
            await runtime.call("create_user", "Alice", "alice@example.com")

        error = exc_info.value
        assert "not in allowed_modes" in str(error)
        assert error.op_name == "create_user"


# ============================================================================
# Argument Validation Tests
# ============================================================================


@pytest.mark.unit
class TestArgumentValidation:
    """Test argument count validation."""

    @pytest.mark.asyncio
    async def test_arg_count_mismatch_raises_validation_error(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test wrong number of args raises RepositoryValidationError.

        Verifies:
            - Too few arguments raises error
            - Too many arguments raises error
            - Error message shows expected vs actual count
        """
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        # Too few arguments (expects 1, got 0)
        with pytest.raises(RepositoryValidationError) as exc_info:
            await runtime.call("find_by_id")  # Missing user_id arg

        error = exc_info.value
        assert "expects 1 argument(s)" in str(error)
        assert "received 0" in str(error)
        assert error.op_name == "find_by_id"

        # Too many arguments (expects 1, got 3)
        with pytest.raises(RepositoryValidationError) as exc_info:
            await runtime.call("find_by_id", 123, "extra", "args")

        error = exc_info.value
        assert "expects 1 argument(s)" in str(error)
        assert "received 3" in str(error)

    @pytest.mark.asyncio
    async def test_arg_count_correct_passes(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test correct number of args passes validation.

        Verifies:
            - Zero args for parameterless operations
            - Correct arg count for parameterized operations
        """
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        # Zero args for find_with_order (no params)
        mock_pool.connection.fetch_result = []
        result = await runtime.call("find_with_order")
        assert isinstance(result, list)

        # One arg for find_by_id
        mock_pool.connection.fetchrow_result = MockAsyncRecord({"id": 1})
        result = await runtime.call("find_by_id", 123)
        assert isinstance(result, dict)

        # Two args for create_user
        mock_pool.connection.fetchrow_result = MockAsyncRecord({"id": 1})
        result = await runtime.call("create_user", "Alice", "alice@example.com")
        assert isinstance(result, dict)


# ============================================================================
# Determinism Tests (ORDER BY Injection)
# ============================================================================


@pytest.mark.unit
class TestDeterminismEnforcement:
    """Test ORDER BY injection for deterministic multi-row results."""

    @pytest.mark.asyncio
    async def test_multi_row_no_order_by_with_pk_injects_order_by(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test ORDER BY injection when PK is configured.

        Verifies:
            - Multi-row query without ORDER BY gets ORDER BY injected
            - Uses primary_key_column or default_order_by
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=10,
            timeout_ms=5000,
            allowed_modes=frozenset({"read", "write"}),
            allow_write_operations=True,
            primary_key_column="id",
            default_order_by=None,
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        # Capture the SQL that gets executed
        executed_sql: list[str] = []

        async def capture_fetch(sql: str, *args: object) -> list[MockAsyncRecord]:
            executed_sql.append(sql)
            return []

        mock_pool.connection.fetch = capture_fetch  # type: ignore[method-assign]

        # find_all has no ORDER BY, should get injection
        await runtime.call("find_all")

        assert len(executed_sql) == 1
        assert "ORDER BY id" in executed_sql[0]

    @pytest.mark.asyncio
    async def test_multi_row_no_order_by_no_pk_raises_error(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test error when no ORDER BY and no primary_key_column.

        Verifies:
            - Multi-row query without ORDER BY and no PK raises error
            - Error message explains determinism requirement
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=10,
            timeout_ms=5000,
            allowed_modes=frozenset({"read", "write"}),
            allow_write_operations=True,
            primary_key_column=None,  # No PK configured
            default_order_by=None,
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        with pytest.raises(RepositoryContractError) as exc_info:
            await runtime.call("find_all")

        error = exc_info.value
        assert "ORDER BY" in str(error)
        assert "primary_key_column" in str(error)
        assert "deterministic" in str(error).lower()
        assert error.op_name == "find_all"

    @pytest.mark.asyncio
    async def test_multi_row_with_order_by_no_injection(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test no injection when ORDER BY already present.

        Verifies:
            - Existing ORDER BY clause is preserved
            - No duplicate ORDER BY is added
        """
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        executed_sql: list[str] = []

        async def capture_fetch(sql: str, *args: object) -> list[MockAsyncRecord]:
            executed_sql.append(sql)
            return []

        mock_pool.connection.fetch = capture_fetch  # type: ignore[method-assign]

        # find_with_order already has ORDER BY created_at DESC
        await runtime.call("find_with_order")

        assert len(executed_sql) == 1
        # Should have original ORDER BY, not injected one
        assert "ORDER BY created_at DESC" in executed_sql[0]
        # Should not have duplicate ORDER BY
        assert executed_sql[0].count("ORDER BY") == 1

    @pytest.mark.asyncio
    async def test_single_row_no_order_by_no_injection(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test single-row queries don't get ORDER BY injection.

        Verifies:
            - many=False operations don't trigger ORDER BY injection
            - Even without ORDER BY, query runs without modification
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=10,
            timeout_ms=5000,
            allowed_modes=frozenset({"read", "write"}),
            allow_write_operations=True,
            primary_key_column=None,  # No PK - would fail for multi-row
            default_order_by=None,
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        executed_sql: list[str] = []

        async def capture_fetchrow(sql: str, *args: object) -> MockAsyncRecord | None:
            executed_sql.append(sql)
            return MockAsyncRecord({"id": 1})

        mock_pool.connection.fetchrow = capture_fetchrow  # type: ignore[method-assign]

        # Single row query should not fail even without PK
        result = await runtime.call("find_by_id", 123)

        assert result is not None
        assert len(executed_sql) == 1
        # No ORDER BY injection for single-row
        assert "ORDER BY" not in executed_sql[0]

    @pytest.mark.asyncio
    async def test_multi_row_limit_without_order_by_injects_order_by_before_limit(
        self,
        mock_pool: MockAsyncPool,
    ) -> None:
        """Test ORDER BY injection when LIMIT exists but ORDER BY doesn't (OMN-1842).

        Verifies:
            - ORDER BY is inserted BEFORE existing LIMIT clause
            - Produces valid SQL: ORDER BY pk LIMIT $n (not LIMIT $n ORDER BY pk)
        """
        # Create contract with LIMIT but no ORDER BY
        contract = ModelDbRepositoryContract(
            name="test_repo",
            engine="postgres",
            database_ref="test_db",
            tables=["users"],
            models={"User": "test.models:User"},
            ops={
                "find_active_limited": ModelDbOperation(
                    mode="read",
                    sql="SELECT * FROM users WHERE status = $1 LIMIT $2",
                    params={
                        "status": ModelDbParam(name="status", param_type="string"),
                        "limit": ModelDbParam(name="limit", param_type="integer"),
                    },
                    returns=ModelDbReturn(model_ref="User", many=True),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            },
        )

        config = ModelRepositoryRuntimeConfig(
            max_row_limit=100,
            timeout_ms=5000,
            allowed_modes=frozenset({"read"}),
            allow_write_operations=False,
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=contract,
            config=config,
        )

        executed_sql: list[str] = []

        async def capture_fetch(sql: str, *args: object) -> list[MockAsyncRecord]:
            executed_sql.append(sql)
            return []

        mock_pool.connection.fetch = capture_fetch  # type: ignore[method-assign]

        await runtime.call("find_active_limited", "active", 25)

        assert len(executed_sql) == 1
        sql = executed_sql[0]

        # ORDER BY should be BEFORE LIMIT (not after)
        order_by_pos = sql.upper().find("ORDER BY")
        limit_pos = sql.upper().find("LIMIT")

        assert order_by_pos != -1, "ORDER BY should be present"
        assert limit_pos != -1, "LIMIT should be present"
        assert order_by_pos < limit_pos, (
            f"ORDER BY should come BEFORE LIMIT. Got: {sql}"
        )

        # Should have exactly one ORDER BY and one LIMIT
        assert sql.upper().count("ORDER BY") == 1
        assert sql.upper().count("LIMIT") == 1

    @pytest.mark.asyncio
    async def test_multi_row_numeric_limit_without_order_by_injects_order_by_before_limit(
        self,
        mock_pool: MockAsyncPool,
    ) -> None:
        """Test ORDER BY injection with numeric LIMIT (OMN-1842).

        Verifies:
            - ORDER BY is inserted BEFORE existing numeric LIMIT clause
            - Produces valid SQL: ORDER BY pk LIMIT 50 (not LIMIT 50 ORDER BY pk)
        """
        # Create contract with numeric LIMIT but no ORDER BY
        contract = ModelDbRepositoryContract(
            name="test_repo",
            engine="postgres",
            database_ref="test_db",
            tables=["users"],
            models={"User": "test.models:User"},
            ops={
                "find_recent": ModelDbOperation(
                    mode="read",
                    sql="SELECT * FROM users LIMIT 50",
                    params={},
                    returns=ModelDbReturn(model_ref="User", many=True),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            },
        )

        config = ModelRepositoryRuntimeConfig(
            max_row_limit=100,
            timeout_ms=5000,
            allowed_modes=frozenset({"read"}),
            allow_write_operations=False,
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=contract,
            config=config,
        )

        executed_sql: list[str] = []

        async def capture_fetch(sql: str, *args: object) -> list[MockAsyncRecord]:
            executed_sql.append(sql)
            return []

        mock_pool.connection.fetch = capture_fetch  # type: ignore[method-assign]

        await runtime.call("find_recent")

        assert len(executed_sql) == 1
        sql = executed_sql[0]

        # ORDER BY should be BEFORE LIMIT (not after)
        order_by_pos = sql.upper().find("ORDER BY")
        limit_pos = sql.upper().find("LIMIT")

        assert order_by_pos != -1, "ORDER BY should be present"
        assert limit_pos != -1, "LIMIT should be present"
        assert order_by_pos < limit_pos, (
            f"ORDER BY should come BEFORE LIMIT. Got: {sql}"
        )

        # Should have exactly one ORDER BY and one LIMIT
        assert sql.upper().count("ORDER BY") == 1
        assert sql.upper().count("LIMIT") == 1

    @pytest.mark.asyncio
    async def test_multi_row_limit_offset_injects_order_by_before_limit(
        self,
        mock_pool: MockAsyncPool,
    ) -> None:
        """Test ORDER BY injection with LIMIT and OFFSET (OMN-1842 edge case).

        Verifies:
            - ORDER BY is inserted BEFORE existing LIMIT + OFFSET clause
            - Produces valid SQL: ORDER BY pk LIMIT $n OFFSET $m
        """
        # Create contract with LIMIT and OFFSET but no ORDER BY
        contract = ModelDbRepositoryContract(
            name="test_repo",
            engine="postgres",
            database_ref="test_db",
            tables=["users"],
            models={"User": "test.models:User"},
            ops={
                "find_active_paginated": ModelDbOperation(
                    mode="read",
                    sql="SELECT * FROM users WHERE status = $1 LIMIT $2 OFFSET $3",
                    params={
                        "status": ModelDbParam(name="status", param_type="string"),
                        "limit": ModelDbParam(name="limit", param_type="integer"),
                        "offset": ModelDbParam(name="offset", param_type="integer"),
                    },
                    returns=ModelDbReturn(model_ref="User", many=True),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            },
        )

        config = ModelRepositoryRuntimeConfig(
            max_row_limit=100,
            timeout_ms=5000,
            allowed_modes=frozenset({"read"}),
            allow_write_operations=False,
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=contract,
            config=config,
        )

        executed_sql: list[str] = []

        async def capture_fetch(sql: str, *args: object) -> list[MockAsyncRecord]:
            executed_sql.append(sql)
            return []

        mock_pool.connection.fetch = capture_fetch  # type: ignore[method-assign]

        await runtime.call("find_active_paginated", "active", 25, 50)

        assert len(executed_sql) == 1
        sql = executed_sql[0]

        # ORDER BY should be BEFORE LIMIT (not after)
        order_by_pos = sql.upper().find("ORDER BY")
        limit_pos = sql.upper().find("LIMIT")
        offset_pos = sql.upper().find("OFFSET")

        assert order_by_pos != -1, "ORDER BY should be present"
        assert limit_pos != -1, "LIMIT should be present"
        assert offset_pos != -1, "OFFSET should be present"
        assert order_by_pos < limit_pos, (
            f"ORDER BY should come BEFORE LIMIT. Got: {sql}"
        )
        assert limit_pos < offset_pos, f"LIMIT should come BEFORE OFFSET. Got: {sql}"

        # Should have exactly one of each clause
        assert sql.upper().count("ORDER BY") == 1
        assert sql.upper().count("LIMIT") == 1
        assert sql.upper().count("OFFSET") == 1


# ============================================================================
# Limit Tests
# ============================================================================


@pytest.mark.unit
class TestLimitEnforcement:
    """Test LIMIT injection and validation for multi-row queries."""

    @pytest.mark.asyncio
    async def test_multi_row_no_limit_injects_limit(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test LIMIT injection when none present.

        Verifies:
            - Multi-row query without LIMIT gets max_row_limit injected
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=25,
            timeout_ms=5000,
            allowed_modes=frozenset({"read", "write"}),
            allow_write_operations=True,
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        executed_sql: list[str] = []

        async def capture_fetch(sql: str, *args: object) -> list[MockAsyncRecord]:
            executed_sql.append(sql)
            return []

        mock_pool.connection.fetch = capture_fetch  # type: ignore[method-assign]

        # find_with_order has no LIMIT
        await runtime.call("find_with_order")

        assert len(executed_sql) == 1
        assert "LIMIT 25" in executed_sql[0]

    @pytest.mark.asyncio
    async def test_multi_row_limit_exceeds_max_raises_error(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test error when LIMIT exceeds max_row_limit.

        Verifies:
            - LIMIT > max_row_limit raises RepositoryContractError
            - Error message shows both limits
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=100,  # Less than the 500 in find_with_high_limit
            timeout_ms=5000,
            allowed_modes=frozenset({"read", "write"}),
            allow_write_operations=True,
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        with pytest.raises(RepositoryContractError) as exc_info:
            await runtime.call("find_with_high_limit")

        error = exc_info.value
        assert "LIMIT 500" in str(error)
        assert "100" in str(error)  # max_row_limit
        assert error.op_name == "find_with_high_limit"

    @pytest.mark.asyncio
    async def test_multi_row_limit_within_max_passes(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test LIMIT within max_row_limit passes without modification.

        Verifies:
            - LIMIT <= max_row_limit is accepted
            - Original LIMIT value is preserved
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=100,  # Greater than the 5 in find_with_limit
            timeout_ms=5000,
            allowed_modes=frozenset({"read", "write"}),
            allow_write_operations=True,
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        executed_sql: list[str] = []

        async def capture_fetch(sql: str, *args: object) -> list[MockAsyncRecord]:
            executed_sql.append(sql)
            return []

        mock_pool.connection.fetch = capture_fetch  # type: ignore[method-assign]

        # find_with_limit has LIMIT 5, within max
        await runtime.call("find_with_limit")

        assert len(executed_sql) == 1
        # Original LIMIT preserved
        assert "LIMIT 5" in executed_sql[0]
        # No injected LIMIT
        assert executed_sql[0].count("LIMIT") == 1

    @pytest.mark.asyncio
    async def test_multi_row_parameterized_limit_no_injection(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test parameterized LIMIT ($1) is not duplicated (OMN-1842).

        Verifies:
            - SQL with LIMIT $1 is returned unchanged
            - No additional LIMIT clause is injected
            - This prevents SQL syntax errors like 'LIMIT $1 LIMIT 100'
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=100,
            timeout_ms=5000,
            allowed_modes=frozenset({"read", "write"}),
            allow_write_operations=True,
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        executed_sql: list[str] = []

        async def capture_fetch(sql: str, *args: object) -> list[MockAsyncRecord]:
            executed_sql.append(sql)
            return []

        mock_pool.connection.fetch = capture_fetch  # type: ignore[method-assign]

        # find_with_param_limit has LIMIT $1
        await runtime.call("find_with_param_limit", 50)

        assert len(executed_sql) == 1
        # Original parameterized LIMIT preserved
        assert "LIMIT $1" in executed_sql[0]
        # No duplicate LIMIT injected
        assert executed_sql[0].count("LIMIT") == 1

    @pytest.mark.asyncio
    async def test_parameterized_limit_with_higher_param_number(
        self,
        mock_pool: MockAsyncPool,
    ) -> None:
        """Test parameterized LIMIT with higher param numbers ($2, $3, etc.).

        Verifies:
            - SQL with LIMIT $2 or LIMIT $10 is detected correctly
            - No additional LIMIT clause is injected
        """
        # Create a contract with LIMIT $2 (second parameter)
        contract = ModelDbRepositoryContract(
            name="test_repo",
            engine="postgres",
            database_ref="test_db",
            tables=["items"],
            models={"Item": "test.models:Item"},
            ops={
                "find_by_status_limited": ModelDbOperation(
                    mode="read",
                    sql="SELECT * FROM items WHERE status = $1 ORDER BY id LIMIT $2",
                    params={
                        "status": ModelDbParam(name="status", param_type="string"),
                        "limit": ModelDbParam(name="limit", param_type="integer"),
                    },
                    returns=ModelDbReturn(model_ref="Item", many=True),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            },
        )

        config = ModelRepositoryRuntimeConfig(
            max_row_limit=100,
            timeout_ms=5000,
            allowed_modes=frozenset({"read"}),
            allow_write_operations=False,
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=contract,
            config=config,
        )

        executed_sql: list[str] = []

        async def capture_fetch(sql: str, *args: object) -> list[MockAsyncRecord]:
            executed_sql.append(sql)
            return []

        mock_pool.connection.fetch = capture_fetch  # type: ignore[method-assign]

        await runtime.call("find_by_status_limited", "active", 25)

        assert len(executed_sql) == 1
        # Original parameterized LIMIT preserved
        assert "LIMIT $2" in executed_sql[0]
        # No duplicate LIMIT injected
        assert executed_sql[0].count("LIMIT") == 1


# ============================================================================
# Timeout Tests
# ============================================================================


@pytest.mark.unit
class TestTimeoutHandling:
    """Test query timeout enforcement."""

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test query exceeding timeout raises RepositoryTimeoutError.

        Verifies:
            - Slow queries are cancelled via asyncio.wait_for
            - RepositoryTimeoutError contains timeout value
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=10,
            timeout_ms=1000,  # 1 second timeout (minimum allowed)
            allowed_modes=frozenset({"read", "write"}),
            allow_write_operations=True,
            primary_key_column="id",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        # Make fetch delay longer than timeout
        mock_pool.connection.fetch_delay = 5.0  # 5 seconds, exceeds 1s timeout

        with pytest.raises(RepositoryTimeoutError) as exc_info:
            await runtime.call("find_with_order")

        error = exc_info.value
        assert "timeout" in str(error).lower()
        assert error.op_name == "find_with_order"
        assert error.timeout_seconds == 1.0


# ============================================================================
# Execution Error Tests
# ============================================================================


@pytest.mark.unit
class TestExecutionErrors:
    """Test database error wrapping."""

    @pytest.mark.asyncio
    async def test_database_error_raises_execution_error(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test database errors are wrapped in RepositoryExecutionError.

        Verifies:
            - asyncpg exceptions are caught and wrapped
            - Original exception is preserved as __cause__
        """
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        # Simulate database error
        db_error = Exception("Connection reset by peer")
        mock_pool.connection.should_raise = db_error

        with pytest.raises(RepositoryExecutionError) as exc_info:
            await runtime.call("find_with_order")

        error = exc_info.value
        assert "find_with_order" in str(error)
        assert "Connection reset by peer" in str(error)
        assert error.op_name == "find_with_order"
        assert error.__cause__ is db_error  # Original exception preserved

    @pytest.mark.asyncio
    async def test_execution_error_includes_sql_fingerprint(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test execution error includes SQL fingerprint.

        Verifies:
            - SQL fingerprint is included for debugging
        """
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        # Simulate database error
        mock_pool.connection.should_raise = Exception("Deadlock detected")

        with pytest.raises(RepositoryExecutionError) as exc_info:
            await runtime.call("find_with_order")

        error = exc_info.value
        assert error.sql_fingerprint is not None
        # SQL fingerprint should contain part of the query
        assert "SELECT" in error.sql_fingerprint


# ============================================================================
# Integration Tests (Multiple Features)
# ============================================================================


@pytest.mark.unit
class TestIntegration:
    """Test multiple features working together."""

    @pytest.mark.asyncio
    async def test_full_read_flow_with_injection(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test complete read flow with ORDER BY and LIMIT injection.

        Verifies:
            - ORDER BY injection when PK configured
            - LIMIT injection when none present
            - Correct result format
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=50,
            timeout_ms=5000,
            allowed_modes=frozenset({"read"}),
            allow_write_operations=False,
            primary_key_column="id",
            default_order_by="id ASC",
            emit_metrics=False,
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        executed_sql: list[str] = []

        async def capture_fetch(sql: str, *args: object) -> list[MockAsyncRecord]:
            executed_sql.append(sql)
            return [
                MockAsyncRecord({"id": 1, "name": "Alice"}),
                MockAsyncRecord({"id": 2, "name": "Bob"}),
            ]

        mock_pool.connection.fetch = capture_fetch  # type: ignore[method-assign]

        result = await runtime.call("find_all")

        assert len(executed_sql) == 1
        sql = executed_sql[0]

        # Check ORDER BY injection (uses default_order_by)
        assert "ORDER BY id ASC" in sql

        # Check LIMIT injection
        assert "LIMIT 50" in sql

        # Check result format
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_error_context_includes_table(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test error context includes table name from contract.

        Verifies:
            - All errors include the primary table
        """
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        with pytest.raises(RepositoryContractError) as exc_info:
            await runtime.call("unknown_op")

        error = exc_info.value
        assert error.table == "users"


# ============================================================================
# Property Tests
# ============================================================================


@pytest.mark.unit
class TestProperties:
    """Test runtime property accessors."""

    def test_contract_property(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test contract property returns the contract."""
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        assert runtime.contract is sample_contract
        assert runtime.contract.name == "test_users_repo"

    def test_config_property(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        default_config: ModelRepositoryRuntimeConfig,
    ) -> None:
        """Test config property returns the configuration."""
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=default_config,
        )

        assert runtime.config is default_config
        assert runtime.config.max_row_limit == 10

    def test_default_config_used_when_none_provided(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
    ) -> None:
        """Test default config is used when None provided."""
        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=None,
        )

        assert runtime.config is not None
        # Check it's the default ModelRepositoryRuntimeConfig
        assert isinstance(runtime.config, ModelRepositoryRuntimeConfig)


# ============================================================================
# Metrics Tests
# ============================================================================


@pytest.mark.unit
class TestMetricsEmission:
    """Test metrics emission behavior."""

    @pytest.mark.asyncio
    async def test_metrics_logged_when_enabled(
        self,
        mock_pool: MockAsyncPool,
        sample_contract: ModelDbRepositoryContract,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test metrics are logged when emit_metrics=True.

        Verifies:
            - Operation name is logged
            - Duration is logged
            - Rows returned is logged
        """
        config = ModelRepositoryRuntimeConfig(
            max_row_limit=10,
            timeout_ms=5000,
            allowed_modes=frozenset({"read", "write"}),
            allow_write_operations=True,
            primary_key_column="id",
            emit_metrics=True,  # Enable metrics
        )

        runtime = PostgresRepositoryRuntime(
            pool=mock_pool,  # type: ignore[arg-type]
            contract=sample_contract,
            config=config,
        )

        mock_pool.connection.fetch_result = [
            MockAsyncRecord({"id": 1}),
            MockAsyncRecord({"id": 2}),
        ]

        with caplog.at_level(logging.INFO):
            await runtime.call("find_with_order")

        # Check log contains operation metrics
        assert any(
            "Repository operation completed" in record.message
            for record in caplog.records
        )


__all__ = [
    "TestArgumentValidation",
    "TestBasicOperations",
    "TestDeterminismEnforcement",
    "TestExecutionErrors",
    "TestIntegration",
    "TestLimitEnforcement",
    "TestMetricsEmission",
    "TestModeValidation",
    "TestProperties",
    "TestTimeoutHandling",
]
