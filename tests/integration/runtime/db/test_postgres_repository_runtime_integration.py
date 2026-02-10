# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for PostgresRepositoryRuntime against real PostgreSQL.

These tests require a running PostgreSQL instance configured via environment variables:
    OMNIBASE_INFRA_DB_URL (preferred) or POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD

Run with: pytest -m postgres tests/integration/runtime/db/

Security Note (S608 Suppression):
----------------------------------
This file uses f-strings for SQL table names, which triggers Bandit's S608
(hardcoded-sql-expression) warning. This is SAFE in this context because:

1. The table name `_TEST_TABLE_NAME` is a MODULE-LEVEL CONSTANT with a random
   UUID suffix (e.g., "test_runtime_a1b2c3d4"), generated once at module load.

2. The table name is NEVER derived from user input, external configuration,
   or any runtime data that could be influenced by an attacker.

3. SQL injection requires attacker-controlled input to be interpolated into
   queries. Since the table name is hardcoded at import time with a random
   suffix, there is no attack vector.

4. The UUID suffix ensures test isolation across parallel test runs without
   introducing any security risk.

The S608 suppression for this file is configured in pyproject.toml under
[tool.ruff.lint.per-file-ignores].
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import pytest

from omnibase_infra.errors.repository import (
    RepositoryContractError,
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

if TYPE_CHECKING:
    import asyncpg

# Skip all tests if postgres marker not enabled or env vars missing
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.integration,
    pytest.mark.asyncio,
]


@pytest.fixture(scope="module")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Create module-scoped event loop for async fixtures.

    Required because db_pool and cleanup_test_table fixtures are module-scoped,
    but pytest-asyncio's default event_loop fixture is function-scoped.
    This prevents ScopeMismatch errors when module-scoped async fixtures
    try to use the event loop.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def _check_postgres_env_vars() -> tuple[bool, str]:
    """Check if required PostgreSQL environment variables are set.

    Primary: ``OMNIBASE_INFRA_DB_URL``.
    Fallback: individual ``POSTGRES_*`` vars.

    Returns:
        Tuple of (all_set, missing_vars_message)
    """
    if os.getenv("OMNIBASE_INFRA_DB_URL"):
        return True, ""

    required_vars = [
        "POSTGRES_HOST",
        "POSTGRES_PASSWORD",
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        return False, (
            f"Missing OMNIBASE_INFRA_DB_URL and required fallback variables: "
            f"{', '.join(missing)}"
        )
    return True, ""


def get_dsn() -> str:
    """Build PostgreSQL DSN from environment variables.

    Primary: ``OMNIBASE_INFRA_DB_URL`` (full DSN).
    Fallback: individual ``POSTGRES_*`` env vars.

    Raises:
        ValueError: If neither OMNIBASE_INFRA_DB_URL nor required
            individual vars are set, or if the DSN has an invalid scheme
            or is missing a database name.
    """
    db_url = os.getenv("OMNIBASE_INFRA_DB_URL")
    if db_url:
        # Validate scheme and database name (consistent with other DSN consumers)
        parsed = urlparse(db_url)
        if parsed.scheme not in ("postgresql", "postgres"):
            msg = (
                f"OMNIBASE_INFRA_DB_URL has invalid scheme '{parsed.scheme}', "
                "expected 'postgresql' or 'postgres'"
            )
            raise ValueError(msg)
        database = (parsed.path or "").lstrip("/")
        if not database:
            msg = "OMNIBASE_INFRA_DB_URL is missing a database name"
            raise ValueError(msg)
        return db_url

    is_configured, error_msg = _check_postgres_env_vars()
    if not is_configured:
        raise ValueError(error_msg)

    from urllib.parse import quote_plus

    host = os.environ["POSTGRES_HOST"]
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.environ["POSTGRES_PASSWORD"]

    # URL-encode credentials to handle special characters (@, :, /, %, etc.)
    encoded_user = quote_plus(user, safe="")
    encoded_password = quote_plus(password, safe="")

    return (
        f"postgresql://{encoded_user}:{encoded_password}@{host}:{port}/omnibase_infra"
    )


# Module-constant table name with random UUID suffix for test isolation.
# SECURITY: This is safe for f-string SQL interpolation because:
#   - Generated ONCE at module load (not from user/external input)
#   - UUID suffix is cryptographically random, not attacker-controllable
#   - No code path allows modification after initialization
# See module docstring for full S608 suppression rationale.
_TEST_TABLE_NAME = f"test_runtime_{uuid.uuid4().hex[:8]}"
_TABLE_CREATED = False


@pytest.fixture(scope="module")
async def db_pool():
    """Create a connection pool for the test module.

    Skips all tests if PostgreSQL environment variables are not configured.
    """
    is_configured, error_msg = _check_postgres_env_vars()
    if not is_configured:
        pytest.skip(f"PostgreSQL integration tests skipped: {error_msg}")

    import asyncpg

    dsn = get_dsn()
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    yield pool
    await pool.close()


@pytest.fixture(scope="module", autouse=True)
async def cleanup_test_table(db_pool):
    """Clean up test table after all tests complete."""
    yield
    async with db_pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {_TEST_TABLE_NAME}")


@pytest.fixture
async def test_table(db_pool: asyncpg.Pool):
    """Create a temporary test table for isolation (reuses same table across tests)."""
    global _TABLE_CREATED  # noqa: PLW0603  # Module-scoped fixture state

    table_name = _TEST_TABLE_NAME

    async with db_pool.acquire() as conn:
        if not _TABLE_CREATED:
            # Create table only once
            await conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            await conn.execute(f"""
                CREATE TABLE {table_name} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name TEXT NOT NULL,
                    score FLOAT NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # Insert test data
            await conn.execute(f"""
                INSERT INTO {table_name} (id, name, score, status) VALUES
                ('11111111-1111-1111-1111-111111111111', 'alpha', 0.9, 'active'),
                ('22222222-2222-2222-2222-222222222222', 'beta', 0.8, 'active'),
                ('33333333-3333-3333-3333-333333333333', 'gamma', 0.7, 'inactive'),
                ('44444444-4444-4444-4444-444444444444', 'delta', 0.6, 'active')
            """)
            _TABLE_CREATED = True

    return table_name


@pytest.fixture
def make_contract(test_table: str):
    """Factory to create contracts for the test table."""

    def _make(ops: dict[str, ModelDbOperation]) -> ModelDbRepositoryContract:
        return ModelDbRepositoryContract(
            name="test_repo",
            engine="postgres",
            database_ref="test_db",
            tables=[test_table],
            # Provide model mappings for any model_ref used
            models={
                "TestRow": "dict",  # Using dict as placeholder
                "WriteResult": "dict",
            },
            ops=ops,
        )

    return _make


class TestReadOperations:
    """Test read operations against real PostgreSQL."""

    async def test_select_single_row_by_id(
        self, db_pool: asyncpg.Pool, test_table: str, make_contract
    ):
        """Test selecting a single row by primary key."""
        contract = make_contract(
            {
                "find_by_id": ModelDbOperation(
                    mode="read",
                    sql=f"SELECT * FROM {test_table} WHERE id = $1",
                    params={"id": ModelDbParam(name="id", param_type="uuid")},
                    returns=ModelDbReturn(model_ref="TestRow", many=False),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            }
        )

        runtime = PostgresRepositoryRuntime(db_pool, contract)
        result = await runtime.call(
            "find_by_id", uuid.UUID("11111111-1111-1111-1111-111111111111")
        )

        assert result is not None
        assert result["name"] == "alpha"
        assert result["score"] == 0.9

    async def test_select_single_row_not_found(
        self, db_pool: asyncpg.Pool, test_table: str, make_contract
    ):
        """Test selecting a non-existent row returns None."""
        contract = make_contract(
            {
                "find_by_id": ModelDbOperation(
                    mode="read",
                    sql=f"SELECT * FROM {test_table} WHERE id = $1",
                    params={"id": ModelDbParam(name="id", param_type="uuid")},
                    returns=ModelDbReturn(model_ref="TestRow", many=False),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            }
        )

        runtime = PostgresRepositoryRuntime(db_pool, contract)
        result = await runtime.call(
            "find_by_id", uuid.UUID("99999999-9999-9999-9999-999999999999")
        )

        assert result is None

    async def test_select_multiple_rows_with_order_by(
        self, db_pool: asyncpg.Pool, test_table: str, make_contract
    ):
        """Test selecting multiple rows with explicit ORDER BY."""
        contract = make_contract(
            {
                "find_active": ModelDbOperation(
                    mode="read",
                    sql=f"SELECT * FROM {test_table} WHERE status = $1 ORDER BY score DESC",
                    params={"status": ModelDbParam(name="status", param_type="string")},
                    returns=ModelDbReturn(model_ref="TestRow", many=True),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            }
        )

        config = ModelRepositoryRuntimeConfig(max_row_limit=100)
        runtime = PostgresRepositoryRuntime(db_pool, contract, config)
        results = await runtime.call("find_active", "active")

        assert isinstance(results, list)
        assert len(results) == 3
        # Verify ordering (score DESC)
        assert results[0]["name"] == "alpha"
        assert results[1]["name"] == "beta"
        assert results[2]["name"] == "delta"

    async def test_select_with_limit_injection(
        self, db_pool: asyncpg.Pool, test_table: str, make_contract
    ):
        """Test that LIMIT is injected for multi-row queries."""
        contract = make_contract(
            {
                "find_all": ModelDbOperation(
                    mode="read",
                    sql=f"SELECT * FROM {test_table} ORDER BY score DESC",
                    params={},
                    returns=ModelDbReturn(model_ref="TestRow", many=True),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            }
        )

        # Config with max_row_limit=2
        config = ModelRepositoryRuntimeConfig(max_row_limit=2)
        runtime = PostgresRepositoryRuntime(db_pool, contract, config)
        results = await runtime.call("find_all")

        # Should only get 2 rows due to injected LIMIT
        assert len(results) == 2


class TestDeterminismEnforcement:
    """Test determinism enforcement with real database."""

    async def test_order_by_injection(
        self, db_pool: asyncpg.Pool, test_table: str, make_contract
    ):
        """Test ORDER BY is injected when missing and PK is configured."""
        contract = make_contract(
            {
                "find_all": ModelDbOperation(
                    mode="read",
                    sql=f"SELECT * FROM {test_table}",  # No ORDER BY
                    params={},
                    returns=ModelDbReturn(model_ref="TestRow", many=True),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            }
        )

        # Configure with primary_key_column
        config = ModelRepositoryRuntimeConfig(
            primary_key_column="id",
            default_order_by="score DESC, id ASC",
            max_row_limit=100,
        )
        runtime = PostgresRepositoryRuntime(db_pool, contract, config)
        results = await runtime.call("find_all")

        # Should work and return ordered results
        assert len(results) == 4
        # Results should be ordered by score DESC
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_no_pk_raises_error(
        self, db_pool: asyncpg.Pool, test_table: str, make_contract
    ):
        """Test error when no ORDER BY and no PK configured."""
        contract = make_contract(
            {
                "find_all": ModelDbOperation(
                    mode="read",
                    sql=f"SELECT * FROM {test_table}",  # No ORDER BY
                    params={},
                    returns=ModelDbReturn(model_ref="TestRow", many=True),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            }
        )

        # Config WITHOUT primary_key_column
        config = ModelRepositoryRuntimeConfig(
            primary_key_column=None,
            default_order_by=None,
        )
        runtime = PostgresRepositoryRuntime(db_pool, contract, config)

        with pytest.raises(RepositoryContractError) as exc_info:
            await runtime.call("find_all")

        assert "ORDER BY" in str(exc_info.value)
        assert "primary_key_column" in str(exc_info.value)


class TestWriteOperations:
    """Test write operations against real PostgreSQL."""

    async def test_insert_row(
        self, db_pool: asyncpg.Pool, test_table: str, make_contract
    ):
        """Test inserting a new row."""
        new_id = uuid.uuid4()
        contract = make_contract(
            {
                "create": ModelDbOperation(
                    mode="write",
                    sql=f"INSERT INTO {test_table} (id, name, score) VALUES ($1, $2, $3)",
                    params={
                        "id": ModelDbParam(name="id", param_type="uuid"),
                        "name": ModelDbParam(name="name", param_type="string"),
                        "score": ModelDbParam(name="score", param_type="float"),
                    },
                    returns=ModelDbReturn(model_ref="WriteResult", many=False),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
                "find_by_id": ModelDbOperation(
                    mode="read",
                    sql=f"SELECT * FROM {test_table} WHERE id = $1",
                    params={"id": ModelDbParam(name="id", param_type="uuid")},
                    returns=ModelDbReturn(model_ref="TestRow", many=False),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            }
        )

        runtime = PostgresRepositoryRuntime(db_pool, contract)

        # Insert
        await runtime.call("create", new_id, "epsilon", 0.5)

        # Verify
        result = await runtime.call("find_by_id", new_id)
        assert result is not None
        assert result["name"] == "epsilon"
        assert result["score"] == 0.5

        # Cleanup
        async with db_pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {test_table} WHERE id = $1", new_id)

    async def test_write_blocked_when_disabled(
        self, db_pool: asyncpg.Pool, test_table: str, make_contract
    ):
        """Test write operations blocked when allow_write_operations=False."""
        contract = make_contract(
            {
                "create": ModelDbOperation(
                    mode="write",
                    sql=f"INSERT INTO {test_table} (name, score) VALUES ($1, $2)",
                    params={
                        "name": ModelDbParam(name="name", param_type="string"),
                        "score": ModelDbParam(name="score", param_type="float"),
                    },
                    returns=ModelDbReturn(model_ref="WriteResult", many=False),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            }
        )

        # Disable write operations
        config = ModelRepositoryRuntimeConfig(allow_write_operations=False)
        runtime = PostgresRepositoryRuntime(db_pool, contract, config)

        with pytest.raises(RepositoryContractError) as exc_info:
            await runtime.call("create", "test", 0.5)

        assert "write" in str(exc_info.value).lower()


class TestArgumentValidation:
    """Test argument validation against real PostgreSQL."""

    async def test_wrong_arg_count(
        self, db_pool: asyncpg.Pool, test_table: str, make_contract
    ):
        """Test that wrong argument count raises validation error."""
        contract = make_contract(
            {
                "find_by_id": ModelDbOperation(
                    mode="read",
                    sql=f"SELECT * FROM {test_table} WHERE id = $1",
                    params={"id": ModelDbParam(name="id", param_type="uuid")},
                    returns=ModelDbReturn(model_ref="TestRow", many=False),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            }
        )

        runtime = PostgresRepositoryRuntime(db_pool, contract)

        with pytest.raises(RepositoryValidationError) as exc_info:
            await runtime.call("find_by_id")  # Missing argument

        assert "expects 1 argument" in str(exc_info.value)


class TestLearnedPatternsTable:
    """Test against the real learned_patterns table schema."""

    async def test_query_learned_patterns_validated(self, db_pool: asyncpg.Pool):
        """Test querying the actual learned_patterns table for validated patterns."""
        contract = ModelDbRepositoryContract(
            name="learned_patterns_repo",
            engine="postgres",
            database_ref="omnibase_infra",
            tables=["learned_patterns"],
            models={"LearnedPattern": "dict"},
            ops={
                "find_validated": ModelDbOperation(
                    mode="read",
                    sql="""
                        SELECT id, pattern_signature, domain_id, confidence, status
                        FROM learned_patterns
                        WHERE status = $1
                        ORDER BY confidence DESC, id ASC
                    """,
                    params={"status": ModelDbParam(name="status", param_type="string")},
                    returns=ModelDbReturn(model_ref="LearnedPattern", many=True),
                    safety_policy=ModelDbSafetyPolicy(),
                ),
            },
        )

        config = ModelRepositoryRuntimeConfig(max_row_limit=10)
        runtime = PostgresRepositoryRuntime(db_pool, contract, config)

        # Query for validated patterns - validates the runtime executes correctly
        results = await runtime.call("find_validated", "validated")

        # Validate the query returns a well-formed list of dicts
        assert isinstance(results, list)

        expected_keys = {"id", "pattern_signature", "domain_id", "confidence", "status"}
        for row in results:
            assert isinstance(row, dict), f"Expected dict row, got {type(row)}"
            assert expected_keys.issubset(row.keys()), (
                f"Row missing expected keys: {expected_keys - row.keys()}"
            )
            assert row["status"] == "validated", (
                f"Expected status='validated', got status='{row['status']}'"
            )

        # Verify ORDER BY confidence DESC is respected
        if len(results) > 1:
            confidences = [row["confidence"] for row in results]
            assert confidences == sorted(confidences, reverse=True), (
                "Results are not ordered by confidence DESC"
            )
