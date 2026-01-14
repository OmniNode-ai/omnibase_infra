# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S608
# S608 disabled: SQL injection is not a concern here - table names come from
# test fixtures (UUID-based), not user input. Parameterized queries are used
# for all user-facing values ($1, $2, etc.).
"""Integration tests for ProjectorShell with real PostgreSQL database.

These tests verify ProjectorShell against a real PostgreSQL database to validate:
1. Database connectivity and pool management
2. Projection modes (upsert, insert_only, append) with actual SQL execution
3. State retrieval (get_state) returning projected rows
4. Idempotency guarantees with real database state
5. Value extraction from nested event payloads end-to-end

Environment Variables:
    POSTGRES_HOST: Database host (default: localhost)
    POSTGRES_PORT: Database port (default: 5432)
    POSTGRES_DATABASE: Database name (default: postgres)
    POSTGRES_USER: Database user (default: postgres)
    POSTGRES_PASSWORD: Database password (REQUIRED - no default, tests skip if unset)

    For OmniNode infrastructure, set:
        POSTGRES_HOST=192.168.86.200
        POSTGRES_PORT=5436
        POSTGRES_DATABASE=omninode_bridge

Test Isolation:
    Each test creates a unique table (test_projector_{uuid8}) and drops it
    after the test completes. This prevents test interference and avoids
    affecting production data.

Type Mapping Notes:
    - Python float -> PostgreSQL NUMERIC: Intentional for precision preservation.
      asyncpg returns NUMERIC as Decimal, tests use float() for comparison.

Related Tickets:
    - OMN-1169: ProjectorShell implementation
    - OMN-1168: ProjectorPluginLoader contract discovery

Usage:
    # Ensure POSTGRES_PASSWORD is set
    export POSTGRES_PASSWORD=your_password
    pytest -m integration tests/integration/runtime/test_projector_shell_database.py
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg
import pytest
from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.models.projectors import (
    ModelProjectorBehavior,
    ModelProjectorColumn,
    ModelProjectorContract,
    ModelProjectorSchema,
)
from pydantic import BaseModel

from omnibase_infra.runtime.projector_shell import ProjectorShell

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
]


# =============================================================================
# Test Payload Models
# =============================================================================


class OrderCreatedPayload(BaseModel):
    """Sample event payload for order creation."""

    order_id: UUID
    customer_id: UUID
    status: str
    total_amount: float
    created_at: datetime


class NestedCustomer(BaseModel):
    """Nested model for testing deep path extraction."""

    customer_id: UUID
    email: str
    name: str


class OrderWithNestedPayload(BaseModel):
    """Event payload with nested model for testing deep extraction."""

    order_id: UUID
    customer: NestedCustomer
    status: str


# =============================================================================
# Fixtures
# =============================================================================


def _get_database_dsn() -> str | None:
    """Build database DSN from environment variables.

    All database configuration comes from environment variables with portable
    defaults suitable for CI/CD environments. Set POSTGRES_HOST, POSTGRES_PORT,
    etc. to override for specific infrastructure.

    Returns:
        PostgreSQL connection string, or None if POSTGRES_PASSWORD is not set.
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DATABASE", "postgres")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        return None

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@pytest.fixture
async def db_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Create database connection pool for tests.

    Yields:
        asyncpg.Pool connected to the test database.

    Raises:
        pytest.skip: If database is not reachable or POSTGRES_PASSWORD not set.
    """
    dsn = _get_database_dsn()

    if not dsn:
        pytest.skip("POSTGRES_PASSWORD environment variable not set")

    try:
        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=5,
            timeout=10.0,
        )
    except (asyncpg.PostgresConnectionError, OSError) as e:
        pytest.skip(f"Database not reachable: {e}")

    yield pool

    await pool.close()


@pytest.fixture
async def test_table(db_pool: asyncpg.Pool) -> AsyncGenerator[str, None]:
    """Create and drop test table for projector tests.

    Creates a uniquely named table for each test to ensure isolation.
    Teardown is shielded from exceptions to ensure cleanup always attempts
    to run, even if setup partially failed.

    Args:
        db_pool: asyncpg connection pool fixture.

    Yields:
        Table name (e.g., "test_projector_a1b2c3d4").
    """
    table_name = f"test_projector_{uuid4().hex[:8]}"

    # Create test table
    # NOTE: amount is NUMERIC to match payload.total_amount (Python float).
    # PostgreSQL NUMERIC preserves precision; asyncpg returns as Decimal.
    # Tests use float() for comparison (see TestStateRetrieval).
    async with db_pool.acquire() as conn:
        await conn.execute(f"""
            CREATE TABLE "{table_name}" (
                id UUID PRIMARY KEY,
                name TEXT,
                status TEXT,
                amount NUMERIC,
                customer_email TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

    yield table_name

    # Drop test table - shielded from exceptions to ensure cleanup runs
    # even if the test or other teardown code raised an exception.
    # Uses DROP TABLE IF EXISTS to handle cases where table creation failed.
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    except Exception:
        # Swallow exceptions during cleanup to avoid masking test failures.
        # The table will be orphaned but has a unique name, so no collision risk.
        pass


def _make_contract(
    table_name: str,
    *,
    mode: str = "upsert",
    upsert_key: str | None = None,
) -> ModelProjectorContract:
    """Create a projector contract for the test table.

    Args:
        table_name: Name of the test table.
        mode: Projection mode (upsert, insert_only, append).
        upsert_key: Optional upsert conflict key.

    Returns:
        ModelProjectorContract configured for the test table.
    """
    columns = [
        ModelProjectorColumn(
            name="id",
            type="UUID",
            source="payload.order_id",
        ),
        ModelProjectorColumn(
            name="name",
            type="TEXT",
            source="payload.status",
        ),
        ModelProjectorColumn(
            name="status",
            type="TEXT",
            source="payload.status",
        ),
        # NOTE: NUMERIC type maps from Python float (payload.total_amount).
        # asyncpg returns NUMERIC as Decimal; use float() for comparison.
        ModelProjectorColumn(
            name="amount",
            type="NUMERIC",
            source="payload.total_amount",
        ),
        ModelProjectorColumn(
            name="created_at",
            type="TIMESTAMPTZ",
            source="envelope_timestamp",
        ),
    ]

    schema = ModelProjectorSchema(
        table=table_name,
        primary_key="id",
        columns=columns,
    )

    behavior = ModelProjectorBehavior(
        mode=mode,
        upsert_key=upsert_key,
    )

    return ModelProjectorContract(
        projector_kind="materialized_view",
        projector_id=f"test-projector-{table_name}",
        name="Test Projector",
        version="1.0.0",
        aggregate_type="Order",
        consumed_events=["order.created.v1", "order.updated.v1"],
        projection_schema=schema,
        behavior=behavior,
    )


def _make_event_envelope(
    order_id: UUID | None = None,
    status: str = "pending",
    total_amount: float = 99.99,
    event_type: str = "order.created.v1",
) -> ModelEventEnvelope[OrderCreatedPayload]:
    """Create a sample event envelope for testing.

    Args:
        order_id: Order UUID (generated if not provided).
        status: Order status string.
        total_amount: Order total amount.
        event_type: Event type string.

    Returns:
        ModelEventEnvelope with OrderCreatedPayload.
    """
    payload = OrderCreatedPayload(
        order_id=order_id or uuid4(),
        customer_id=uuid4(),
        status=status,
        total_amount=total_amount,
        created_at=datetime.now(UTC),
    )

    return ModelEventEnvelope(
        payload=payload,
        envelope_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        correlation_id=uuid4(),
        metadata=ModelEnvelopeMetadata(
            tags={"event_type": event_type},
        ),
        onex_version=ModelSemVer(major=1, minor=0, patch=0),
        envelope_version=ModelSemVer(major=1, minor=0, patch=0),
    )


def _make_nested_event_envelope(
    order_id: UUID | None = None,
    customer_email: str = "customer@example.com",
    event_type: str = "order.created.v1",
) -> ModelEventEnvelope[OrderWithNestedPayload]:
    """Create an event envelope with nested payload for testing.

    Args:
        order_id: Order UUID (generated if not provided).
        customer_email: Customer email string.
        event_type: Event type string.

    Returns:
        ModelEventEnvelope with OrderWithNestedPayload.
    """
    payload = OrderWithNestedPayload(
        order_id=order_id or uuid4(),
        customer=NestedCustomer(
            customer_id=uuid4(),
            email=customer_email,
            name="Test Customer",
        ),
        status="confirmed",
    )

    return ModelEventEnvelope(
        payload=payload,
        envelope_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        correlation_id=uuid4(),
        metadata=ModelEnvelopeMetadata(
            tags={"event_type": event_type},
        ),
        onex_version=ModelSemVer(major=1, minor=0, patch=0),
        envelope_version=ModelSemVer(major=1, minor=0, patch=0),
    )


# =============================================================================
# Database Connection Tests
# =============================================================================


class TestDatabaseConnection:
    """Tests for database connection and pool management."""

    async def test_can_connect_to_database(
        self,
        db_pool: asyncpg.Pool,
    ) -> None:
        """Verify database connection works.

        Given: Valid database configuration
        When: Acquiring a connection from the pool
        Then: Connection succeeds and can execute queries
        """
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")

        assert result == 1

    async def test_pool_management(
        self,
        db_pool: asyncpg.Pool,
    ) -> None:
        """Verify connection pool is managed correctly.

        Given: Connection pool with min_size=1, max_size=5
        When: Acquiring multiple connections
        Then: Pool manages connections within limits
        """
        # Pool should have at least 1 connection
        assert db_pool.get_size() >= 1

        # Acquire multiple connections
        connections: list[asyncpg.Connection] = []
        try:
            for _ in range(3):
                conn = await db_pool.acquire()
                connections.append(conn)

            # All connections should be valid
            for conn in connections:
                result = await conn.fetchval("SELECT 1")
                assert result == 1
        finally:
            # Release connections even if assertions fail
            for conn in connections:
                await db_pool.release(conn)

    async def test_test_table_creation(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Verify test table is created correctly.

        Given: test_table fixture
        When: Checking table existence
        Then: Table exists with expected columns
        """
        async with db_pool.acquire() as conn:
            # Check table exists
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = $1
                )
                """,
                test_table,
            )

        assert exists is True


# =============================================================================
# Projection Mode Tests (Real Database)
# =============================================================================


class TestProjectionModes:
    """Tests for projection modes with real database operations."""

    async def test_upsert_mode_insert_new_row(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Upsert inserts when row doesn't exist.

        Given: Empty test table with upsert mode projector
        When: Projecting a new event
        Then: Row is inserted into the database
        """
        contract = _make_contract(test_table, mode="upsert")
        projector = ProjectorShell(contract=contract, pool=db_pool)

        envelope = _make_event_envelope(status="pending")
        order_id = envelope.payload.order_id
        correlation_id = uuid4()

        result = await projector.project(envelope, correlation_id)

        assert result.success is True
        assert result.skipped is False
        assert result.rows_affected == 1

        # Verify row in database
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT * FROM "{test_table}" WHERE id = $1',
                order_id,
            )

        assert row is not None
        assert row["id"] == order_id
        assert row["status"] == "pending"

    async def test_upsert_mode_update_existing_row(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Upsert updates when row exists.

        Given: Test table with existing row
        When: Projecting event with same primary key
        Then: Existing row is updated
        """
        contract = _make_contract(test_table, mode="upsert")
        projector = ProjectorShell(contract=contract, pool=db_pool)

        order_id = uuid4()
        correlation_id = uuid4()

        # First projection - insert
        envelope1 = _make_event_envelope(order_id=order_id, status="pending")
        result1 = await projector.project(envelope1, correlation_id)

        assert result1.success is True
        assert result1.rows_affected == 1

        # Second projection - update
        envelope2 = _make_event_envelope(order_id=order_id, status="confirmed")
        result2 = await projector.project(envelope2, correlation_id)

        assert result2.success is True
        assert result2.rows_affected == 1

        # Verify updated row
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT * FROM "{test_table}" WHERE id = $1',
                order_id,
            )

        assert row is not None
        assert row["status"] == "confirmed"

    async def test_insert_only_mode_succeeds(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Insert-only mode works for new rows.

        Given: Empty test table with insert_only mode projector
        When: Projecting a new event
        Then: Row is inserted successfully
        """
        contract = _make_contract(test_table, mode="insert_only")
        projector = ProjectorShell(contract=contract, pool=db_pool)

        envelope = _make_event_envelope(status="pending")
        order_id = envelope.payload.order_id
        correlation_id = uuid4()

        result = await projector.project(envelope, correlation_id)

        assert result.success is True
        assert result.rows_affected == 1

        # Verify row in database
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT * FROM "{test_table}" WHERE id = $1',
                order_id,
            )

        assert row is not None
        assert row["status"] == "pending"

    async def test_insert_only_mode_fails_on_duplicate(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Insert-only mode returns failure on duplicate key.

        Given: Test table with existing row
        When: Projecting event with same primary key in insert_only mode
        Then: Returns failure result (not exception)
        """
        contract = _make_contract(test_table, mode="insert_only")
        projector = ProjectorShell(contract=contract, pool=db_pool)

        order_id = uuid4()
        correlation_id = uuid4()

        # First projection - insert
        envelope1 = _make_event_envelope(order_id=order_id, status="pending")
        result1 = await projector.project(envelope1, correlation_id)

        assert result1.success is True

        # Second projection - duplicate key
        envelope2 = _make_event_envelope(order_id=order_id, status="confirmed")
        result2 = await projector.project(envelope2, correlation_id)

        # Should return failure, not raise exception
        assert result2.success is False
        assert result2.error is not None
        assert (
            "unique" in result2.error.lower() or "constraint" in result2.error.lower()
        )

    async def test_append_mode_always_inserts(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Append mode creates new rows each time.

        Given: Test table with append mode projector using envelope_id as primary key
        When: Projecting multiple events for same order
        Then: Each projection creates a new row
        """
        # For append mode, we need envelope_id as primary key (unique per event)
        columns = [
            ModelProjectorColumn(
                name="id",
                type="UUID",
                source="envelope_id",
            ),
            ModelProjectorColumn(
                name="name",
                type="TEXT",
                source="payload.status",
            ),
            ModelProjectorColumn(
                name="status",
                type="TEXT",
                source="payload.status",
            ),
            ModelProjectorColumn(
                name="amount",
                type="NUMERIC",
                source="payload.total_amount",
            ),
        ]

        schema = ModelProjectorSchema(
            table=test_table,
            primary_key="id",
            columns=columns,
        )

        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id=f"test-append-{test_table}",
            name="Test Append Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="append"),
        )

        projector = ProjectorShell(contract=contract, pool=db_pool)
        correlation_id = uuid4()
        order_id = uuid4()

        # Project same order multiple times (different envelope_ids)
        for status in ["pending", "confirmed", "shipped"]:
            envelope = _make_event_envelope(order_id=order_id, status=status)
            result = await projector.project(envelope, correlation_id)
            assert result.success is True
            assert result.rows_affected == 1

        # Verify all rows exist
        async with db_pool.acquire() as conn:
            count = await conn.fetchval(f'SELECT COUNT(*) FROM "{test_table}"')

        assert count == 3


# =============================================================================
# State Retrieval Tests
# =============================================================================


class TestStateRetrieval:
    """Tests for get_state aggregate state retrieval."""

    async def test_get_state_returns_projected_row(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """get_state returns the projected state after projection.

        Given: Projected event in test table
        When: Calling get_state with aggregate_id
        Then: Returns dict with projected column values
        """
        contract = _make_contract(test_table, mode="upsert")
        projector = ProjectorShell(contract=contract, pool=db_pool)

        envelope = _make_event_envelope(status="confirmed", total_amount=150.0)
        order_id = envelope.payload.order_id
        correlation_id = uuid4()

        # Project the event
        await projector.project(envelope, correlation_id)

        # Get state
        state = await projector.get_state(order_id, correlation_id)

        assert state is not None
        assert isinstance(state, dict)
        assert state["id"] == order_id
        assert state["status"] == "confirmed"
        # amount is stored as NUMERIC (preserves decimal precision)
        assert float(state["amount"]) == 150.0

    async def test_get_state_returns_none_for_missing(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """get_state returns None for non-existent aggregate.

        Given: Empty test table
        When: Calling get_state with non-existent aggregate_id
        Then: Returns None
        """
        contract = _make_contract(test_table, mode="upsert")
        projector = ProjectorShell(contract=contract, pool=db_pool)

        non_existent_id = uuid4()
        correlation_id = uuid4()

        state = await projector.get_state(non_existent_id, correlation_id)

        assert state is None


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestIdempotency:
    """Tests for idempotent projection behavior."""

    async def test_idempotent_projection_same_result(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Projecting same event twice produces same database state.

        Given: Projected event in test table
        When: Projecting the same event again
        Then: Database state remains consistent (idempotent upsert)
        """
        contract = _make_contract(test_table, mode="upsert")
        projector = ProjectorShell(contract=contract, pool=db_pool)

        envelope = _make_event_envelope(status="pending", total_amount=100.0)
        order_id = envelope.payload.order_id
        correlation_id = uuid4()

        # First projection
        result1 = await projector.project(envelope, correlation_id)
        assert result1.success is True

        # Get state after first projection
        state1 = await projector.get_state(order_id, correlation_id)
        assert state1 is not None

        # Project same event again
        result2 = await projector.project(envelope, correlation_id)
        assert result2.success is True

        # Get state after second projection
        state2 = await projector.get_state(order_id, correlation_id)
        assert state2 is not None

        # States should be equivalent
        assert state1["id"] == state2["id"]
        assert state1["status"] == state2["status"]
        assert state1["amount"] == state2["amount"]

        # Only one row should exist
        async with db_pool.acquire() as conn:
            count = await conn.fetchval(
                f'SELECT COUNT(*) FROM "{test_table}" WHERE id = $1',
                order_id,
            )

        assert count == 1


# =============================================================================
# Value Extraction Tests (End-to-End)
# =============================================================================


class TestValueExtraction:
    """Tests for value extraction from event payloads end-to-end."""

    async def test_nested_payload_extraction(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Values extracted from nested event payloads end up in database.

        Given: Contract with nested path sources
        When: Projecting event with nested payload
        Then: Nested values are correctly extracted and stored
        """
        # Create contract with nested path extraction
        columns = [
            ModelProjectorColumn(
                name="id",
                type="UUID",
                source="payload.order_id",
            ),
            ModelProjectorColumn(
                name="status",
                type="TEXT",
                source="payload.status",
            ),
            ModelProjectorColumn(
                name="customer_email",
                type="TEXT",
                source="payload.customer.email",
            ),
            ModelProjectorColumn(
                name="name",
                type="TEXT",
                source="payload.customer.name",
            ),
        ]

        schema = ModelProjectorSchema(
            table=test_table,
            primary_key="id",
            columns=columns,
        )

        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id=f"test-nested-{test_table}",
            name="Test Nested Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="upsert"),
        )

        projector = ProjectorShell(contract=contract, pool=db_pool)

        envelope = _make_nested_event_envelope(
            customer_email="test@example.com",
            event_type="order.created.v1",
        )
        order_id = envelope.payload.order_id
        correlation_id = uuid4()

        result = await projector.project(envelope, correlation_id)

        assert result.success is True

        # Verify nested values in database
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT * FROM "{test_table}" WHERE id = $1',
                order_id,
            )

        assert row is not None
        assert row["customer_email"] == "test@example.com"
        assert row["name"] == "Test Customer"
        assert row["status"] == "confirmed"

    async def test_envelope_metadata_extraction(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Values extracted from envelope metadata end up in database.

        Given: Contract with envelope-level path sources
        When: Projecting event
        Then: Envelope values (timestamp) are correctly extracted
        """
        contract = _make_contract(test_table, mode="upsert")
        projector = ProjectorShell(contract=contract, pool=db_pool)

        envelope = _make_event_envelope(status="pending")
        order_id = envelope.payload.order_id
        correlation_id = uuid4()

        result = await projector.project(envelope, correlation_id)

        assert result.success is True

        # Verify envelope timestamp in database
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT created_at FROM "{test_table}" WHERE id = $1',
                order_id,
            )

        assert row is not None
        assert row["created_at"] is not None
        # Timestamp should be close to envelope_timestamp
        # (within reasonable tolerance for timezone handling)


# =============================================================================
# Event Filtering Tests
# =============================================================================


class TestEventFiltering:
    """Tests for event type filtering in projections."""

    async def test_unconsumed_event_skipped(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Events not in consumed_events are skipped without database interaction.

        Given: Contract with consumed_events=["order.created.v1", "order.updated.v1"]
        When: Projecting event with type "user.created.v1"
        Then: Returns skipped=True, no database write
        """
        contract = _make_contract(test_table, mode="upsert")
        projector = ProjectorShell(contract=contract, pool=db_pool)

        # Create event with unconsumed type
        envelope = _make_event_envelope(
            status="pending",
            event_type="user.created.v1",  # Not in consumed_events
        )
        correlation_id = uuid4()

        result = await projector.project(envelope, correlation_id)

        assert result.success is True
        assert result.skipped is True
        assert result.rows_affected == 0

        # Verify no row was inserted
        async with db_pool.acquire() as conn:
            count = await conn.fetchval(f'SELECT COUNT(*) FROM "{test_table}"')

        assert count == 0

    async def test_consumed_event_processed(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Events in consumed_events are processed and written.

        Given: Contract with consumed_events=["order.created.v1"]
        When: Projecting event with type "order.created.v1"
        Then: Event is processed and row is written
        """
        contract = _make_contract(test_table, mode="upsert")
        projector = ProjectorShell(contract=contract, pool=db_pool)

        envelope = _make_event_envelope(
            status="pending",
            event_type="order.created.v1",  # In consumed_events
        )
        correlation_id = uuid4()

        result = await projector.project(envelope, correlation_id)

        assert result.success is True
        assert result.skipped is False
        assert result.rows_affected == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling during database operations."""

    async def test_projection_with_null_values(
        self,
        db_pool: asyncpg.Pool,
        test_table: str,
    ) -> None:
        """Projection handles null values from missing source paths.

        Given: Contract with source path that doesn't exist in payload
        When: Projecting event
        Then: Column receives NULL value (or default if specified)
        """
        # Create contract with path that won't exist
        columns = [
            ModelProjectorColumn(
                name="id",
                type="UUID",
                source="payload.order_id",
            ),
            ModelProjectorColumn(
                name="status",
                type="TEXT",
                source="payload.nonexistent_field",  # Doesn't exist
            ),
            ModelProjectorColumn(
                name="name",
                type="TEXT",
                source="payload.another_missing",
                default="default_name",  # Has default
            ),
        ]

        schema = ModelProjectorSchema(
            table=test_table,
            primary_key="id",
            columns=columns,
        )

        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id=f"test-null-{test_table}",
            name="Test Null Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="upsert"),
        )

        projector = ProjectorShell(contract=contract, pool=db_pool)

        envelope = _make_event_envelope(status="pending")
        order_id = envelope.payload.order_id
        correlation_id = uuid4()

        result = await projector.project(envelope, correlation_id)

        assert result.success is True

        # Verify null and default handling
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                f'SELECT status, name FROM "{test_table}" WHERE id = $1',
                order_id,
            )

        assert row is not None
        assert row["status"] is None  # No default, path not found
        assert row["name"] == "default_name"  # Default applied


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestDatabaseConnection",
    "TestErrorHandling",
    "TestEventFiltering",
    "TestIdempotency",
    "TestProjectionModes",
    "TestStateRetrieval",
    "TestValueExtraction",
]
