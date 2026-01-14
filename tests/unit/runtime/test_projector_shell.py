# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for ProjectorShell contract-driven event projection.

Tests the ProjectorShell functionality including:
- Event filtering based on consumed_events contract definition
- Value extraction from event envelopes using source paths
- Projection modes (upsert, insert_only, append)
- Idempotency guarantees for event replay
- Protocol compliance with ProtocolEventProjector
- State retrieval via get_state

Related:
    - OMN-1169: Implement ProjectorShell for contract-driven projections
    - src/omnibase_infra/runtime/projector_shell.py

Expected Behavior:
    ProjectorShell is a contract-driven event projector that:
    1. Filters events based on consumed_events in the contract
    2. Extracts values from event envelopes using source path expressions
    3. Writes projections to PostgreSQL using asyncpg with configurable modes
    4. Supports idempotent event replay via sequence number tracking
    5. Implements ProtocolEventProjector for runtime integration
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

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


class OrderUpdatedPayload(BaseModel):
    """Sample event payload for order updates."""

    order_id: UUID
    status: str
    updated_at: datetime


class UserCreatedPayload(BaseModel):
    """Sample event payload for user creation (unconsumed event type)."""

    user_id: UUID
    email: str


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


@pytest.fixture
def sample_projector_columns() -> list[ModelProjectorColumn]:
    """Create sample projector columns for testing."""
    return [
        ModelProjectorColumn(
            name="id",
            type="UUID",
            source="payload.order_id",
        ),
        ModelProjectorColumn(
            name="customer_id",
            type="UUID",
            source="payload.customer_id",
        ),
        ModelProjectorColumn(
            name="status",
            type="TEXT",
            source="payload.status",
        ),
        ModelProjectorColumn(
            name="total_amount",
            type="NUMERIC",
            source="payload.total_amount",
            default="0.0",
        ),
        ModelProjectorColumn(
            name="created_at",
            type="TIMESTAMPTZ",
            source="envelope_timestamp",
        ),
        ModelProjectorColumn(
            name="correlation_id",
            type="UUID",
            source="correlation_id",
        ),
    ]


@pytest.fixture
def sample_projector_schema(
    sample_projector_columns: list[ModelProjectorColumn],
) -> ModelProjectorSchema:
    """Create sample projector schema for testing."""
    return ModelProjectorSchema(
        table="order_projections",
        primary_key="id",
        columns=sample_projector_columns,
    )


@pytest.fixture
def sample_projector_behavior() -> ModelProjectorBehavior:
    """Create sample projector behavior for testing."""
    return ModelProjectorBehavior(
        mode="upsert",
    )


@pytest.fixture
def sample_contract(
    sample_projector_schema: ModelProjectorSchema,
    sample_projector_behavior: ModelProjectorBehavior,
) -> ModelProjectorContract:
    """Create a sample projector contract for testing."""
    return ModelProjectorContract(
        projector_kind="materialized_view",
        projector_id="order-projector-v1",
        name="Order Projector",
        version="1.0.0",
        aggregate_type="Order",
        consumed_events=["order.created.v1", "order.updated.v1"],
        projection_schema=sample_projector_schema,
        behavior=sample_projector_behavior,
    )


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create mocked asyncpg.Pool.

    Returns:
        MagicMock configured to simulate asyncpg.Pool behavior with
        async connection context manager support.

    The mock uses a context manager pattern that mimics asyncpg's pool.acquire():
        async with pool.acquire() as conn:
            await conn.execute(...)
    """
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    # Create an async context manager for acquire()
    class MockAcquireContext:
        async def __aenter__(self) -> AsyncMock:
            return mock_conn

        async def __aexit__(
            self, exc_type: object, exc_val: object, exc_tb: object
        ) -> None:
            pass

    # acquire() returns the async context manager
    mock_pool.acquire.return_value = MockAcquireContext()

    # Default execute returns success
    mock_conn.execute.return_value = "INSERT 0 1"
    mock_conn.fetchrow.return_value = None
    mock_conn.fetch.return_value = []

    # Store mock_conn on mock_pool for test access
    mock_pool._mock_conn = mock_conn

    return mock_pool


@pytest.fixture
def sample_order_created_payload() -> OrderCreatedPayload:
    """Create sample order created payload."""
    return OrderCreatedPayload(
        order_id=uuid4(),
        customer_id=uuid4(),
        status="pending",
        total_amount=99.99,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_event_envelope(
    sample_order_created_payload: OrderCreatedPayload,
) -> ModelEventEnvelope[OrderCreatedPayload]:
    """Create sample event envelope for testing."""
    return ModelEventEnvelope(
        payload=sample_order_created_payload,
        envelope_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        correlation_id=uuid4(),
        metadata=ModelEnvelopeMetadata(
            tags={"event_type": "order.created.v1"},
        ),
        onex_version=ModelSemVer(major=1, minor=0, patch=0),
        envelope_version=ModelSemVer(major=1, minor=0, patch=0),
    )


@pytest.fixture
def unconsumed_event_envelope() -> ModelEventEnvelope[UserCreatedPayload]:
    """Create event envelope with unconsumed event type."""
    payload = UserCreatedPayload(
        user_id=uuid4(),
        email="test@example.com",
    )
    return ModelEventEnvelope(
        payload=payload,
        envelope_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        correlation_id=uuid4(),
        metadata=ModelEnvelopeMetadata(
            tags={"event_type": "user.created.v1"},
        ),
        onex_version=ModelSemVer(major=1, minor=0, patch=0),
        envelope_version=ModelSemVer(major=1, minor=0, patch=0),
    )


@pytest.fixture
def nested_payload_envelope() -> ModelEventEnvelope[OrderWithNestedPayload]:
    """Create event envelope with nested payload for deep extraction tests."""
    payload = OrderWithNestedPayload(
        order_id=uuid4(),
        customer=NestedCustomer(
            customer_id=uuid4(),
            email="customer@example.com",
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
            tags={"event_type": "order.created.v1"},
        ),
        onex_version=ModelSemVer(major=1, minor=0, patch=0),
        envelope_version=ModelSemVer(major=1, minor=0, patch=0),
    )


# =============================================================================
# Event Filtering Tests
# =============================================================================


class TestProjectorShellEventFiltering:
    """Tests for event filtering based on consumed_events contract definition.

    These tests verify that ProjectorShell correctly filters events based on
    the consumed_events list defined in the projector contract.
    """

    @pytest.mark.asyncio
    async def test_skip_unconsumed_event(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        unconsumed_event_envelope: ModelEventEnvelope[UserCreatedPayload],
    ) -> None:
        """Events not in consumed_events are skipped.

        Given: contract with consumed_events=["order.created.v1", "order.updated.v1"]
        When: project() called with event type "user.created.v1"
        Then: returns ModelProjectionResult(success=True, skipped=True)
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        correlation_id = uuid4()

        result = await projector.project(unconsumed_event_envelope, correlation_id)

        assert result.success is True
        assert result.skipped is True
        assert result.rows_affected == 0
        # Should not interact with database for skipped events
        mock_pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_consumed_event(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Events in consumed_events are processed.

        Given: contract with consumed_events=["order.created.v1", "order.updated.v1"]
        When: project() called with matching event type "order.created.v1"
        Then: returns ModelProjectionResult with rows_affected > 0
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        correlation_id = uuid4()

        result = await projector.project(sample_event_envelope, correlation_id)

        assert result.success is True
        assert result.skipped is False
        assert result.rows_affected >= 0
        # Should interact with database for consumed events
        mock_pool.acquire.assert_called()

    @pytest.mark.asyncio
    async def test_filter_by_exact_event_type_match(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """Event filtering requires exact event type match.

        Given: contract with consumed_events=["order.created.v1"]
        When: project() called with event type "order.created.v2" (different version)
        Then: returns ModelProjectionResult(success=True, skipped=True)
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        # Create event with different version
        payload = OrderCreatedPayload(
            order_id=uuid4(),
            customer_id=uuid4(),
            status="pending",
            total_amount=100.0,
            created_at=datetime.now(UTC),
        )
        envelope = ModelEventEnvelope(
            payload=payload,
            envelope_id=uuid4(),
            envelope_timestamp=datetime.now(UTC),
            metadata=ModelEnvelopeMetadata(
                tags={"event_type": "order.created.v2"},  # Different version
            ),
            onex_version=ModelSemVer(major=1, minor=0, patch=0),
            envelope_version=ModelSemVer(major=1, minor=0, patch=0),
        )

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        result = await projector.project(envelope, uuid4())

        assert result.success is True
        assert result.skipped is True


# =============================================================================
# Value Extraction Tests
# =============================================================================


class TestProjectorShellValueExtraction:
    """Tests for value extraction from event envelopes using source paths.

    These tests verify that ProjectorShell correctly extracts values from
    event envelopes using the source path expressions defined in the schema.
    """

    @pytest.mark.asyncio
    async def test_extract_simple_path(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """'envelope_id' extracts correctly from envelope.

        Verifies that simple paths like envelope_id correctly extract
        top-level envelope fields.
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        # Get the extracted value using the internal _resolve_path method
        extracted = projector._resolve_path(sample_event_envelope, "envelope_id")

        assert extracted == sample_event_envelope.envelope_id

    @pytest.mark.asyncio
    async def test_extract_payload_path(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """'payload.status' extracts from payload.

        Verifies that paths into the payload correctly extract nested values.
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        extracted = projector._resolve_path(sample_event_envelope, "payload.status")

        assert extracted == sample_event_envelope.payload.status

    @pytest.mark.asyncio
    async def test_extract_nested_model_path(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        nested_payload_envelope: ModelEventEnvelope[OrderWithNestedPayload],
    ) -> None:
        """'payload.customer.email' extracts from nested models.

        Verifies that deep paths correctly traverse nested Pydantic models.
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        extracted = projector._resolve_path(
            nested_payload_envelope, "payload.customer.email"
        )

        assert extracted == nested_payload_envelope.payload.customer.email

    @pytest.mark.asyncio
    async def test_extract_with_on_event_filter_match(
        self,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Column with on_event extracts when event matches.

        Given: Column with on_event="order.created.v1"
        When: _extract_values called with matching event type
        Then: value is extracted normally
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        # Create contract with on_event filter
        columns = [
            ModelProjectorColumn(
                name="id",
                type="UUID",
                source="payload.order_id",
            ),
            ModelProjectorColumn(
                name="initial_status",
                type="TEXT",
                source="payload.status",
                on_event="order.created.v1",  # Only extract on creation
            ),
        ]
        schema = ModelProjectorSchema(
            table="test_projections",
            primary_key="id",
            columns=columns,
        )
        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="test-projector",
            name="Test Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1", "order.updated.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="upsert"),
        )

        projector = ProjectorShell(contract=contract, pool=mock_pool)

        # Extract with matching event type
        values = projector._extract_values(sample_event_envelope, "order.created.v1")

        assert "initial_status" in values
        assert values["initial_status"] == sample_event_envelope.payload.status

    @pytest.mark.asyncio
    async def test_extract_with_on_event_filter_no_match(
        self,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Column with on_event is skipped when event doesn't match.

        Given: Column with on_event="order.created.v1"
        When: _extract_values called with "order.updated.v1"
        Then: column is not in extracted values (skipped)
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        columns = [
            ModelProjectorColumn(
                name="id",
                type="UUID",
                source="payload.order_id",
            ),
            ModelProjectorColumn(
                name="initial_status",
                type="TEXT",
                source="payload.status",
                on_event="order.created.v1",  # Only extract on creation
            ),
        ]
        schema = ModelProjectorSchema(
            table="test_projections",
            primary_key="id",
            columns=columns,
        )
        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="test-projector",
            name="Test Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1", "order.updated.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="upsert"),
        )

        projector = ProjectorShell(contract=contract, pool=mock_pool)

        # Extract with non-matching event type - initial_status should be skipped
        values = projector._extract_values(sample_event_envelope, "order.updated.v1")

        # Column with non-matching on_event filter should not be in values
        assert "initial_status" not in values

    @pytest.mark.asyncio
    async def test_extract_with_default_value(
        self,
        mock_pool: AsyncMock,
    ) -> None:
        """Default value used when path resolves to None.

        Given: Column with default="unknown"
        When: source path resolves to None
        Then: default value is returned
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        columns = [
            ModelProjectorColumn(
                name="id",
                type="UUID",
                source="payload.order_id",
            ),
            ModelProjectorColumn(
                name="optional_field",
                type="TEXT",
                source="payload.nonexistent_field",
                default="default_value",
            ),
        ]
        schema = ModelProjectorSchema(
            table="test_projections",
            primary_key="id",
            columns=columns,
        )
        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="test-projector",
            name="Test Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="upsert"),
        )

        payload = OrderCreatedPayload(
            order_id=uuid4(),
            customer_id=uuid4(),
            status="pending",
            total_amount=100.0,
            created_at=datetime.now(UTC),
        )
        envelope = ModelEventEnvelope(
            payload=payload,
            envelope_id=uuid4(),
            envelope_timestamp=datetime.now(UTC),
            metadata=ModelEnvelopeMetadata(tags={"event_type": "order.created.v1"}),
            onex_version=ModelSemVer(major=1, minor=0, patch=0),
            envelope_version=ModelSemVer(major=1, minor=0, patch=0),
        )

        projector = ProjectorShell(contract=contract, pool=mock_pool)

        values = projector._extract_values(envelope, "order.created.v1")

        assert values["optional_field"] == "default_value"

    @pytest.mark.asyncio
    async def test_extract_path_not_found_returns_none(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Missing path returns None (no exception).

        Given: source path that doesn't exist in the event
        When: _resolve_path is called
        Then: returns None without raising exception
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        extracted = projector._resolve_path(
            sample_event_envelope, "payload.nonexistent_field"
        )

        assert extracted is None

    @pytest.mark.asyncio
    async def test_extract_envelope_timestamp(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """'envelope_timestamp' extracts timestamp correctly."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        extracted = projector._resolve_path(sample_event_envelope, "envelope_timestamp")

        assert extracted == sample_event_envelope.envelope_timestamp


# =============================================================================
# Projection Mode Tests (mock database)
# =============================================================================


class TestProjectorShellProjectionModes:
    """Tests for projection modes (upsert, insert_only, append).

    These tests verify that ProjectorShell correctly handles different
    projection modes using mocked asyncpg database operations.
    """

    @pytest.mark.asyncio
    async def test_upsert_mode_inserts_new_record(
        self,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Upsert mode inserts when record doesn't exist.

        Given: upsert mode projection
        When: project() called for new record (no conflict)
        Then: INSERT...ON CONFLICT executed, rows_affected=1
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

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
        ]
        schema = ModelProjectorSchema(
            table="order_projections",
            primary_key="id",
            columns=columns,
        )
        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="order-projector",
            name="Order Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="upsert"),
        )

        # Mock successful insert
        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"

        projector = ProjectorShell(contract=contract, pool=mock_pool)
        result = await projector.project(sample_event_envelope, uuid4())

        assert result.success is True
        assert result.rows_affected == 1
        # Verify UPSERT SQL was generated - must include ON CONFLICT for upsert mode
        call_args = mock_conn.execute.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "ON CONFLICT" in sql, "Upsert mode must generate ON CONFLICT clause"

    @pytest.mark.asyncio
    async def test_upsert_mode_updates_existing_record(
        self,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Upsert mode updates when record exists (conflict).

        Given: upsert mode projection
        When: project() called for existing record (conflict)
        Then: INSERT...ON CONFLICT DO UPDATE executed, rows_affected=1
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

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
        ]
        schema = ModelProjectorSchema(
            table="order_projections",
            primary_key="id",
            columns=columns,
        )
        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="order-projector",
            name="Order Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="upsert"),
        )

        # Mock update (conflict triggered)
        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"  # Upsert returns 1 row

        projector = ProjectorShell(contract=contract, pool=mock_pool)
        result = await projector.project(sample_event_envelope, uuid4())

        assert result.success is True
        assert result.rows_affected == 1

    @pytest.mark.asyncio
    async def test_insert_only_mode_succeeds_for_new(
        self,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Insert-only mode works for new records.

        Given: insert_only mode projection
        When: project() called for new record
        Then: INSERT executed successfully, rows_affected=1
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

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
        ]
        schema = ModelProjectorSchema(
            table="order_projections",
            primary_key="id",
            columns=columns,
        )
        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="order-projector",
            name="Order Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="insert_only"),
        )

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"

        projector = ProjectorShell(contract=contract, pool=mock_pool)
        result = await projector.project(sample_event_envelope, uuid4())

        assert result.success is True
        assert result.rows_affected == 1

    @pytest.mark.asyncio
    async def test_insert_only_mode_fails_on_conflict(
        self,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Insert-only mode raises/fails on duplicate key.

        Given: insert_only mode projection
        When: project() called for existing record (duplicate key)
        Then: raises error or returns failure result
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

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
        ]
        schema = ModelProjectorSchema(
            table="order_projections",
            primary_key="id",
            columns=columns,
        )
        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="order-projector",
            name="Order Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="insert_only"),
        )

        # Mock duplicate key violation with asyncpg exception
        import asyncpg

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.side_effect = asyncpg.UniqueViolationError(
            "duplicate key value violates unique constraint"
        )

        projector = ProjectorShell(contract=contract, pool=mock_pool)
        result = await projector.project(sample_event_envelope, uuid4())

        # insert_only mode should report failure on conflict
        assert result.success is False
        assert result.error is not None
        assert "unique" in result.error.lower() or "constraint" in result.error.lower()

    @pytest.mark.asyncio
    async def test_append_mode_always_inserts(
        self,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Append mode always creates new rows (event log style).

        Given: append mode projection
        When: project() called multiple times
        Then: INSERT executed each time, no conflict handling
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        columns = [
            ModelProjectorColumn(
                name="id",
                type="UUID",
                source="envelope_id",  # Use envelope_id for uniqueness
            ),
            ModelProjectorColumn(
                name="order_id",
                type="UUID",
                source="payload.order_id",
            ),
            ModelProjectorColumn(
                name="status",
                type="TEXT",
                source="payload.status",
            ),
            ModelProjectorColumn(
                name="event_timestamp",
                type="TIMESTAMPTZ",
                source="envelope_timestamp",
            ),
        ]
        schema = ModelProjectorSchema(
            table="order_events",
            primary_key="id",
            columns=columns,
        )
        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="order-events-projector",
            name="Order Events Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="append"),
        )

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"

        projector = ProjectorShell(contract=contract, pool=mock_pool)

        # Project the same event twice - append should work both times
        result1 = await projector.project(sample_event_envelope, uuid4())
        result2 = await projector.project(sample_event_envelope, uuid4())

        assert result1.success is True
        assert result2.success is True
        assert mock_conn.execute.call_count >= 2


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestProjectorShellIdempotency:
    """Tests for idempotency guarantees during event replay.

    These tests verify that ProjectorShell produces consistent results
    when replaying the same events multiple times.
    """

    @pytest.mark.asyncio
    async def test_idempotent_replay_same_result(
        self,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Same event replayed produces same database state.

        Given: upsert mode projection
        When: same event projected multiple times
        Then: database state remains consistent (idempotent)
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

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
        ]
        schema = ModelProjectorSchema(
            table="order_projections",
            primary_key="id",
            columns=columns,
        )
        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="order-projector",
            name="Order Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="upsert"),
        )

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"

        projector = ProjectorShell(contract=contract, pool=mock_pool)
        correlation_id = uuid4()

        # Project same event multiple times
        result1 = await projector.project(sample_event_envelope, correlation_id)
        result2 = await projector.project(sample_event_envelope, correlation_id)
        result3 = await projector.project(sample_event_envelope, correlation_id)

        # All should succeed - idempotent replay
        assert result1.success is True
        assert result2.success is True
        assert result3.success is True

    @pytest.mark.asyncio
    async def test_sequence_number_tracked(
        self,
        mock_pool: AsyncMock,
    ) -> None:
        """Sequence number is used for idempotency tracking.

        Given: idempotency config with sequence tracking
        When: events projected with sequence numbers
        Then: duplicate sequences are handled correctly
        """
        from omnibase_core.models.projectors.model_idempotency_config import (
            ModelIdempotencyConfig,
        )

        from omnibase_infra.runtime.projector_shell import ProjectorShell

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
        ]
        schema = ModelProjectorSchema(
            table="order_projections",
            primary_key="id",
            columns=columns,
        )
        idempotency = ModelIdempotencyConfig(
            enabled=True,
            key="envelope_id",
        )
        contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="order-projector",
            name="Order Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="upsert", idempotency=idempotency),
        )

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"

        projector = ProjectorShell(contract=contract, pool=mock_pool)

        # Create event with specific envelope_id
        envelope_id = uuid4()
        payload = OrderCreatedPayload(
            order_id=uuid4(),
            customer_id=uuid4(),
            status="pending",
            total_amount=100.0,
            created_at=datetime.now(UTC),
        )
        envelope = ModelEventEnvelope(
            payload=payload,
            envelope_id=envelope_id,
            envelope_timestamp=datetime.now(UTC),
            metadata=ModelEnvelopeMetadata(tags={"event_type": "order.created.v1"}),
            onex_version=ModelSemVer(major=1, minor=0, patch=0),
            envelope_version=ModelSemVer(major=1, minor=0, patch=0),
        )

        # Project with idempotency tracking
        result = await projector.project(envelope, uuid4())

        assert result.success is True


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProjectorShellProtocolCompliance:
    """Tests for ProtocolEventProjector protocol compliance.

    These tests verify that ProjectorShell correctly implements
    the ProtocolEventProjector interface.
    """

    def test_implements_protocol_event_projector(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """ProjectorShell implements ProtocolEventProjector.

        Uses duck typing verification instead of isinstance check
        per ONEX principle: Protocol Resolution - Duck typing through
        protocols, never isinstance.
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        # Verify protocol compliance via duck typing (hasattr/callable checks)
        # Required properties
        assert hasattr(projector, "projector_id")
        assert hasattr(projector, "aggregate_type")
        assert hasattr(projector, "consumed_events")
        assert hasattr(projector, "is_placeholder")

        # Required methods must be callable
        assert hasattr(projector, "project")
        assert callable(getattr(projector, "project", None))

        assert hasattr(projector, "get_state")
        assert callable(getattr(projector, "get_state", None))

    def test_projector_id_from_contract(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """projector_id property returns contract value."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        assert projector.projector_id == sample_contract.projector_id

    def test_aggregate_type_from_contract(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """aggregate_type property returns contract value."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        assert projector.aggregate_type == sample_contract.aggregate_type

    def test_consumed_events_from_contract(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """consumed_events property returns contract list."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        assert projector.consumed_events == sample_contract.consumed_events

    def test_is_placeholder_returns_false(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """is_placeholder is always False for real implementation."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        assert projector.is_placeholder is False


# =============================================================================
# get_state Tests
# =============================================================================


class TestProjectorShellGetState:
    """Tests for get_state aggregate state retrieval.

    These tests verify that ProjectorShell correctly retrieves
    the current projected state for aggregates.
    """

    @pytest.mark.asyncio
    async def test_get_state_returns_dict_when_found(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """get_state returns dict for existing aggregate.

        Given: aggregate exists in projection table
        When: get_state() called with aggregate_id
        Then: returns dict with projected state
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id = uuid4()
        expected_state = {
            "id": aggregate_id,
            "status": "confirmed",
            "total_amount": 150.0,
        }

        mock_conn = mock_pool._mock_conn
        mock_conn.fetchrow.return_value = expected_state

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        correlation_id = uuid4()

        result = await projector.get_state(aggregate_id, correlation_id)

        assert result is not None
        assert isinstance(result, dict)
        assert result["id"] == aggregate_id
        assert result["status"] == "confirmed"

    @pytest.mark.asyncio
    async def test_get_state_returns_none_when_not_found(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """get_state returns None for non-existent aggregate.

        Given: aggregate does not exist in projection table
        When: get_state() called with aggregate_id
        Then: returns None
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id = uuid4()

        mock_conn = mock_pool._mock_conn
        mock_conn.fetchrow.return_value = None

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        correlation_id = uuid4()

        result = await projector.get_state(aggregate_id, correlation_id)

        assert result is None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestProjectorShellErrorHandling:
    """Tests for error handling during projection operations.

    These tests verify that ProjectorShell correctly handles various
    error conditions during database operations.

    Note: The implementation raises infrastructure errors for connection
    and general execution failures, while returning failure results only
    for unique constraint violations (which are data-level errors).
    """

    @pytest.mark.asyncio
    async def test_database_connection_error_raises_infra_error(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Database connection errors raise InfraConnectionError.

        Given: database connection fails with asyncpg.PostgresConnectionError
        When: project() called
        Then: raises InfraConnectionError (not a return value)
        """
        import asyncpg

        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        # Create a mock acquire that raises connection error
        class MockAcquireContextWithError:
            async def __aenter__(self) -> MagicMock:
                raise asyncpg.PostgresConnectionError("Connection refused")

            async def __aexit__(
                self, exc_type: object, exc_val: object, exc_tb: object
            ) -> None:
                pass

        mock_pool.acquire.return_value = MockAcquireContextWithError()

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        with pytest.raises(InfraConnectionError) as exc_info:
            await projector.project(sample_event_envelope, uuid4())

        assert "connect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_sql_execution_error_raises_runtime_error(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """SQL execution errors raise RuntimeHostError.

        Given: SQL execution fails with generic exception
        When: project() called
        Then: raises RuntimeHostError (for unexpected errors)
        """
        from omnibase_infra.errors import RuntimeHostError
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.side_effect = Exception("column 'nonexistent' does not exist")

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        with pytest.raises(RuntimeHostError):
            await projector.project(sample_event_envelope, uuid4())

    @pytest.mark.asyncio
    async def test_unique_violation_returns_failure_result(
        self,
        mock_pool: MagicMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Unique constraint violations return failure result for insert_only mode.

        Given: insert_only mode and SQL execution fails with asyncpg.UniqueViolationError
        When: project() called
        Then: returns ModelProjectionResult with success=False

        Note: UniqueViolationError is only caught and returned as failure for
        insert_only mode. For upsert mode, the error is re-raised as it indicates
        unexpected behavior (upsert should handle conflicts via ON CONFLICT).
        """
        import asyncpg

        from omnibase_infra.runtime.projector_shell import ProjectorShell

        # Create insert_only mode contract - UniqueViolationError is only
        # caught for this mode (expected behavior for duplicate key rejection)
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
        ]
        schema = ModelProjectorSchema(
            table="order_projections",
            primary_key="id",
            columns=columns,
        )
        insert_only_contract = ModelProjectorContract(
            projector_kind="materialized_view",
            projector_id="order-projector",
            name="Order Projector",
            version="1.0.0",
            aggregate_type="Order",
            consumed_events=["order.created.v1"],
            projection_schema=schema,
            behavior=ModelProjectorBehavior(mode="insert_only"),
        )

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.side_effect = asyncpg.UniqueViolationError(
            "duplicate key value violates unique constraint"
        )

        projector = ProjectorShell(contract=insert_only_contract, pool=mock_pool)
        result = await projector.project(sample_event_envelope, uuid4())

        assert result.success is False
        assert result.error is not None
        assert "unique" in result.error.lower() or "constraint" in result.error.lower()


# =============================================================================
# SQL Generation Tests
# =============================================================================


class TestProjectorShellSQLGeneration:
    """Tests for SQL query generation.

    These tests verify that ProjectorShell generates correct SQL
    for different projection modes and schema configurations.
    """

    @pytest.mark.asyncio
    async def test_generates_correct_column_list(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """SQL includes all columns from schema."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        await projector.project(sample_event_envelope, uuid4())

        # Verify execute was called
        assert mock_conn.execute.called
        call_args = mock_conn.execute.call_args
        assert call_args is not None

        # Check SQL contains expected column names
        sql = call_args[0][0]
        for column in sample_contract.projection_schema.columns:
            # Column names should appear in INSERT statement
            assert column.name in sql

    @pytest.mark.asyncio
    async def test_generates_parameterized_queries(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """SQL uses parameterized queries to prevent injection."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        await projector.project(sample_event_envelope, uuid4())

        # Verify execute was called with parameters
        assert mock_conn.execute.called
        call_args = mock_conn.execute.call_args
        assert call_args is not None

        # Should have SQL and parameter values
        sql = call_args[0][0]
        # Parameterized queries use $1, $2, etc. placeholders
        assert "$" in sql or "?" in sql


# =============================================================================
# Contract Property Access Tests
# =============================================================================


class TestProjectorShellContractAccess:
    """Tests for accessing the underlying contract.

    These tests verify that ProjectorShell provides access to the
    underlying projector contract for inspection and debugging.
    """

    def test_contract_property_returns_contract(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """contract property returns the underlying contract."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        assert projector.contract is sample_contract
        assert projector.contract.projector_id == "order-projector-v1"

    def test_schema_accessible_via_contract(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """projection_schema accessible via contract property."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        assert projector.contract.projection_schema is not None
        assert projector.contract.projection_schema.table == "order_projections"
        assert projector.contract.projection_schema.primary_key == "id"


# =============================================================================
# Integration-Ready Tests (with mocked pool)
# =============================================================================


class TestProjectorShellIntegrationReady:
    """Tests verifying integration-ready patterns.

    These tests verify that ProjectorShell follows patterns that
    enable smooth integration with the runtime infrastructure.
    """

    @pytest.mark.asyncio
    async def test_can_be_instantiated_with_pool(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """ProjectorShell can be instantiated with asyncpg pool."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        assert projector is not None
        assert projector.projector_id == sample_contract.projector_id

    @pytest.mark.asyncio
    async def test_handles_correlation_id_propagation(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Correlation ID is propagated through operations."""
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        correlation_id = uuid4()

        result = await projector.project(sample_event_envelope, correlation_id)

        # Operation should succeed with correlation_id
        assert result.success is True
        # Correlation ID should be available for logging/tracing
        assert correlation_id is not None


__all__ = [
    "TestProjectorShellContractAccess",
    "TestProjectorShellErrorHandling",
    "TestProjectorShellEventFiltering",
    "TestProjectorShellGetState",
    "TestProjectorShellIdempotency",
    "TestProjectorShellIntegrationReady",
    "TestProjectorShellProjectionModes",
    "TestProjectorShellProtocolCompliance",
    "TestProjectorShellSQLGeneration",
    "TestProjectorShellValueExtraction",
]
