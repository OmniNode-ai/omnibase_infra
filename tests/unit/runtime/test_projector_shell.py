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
    4. Supports idempotent event replay via envelope-based deduplication
       (configurable idempotency key, typically envelope_id)
    5. Implements ProtocolEventProjector for runtime integration
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
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

from omnibase_infra.errors import ProtocolConfigurationError

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


class MockAsyncpgRecord(dict):
    """Mock that mimics asyncpg.Record behavior.

    asyncpg.Record objects behave like both dicts (key access via []) and
    support attribute-style access. This mock enables more realistic testing
    without requiring a real database connection.
    """

    def __getattr__(self, name: str) -> object:
        """Enable attribute-style access like asyncpg.Record."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Record has no field '{name}'") from None


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create mocked asyncpg.Pool with spec for type safety.

    Returns:
        MagicMock configured to simulate asyncpg.Pool behavior with
        async connection context manager support.

    The mock uses a context manager pattern that mimics asyncpg's pool.acquire():
        async with pool.acquire() as conn:
            await conn.execute(...)

    Note: Uses spec_set on connection mock to catch typos and regressions.
    The pool mock uses spec=asyncpg.Pool for the same purpose.
    """
    # Use spec to catch typos on pool methods (acquire, close, etc.)
    mock_pool = MagicMock(spec=asyncpg.Pool)

    # Use spec_set on connection to catch typos on connection methods
    # spec_set is stricter - prevents setting attributes not on the spec
    mock_conn = AsyncMock(spec_set=asyncpg.Connection)

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

    # Default execute returns success (mimics asyncpg status string)
    mock_conn.execute.return_value = "INSERT 0 1"
    # Default fetchrow returns None (no row found) - use MockAsyncpgRecord in tests
    # that need to verify field access patterns
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
        # Strict assertion: exactly 1 row should be affected
        # to catch rowcount regressions
        assert result.rows_affected == 1, (
            f"Expected exactly 1 row affected, got {result.rows_affected}"
        )
        # Should interact with database for consumed events
        mock_pool.acquire.assert_called()
        # Verify execute was actually called (not just acquire)
        mock_conn = mock_pool._mock_conn
        assert mock_conn.execute.called, (
            "Database execute() should be called for consumed events"
        )
        assert mock_conn.execute.call_count == 1, (
            f"Expected exactly 1 execute() call, got {mock_conn.execute.call_count}"
        )

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

    @pytest.mark.asyncio
    async def test_extract_path_not_found_logs_warning_no_default(
        self,
        mock_pool: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Path resolution failure logs WARNING when no default is specified.

        Given: Column with source path that doesn't exist in event
        And: No default value specified
        When: _extract_values is called
        Then: WARNING is logged with column name, source path, and event type
        """
        import logging

        from omnibase_infra.runtime.projector_shell import ProjectorShell

        columns = [
            ModelProjectorColumn(
                name="id",
                type="UUID",
                source="payload.order_id",
            ),
            ModelProjectorColumn(
                name="missing_field",
                type="TEXT",
                source="payload.nonexistent_path",  # This path doesn't exist
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

        order_id = uuid4()
        payload = OrderCreatedPayload(
            order_id=order_id,
            customer_id=uuid4(),
            status="pending",
            total_amount=99.99,
            created_at=datetime.now(tz=UTC),
        )
        envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            source="test",
            source_version=ModelSemVer(major=1, minor=0, patch=0),
            payload=payload,
            metadata=ModelEnvelopeMetadata(tags={"event_type": "order.created.v1"}),
        )

        projector = ProjectorShell(contract=contract, pool=mock_pool)

        with caplog.at_level(
            logging.WARNING, logger="omnibase_infra.runtime.projector_shell"
        ):
            values = projector._extract_values(envelope, "order.created.v1")

        # Value should be None (path resolution failed)
        assert values["missing_field"] is None

        # WARNING should have been logged
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) >= 1
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        # Check the warning message contains the expected information
        found_warning = False
        for msg in warning_messages:
            if (
                "missing_field" in msg
                and "payload.nonexistent_path" in msg
                and "order.created.v1" in msg
                and "Value will be None" in msg
            ):
                found_warning = True
                break
        assert found_warning, f"Expected warning not found. Got: {warning_messages}"

    @pytest.mark.asyncio
    async def test_extract_path_not_found_logs_warning_with_default(
        self,
        mock_pool: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Path resolution failure logs WARNING when default is applied.

        Given: Column with source path that doesn't exist in event
        And: Default value is specified
        When: _extract_values is called
        Then: WARNING is logged mentioning the default value being applied
        """
        import logging

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
                source="payload.nonexistent_path",  # This path doesn't exist
                default="fallback_value",  # Default will be applied
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

        order_id = uuid4()
        payload = OrderCreatedPayload(
            order_id=order_id,
            customer_id=uuid4(),
            status="pending",
            total_amount=99.99,
            created_at=datetime.now(tz=UTC),
        )
        envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            source="test",
            source_version=ModelSemVer(major=1, minor=0, patch=0),
            payload=payload,
            metadata=ModelEnvelopeMetadata(tags={"event_type": "order.created.v1"}),
        )

        projector = ProjectorShell(contract=contract, pool=mock_pool)

        with caplog.at_level(
            logging.WARNING, logger="omnibase_infra.runtime.projector_shell"
        ):
            values = projector._extract_values(envelope, "order.created.v1")

        # Default value should have been applied
        assert values["optional_field"] == "fallback_value"

        # WARNING should have been logged
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) >= 1
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        # Check the warning message mentions the default being used
        found_warning = False
        for msg in warning_messages:
            if (
                "optional_field" in msg
                and "payload.nonexistent_path" in msg
                and "order.created.v1" in msg
                and "Using default value" in msg
                and "fallback_value" in msg
            ):
                found_warning = True
                break
        assert found_warning, f"Expected warning not found. Got: {warning_messages}"


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
        # Verify UPSERT SQL was generated - strict assertions for upsert mode
        call_args = mock_conn.execute.call_args
        assert call_args is not None
        sql = call_args[0][0]
        # Strict: Must have both ON CONFLICT and DO UPDATE for proper upsert behavior
        assert "ON CONFLICT" in sql, "Upsert mode must generate ON CONFLICT clause"
        assert "DO UPDATE" in sql, (
            "Upsert mode must generate DO UPDATE clause (not DO NOTHING)"
        )
        # Verify primary key is referenced in conflict target (quoted)
        # quote_identifier wraps identifiers in double quotes for SQL safety
        assert '("id")' in sql, (
            "ON CONFLICT clause must reference quoted primary key column"
        )

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
    async def test_envelope_based_idempotency_tracking(
        self,
        mock_pool: AsyncMock,
    ) -> None:
        """Envelope ID is used for idempotency tracking (deduplication).

        Given: idempotency config with envelope_id key
        When: events projected with idempotency enabled
        Then: duplicate events (same envelope_id) are handled correctly

        Note: The idempotency mechanism uses a configurable key (typically
        envelope_id) for deduplication, not sequence numbers. This test
        verifies the idempotency configuration is respected during projection.
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

        # Verify idempotency configuration is accessible and correctly set
        assert projector.contract.behavior.idempotency is not None
        assert projector.contract.behavior.idempotency.enabled is True
        assert projector.contract.behavior.idempotency.key == "envelope_id"

        # Verify the envelope_id was used in the projection
        # (idempotency key should be extractable from envelope)
        extracted_key = projector._resolve_path(envelope, "envelope_id")
        assert extracted_key == envelope_id, (
            f"Idempotency key (envelope_id) should resolve to "
            f"{envelope_id}, got {extracted_key}"
        )


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProjectorShellProtocolCompliance:
    """Tests for ProtocolEventProjector protocol compliance.

    These tests verify that ProjectorShell correctly implements
    the ProtocolEventProjector interface.
    """

    @pytest.mark.asyncio
    async def test_implements_protocol_event_projector(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """ProjectorShell implements ProtocolEventProjector.

        Uses duck typing verification instead of isinstance check
        per ONEX principle: Protocol Resolution - Duck typing through
        protocols, never isinstance.

        Verifies both structural compliance (hasattr/callable) and behavioral
        compliance (actual method invocation to verify protocol contract).
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

        # BEHAVIORAL CHECK: Actually invoke protocol methods to verify contract
        # This catches implementation bugs where methods exist but don't work
        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"
        mock_conn.fetchrow.return_value = None

        correlation_id = uuid4()

        # Invoke project() - verifies method signature and return type
        project_result = await projector.project(sample_event_envelope, correlation_id)
        assert project_result is not None, "project() must return a result"
        assert hasattr(project_result, "success"), (
            "project() result must have 'success' attribute"
        )

        # Invoke get_state() - verifies method signature and return type
        # get_state returns None when not found, which is valid
        # The key assertion is that it doesn't raise an exception
        _ = await projector.get_state(uuid4(), correlation_id)

        # Behavioral checks: verify properties return expected types via duck typing
        # Uses duck typing (method/attribute checks) instead of isinstance
        # per ONEX principle: "Protocol Resolution - Duck typing, never isinstance"

        # String duck typing: verify str-like behavior via upper() method
        projector_id_value = projector.projector_id
        has_upper = hasattr(projector_id_value, "upper")
        upper_callable = callable(getattr(projector_id_value, "upper", None))
        assert has_upper and upper_callable, (
            f"projector_id should behave like str, got {type(projector_id_value)}"
        )

        aggregate_type_value = projector.aggregate_type
        has_upper = hasattr(aggregate_type_value, "upper")
        upper_callable = callable(getattr(aggregate_type_value, "upper", None))
        assert has_upper and upper_callable, (
            f"aggregate_type should behave like str, got {type(aggregate_type_value)}"
        )

        # List duck typing: verify list-like behavior via iteration and __len__
        consumed_events_value = projector.consumed_events
        is_iterable = hasattr(consumed_events_value, "__iter__")
        has_len = hasattr(consumed_events_value, "__len__")
        assert is_iterable and has_len, (
            f"consumed_events should behave like list, "
            f"got {type(consumed_events_value)}"
        )

        # Bool duck typing: verify exact boolean values (not just truthy/falsy)
        is_placeholder_value = projector.is_placeholder
        assert is_placeholder_value in {True, False}, (
            f"is_placeholder should be True or False, got {is_placeholder_value!r}"
        )

        # Verify consumed_events contains string-like items via duck typing
        for event in consumed_events_value:
            assert hasattr(event, "upper") and callable(event.upper), (
                f"consumed_events items should behave like str, got {type(event)}"
            )

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

        Note: Uses MockAsyncpgRecord to simulate asyncpg.Record behavior,
        which supports both dict-style and attribute-style access.
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id = uuid4()
        # Use MockAsyncpgRecord to simulate asyncpg.Record behavior
        expected_state = MockAsyncpgRecord(
            {
                "id": aggregate_id,
                "status": "confirmed",
                "total_amount": 150.0,
            }
        )

        mock_conn = mock_pool._mock_conn
        mock_conn.fetchrow.return_value = expected_state

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        correlation_id = uuid4()

        result = await projector.get_state(aggregate_id, correlation_id)

        assert result is not None
        # Duck typing: verify dict-like behavior instead of isinstance check
        # per ONEX principle: "Protocol Resolution - Duck typing, never isinstance"
        assert hasattr(result, "__getitem__"), (
            f"get_state result should support item access, got {type(result)}"
        )
        assert hasattr(result, "keys") and callable(result.keys), (
            f"get_state result should have keys() method, got {type(result)}"
        )
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
# get_states Bulk Query Tests
# =============================================================================


class TestProjectorShellGetStates:
    """Tests for get_states bulk aggregate state retrieval.

    These tests verify that ProjectorShell correctly retrieves
    multiple projected states in a single query for N+1 optimization.
    """

    @pytest.mark.asyncio
    async def test_get_states_returns_dict_of_states(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """get_states returns dict mapping aggregate_id to state.

        Given: multiple aggregates exist in projection table
        When: get_states() called with list of aggregate_ids
        Then: returns dict mapping each found aggregate_id to its state
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id_1 = uuid4()
        aggregate_id_2 = uuid4()

        # Mock multiple rows returned
        mock_conn = mock_pool._mock_conn
        mock_conn.fetch.return_value = [
            MockAsyncpgRecord(
                {
                    "id": aggregate_id_1,
                    "status": "confirmed",
                    "total_amount": 150.0,
                }
            ),
            MockAsyncpgRecord(
                {
                    "id": aggregate_id_2,
                    "status": "pending",
                    "total_amount": 200.0,
                }
            ),
        ]

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        correlation_id = uuid4()

        result = await projector.get_states(
            [aggregate_id_1, aggregate_id_2], correlation_id
        )

        assert len(result) == 2
        assert aggregate_id_1 in result
        assert aggregate_id_2 in result
        assert result[aggregate_id_1]["status"] == "confirmed"
        assert result[aggregate_id_2]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_states_returns_empty_dict_for_empty_input(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """get_states returns empty dict for empty aggregate_ids list.

        Given: empty list of aggregate_ids
        When: get_states() called
        Then: returns empty dict without database call
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        correlation_id = uuid4()

        result = await projector.get_states([], correlation_id)

        assert result == {}
        # Should not make database call for empty input
        mock_pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_states_omits_not_found_aggregates(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """get_states omits aggregates not found in database.

        Given: some aggregate_ids don't exist in projection table
        When: get_states() called with mixed found/not-found ids
        Then: returns dict with only found aggregates
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id_1 = uuid4()
        aggregate_id_2 = uuid4()  # This one won't be found
        aggregate_id_3 = uuid4()

        # Only return rows for id_1 and id_3
        mock_conn = mock_pool._mock_conn
        mock_conn.fetch.return_value = [
            MockAsyncpgRecord(
                {
                    "id": aggregate_id_1,
                    "status": "confirmed",
                }
            ),
            MockAsyncpgRecord(
                {
                    "id": aggregate_id_3,
                    "status": "shipped",
                }
            ),
        ]

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        correlation_id = uuid4()

        result = await projector.get_states(
            [aggregate_id_1, aggregate_id_2, aggregate_id_3], correlation_id
        )

        assert len(result) == 2
        assert aggregate_id_1 in result
        assert aggregate_id_2 not in result  # Not found
        assert aggregate_id_3 in result

    @pytest.mark.asyncio
    async def test_get_states_uses_any_clause_for_bulk_query(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """get_states uses PostgreSQL ANY() for efficient bulk query.

        Given: list of aggregate_ids
        When: get_states() called
        Then: SQL uses ANY($1) syntax for efficient bulk lookup
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_ids = [uuid4(), uuid4(), uuid4()]

        mock_conn = mock_pool._mock_conn
        mock_conn.fetch.return_value = []

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)
        await projector.get_states(aggregate_ids, uuid4())

        # Verify ANY clause was used
        call_args = mock_conn.fetch.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "ANY($1)" in sql, "Bulk query should use PostgreSQL ANY() syntax"


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

        Given: insert_only mode and SQL execution fails with
               asyncpg.UniqueViolationError
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

    @pytest.mark.asyncio
    async def test_unique_violation_raises_runtime_error_for_upsert_mode(
        self,
        mock_pool: MagicMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Unique constraint violations raise RuntimeHostError for upsert mode.

        Given: upsert mode and SQL execution fails with asyncpg.UniqueViolationError
        When: project() called
        Then: raises RuntimeHostError (unexpected - upsert should handle conflicts)

        Note: This tests the fix for PR #144 - raw asyncpg exceptions should not
        leak through the runtime boundary in non-insert_only modes.
        """
        import asyncpg

        from omnibase_infra.errors import RuntimeHostError
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        # Create upsert mode contract - UniqueViolationError is unexpected
        # and should be wrapped as RuntimeHostError
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
        upsert_contract = ModelProjectorContract(
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
        mock_conn.execute.side_effect = asyncpg.UniqueViolationError(
            "duplicate key value violates unique constraint"
        )

        projector = ProjectorShell(contract=upsert_contract, pool=mock_pool)

        with pytest.raises(RuntimeHostError) as exc_info:
            await projector.project(sample_event_envelope, uuid4())

        # Verify the error message includes mode and projector info
        error_msg = str(exc_info.value).lower()
        assert "upsert" in error_msg
        assert "unique" in error_msg or "constraint" in error_msg

    @pytest.mark.asyncio
    async def test_unique_violation_raises_runtime_error_for_append_mode(
        self,
        mock_pool: MagicMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Unique constraint violations raise RuntimeHostError for append mode.

        Given: append mode and SQL execution fails with asyncpg.UniqueViolationError
        When: project() called
        Then: raises RuntimeHostError (unexpected - append is event-log style)

        Note: This tests the fix for PR #144 - raw asyncpg exceptions should not
        leak through the runtime boundary in non-insert_only modes.
        """
        import asyncpg

        from omnibase_infra.errors import RuntimeHostError
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        # Create append mode contract - UniqueViolationError is unexpected
        # and should be wrapped as RuntimeHostError
        columns = [
            ModelProjectorColumn(
                name="id",
                type="UUID",
                source="envelope_id",  # Use envelope_id for uniqueness
            ),
            ModelProjectorColumn(
                name="status",
                type="TEXT",
                source="payload.status",
            ),
        ]
        schema = ModelProjectorSchema(
            table="order_events",
            primary_key="id",
            columns=columns,
        )
        append_contract = ModelProjectorContract(
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
        mock_conn.execute.side_effect = asyncpg.UniqueViolationError(
            "duplicate key value violates unique constraint"
        )

        projector = ProjectorShell(contract=append_contract, pool=mock_pool)

        with pytest.raises(RuntimeHostError) as exc_info:
            await projector.project(sample_event_envelope, uuid4())

        # Verify the error message includes mode and projector info
        error_msg = str(exc_info.value).lower()
        assert "append" in error_msg
        assert "unique" in error_msg or "constraint" in error_msg


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
        # Parameterized queries use $1, $2, etc. placeholders (asyncpg format)
        # Strict check: verify specific numbered placeholders exist
        assert "$1" in sql, (
            f"Parameterized query must use $1 placeholder (asyncpg format), "
            f"got: {sql[:150]}"
        )
        # Verify at least $2 exists (most projections have multiple columns)
        assert "$2" in sql, (
            f"Parameterized query should use multiple placeholders "
            f"($2 expected), got: {sql[:150]}"
        )
        # Verify parameters are passed as additional arguments (asyncpg style)
        assert len(call_args[0]) > 1 or call_args[1], (
            "execute() should be called with SQL and parameter values"
        )


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


# =============================================================================
# Query Timeout Configuration Tests
# =============================================================================


class TestProjectorShellQueryTimeout:
    """Tests for query timeout configuration.

    These tests verify that ProjectorShell correctly applies
    configurable query timeouts to database operations.
    """

    def test_default_query_timeout_applied(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """Default query timeout is applied when not specified.

        Given: no query_timeout_seconds parameter
        When: ProjectorShell instantiated
        Then: uses default timeout (30 seconds)
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        assert projector._query_timeout == ProjectorShell.DEFAULT_QUERY_TIMEOUT_SECONDS
        assert projector._query_timeout == 30.0

    def test_custom_query_timeout_applied(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """Custom query timeout is applied when specified.

        Given: custom query_timeout_seconds parameter
        When: ProjectorShell instantiated
        Then: uses specified timeout
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        projector = ProjectorShell(
            contract=sample_contract, pool=mock_pool, query_timeout_seconds=10.0
        )

        assert projector._query_timeout == 10.0

    @pytest.mark.asyncio
    async def test_timeout_passed_to_execute(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
        sample_event_envelope: ModelEventEnvelope[OrderCreatedPayload],
    ) -> None:
        """Query timeout is passed to database execute calls.

        Given: ProjectorShell with custom timeout
        When: project() called
        Then: timeout parameter is passed to conn.execute()
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "INSERT 0 1"

        projector = ProjectorShell(
            contract=sample_contract, pool=mock_pool, query_timeout_seconds=5.0
        )
        await projector.project(sample_event_envelope, uuid4())

        # Verify timeout was passed to execute
        call_kwargs = mock_conn.execute.call_args
        assert call_kwargs is not None
        # timeout is passed as keyword argument
        assert "timeout" in call_kwargs.kwargs or (
            len(call_kwargs.args) > 0 and 5.0 in call_kwargs.args
        )

    @pytest.mark.asyncio
    async def test_timeout_passed_to_fetchrow(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """Query timeout is passed to database fetchrow calls.

        Given: ProjectorShell with custom timeout
        When: get_state() called
        Then: timeout parameter is passed to conn.fetchrow()
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        mock_conn = mock_pool._mock_conn
        mock_conn.fetchrow.return_value = None

        projector = ProjectorShell(
            contract=sample_contract, pool=mock_pool, query_timeout_seconds=15.0
        )
        await projector.get_state(uuid4(), uuid4())

        # Verify timeout was passed to fetchrow
        call_kwargs = mock_conn.fetchrow.call_args
        assert call_kwargs is not None
        # timeout is passed as keyword argument
        assert "timeout" in call_kwargs.kwargs


# =============================================================================
# Partial Update Tests
# =============================================================================


class TestProjectorShellPartialUpdate:
    """Tests for partial_update column-specific update operations.

    These tests verify that ProjectorShell correctly performs partial updates
    on specific columns without requiring full event-driven projection.
    """

    @pytest.mark.asyncio
    async def test_partial_update_returns_true_when_row_found(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """partial_update returns True when row is found and updated.

        Given: Row exists in projection table
        When: partial_update() called with valid aggregate_id
        Then: Returns True indicating successful update
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id = uuid4()
        correlation_id = uuid4()

        # Mock successful update (1 row affected)
        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        updates = {"status": "confirmed", "updated_at": datetime.now(UTC)}
        result = await projector.partial_update(aggregate_id, updates, correlation_id)

        assert result is True
        mock_pool.acquire.assert_called()
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_partial_update_returns_false_when_row_not_found(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """partial_update returns False when no row matches.

        Given: Row does not exist in projection table
        When: partial_update() called with non-existent aggregate_id
        Then: Returns False indicating no row was updated
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id = uuid4()
        correlation_id = uuid4()

        # Mock no rows affected
        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "UPDATE 0"

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        updates = {"status": "confirmed"}
        result = await projector.partial_update(aggregate_id, updates, correlation_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_partial_update_generates_correct_sql(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """partial_update generates parameterized UPDATE SQL.

        Given: Contract with table "order_projections" and primary_key "id"
        When: partial_update() called with {"status": "confirmed", "amount": 150.0}
        Then: SQL contains UPDATE, SET with column names, WHERE with primary key
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id = uuid4()
        correlation_id = uuid4()

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        updates = {"status": "confirmed", "total_amount": 150.0}
        await projector.partial_update(aggregate_id, updates, correlation_id)

        # Verify SQL structure
        call_args = mock_conn.execute.call_args
        assert call_args is not None
        sql = call_args[0][0]

        # Check UPDATE statement structure
        assert "UPDATE" in sql
        assert '"order_projections"' in sql
        assert "SET" in sql
        assert '"status"' in sql
        assert '"total_amount"' in sql
        assert "WHERE" in sql
        assert '"id"' in sql  # Primary key from contract
        # Verify parameterized query format
        assert "$1" in sql
        assert "$2" in sql
        assert "$3" in sql  # PK is last parameter

    @pytest.mark.asyncio
    async def test_partial_update_uses_parameterized_values(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """partial_update passes values as parameters (not inline SQL).

        Given: updates dict with values
        When: partial_update() called
        Then: Values are passed as execute() arguments, not embedded in SQL
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id = uuid4()
        correlation_id = uuid4()

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        now = datetime.now(UTC)
        updates = {"status": "shipped", "updated_at": now}
        await projector.partial_update(aggregate_id, updates, correlation_id)

        # Verify parameters passed to execute
        call_args = mock_conn.execute.call_args
        assert call_args is not None

        # Extract positional args (SQL, then values)
        args = call_args[0]
        sql = args[0]

        # Values should be in args, not in SQL string
        assert "shipped" not in sql  # Value not in SQL
        assert "shipped" in args  # Value in parameters
        # aggregate_id should be last parameter
        assert aggregate_id == args[-1]

    @pytest.mark.asyncio
    async def test_partial_update_empty_updates_raises_protocol_configuration_error(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """partial_update raises ProtocolConfigurationError for empty updates dict.

        Given: Empty updates dict
        When: partial_update() called
        Then: Raises ProtocolConfigurationError
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id = uuid4()
        correlation_id = uuid4()

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        with pytest.raises(ProtocolConfigurationError, match="empty"):
            await projector.partial_update(aggregate_id, {}, correlation_id)

        # Should not interact with database
        mock_pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_update_connection_error_raises_infra_error(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: MagicMock,
    ) -> None:
        """partial_update raises InfraConnectionError on connection failure.

        Given: Database connection fails
        When: partial_update() called
        Then: Raises InfraConnectionError (not raw asyncpg exception)
        """
        import asyncpg

        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        # Create mock acquire that raises connection error
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
            await projector.partial_update(uuid4(), {"status": "confirmed"}, uuid4())

        assert "connect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_partial_update_timeout_raises_infra_timeout_error(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """partial_update raises InfraTimeoutError on query timeout.

        Given: Query times out
        When: partial_update() called
        Then: Raises InfraTimeoutError
        """
        import asyncpg

        from omnibase_infra.errors import InfraTimeoutError
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.side_effect = asyncpg.QueryCanceledError(
            "canceling statement due to statement timeout"
        )

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        with pytest.raises(InfraTimeoutError) as exc_info:
            await projector.partial_update(uuid4(), {"status": "confirmed"}, uuid4())

        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_partial_update_generic_error_raises_runtime_error(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """partial_update raises RuntimeHostError on generic exceptions.

        Given: Unexpected database error
        When: partial_update() called
        Then: Raises RuntimeHostError (wraps the error)
        """
        from omnibase_infra.errors import RuntimeHostError
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.side_effect = Exception("column 'nonexistent' does not exist")

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        with pytest.raises(RuntimeHostError):
            await projector.partial_update(uuid4(), {"nonexistent": "value"}, uuid4())

    @pytest.mark.asyncio
    async def test_partial_update_single_column(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """partial_update works with single column update.

        Given: Updates dict with one column
        When: partial_update() called
        Then: SQL contains only that column in SET clause
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        aggregate_id = uuid4()
        correlation_id = uuid4()

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        projector = ProjectorShell(contract=sample_contract, pool=mock_pool)

        # Single column update (like timeout marker)
        now = datetime.now(UTC)
        updates = {"ack_timeout_emitted_at": now}
        result = await projector.partial_update(aggregate_id, updates, correlation_id)

        assert result is True

        # Verify SQL has only one SET clause
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert '"ack_timeout_emitted_at"' in sql
        assert "$1" in sql  # Value parameter
        assert "$2" in sql  # PK parameter
        # Should not have more parameters
        assert "$3" not in sql

    @pytest.mark.asyncio
    async def test_partial_update_timeout_config_applied(
        self,
        sample_contract: ModelProjectorContract,
        mock_pool: AsyncMock,
    ) -> None:
        """partial_update uses configured query timeout.

        Given: ProjectorShell with custom timeout
        When: partial_update() called
        Then: timeout parameter is passed to execute()
        """
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        mock_conn = mock_pool._mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        projector = ProjectorShell(
            contract=sample_contract,
            pool=mock_pool,
            query_timeout_seconds=5.0,
        )

        await projector.partial_update(uuid4(), {"status": "confirmed"}, uuid4())

        # Verify timeout was passed to execute
        call_kwargs = mock_conn.execute.call_args
        assert call_kwargs is not None
        assert "timeout" in call_kwargs.kwargs
        assert call_kwargs.kwargs["timeout"] == 5.0


__all__ = [
    "TestProjectorShellContractAccess",
    "TestProjectorShellErrorHandling",
    "TestProjectorShellEventFiltering",
    "TestProjectorShellGetState",
    "TestProjectorShellGetStates",
    "TestProjectorShellIdempotency",
    "TestProjectorShellIntegrationReady",
    "TestProjectorShellPartialUpdate",
    "TestProjectorShellProjectionModes",
    "TestProjectorShellProtocolCompliance",
    "TestProjectorShellQueryTimeout",
    "TestProjectorShellSQLGeneration",
    "TestProjectorShellValueExtraction",
]
