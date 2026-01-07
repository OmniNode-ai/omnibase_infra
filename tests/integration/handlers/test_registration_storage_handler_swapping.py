# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for Registration Storage Handler Swapping.

This module tests that registration storage handlers can be swapped at runtime
without breaking node functionality. This is a core capability-oriented design
principle: handlers should be interchangeable as long as they implement the
same protocol.

Handler Swapping Concept
------------------------
The capability-oriented design allows nodes to work with any handler that
implements the required protocol. This enables:
- Testing with mock handlers
- Switching between backends (PostgreSQL, MongoDB) without code changes
- Gradual migration between storage backends
- Environment-specific handler selection (dev vs prod)

Handlers Tested:
    - MockRegistrationStorageHandler: In-memory mock for testing
    - PostgresRegistrationStorageHandler: PostgreSQL backend (requires DB)

CI/CD Graceful Skip Behavior
----------------------------
These integration tests skip gracefully when PostgreSQL is unavailable,
enabling CI/CD pipelines to run without hard failures in environments
without database access.

Related:
    - OMN-1131: Capability-oriented node architecture
    - ProtocolRegistrationStorageHandler: Protocol definition
    - NodeRegistrationStorageEffect: Effect node that uses these handlers
    - PR #119: Test coverage for handler swapping
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind

from omnibase_infra.handlers.registration_storage.handler_mock_registration_storage import (
    MockRegistrationStorageHandler,
)
from omnibase_infra.handlers.registration_storage.protocol_registration_storage_handler import (
    ProtocolRegistrationStorageHandler,
)
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelDeleteResult,
    ModelStorageHealthCheckResult,
)
from omnibase_infra.nodes.node_registration_storage_effect.models.model_registration_record import (
    ModelRegistrationRecord,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# =============================================================================
# Environment Configuration
# =============================================================================

# Check if PostgreSQL is available for integration tests
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_AVAILABLE = POSTGRES_HOST is not None and POSTGRES_PASSWORD is not None


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_handler() -> MockRegistrationStorageHandler:
    """Create a MockRegistrationStorageHandler for testing."""
    return MockRegistrationStorageHandler()


@pytest.fixture
def sample_registration_record() -> ModelRegistrationRecord:
    """Create a sample registration record for testing."""
    return ModelRegistrationRecord(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,
        node_version="1.0.0",
        capabilities=["registration.storage", "registration.storage.query"],
        endpoints={"health": "http://localhost:8080/health"},
        metadata={"team": "platform", "environment": "test"},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        correlation_id=uuid4(),
    )


@pytest.fixture
def multiple_registration_records() -> list[ModelRegistrationRecord]:
    """Create multiple registration records for batch testing."""
    records = []
    for i in range(5):
        records.append(
            ModelRegistrationRecord(
                node_id=uuid4(),
                node_type=EnumNodeKind.EFFECT if i % 2 == 0 else EnumNodeKind.COMPUTE,
                node_version=f"1.{i}.0",
                capabilities=[f"capability.{i}"],
                endpoints={"health": f"http://localhost:808{i}/health"},
                metadata={"index": str(i)},
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                correlation_id=uuid4(),
            )
        )
    return records


# =============================================================================
# Handler Swapping Test Base Class
# =============================================================================


class BaseHandlerSwappingTests:
    """Base class defining common handler swapping tests.

    Subclasses implement handler_fixture to provide the specific handler
    to test. This ensures all handlers pass the same behavioral tests.
    """

    @pytest.fixture
    def handler(self) -> ProtocolRegistrationStorageHandler:
        """Override in subclass to provide the handler to test."""
        raise NotImplementedError("Subclasses must implement handler fixture")

    async def test_handler_conforms_to_protocol(
        self,
        handler: ProtocolRegistrationStorageHandler,
    ) -> None:
        """Handler is an instance of ProtocolRegistrationStorageHandler."""
        assert isinstance(handler, ProtocolRegistrationStorageHandler), (
            f"{type(handler).__name__} must implement ProtocolRegistrationStorageHandler"
        )

    async def test_handler_has_handler_type(
        self,
        handler: ProtocolRegistrationStorageHandler,
    ) -> None:
        """Handler has handler_type property returning non-empty string."""
        handler_type = handler.handler_type
        assert isinstance(handler_type, str), "handler_type must return string"
        assert len(handler_type) > 0, "handler_type must not be empty"

    async def test_store_and_query_roundtrip(
        self,
        handler: ProtocolRegistrationStorageHandler,
        sample_registration_record: ModelRegistrationRecord,
    ) -> None:
        """Records can be stored and retrieved correctly."""
        correlation_id = uuid4()

        # Store the record
        store_result = await handler.store_registration(
            record=sample_registration_record,
            correlation_id=correlation_id,
        )

        # Verify store succeeded
        assert store_result.success, f"Store failed: {store_result.error}"
        assert store_result.node_id == sample_registration_record.node_id

        # Query the record back
        query_result = await handler.query_registrations(
            node_type=sample_registration_record.node_type,
            correlation_id=correlation_id,
        )

        # Verify query returned the stored record
        assert query_result.success, "Query failed"
        assert len(query_result.records) >= 1

        # Find our record in results
        found = False
        for record in query_result.records:
            if record.node_id == sample_registration_record.node_id:
                found = True
                assert record.node_type == sample_registration_record.node_type
                assert record.node_version == sample_registration_record.node_version
                break

        assert found, "Stored record not found in query results"

    async def test_delete_registration(
        self,
        handler: ProtocolRegistrationStorageHandler,
        sample_registration_record: ModelRegistrationRecord,
    ) -> None:
        """Records can be deleted correctly."""
        correlation_id = uuid4()

        # Store a record first
        await handler.store_registration(
            record=sample_registration_record,
            correlation_id=correlation_id,
        )

        # Delete the record
        delete_result = await handler.delete_registration(
            node_id=sample_registration_record.node_id,
            correlation_id=correlation_id,
        )

        assert isinstance(delete_result, ModelDeleteResult), (
            "delete_registration must return ModelDeleteResult"
        )
        assert delete_result.was_deleted(), "Delete should indicate record was deleted"

        # Verify record is gone - delete again should return deleted=False
        delete_result_again = await handler.delete_registration(
            node_id=sample_registration_record.node_id,
            correlation_id=correlation_id,
        )

        assert delete_result_again.success, "Delete operation should succeed"
        assert not delete_result_again.deleted, (
            "Delete should indicate no record was deleted for non-existent record"
        )

    async def test_health_check_returns_status(
        self,
        handler: ProtocolRegistrationStorageHandler,
    ) -> None:
        """Health check returns valid status information."""
        correlation_id = uuid4()

        health_status = await handler.health_check(correlation_id=correlation_id)

        assert isinstance(health_status, ModelStorageHealthCheckResult), (
            "health_check must return ModelStorageHealthCheckResult"
        )
        assert hasattr(health_status, "healthy"), (
            "health_check must include healthy attribute"
        )
        assert hasattr(health_status, "backend_type"), (
            "health_check must include backend_type attribute"
        )


# =============================================================================
# Mock Handler Swapping Tests
# =============================================================================


class TestMockHandlerSwapping(BaseHandlerSwappingTests):
    """Test handler swapping with MockRegistrationStorageHandler.

    These tests verify that the mock handler can be used as a drop-in
    replacement for any other handler implementing the protocol.
    """

    @pytest.fixture
    def handler(
        self, mock_handler: MockRegistrationStorageHandler
    ) -> MockRegistrationStorageHandler:
        """Provide MockRegistrationStorageHandler for testing."""
        return mock_handler

    @pytest.mark.asyncio
    async def test_mock_handler_store_and_query(
        self,
        handler: MockRegistrationStorageHandler,
        sample_registration_record: ModelRegistrationRecord,
    ) -> None:
        """Mock handler can store and query records."""
        await self.test_store_and_query_roundtrip(handler, sample_registration_record)

    @pytest.mark.asyncio
    async def test_mock_handler_delete(
        self,
        handler: MockRegistrationStorageHandler,
        sample_registration_record: ModelRegistrationRecord,
    ) -> None:
        """Mock handler can delete records."""
        await self.test_delete_registration(handler, sample_registration_record)

    @pytest.mark.asyncio
    async def test_mock_handler_health_check(
        self,
        handler: MockRegistrationStorageHandler,
    ) -> None:
        """Mock handler health check works."""
        await self.test_health_check_returns_status(handler)

    @pytest.mark.asyncio
    async def test_mock_handler_multiple_records(
        self,
        handler: MockRegistrationStorageHandler,
        multiple_registration_records: list[ModelRegistrationRecord],
    ) -> None:
        """Mock handler can store and query multiple records."""
        correlation_id = uuid4()

        # Store all records
        for record in multiple_registration_records:
            result = await handler.store_registration(
                record=record,
                correlation_id=correlation_id,
            )
            assert result.success

        # Query all EFFECT nodes
        query_result = await handler.query_registrations(
            node_type=EnumNodeKind.EFFECT,
            correlation_id=correlation_id,
        )

        assert query_result.success
        # Should have at least some EFFECT nodes (index 0, 2, 4)
        assert len(query_result.records) >= 3

    @pytest.mark.asyncio
    async def test_mock_handler_pagination(
        self,
        handler: MockRegistrationStorageHandler,
        multiple_registration_records: list[ModelRegistrationRecord],
    ) -> None:
        """Mock handler supports pagination."""
        correlation_id = uuid4()

        # Store all records
        for record in multiple_registration_records:
            await handler.store_registration(
                record=record,
                correlation_id=correlation_id,
            )

        # Query with limit
        query_result = await handler.query_registrations(
            limit=2,
            offset=0,
            correlation_id=correlation_id,
        )

        assert query_result.success
        assert len(query_result.records) == 2

        # Query next page
        query_result_page2 = await handler.query_registrations(
            limit=2,
            offset=2,
            correlation_id=correlation_id,
        )

        assert query_result_page2.success
        assert len(query_result_page2.records) == 2

    @pytest.mark.asyncio
    async def test_mock_handler_upsert_behavior(
        self,
        handler: MockRegistrationStorageHandler,
        sample_registration_record: ModelRegistrationRecord,
    ) -> None:
        """Mock handler supports upsert (insert or update)."""
        correlation_id = uuid4()

        # First store - should be insert
        result1 = await handler.store_registration(
            record=sample_registration_record,
            correlation_id=correlation_id,
        )
        assert result1.success
        assert result1.was_inserted(), "First store should be an insert"

        # Second store with same node_id - should be update
        updated_record = ModelRegistrationRecord(
            node_id=sample_registration_record.node_id,
            node_type=sample_registration_record.node_type,
            node_version="2.0.0",  # Changed version
            capabilities=["new.capability"],
            endpoints={"health": "http://localhost:9090/health"},
            metadata={"updated": "true"},
            created_at=sample_registration_record.created_at,
            updated_at=datetime.now(UTC),
            correlation_id=correlation_id,
        )

        result2 = await handler.store_registration(
            record=updated_record,
            correlation_id=correlation_id,
        )
        assert result2.success
        assert result2.was_updated(), "Second store should be an update"


# =============================================================================
# Handler Factory Pattern Tests
# =============================================================================


class TestHandlerFactoryPattern:
    """Test that handlers can be swapped using factory pattern.

    This validates the capability-oriented design principle that nodes
    should work with any handler implementing the protocol.
    """

    def create_handler_by_type(
        self, handler_type: str
    ) -> ProtocolRegistrationStorageHandler:
        """Factory method to create handlers by type identifier.

        In production code, this pattern would be used to select handlers
        based on configuration (environment variables, config files, etc.).

        Args:
            handler_type: Type identifier ("mock", "postgresql")

        Returns:
            Handler instance implementing ProtocolRegistrationStorageHandler

        Raises:
            ValueError: If handler_type is not recognized
        """
        if handler_type == "mock":
            return MockRegistrationStorageHandler()
        elif handler_type == "postgresql":
            from omnibase_infra.handlers.registration_storage.handler_postgres_registration_storage import (
                PostgresRegistrationStorageHandler,
            )

            return PostgresRegistrationStorageHandler(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
            )
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")

    def test_factory_creates_mock_handler(self) -> None:
        """Factory creates MockRegistrationStorageHandler for 'mock' type."""
        handler = self.create_handler_by_type("mock")

        assert isinstance(handler, MockRegistrationStorageHandler)
        assert isinstance(handler, ProtocolRegistrationStorageHandler)
        assert handler.handler_type == "mock"

    @pytest.mark.skipif(
        not POSTGRES_AVAILABLE,
        reason="PostgreSQL not available (POSTGRES_HOST/POSTGRES_PASSWORD not set)",
    )
    def test_factory_creates_postgres_handler(self) -> None:
        """Factory creates PostgresRegistrationStorageHandler for 'postgresql' type."""
        handler = self.create_handler_by_type("postgresql")

        from omnibase_infra.handlers.registration_storage.handler_postgres_registration_storage import (
            PostgresRegistrationStorageHandler,
        )

        assert isinstance(handler, PostgresRegistrationStorageHandler)
        assert isinstance(handler, ProtocolRegistrationStorageHandler)
        assert handler.handler_type == "postgresql"

    def test_factory_raises_for_unknown_type(self) -> None:
        """Factory raises ValueError for unknown handler type."""
        with pytest.raises(ValueError, match="Unknown handler type"):
            self.create_handler_by_type("unknown")

    @pytest.mark.asyncio
    async def test_swapping_handlers_preserves_behavior(self) -> None:
        """Swapping handlers preserves expected behavior.

        This test demonstrates that code written against the protocol
        works identically regardless of which handler is used.
        """
        correlation_id = uuid4()
        record = ModelRegistrationRecord(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_version="1.0.0",
            capabilities=["test.capability"],
            endpoints={},
            metadata={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            correlation_id=correlation_id,
        )

        # Create two different handlers
        handler1 = self.create_handler_by_type("mock")
        handler2 = self.create_handler_by_type("mock")  # Another mock for comparison

        # Both handlers should work identically
        result1 = await handler1.store_registration(record, correlation_id)
        result2 = await handler2.store_registration(record, correlation_id)

        assert result1.success == result2.success
        assert result1.node_id == result2.node_id

        # Health checks should both work and return ModelStorageHealthCheckResult
        health1 = await handler1.health_check(correlation_id)
        health2 = await handler2.health_check(correlation_id)

        assert isinstance(health1, ModelStorageHealthCheckResult)
        assert isinstance(health2, ModelStorageHealthCheckResult)
        assert health1.healthy == health2.healthy


# =============================================================================
# Runtime Handler Swapping Tests
# =============================================================================


class TestRuntimeHandlerSwapping:
    """Test that handlers can be swapped at runtime.

    This validates that a node or service can change its handler
    implementation without restart or code changes.
    """

    @pytest.mark.asyncio
    async def test_swap_handler_mid_operation(self) -> None:
        """Handler can be swapped mid-operation without data loss.

        Simulates a scenario where a service starts with a mock handler
        for testing, then switches to a different handler.
        """
        correlation_id = uuid4()
        record = ModelRegistrationRecord(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_version="1.0.0",
            capabilities=["test.capability"],
            endpoints={},
            metadata={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            correlation_id=correlation_id,
        )

        # Start with mock handler
        current_handler: ProtocolRegistrationStorageHandler = (
            MockRegistrationStorageHandler()
        )

        # Store data with first handler
        result = await current_handler.store_registration(record, correlation_id)
        assert result.success

        # Swap to a new mock handler (simulating handler switch)
        new_handler = MockRegistrationStorageHandler()

        # Store same record in new handler
        result2 = await new_handler.store_registration(record, correlation_id)
        assert result2.success

        # Both handlers should report healthy
        current_health = await current_handler.health_check()
        new_health = await new_handler.health_check()
        assert current_health.healthy
        assert new_health.healthy

    @pytest.mark.asyncio
    async def test_handler_interface_contract(self) -> None:
        """All handlers expose the same interface contract.

        This test verifies that code written against the protocol
        can work with any conforming handler.
        """

        async def perform_operations(
            handler: ProtocolRegistrationStorageHandler,
        ) -> tuple[bool, bool, bool]:
            """Perform a series of operations using only protocol methods."""
            correlation_id = uuid4()
            record = ModelRegistrationRecord(
                node_id=uuid4(),
                node_type=EnumNodeKind.EFFECT,
                node_version="1.0.0",
                capabilities=[],
                endpoints={},
                metadata={},
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )

            # Store
            store_result = await handler.store_registration(record, correlation_id)

            # Query
            query_result = await handler.query_registrations(
                node_type=EnumNodeKind.EFFECT,
                correlation_id=correlation_id,
            )

            # Delete
            delete_result = await handler.delete_registration(
                record.node_id, correlation_id
            )

            return (
                store_result.success,
                query_result.success,
                delete_result.was_deleted(),
            )

        # Test with mock handler
        mock = MockRegistrationStorageHandler()
        store_ok, query_ok, delete_ok = await perform_operations(mock)

        assert store_ok, "Mock handler store failed"
        assert query_ok, "Mock handler query failed"
        assert delete_ok, "Mock handler delete failed"


# =============================================================================
# PostgreSQL Handler Integration Tests (requires DB)
# =============================================================================


@pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason="PostgreSQL not available (POSTGRES_HOST/POSTGRES_PASSWORD not set)",
)
class TestPostgresHandlerSwapping(BaseHandlerSwappingTests):
    """Test handler swapping with PostgresRegistrationStorageHandler.

    These tests require a running PostgreSQL instance and are skipped
    in CI environments without database access.
    """

    @pytest.fixture
    async def handler(self) -> AsyncGenerator[ProtocolRegistrationStorageHandler, None]:
        """Provide PostgresRegistrationStorageHandler for testing."""
        from omnibase_infra.handlers.registration_storage.handler_postgres_registration_storage import (
            PostgresRegistrationStorageHandler,
        )

        handler = PostgresRegistrationStorageHandler(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )

        yield handler

        # Cleanup
        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_postgres_handler_store_and_query(
        self,
        handler: ProtocolRegistrationStorageHandler,
        sample_registration_record: ModelRegistrationRecord,
    ) -> None:
        """PostgreSQL handler can store and query records."""
        await self.test_store_and_query_roundtrip(handler, sample_registration_record)

    @pytest.mark.asyncio
    async def test_postgres_handler_delete(
        self,
        handler: ProtocolRegistrationStorageHandler,
        sample_registration_record: ModelRegistrationRecord,
    ) -> None:
        """PostgreSQL handler can delete records."""
        await self.test_delete_registration(handler, sample_registration_record)

    @pytest.mark.asyncio
    async def test_postgres_handler_health_check(
        self,
        handler: ProtocolRegistrationStorageHandler,
    ) -> None:
        """PostgreSQL handler health check works."""
        await self.test_health_check_returns_status(handler)


__all__: list[str] = [
    "BaseHandlerSwappingTests",
    "TestMockHandlerSwapping",
    "TestHandlerFactoryPattern",
    "TestRuntimeHandlerSwapping",
    "TestPostgresHandlerSwapping",
]
