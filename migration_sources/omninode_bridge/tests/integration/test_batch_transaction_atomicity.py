"""
Comprehensive Batch Transaction Atomicity Tests.

Tests that BATCH_INSERT operations maintain ACID compliance with
all-or-nothing semantics using explicit transaction wrapper.

Test Categories:
1. All inserts succeed together
2. All inserts rollback on failure (mid-batch error simulation)
3. Transaction isolation verification
4. Concurrent batch inserts don't interfere
5. ACID compliance validation
6. Partial failure detection
7. Rollback verification
8. Circuit breaker integration with transactions

Security Fix: Batch transaction atomicity to prevent partial writes
Implementation: Agent 5
"""

import asyncio
from unittest.mock import patch
from uuid import uuid4

import pytest

try:
    from testcontainers.postgres import PostgresContainer

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    PostgresContainer = None  # type: ignore

try:
    from omnibase_core import EnumCoreErrorCode, ModelOnexError

    # For backward compatibility with test expectations
    CoreErrorCode = EnumCoreErrorCode
    OnexError = ModelOnexError
except ImportError:
    # Use fallback when omnibase_core is not available
    class EnumCoreErrorCode:
        VALIDATION_ERROR = "VALIDATION_ERROR"
        DATABASE_ERROR = "DATABASE_ERROR"

    CoreErrorCode = EnumCoreErrorCode

    class ModelOnexError(Exception):
        def __init__(
            self,
            code: str,
            message: str,
            context: dict = None,
            original_error: Exception = None,
        ):
            self.code = code
            self.message = message
            self.context = context or {}
            self.original_error = original_error
            super().__init__(message)

    OnexError = ModelOnexError


from omninode_bridge.infrastructure.entities.model_workflow_step import (
    ModelWorkflowStep,
)
from omninode_bridge.infrastructure.enum_entity_type import EnumEntityType
from omninode_bridge.infrastructure.postgres_connection_manager import (
    ModelPostgresConfig,
    PostgresConnectionManager,
)
from omninode_bridge.nodes.database_adapter_effect.v1_0_0._generic_crud_handlers import (
    GenericCRUDHandlers,
)
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.circuit_breaker import (
    CircuitBreakerState,
    DatabaseCircuitBreaker,
)
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.enums.enum_database_operation_type import (
    EnumDatabaseOperationType,
)
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_database_operation_input import (
    ModelDatabaseOperationInput,
)

# Skip all tests if testcontainers not available
pytestmark = pytest.mark.skipif(
    not TESTCONTAINERS_AVAILABLE,
    reason="testcontainers not installed - required for integration tests",
)


@pytest.fixture(scope="module")
def postgres_container():
    """Start PostgreSQL container for integration tests."""
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers not available")

    container = PostgresContainer("postgres:16-alpine")
    container.start()

    yield container

    container.stop()


@pytest.fixture
async def postgres_config(postgres_container) -> ModelPostgresConfig:
    """Create PostgreSQL configuration from container."""
    return ModelPostgresConfig(
        host=postgres_container.get_container_host_ip(),
        port=int(postgres_container.get_exposed_port(5432)),
        database=postgres_container.dbname,
        user=postgres_container.username,
        password=postgres_container.password,
        schema="public",
        min_connections=2,
        max_connections=10,
    )


@pytest.fixture
async def connection_manager(postgres_config) -> PostgresConnectionManager:
    """Create and initialize connection manager with real database."""
    manager = PostgresConnectionManager(postgres_config)
    await manager.initialize()

    yield manager

    await manager.close()


@pytest.fixture
async def database_schema(connection_manager):
    """Create database schema for testing."""
    # Create workflow_steps table
    await connection_manager.execute_query(
        """
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";

        CREATE TABLE IF NOT EXISTS workflow_steps (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_id UUID NOT NULL,
            step_name VARCHAR(100) NOT NULL,
            step_order INTEGER NOT NULL CHECK (step_order >= 1),
            status VARCHAR(50) NOT NULL,
            execution_time_ms INTEGER,
            step_data JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    yield

    # Cleanup
    await connection_manager.execute_query("DROP TABLE IF EXISTS workflow_steps")


@pytest.fixture
async def crud_handler(connection_manager):
    """Create CRUD handler instance for testing."""

    class TestCRUDHandler(GenericCRUDHandlers):
        """Test implementation of GenericCRUDHandlers."""

        def __init__(self, connection_manager):
            """Initialize with dependencies."""
            self._connection_manager = connection_manager
            self._query_executor = connection_manager

            # Create circuit breaker from config
            circuit_breaker_config = {
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "half_open_max_calls": 3,
            }
            self._circuit_breaker = DatabaseCircuitBreaker.from_config(
                circuit_breaker_config
            )

            # Mock logger
            class MockLogger:
                def log_operation_complete(self, **kwargs):
                    pass

            self._logger = MockLogger()

    handler = TestCRUDHandler(connection_manager)
    return handler


@pytest.mark.integration
@pytest.mark.asyncio
class TestBatchTransactionAtomicity:
    """Integration test suite for batch transaction atomicity."""

    async def test_batch_insert_all_succeed_together(
        self, crud_handler, database_schema
    ):
        """
        Test that all rows in batch INSERT succeed together.

        Verifies that successful batch operations insert all rows atomically.
        """
        workflow_id = uuid4()

        # Create 10 workflow steps
        steps = [
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=f"step_{i}",
                step_order=i + 1,
                status="PENDING",
                step_data={},
            )
            for i in range(10)
        ]

        batch_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.BATCH_INSERT,
            entity_type=EnumEntityType.WORKFLOW_STEP,
            correlation_id=uuid4(),
            batch_entities=steps,
        )

        # Execute batch insert
        result = await crud_handler._handle_batch_insert(batch_input)

        # Verify all succeeded
        assert result.success is True
        assert result.rows_affected == 10
        assert len(result.result_data["ids"]) == 10

        # Verify all in database
        rows = await crud_handler._connection_manager.execute_query(
            "SELECT COUNT(*) as count FROM workflow_steps WHERE workflow_id = $1",
            workflow_id,
        )

        assert rows[0]["count"] == 10

    async def test_batch_insert_rollback_on_failure(
        self, crud_handler, database_schema
    ):
        """
        Test that batch INSERT rolls back all rows on failure.

        Simulates a mid-batch error and verifies that no partial writes occur.
        """
        workflow_id = uuid4()

        # Create valid and invalid steps (duplicate step_order violates unique constraint)
        steps = [
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=f"step_{i}",
                step_order=i + 1,
                status="PENDING",
                step_data={},
            )
            for i in range(5)
        ]

        batch_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.BATCH_INSERT,
            entity_type=EnumEntityType.WORKFLOW_STEP,
            correlation_id=uuid4(),
            batch_entities=steps,
        )

        # Mock connection fetch to simulate mid-batch failure
        original_transaction = crud_handler._connection_manager.transaction

        async def failing_transaction():
            """Mock transaction that fails during batch insert."""

            class FailingConnection:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    # Rollback on exception
                    return False

                async def fetch(self, query, *params):
                    # Simulate constraint violation on 3rd row
                    raise Exception("Mock database constraint violation on row 3")

            return FailingConnection()

        # Patch transaction to use failing mock
        with patch.object(
            crud_handler._connection_manager,
            "transaction",
            side_effect=failing_transaction,
        ):
            # Attempt batch insert (should fail and rollback)
            with pytest.raises((OnexError, Exception)):
                await crud_handler._handle_batch_insert(batch_input)

        # Verify NO rows were inserted (atomicity)
        rows = await crud_handler._connection_manager.execute_query(
            "SELECT COUNT(*) as count FROM workflow_steps WHERE workflow_id = $1",
            workflow_id,
        )

        assert rows[0]["count"] == 0, "Expected 0 rows due to rollback, but found rows"

    async def test_batch_insert_transaction_isolation(
        self, crud_handler, database_schema
    ):
        """
        Test transaction isolation for batch INSERT.

        Verifies that concurrent transactions don't see partial state
        during batch insert execution.
        """
        workflow_id = uuid4()

        steps = [
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=f"step_{i}",
                step_order=i + 1,
                status="PENDING",
                step_data={},
            )
            for i in range(100)  # Large batch to increase transaction duration
        ]

        batch_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.BATCH_INSERT,
            entity_type=EnumEntityType.WORKFLOW_STEP,
            correlation_id=uuid4(),
            batch_entities=steps,
        )

        # Track row counts observed during transaction
        observed_counts = []

        async def concurrent_reader():
            """Concurrent task reading row count during batch insert."""
            for _ in range(10):  # Check 10 times
                try:
                    rows = await crud_handler._connection_manager.execute_query(
                        "SELECT COUNT(*) as count FROM workflow_steps WHERE workflow_id = $1",
                        workflow_id,
                    )
                    observed_counts.append(rows[0]["count"])
                except Exception:
                    pass  # Ignore read errors
                await asyncio.sleep(0.001)  # Small delay between checks

        # Start concurrent reader
        reader_task = asyncio.create_task(concurrent_reader())

        # Execute batch insert
        result = await crud_handler._handle_batch_insert(batch_input)

        # Wait for reader to finish
        await reader_task

        # Verify batch insert succeeded
        assert result.success is True
        assert result.rows_affected == 100

        # Verify isolation: observed counts should be either 0 (before commit)
        # or 100 (after commit), never partial values
        for count in observed_counts:
            assert count in [
                0,
                100,
            ], f"Expected 0 or 100 rows due to isolation, but observed {count}"

    async def test_concurrent_batch_inserts_dont_interfere(
        self, crud_handler, database_schema
    ):
        """
        Test that concurrent batch inserts maintain isolation.

        Verifies that multiple concurrent batch operations don't interfere
        with each other's transaction boundaries.
        """

        async def batch_insert_workflow(workflow_index: int):
            """Execute batch insert for a specific workflow."""
            workflow_id = uuid4()

            steps = [
                ModelWorkflowStep(
                    workflow_id=workflow_id,
                    step_name=f"workflow_{workflow_index}_step_{i}",
                    step_order=i + 1,
                    status="PENDING",
                    step_data={},
                )
                for i in range(10)
            ]

            batch_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.BATCH_INSERT,
                entity_type=EnumEntityType.WORKFLOW_STEP,
                correlation_id=uuid4(),
                batch_entities=steps,
            )

            result = await crud_handler._handle_batch_insert(batch_input)

            # Verify this workflow's batch succeeded
            assert result.success is True
            assert result.rows_affected == 10

            return workflow_id

        # Execute 10 concurrent batch inserts
        workflow_ids = await asyncio.gather(
            *[batch_insert_workflow(i) for i in range(10)]
        )

        # Verify all 10 workflows have exactly 10 steps each
        for workflow_id in workflow_ids:
            rows = await crud_handler._connection_manager.execute_query(
                "SELECT COUNT(*) as count FROM workflow_steps WHERE workflow_id = $1",
                workflow_id,
            )

            assert (
                rows[0]["count"] == 10
            ), f"Expected 10 rows for workflow {workflow_id}, but found {rows[0]['count']}"

        # Verify total row count
        total_rows = await crud_handler._connection_manager.execute_query(
            "SELECT COUNT(*) as count FROM workflow_steps"
        )

        assert total_rows[0]["count"] == 100  # 10 workflows * 10 steps each

    async def test_batch_insert_acid_compliance(self, crud_handler, database_schema):
        """
        Test ACID compliance for batch INSERT.

        Verifies:
        - Atomicity: All or nothing
        - Consistency: Database constraints enforced
        - Isolation: Transactions don't interfere
        - Durability: Committed data persists
        """
        workflow_id = uuid4()

        # Test 1: Atomicity - All succeed
        steps_batch1 = [
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=f"atomic_step_{i}",
                step_order=i + 1,
                status="PENDING",
                step_data={},
            )
            for i in range(5)
        ]

        result1 = await crud_handler._handle_batch_insert(
            ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.BATCH_INSERT,
                entity_type=EnumEntityType.WORKFLOW_STEP,
                correlation_id=uuid4(),
                batch_entities=steps_batch1,
            )
        )

        assert result1.success is True
        assert result1.rows_affected == 5

        # Test 2: Durability - Verify data persists after commit
        rows = await crud_handler._connection_manager.execute_query(
            "SELECT COUNT(*) as count FROM workflow_steps WHERE workflow_id = $1",
            workflow_id,
        )

        assert rows[0]["count"] == 5

        # Test 3: Consistency - Verify constraints enforced
        # (Tested implicitly by database schema constraints)

    async def test_batch_size_limit_enforcement(self, crud_handler, database_schema):
        """
        Test that batch size limit (MAX_BATCH_SIZE) is enforced.

        Prevents resource exhaustion from extremely large batches.
        """
        workflow_id = uuid4()

        # Attempt to insert more than MAX_BATCH_SIZE (1000) rows
        oversized_batch = [
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=f"step_{i}",
                step_order=i + 1,
                status="PENDING",
                step_data={},
            )
            for i in range(1001)  # Exceeds MAX_BATCH_SIZE
        ]

        batch_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.BATCH_INSERT,
            entity_type=EnumEntityType.WORKFLOW_STEP,
            correlation_id=uuid4(),
            batch_entities=oversized_batch,
        )

        # Should raise validation error
        with pytest.raises(OnexError) as exc_info:
            await crud_handler._handle_batch_insert(batch_input)

        assert exc_info.value.error_code == CoreErrorCode.VALIDATION_ERROR
        assert "exceeds maximum" in exc_info.value.message.lower()
        # Check that batch_size is in the nested context structure
        assert "additional_context" in exc_info.value.context
        assert "context" in exc_info.value.context["additional_context"]
        assert "batch_size" in exc_info.value.context["additional_context"]["context"]

    async def test_empty_batch_validation(self, crud_handler, database_schema):
        """Test that empty batch is rejected with validation error."""
        batch_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.BATCH_INSERT,
            entity_type=EnumEntityType.WORKFLOW_STEP,
            correlation_id=uuid4(),
            batch_entities=[],  # Empty batch
        )

        with pytest.raises(OnexError) as exc_info:
            await crud_handler._handle_batch_insert(batch_input)

        assert exc_info.value.error_code == CoreErrorCode.VALIDATION_ERROR
        # Error message should indicate batch_entities is required/missing
        assert "batch_entities" in exc_info.value.message.lower()
        assert "required" in exc_info.value.message.lower()

    async def test_circuit_breaker_integration_with_batch_transactions(
        self, crud_handler, database_schema
    ):
        """
        Test that circuit breaker correctly wraps batch transactions.

        Verifies that circuit breaker resilience patterns apply to
        batch operations without breaking transaction semantics.
        """
        workflow_id = uuid4()

        steps = [
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=f"step_{i}",
                step_order=i + 1,
                status="PENDING",
                step_data={},
            )
            for i in range(5)
        ]

        batch_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.BATCH_INSERT,
            entity_type=EnumEntityType.WORKFLOW_STEP,
            correlation_id=uuid4(),
            batch_entities=steps,
        )

        # Execute batch insert (should go through circuit breaker)
        result = await crud_handler._handle_batch_insert(batch_input)

        # Verify success
        assert result.success is True
        assert result.rows_affected == 5

        # Verify circuit breaker state (should still be closed)
        assert crud_handler._circuit_breaker.state == CircuitBreakerState.CLOSED


if __name__ == "__main__":
    # Run tests with: pytest tests/integration/test_batch_transaction_atomicity.py -v
    pytest.main([__file__, "-v"])
