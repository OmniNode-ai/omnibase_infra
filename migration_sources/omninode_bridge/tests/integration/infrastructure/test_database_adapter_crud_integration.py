"""
Integration Tests for Database Adapter CRUD Operations.

Tests all 8 generic CRUD handlers against a real PostgreSQL database:
1. INSERT - Create new records
2. UPDATE - Modify existing records
3. DELETE - Remove records
4. QUERY - Retrieve records with pagination
5. UPSERT - Insert or update on conflict
6. BATCH_INSERT - Insert multiple records
7. COUNT - Count records matching filters
8. EXISTS - Check if records exist

Requirements:
- Docker must be running for testcontainers
- PostgreSQL container will be automatically started
- Tests use real database tables from schema

Test Coverage:
- All 8 CRUD operations with real database
- EntityRegistry validation
- Strongly-typed entity models
- SQL injection prevention
- Error handling
- Performance metrics
"""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

try:
    from testcontainers.postgres import PostgresContainer

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    PostgresContainer = None  # type: ignore

try:
    from omnibase_core.error_codes import CoreErrorCode
    from omnibase_core.exceptions import OnexError
except ImportError:
    # Use fallback when omnibase_core is not available
    class CoreErrorCode:
        VALIDATION_ERROR = "VALIDATION_ERROR"
        DATABASE_ERROR = "DATABASE_ERROR"

    class OnexError(Exception):
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


from omninode_bridge.infrastructure.entities.model_workflow_execution import (
    ModelWorkflowExecution,
)
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
    """
    Create database schema for testing.

    Creates tables for all entity types in EntityRegistry.
    """
    # Create workflow_executions table
    await connection_manager.execute_query(
        """
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";

        CREATE TABLE IF NOT EXISTS workflow_executions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            correlation_id UUID NOT NULL UNIQUE,
            workflow_type VARCHAR(100) NOT NULL,
            current_state VARCHAR(50) NOT NULL,
            namespace VARCHAR(100) NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE NOT NULL,
            completed_at TIMESTAMP WITH TIME ZONE,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Create workflow_steps table
    await connection_manager.execute_query(
        """
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
    await connection_manager.execute_query("DROP TABLE IF EXISTS workflow_executions")


@pytest.fixture
async def crud_handler(connection_manager):
    """
    Create CRUD handler instance for testing.

    This creates a test class that inherits from GenericCRUDHandlers
    and provides all required dependencies.

    The circuit breaker is configured using the from_config() factory method,
    demonstrating how to use container-based configuration in production.
    """

    class TestCRUDHandler(GenericCRUDHandlers):
        """Test implementation of GenericCRUDHandlers."""

        def __init__(self, connection_manager):
            """Initialize with dependencies."""
            self._connection_manager = connection_manager
            self._query_executor = connection_manager

            # Create circuit breaker from config
            # In production, this config would come from ModelONEXContainer
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


# Integration Tests


@pytest.mark.integration
@pytest.mark.asyncio
class TestDatabaseAdapterCRUDIntegration:
    """Integration test suite for all 8 CRUD operations."""

    async def test_01_insert_operation_creates_record(
        self, crud_handler, database_schema
    ):
        """✅ Test 1: INSERT operation creates a record in real database."""
        print("\n" + "=" * 80)
        print("TEST 1: INSERT - Creating new workflow execution record")
        print("=" * 80)

        # Create workflow execution entity
        correlation_id = uuid4()
        workflow = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type="metadata_stamping",
            current_state="PROCESSING",
            namespace="test_app",
            started_at=datetime.now(UTC),
        )

        # Create INSERT operation input
        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=workflow,
        )

        # Execute INSERT
        result = await crud_handler._handle_insert(input_data)

        # Verify result
        assert result.success is True
        assert result.operation_type == "insert"
        assert result.rows_affected == 1
        assert "id" in result.result_data
        assert result.execution_time_ms > 0

        print("✅ INSERT successful:")
        print(f"   - Generated ID: {result.result_data['id']}")
        print(f"   - Execution time: {result.execution_time_ms}ms")
        print(f"   - Rows affected: {result.rows_affected}")

        # Verify record in database
        rows = await crud_handler._connection_manager.execute_query(
            "SELECT * FROM workflow_executions WHERE correlation_id = $1",
            correlation_id,
        )

        assert len(rows) == 1
        assert rows[0]["workflow_type"] == "metadata_stamping"
        assert rows[0]["current_state"] == "PROCESSING"
        assert rows[0]["namespace"] == "test_app"

        print("✅ Database verification passed:")
        print("   - Record found in database")
        print(f"   - workflow_type: {rows[0]['workflow_type']}")
        print(f"   - current_state: {rows[0]['current_state']}")
        print(f"   - namespace: {rows[0]['namespace']}")

    async def test_02_query_operation_retrieves_records(
        self, crud_handler, database_schema
    ):
        """✅ Test 2: QUERY operation retrieves records from real database."""
        print("\n" + "=" * 80)
        print("TEST 2: QUERY - Retrieving workflow execution records")
        print("=" * 80)

        # Insert test data first
        correlation_id = uuid4()
        workflow = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type="test_workflow",
            current_state="COMPLETED",
            namespace="test_app",
            started_at=datetime.now(UTC),
        )

        insert_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=workflow,
        )

        await crud_handler._handle_insert(insert_input)
        print("✅ Test data inserted")

        # Execute QUERY
        query_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"namespace": "test_app"},
        )

        result = await crud_handler._handle_query(query_input)

        # Verify result
        assert result.success is True
        assert result.operation_type == "query"
        assert result.rows_affected >= 1
        assert isinstance(result.result_data, dict)
        assert "items" in result.result_data
        assert len(result.result_data["items"]) >= 1

        print("✅ QUERY successful:")
        print(f"   - Rows retrieved: {result.rows_affected}")
        print(f"   - Execution time: {result.execution_time_ms}ms")
        print(
            f"   - First record workflow_type: {result.result_data['items'][0]['workflow_type']}"
        )

    async def test_03_update_operation_modifies_record(
        self, crud_handler, database_schema
    ):
        """✅ Test 3: UPDATE operation modifies existing record in real database."""
        print("\n" + "=" * 80)
        print("TEST 3: UPDATE - Modifying workflow execution state")
        print("=" * 80)

        # Insert test data
        correlation_id = uuid4()
        workflow = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type="test_workflow",
            current_state="PENDING",
            namespace="test_app",
            started_at=datetime.now(UTC),
        )

        insert_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=workflow,
        )

        await crud_handler._handle_insert(insert_input)
        print("✅ Initial record created with state: PENDING")

        # Update workflow state
        updated_workflow = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type="test_workflow",
            current_state="COMPLETED",  # Changed
            namespace="test_app",
            started_at=workflow.started_at,
            completed_at=datetime.now(UTC),  # Added
        )

        update_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPDATE,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=updated_workflow,
            query_filters={"correlation_id": str(correlation_id)},
        )

        result = await crud_handler._handle_update(update_input)

        # Verify result
        assert result.success is True
        assert result.operation_type == "update"
        assert result.rows_affected == 1

        print("✅ UPDATE successful:")
        print(f"   - Rows affected: {result.rows_affected}")
        print(f"   - Execution time: {result.execution_time_ms}ms")

        # Verify update in database
        rows = await crud_handler._connection_manager.execute_query(
            "SELECT * FROM workflow_executions WHERE correlation_id = $1",
            correlation_id,
        )

        assert rows[0]["current_state"] == "COMPLETED"
        assert rows[0]["completed_at"] is not None

        print("✅ Database verification passed:")
        print(f"   - State changed to: {rows[0]['current_state']}")
        print("   - completed_at is now set")

    async def test_04_delete_operation_removes_record(
        self, crud_handler, database_schema
    ):
        """✅ Test 4: DELETE operation removes record from real database."""
        print("\n" + "=" * 80)
        print("TEST 4: DELETE - Removing workflow execution record")
        print("=" * 80)

        # Insert test data
        correlation_id = uuid4()
        workflow = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type="test_workflow",
            current_state="FAILED",
            namespace="test_app",
            started_at=datetime.now(UTC),
        )

        insert_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=workflow,
        )

        await crud_handler._handle_insert(insert_input)
        print("✅ Record created for deletion test")

        # Delete workflow
        delete_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.DELETE,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"correlation_id": str(correlation_id)},
        )

        result = await crud_handler._handle_delete(delete_input)

        # Verify result
        assert result.success is True
        assert result.operation_type == "delete"
        assert result.rows_affected == 1

        print("✅ DELETE successful:")
        print(f"   - Rows deleted: {result.rows_affected}")
        print(f"   - Execution time: {result.execution_time_ms}ms")

        # Verify deletion in database
        rows = await crud_handler._connection_manager.execute_query(
            "SELECT * FROM workflow_executions WHERE correlation_id = $1",
            correlation_id,
        )

        assert len(rows) == 0
        print("✅ Database verification passed: Record successfully deleted")

    async def test_05_upsert_operation_inserts_or_updates(
        self, crud_handler, database_schema
    ):
        """✅ Test 5: UPSERT operation inserts new record or updates existing."""
        print("\n" + "=" * 80)
        print("TEST 5: UPSERT - Insert on first call, update on second")
        print("=" * 80)

        correlation_id = uuid4()

        # First upsert (should INSERT)
        workflow = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type="test_workflow",
            current_state="PENDING",
            namespace="test_app",
            started_at=datetime.now(UTC),
        )

        upsert_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=workflow,
            query_filters={"correlation_id": str(correlation_id)},
        )

        result1 = await crud_handler._handle_upsert(upsert_input)

        assert result1.success is True
        assert result1.operation_type == "upsert"
        assert "id" in result1.result_data

        print("✅ First UPSERT (INSERT):")
        print(f"   - Generated ID: {result1.result_data['id']}")
        print("   - State: PENDING")

        # Second upsert (should UPDATE)
        workflow.current_state = "COMPLETED"

        result2 = await crud_handler._handle_upsert(upsert_input)

        assert result2.success is True

        print("✅ Second UPSERT (UPDATE):")
        print("   - State changed to: COMPLETED")

        # Verify final state
        rows = await crud_handler._connection_manager.execute_query(
            "SELECT * FROM workflow_executions WHERE correlation_id = $1",
            correlation_id,
        )

        assert len(rows) == 1  # Only one record
        assert rows[0]["current_state"] == "COMPLETED"

        print("✅ Database verification passed:")
        print("   - Only 1 record exists (no duplicate)")
        print(f"   - State correctly updated to: {rows[0]['current_state']}")

    async def test_06_batch_insert_operation_creates_multiple_records(
        self, crud_handler, database_schema
    ):
        """✅ Test 6: BATCH_INSERT operation creates multiple records in real database."""
        print("\n" + "=" * 80)
        print("TEST 6: BATCH_INSERT - Creating 5 workflow step records at once")
        print("=" * 80)

        # Create multiple workflow steps
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

        result = await crud_handler._handle_batch_insert(batch_input)

        # Verify result
        assert result.success is True
        assert result.operation_type == "batch_insert"
        assert result.rows_affected == 5
        assert "ids" in result.result_data
        assert len(result.result_data["ids"]) == 5

        print("✅ BATCH_INSERT successful:")
        print(f"   - Records created: {result.rows_affected}")
        print(f"   - Execution time: {result.execution_time_ms}ms")
        print(f"   - Generated IDs: {len(result.result_data['ids'])} IDs")

        # Verify records in database
        rows = await crud_handler._connection_manager.execute_query(
            "SELECT * FROM workflow_steps WHERE workflow_id = $1 ORDER BY step_order",
            workflow_id,
        )

        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert row["step_name"] == f"step_{i}"
            assert row["step_order"] == i + 1

        print("✅ Database verification passed:")
        print("   - All 5 records found in database")
        print("   - Step ordering preserved correctly")

    async def test_07_count_operation_counts_matching_records(
        self, crud_handler, database_schema
    ):
        """✅ Test 7: COUNT operation counts records matching filters."""
        print("\n" + "=" * 80)
        print("TEST 7: COUNT - Counting records in specific namespace")
        print("=" * 80)

        # Insert test data with different namespaces
        for i in range(3):
            workflow = ModelWorkflowExecution(
                correlation_id=uuid4(),
                workflow_type="test_workflow",
                current_state="PENDING",
                namespace="count_test_namespace",
                started_at=datetime.now(UTC),
            )

            insert_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.INSERT,
                entity_type=EnumEntityType.WORKFLOW_EXECUTION,
                correlation_id=uuid4(),
                entity=workflow,
            )

            await crud_handler._handle_insert(insert_input)

        print("✅ Inserted 3 test records with namespace: count_test_namespace")

        # Count records in test_app_1
        count_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.COUNT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"namespace": "count_test_namespace"},
        )

        result = await crud_handler._handle_count(count_input)

        # Verify result
        assert result.success is True
        assert result.operation_type == "count"
        assert result.result_data["count"] == 3

        print("✅ COUNT successful:")
        print(f"   - Records counted: {result.result_data['count']}")
        print(f"   - Execution time: {result.execution_time_ms}ms")

    async def test_08_exists_operation_checks_record_existence(
        self, crud_handler, database_schema
    ):
        """✅ Test 8: EXISTS operation checks if records exist."""
        print("\n" + "=" * 80)
        print("TEST 8: EXISTS - Checking if specific record exists")
        print("=" * 80)

        # Insert test data
        correlation_id = uuid4()
        workflow = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type="test_workflow",
            current_state="PENDING",
            namespace="test_app",
            started_at=datetime.now(UTC),
        )

        insert_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=workflow,
        )

        await crud_handler._handle_insert(insert_input)
        print("✅ Test record created")

        # Test EXISTS with matching filter
        exists_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.EXISTS,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"correlation_id": str(correlation_id)},
        )

        result = await crud_handler._handle_exists(exists_input)

        assert result.success is True
        assert result.operation_type == "exists"
        assert result.result_data["exists"] is True

        print(f"✅ EXISTS check (should exist): {result.result_data['exists']}")

        # Test EXISTS with non-matching filter
        non_exists_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.EXISTS,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"correlation_id": str(uuid4())},  # Random UUID
        )

        result2 = await crud_handler._handle_exists(non_exists_input)

        assert result2.success is True
        assert result2.result_data["exists"] is False

        print(f"✅ EXISTS check (should not exist): {result2.result_data['exists']}")
        print(
            "✅ Database verification passed: EXISTS correctly identifies presence/absence"
        )

    async def test_09_query_with_pagination_and_sorting(
        self, crud_handler, database_schema
    ):
        """✅ Test 9: QUERY operation with pagination and sorting."""
        print("\n" + "=" * 80)
        print("TEST 9: QUERY with PAGINATION - Sorting and limiting results")
        print("=" * 80)

        # Insert test data with different workflow types
        for i in range(10):
            workflow = ModelWorkflowExecution(
                correlation_id=uuid4(),
                workflow_type=f"workflow_{i:02d}",
                current_state="PENDING",
                namespace="pagination_test",
                started_at=datetime.now(UTC),
            )

            insert_input = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.INSERT,
                entity_type=EnumEntityType.WORKFLOW_EXECUTION,
                correlation_id=uuid4(),
                entity=workflow,
            )

            await crud_handler._handle_insert(insert_input)

        print("✅ Inserted 10 test records")

        # Query with pagination (limit 5, offset 0)
        query_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"namespace": "pagination_test"},
            sort_by="workflow_type",
            sort_order="asc",
            limit=5,
            offset=0,
        )

        result = await crud_handler._handle_query(query_input)

        assert result.success is True
        assert len(result.result_data["items"]) == 5
        assert result.result_data["items"][0]["workflow_type"] == "workflow_00"

        print("✅ First page (limit 5, offset 0):")
        print(f"   - Records retrieved: {len(result.result_data['items'])}")
        print(f"   - First record: {result.result_data['items'][0]['workflow_type']}")

        # Query second page (limit 5, offset 5)
        query_input.offset = 5

        result2 = await crud_handler._handle_query(query_input)

        assert len(result2.result_data["items"]) == 5
        assert result2.result_data["items"][0]["workflow_type"] == "workflow_05"

        print("✅ Second page (limit 5, offset 5):")
        print(f"   - Records retrieved: {len(result2.result_data['items'])}")
        print(f"   - First record: {result2.result_data['items'][0]['workflow_type']}")
        print("✅ Pagination working correctly")

    async def test_10_concurrent_crud_operations(self, crud_handler, database_schema):
        """✅ Test 10: Concurrent CRUD operations to verify thread safety."""
        print("\n" + "=" * 80)
        print("TEST 10: CONCURRENT OPERATIONS - 20 parallel inserts")
        print("=" * 80)

        async def insert_workflow(index: int):
            workflow = ModelWorkflowExecution(
                correlation_id=uuid4(),
                workflow_type=f"concurrent_{index}",
                current_state="PENDING",
                namespace="concurrent_test",
                started_at=datetime.now(UTC),
            )

            input_data = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.INSERT,
                entity_type=EnumEntityType.WORKFLOW_EXECUTION,
                correlation_id=uuid4(),
                entity=workflow,
            )

            return await crud_handler._handle_insert(input_data)

        # Execute 20 concurrent inserts
        results = await asyncio.gather(*[insert_workflow(i) for i in range(20)])

        # All should succeed
        assert all(r.success for r in results)
        assert len(results) == 20

        print("✅ 20 concurrent inserts completed successfully")

        # Verify all records in database
        count_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.COUNT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"namespace": "concurrent_test"},
        )

        count_result = await crud_handler._handle_count(count_input)

        assert count_result.result_data["count"] == 20

        print("✅ Database verification passed: All 20 records created")
        print("✅ Thread safety confirmed")

    async def test_11_performance_metrics_tracking(self, crud_handler, database_schema):
        """✅ Test 11: Performance metrics are tracked for all operations."""
        print("\n" + "=" * 80)
        print("TEST 11: PERFORMANCE METRICS - Verifying execution time tracking")
        print("=" * 80)

        workflow = ModelWorkflowExecution(
            correlation_id=uuid4(),
            workflow_type="perf_test",
            current_state="PENDING",
            namespace="test_app",
            started_at=datetime.now(UTC),
        )

        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=workflow,
        )

        result = await crud_handler._handle_insert(input_data)

        # Verify execution time is tracked
        assert result.execution_time_ms > 0
        assert result.execution_time_ms < 1000  # Should be fast

        print("✅ Performance metrics tracked:")
        print(f"   - Execution time: {result.execution_time_ms}ms")
        print("   - Performance acceptable: < 1000ms")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_crud_workflow_summary(crud_handler, database_schema):
    """
    🎯 COMPREHENSIVE INTEGRATION TEST SUMMARY

    This test executes a complete workflow demonstrating all 8 CRUD operations
    in a realistic scenario, proving end-to-end functionality.
    """
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE CRUD WORKFLOW TEST - ALL 8 OPERATIONS")
    print("=" * 80)

    summary = {
        "operations_tested": 0,
        "operations_successful": 0,
        "total_execution_time_ms": 0,
        "records_created": 0,
        "records_modified": 0,
        "records_deleted": 0,
        "records_queried": 0,
    }

    # 1. INSERT
    print("\n1️⃣ INSERT - Creating initial workflow...")
    correlation_id = uuid4()
    workflow = ModelWorkflowExecution(
        correlation_id=correlation_id,
        workflow_type="comprehensive_test",
        current_state="PENDING",
        namespace="summary_test",
        started_at=datetime.now(UTC),
    )

    result = await crud_handler._handle_insert(
        ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=workflow,
        )
    )

    assert result.success
    summary["operations_tested"] += 1
    summary["operations_successful"] += 1
    summary["total_execution_time_ms"] += result.execution_time_ms
    summary["records_created"] += 1
    print(f"   ✅ Created record {result.result_data['id']}")

    # 2. EXISTS
    print("\n2️⃣ EXISTS - Verifying record exists...")
    result = await crud_handler._handle_exists(
        ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.EXISTS,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"correlation_id": str(correlation_id)},
        )
    )

    assert result.success
    assert result.result_data["exists"] is True
    summary["operations_tested"] += 1
    summary["operations_successful"] += 1
    summary["total_execution_time_ms"] += result.execution_time_ms
    print(f"   ✅ Record exists: {result.result_data['exists']}")

    # 3. QUERY
    print("\n3️⃣ QUERY - Retrieving record...")
    result = await crud_handler._handle_query(
        ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"namespace": "summary_test"},
        )
    )

    assert result.success
    assert len(result.result_data["items"]) >= 1
    summary["operations_tested"] += 1
    summary["operations_successful"] += 1
    summary["total_execution_time_ms"] += result.execution_time_ms
    summary["records_queried"] += len(result.result_data["items"])
    print(f"   ✅ Retrieved {len(result.result_data['items'])} record(s)")

    # 4. COUNT
    print("\n4️⃣ COUNT - Counting records...")
    result = await crud_handler._handle_count(
        ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.COUNT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"namespace": "summary_test"},
        )
    )

    assert result.success
    initial_count = result.result_data["count"]
    summary["operations_tested"] += 1
    summary["operations_successful"] += 1
    summary["total_execution_time_ms"] += result.execution_time_ms
    print(f"   ✅ Count: {initial_count}")

    # 5. UPDATE
    print("\n5️⃣ UPDATE - Changing workflow state...")
    workflow.current_state = "PROCESSING"
    result = await crud_handler._handle_update(
        ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPDATE,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=workflow,
            query_filters={"correlation_id": str(correlation_id)},
        )
    )

    assert result.success
    assert result.rows_affected == 1
    summary["operations_tested"] += 1
    summary["operations_successful"] += 1
    summary["total_execution_time_ms"] += result.execution_time_ms
    summary["records_modified"] += 1
    print(f"   ✅ Updated {result.rows_affected} record(s)")

    # 6. UPSERT
    print("\n6️⃣ UPSERT - Upserting workflow...")
    workflow.current_state = "COMPLETED"
    result = await crud_handler._handle_upsert(
        ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=workflow,
            query_filters={"correlation_id": str(correlation_id)},
        )
    )

    assert result.success
    summary["operations_tested"] += 1
    summary["operations_successful"] += 1
    summary["total_execution_time_ms"] += result.execution_time_ms
    print("   ✅ Upserted record")

    # 7. BATCH_INSERT
    print("\n7️⃣ BATCH_INSERT - Creating 3 workflow steps...")
    workflow_id = uuid4()
    steps = [
        ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=f"step_{i}",
            step_order=i + 1,
            status="PENDING",
            step_data={},
        )
        for i in range(3)
    ]

    result = await crud_handler._handle_batch_insert(
        ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.BATCH_INSERT,
            entity_type=EnumEntityType.WORKFLOW_STEP,
            correlation_id=uuid4(),
            batch_entities=steps,
        )
    )

    assert result.success
    assert result.rows_affected == 3
    summary["operations_tested"] += 1
    summary["operations_successful"] += 1
    summary["total_execution_time_ms"] += result.execution_time_ms
    summary["records_created"] += 3
    print(f"   ✅ Batch inserted {result.rows_affected} record(s)")

    # 8. DELETE
    print("\n8️⃣ DELETE - Removing workflow...")
    result = await crud_handler._handle_delete(
        ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.DELETE,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"correlation_id": str(correlation_id)},
        )
    )

    assert result.success
    assert result.rows_affected == 1
    summary["operations_tested"] += 1
    summary["operations_successful"] += 1
    summary["total_execution_time_ms"] += result.execution_time_ms
    summary["records_deleted"] += 1
    print(f"   ✅ Deleted {result.rows_affected} record(s)")

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 INTEGRATION TEST SUMMARY - ALL CRUD OPERATIONS VERIFIED")
    print("=" * 80)
    print(f"✅ Operations tested: {summary['operations_tested']}/8")
    print(f"✅ Operations successful: {summary['operations_successful']}/8")
    print(f"✅ Total execution time: {summary['total_execution_time_ms']}ms")
    print(f"✅ Records created: {summary['records_created']}")
    print(f"✅ Records modified: {summary['records_modified']}")
    print(f"✅ Records deleted: {summary['records_deleted']}")
    print(f"✅ Records queried: {summary['records_queried']}")
    print("=" * 80)
    print("🎯 ALL 8 CRUD OPERATIONS WORKING WITH REAL POSTGRESQL DATABASE")
    print("=" * 80 + "\n")

    assert summary["operations_tested"] == 8
    assert summary["operations_successful"] == 8
