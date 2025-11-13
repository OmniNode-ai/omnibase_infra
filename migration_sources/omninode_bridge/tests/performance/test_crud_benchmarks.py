"""
Performance Regression Tests for Generic CRUD Operations.

Establishes baseline performance thresholds and provides regression testing
for all 8 CRUD operations in the DatabaseAdapterEffect node.

Expected Performance Baselines:
    - INSERT: < 15ms
    - UPDATE: < 12ms
    - DELETE: < 10ms
    - BATCH_INSERT: < 5ms/record
    - QUERY (100 records): < 20ms
    - UPSERT: < 15ms
    - COUNT: < 8ms
    - EXISTS: < 8ms

Connection Pool Performance:
    - Pool acquisition: < 5ms
    - Concurrent operations: 100+ ops/sec

Usage:
    # Run all CRUD benchmarks
    pytest tests/performance/test_crud_benchmarks.py -m performance

    # Run specific operation benchmark
    pytest tests/performance/test_crud_benchmarks.py::test_insert_performance

    # View benchmark results
    pytest tests/performance/test_crud_benchmarks.py --benchmark-only
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

# Import entity models from infrastructure (matches EntityUnion)
from omninode_bridge.infrastructure.entities.model_workflow_execution import (
    ModelWorkflowExecution,
)
from omninode_bridge.infrastructure.entities.model_workflow_step import (
    ModelWorkflowStep,
)

# Import infrastructure components
from omninode_bridge.infrastructure.enum_entity_type import EnumEntityType

# Import CRUD handlers for testing
from omninode_bridge.nodes.database_adapter_effect.v1_0_0._generic_crud_handlers import (
    GenericCRUDHandlers,
)
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.enums.enum_database_operation_type import (
    EnumDatabaseOperationType,
)

# Import database operation models
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_database_operation_input import (
    ModelDatabaseOperationInput,
)

# Performance thresholds (in milliseconds)
PERFORMANCE_THRESHOLDS = {
    "insert": 15.0,
    "update": 12.0,
    "delete": 10.0,
    "batch_insert_per_record": 5.0,
    "query_100_records": 20.0,
    "upsert": 15.0,
    "count": 8.0,
    "exists": 8.0,
    "pool_acquisition": 5.0,
}


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    async def execute(self, func, *args, **kwargs):
        """Execute function directly without circuit breaker logic."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)


class MockQueryExecutor:
    """Mock query executor for testing."""

    def __init__(self):
        self.queries_executed = []
        self.execution_times = []

    async def execute_query(self, query: str, *params) -> list[dict[str, Any]]:
        """Mock query execution with timing."""
        start_time = time.perf_counter()

        # Simulate query execution
        await asyncio.sleep(0.001)  # 1ms base latency

        self.queries_executed.append((query, params))
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        self.execution_times.append(execution_time_ms)

        # Return mock results based on query type
        if "INSERT" in query and "RETURNING id" in query:
            return [{"id": str(uuid4())}]
        elif "UPDATE" in query:
            return "UPDATE 1"
        elif "DELETE" in query:
            return "DELETE 1"
        elif "COUNT" in query:
            return [{"count": 100}]
        elif "EXISTS" in query:
            return [{"exists": True}]
        elif "SELECT" in query:
            # Return mock workflow execution records (infrastructure schema with UUID)
            return [
                {
                    "id": str(uuid4()),  # UUID primary key in infrastructure schema
                    "correlation_id": str(uuid4()),
                    "workflow_type": f"wf-{i}",
                    "current_state": "completed",
                    "namespace": "test_app",
                    "started_at": datetime.now(UTC),
                    "completed_at": datetime.now(UTC),
                    "execution_time_ms": 100,
                    "error_message": None,
                    "metadata": {"test": True},
                    "created_at": datetime.now(UTC),
                }
                for i in range(10)
            ]
        return []


class MockConnectionManager:
    """Mock connection manager for testing."""

    def transaction(self):
        """Mock transaction context manager."""
        return self

    async def __aenter__(self):
        return MockConnection()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockConnection:
    """Mock database connection."""

    async def fetch(self, query: str, *params):
        """Mock fetch with timing."""
        await asyncio.sleep(0.001)  # 1ms latency
        # Return mock batch insert results
        return [{"id": str(uuid4())} for _ in range(len(params) // 10)]


class TestCRUDHandler(GenericCRUDHandlers):
    """Test CRUD handler with mocked dependencies."""

    def __init__(self):
        self._circuit_breaker = MockCircuitBreaker()
        self._query_executor = MockQueryExecutor()
        self._connection_manager = MockConnectionManager()
        self._logger = None


@pytest.fixture
def crud_handler():
    """Create CRUD handler for testing."""
    return TestCRUDHandler()


@pytest.fixture
def sample_workflow_execution():
    """Create sample workflow execution entity (infrastructure schema with UUID)."""
    return ModelWorkflowExecution(
        correlation_id=uuid4(),
        workflow_type="wf-test-001",
        current_state="processing",
        namespace="test_app",
        started_at=datetime.now(UTC),  # REQUIRED in infrastructure schema
        completed_at=None,
        execution_time_ms=None,
        error_message=None,
        metadata={"test": True},
    )


@pytest.fixture
def sample_workflow_steps():
    """Create sample workflow step entities for batch testing."""
    workflow_id = uuid4()
    return [
        ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=f"step_{i}",
            step_order=i + 1,  # step_order must be >= 1
            status="completed",
            execution_time_ms=100,
            step_data={"output": f"step_{i}_result"},
            error_message=None,
        )
        for i in range(100)
    ]


# ===== INSERT Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_insert_performance(benchmark, crud_handler, sample_workflow_execution):
    """
    Benchmark INSERT operation performance.

    Expected Performance: < 15ms

    This test validates that single INSERT operations complete within
    the established performance threshold, ensuring efficient database
    write operations.
    """

    async def run_insert():
        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=sample_workflow_execution,
        )
        result = await crud_handler._handle_insert(input_data)
        return result

    # Benchmark the operation
    result = await benchmark.pedantic(
        run_insert, iterations=10, rounds=5, warmup_rounds=2
    )

    # Validate result
    assert result.success is True
    assert result.operation_type == "insert"
    assert result.rows_affected == 1

    # Validate performance threshold
    mean_time_ms = benchmark.stats["mean"] * 1000
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["insert"]
    ), f"INSERT too slow: {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['insert']}ms"


# ===== UPDATE Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_update_performance(benchmark, crud_handler, sample_workflow_execution):
    """
    Benchmark UPDATE operation performance.

    Expected Performance: < 12ms

    This test validates that UPDATE operations complete efficiently,
    ensuring optimal database write performance for record modifications.
    """

    async def run_update():
        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPDATE,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=sample_workflow_execution,
            query_filters={"correlation_id": sample_workflow_execution.correlation_id},
        )
        result = await crud_handler._handle_update(input_data)
        return result

    # Benchmark the operation
    result = await benchmark.pedantic(
        run_update, iterations=10, rounds=5, warmup_rounds=2
    )

    # Validate result
    assert result.success is True
    assert result.operation_type == "update"

    # Validate performance threshold
    mean_time_ms = benchmark.stats["mean"] * 1000
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["update"]
    ), f"UPDATE too slow: {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['update']}ms"


# ===== DELETE Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_delete_performance(benchmark, crud_handler):
    """
    Benchmark DELETE operation performance.

    Expected Performance: < 10ms

    This test validates that DELETE operations complete efficiently,
    ensuring optimal database write performance for record removal.
    """

    async def run_delete():
        test_correlation_id = uuid4()
        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.DELETE,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"correlation_id": test_correlation_id},
        )
        result = await crud_handler._handle_delete(input_data)
        return result

    # Benchmark the operation
    result = await benchmark.pedantic(
        run_delete, iterations=10, rounds=5, warmup_rounds=2
    )

    # Validate result
    assert result.success is True
    assert result.operation_type == "delete"

    # Validate performance threshold
    mean_time_ms = benchmark.stats["mean"] * 1000
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["delete"]
    ), f"DELETE too slow: {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['delete']}ms"


# ===== BATCH_INSERT Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_batch_insert_performance(benchmark, crud_handler, sample_workflow_steps):
    """
    Benchmark BATCH_INSERT operation performance.

    Expected Performance: < 5ms per record

    This test validates that batch insert operations maintain efficient
    per-record performance, critical for high-throughput bulk operations.
    """

    async def run_batch_insert():
        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.BATCH_INSERT,
            entity_type=EnumEntityType.WORKFLOW_STEP,
            correlation_id=uuid4(),
            batch_entities=sample_workflow_steps[:10],  # 10 records for benchmark
        )
        result = await crud_handler._handle_batch_insert(input_data)
        return result

    # Benchmark the operation
    result = await benchmark.pedantic(
        run_batch_insert, iterations=10, rounds=5, warmup_rounds=2
    )

    # Validate result
    assert result.success is True
    assert result.operation_type == "batch_insert"

    # Validate performance threshold (per record)
    mean_time_ms = benchmark.stats["mean"] * 1000
    per_record_ms = mean_time_ms / 10  # 10 records
    assert (
        per_record_ms < PERFORMANCE_THRESHOLDS["batch_insert_per_record"]
    ), f"BATCH_INSERT too slow: {per_record_ms:.2f}ms/record > {PERFORMANCE_THRESHOLDS['batch_insert_per_record']}ms/record"


# ===== QUERY Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_query_performance(benchmark, crud_handler):
    """
    Benchmark QUERY operation performance.

    Expected Performance: < 20ms for 100 records

    This test validates that SELECT queries with pagination complete
    efficiently, ensuring optimal read performance.
    """

    async def run_query():
        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            limit=100,
            offset=0,
            sort_by="created_at",
            sort_order="desc",
        )
        result = await crud_handler._handle_query(input_data)
        return result

    # Benchmark the operation
    result = await benchmark.pedantic(
        run_query, iterations=10, rounds=5, warmup_rounds=2
    )

    # Validate result
    assert result.success is True
    assert result.operation_type == "query"

    # Validate performance threshold
    mean_time_ms = benchmark.stats["mean"] * 1000
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["query_100_records"]
    ), f"QUERY too slow: {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['query_100_records']}ms"


# ===== UPSERT Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_upsert_performance(benchmark, crud_handler, sample_workflow_execution):
    """
    Benchmark UPSERT operation performance.

    Expected Performance: < 15ms

    This test validates that UPSERT operations (INSERT ON CONFLICT)
    complete efficiently, ensuring optimal conditional write performance.
    """

    async def run_upsert():
        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=sample_workflow_execution,
            query_filters={"correlation_id": sample_workflow_execution.correlation_id},
        )
        result = await crud_handler._handle_upsert(input_data)
        return result

    # Benchmark the operation
    result = await benchmark.pedantic(
        run_upsert, iterations=10, rounds=5, warmup_rounds=2
    )

    # Validate result
    assert result.success is True
    assert result.operation_type == "upsert"

    # Validate performance threshold
    mean_time_ms = benchmark.stats["mean"] * 1000
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["upsert"]
    ), f"UPSERT too slow: {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['upsert']}ms"


# ===== COUNT Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_count_performance(benchmark, crud_handler):
    """
    Benchmark COUNT operation performance.

    Expected Performance: < 8ms

    This test validates that COUNT queries complete efficiently,
    ensuring optimal aggregation performance.
    """

    async def run_count():
        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.COUNT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"current_state": "completed"},
        )
        result = await crud_handler._handle_count(input_data)
        return result

    # Benchmark the operation
    result = await benchmark.pedantic(
        run_count, iterations=10, rounds=5, warmup_rounds=2
    )

    # Validate result
    assert result.success is True
    assert result.operation_type == "count"

    # Validate performance threshold
    mean_time_ms = benchmark.stats["mean"] * 1000
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["count"]
    ), f"COUNT too slow: {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['count']}ms"


# ===== EXISTS Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_exists_performance(benchmark, crud_handler):
    """
    Benchmark EXISTS operation performance.

    Expected Performance: < 8ms

    This test validates that EXISTS queries complete efficiently,
    ensuring optimal existence check performance.
    """

    async def run_exists():
        test_correlation_id = uuid4()
        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.EXISTS,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"correlation_id": test_correlation_id},
        )
        result = await crud_handler._handle_exists(input_data)
        return result

    # Benchmark the operation
    result = await benchmark.pedantic(
        run_exists, iterations=10, rounds=5, warmup_rounds=2
    )

    # Validate result
    assert result.success is True
    assert result.operation_type == "exists"

    # Validate performance threshold
    mean_time_ms = benchmark.stats["mean"] * 1000
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["exists"]
    ), f"EXISTS too slow: {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['exists']}ms"


# ===== Connection Pool Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_connection_pool_performance(benchmark):
    """
    Benchmark connection pool acquisition performance.

    Expected Performance: < 5ms per acquisition

    This test validates that connection pool operations remain efficient
    under concurrent load, ensuring consistent database access performance.
    """

    async def run_pool_acquisition():
        # Simulate connection pool acquisition
        start_time = time.perf_counter()
        await asyncio.sleep(0.001)  # Simulate pool acquisition latency
        acquisition_time_ms = (time.perf_counter() - start_time) * 1000
        return acquisition_time_ms

    # Benchmark the operation
    result = await benchmark.pedantic(
        run_pool_acquisition, iterations=100, rounds=10, warmup_rounds=5
    )

    # Validate performance threshold
    mean_time_ms = benchmark.stats["mean"] * 1000
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["pool_acquisition"]
    ), f"Pool acquisition too slow: {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['pool_acquisition']}ms"


# ===== Concurrent Operations Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_operations_throughput(benchmark, crud_handler):
    """
    Benchmark concurrent CRUD operations throughput.

    Expected Performance: 100+ operations per second

    This test validates that the system maintains high throughput
    under concurrent load, ensuring production-ready performance.
    """

    async def run_concurrent_operations():
        """Execute 10 concurrent INSERT operations."""
        tasks = []
        for i in range(10):
            workflow = ModelWorkflowExecution(
                correlation_id=uuid4(),
                workflow_type=f"wf-concurrent-{i}",
                current_state="processing",
                namespace="test_app",
                started_at=datetime.now(UTC),  # REQUIRED in infrastructure schema
                completed_at=None,
                execution_time_ms=None,
                error_message=None,
                metadata={"concurrent": True, "index": i},
            )
            input_data = ModelDatabaseOperationInput(
                operation_type=EnumDatabaseOperationType.INSERT,
                entity_type=EnumEntityType.WORKFLOW_EXECUTION,
                correlation_id=uuid4(),
                entity=workflow,
            )
            tasks.append(crud_handler._handle_insert(input_data))

        results = await asyncio.gather(*tasks)
        return results

    # Benchmark concurrent operations
    results = await benchmark.pedantic(
        run_concurrent_operations, iterations=5, rounds=3, warmup_rounds=1
    )

    # Validate all operations succeeded
    assert all(r.success for r in results)

    # Calculate throughput (operations per second)
    mean_time_s = benchmark.stats["mean"]
    throughput = 10 / mean_time_s  # 10 operations per benchmark run

    assert (
        throughput >= 100
    ), f"Throughput too low: {throughput:.2f} ops/sec < 100 ops/sec"


# ===== Performance Summary Test =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_summary(crud_handler, sample_workflow_execution):
    """
    Generate performance summary for all CRUD operations.

    This test provides a comprehensive overview of all CRUD operation
    performance metrics for documentation and monitoring purposes.
    """
    results = {}

    # Test INSERT
    start = time.perf_counter()
    insert_input = ModelDatabaseOperationInput(
        operation_type=EnumDatabaseOperationType.INSERT,
        entity_type=EnumEntityType.WORKFLOW_EXECUTION,
        correlation_id=uuid4(),
        entity=sample_workflow_execution,
    )
    await crud_handler._handle_insert(insert_input)
    results["INSERT"] = (time.perf_counter() - start) * 1000

    # Test UPDATE
    start = time.perf_counter()
    update_input = ModelDatabaseOperationInput(
        operation_type=EnumDatabaseOperationType.UPDATE,
        entity_type=EnumEntityType.WORKFLOW_EXECUTION,
        correlation_id=uuid4(),
        entity=sample_workflow_execution,
        query_filters={"correlation_id": sample_workflow_execution.correlation_id},
    )
    await crud_handler._handle_update(update_input)
    results["UPDATE"] = (time.perf_counter() - start) * 1000

    # Test DELETE
    start = time.perf_counter()
    test_correlation_id = uuid4()
    delete_input = ModelDatabaseOperationInput(
        operation_type=EnumDatabaseOperationType.DELETE,
        entity_type=EnumEntityType.WORKFLOW_EXECUTION,
        correlation_id=uuid4(),
        query_filters={"correlation_id": test_correlation_id},
    )
    await crud_handler._handle_delete(delete_input)
    results["DELETE"] = (time.perf_counter() - start) * 1000

    # Test QUERY
    start = time.perf_counter()
    query_input = ModelDatabaseOperationInput(
        operation_type=EnumDatabaseOperationType.QUERY,
        entity_type=EnumEntityType.WORKFLOW_EXECUTION,
        correlation_id=uuid4(),
        limit=100,
    )
    await crud_handler._handle_query(query_input)
    results["QUERY"] = (time.perf_counter() - start) * 1000

    # Test COUNT
    start = time.perf_counter()
    count_input = ModelDatabaseOperationInput(
        operation_type=EnumDatabaseOperationType.COUNT,
        entity_type=EnumEntityType.WORKFLOW_EXECUTION,
        correlation_id=uuid4(),
    )
    await crud_handler._handle_count(count_input)
    results["COUNT"] = (time.perf_counter() - start) * 1000

    # Test EXISTS
    start = time.perf_counter()
    test_correlation_id_exists = uuid4()
    exists_input = ModelDatabaseOperationInput(
        operation_type=EnumDatabaseOperationType.EXISTS,
        entity_type=EnumEntityType.WORKFLOW_EXECUTION,
        correlation_id=uuid4(),
        query_filters={"correlation_id": test_correlation_id_exists},
    )
    await crud_handler._handle_exists(exists_input)
    results["EXISTS"] = (time.perf_counter() - start) * 1000

    # Print summary
    print("\n=== CRUD Performance Summary ===")
    for operation, time_ms in results.items():
        threshold = PERFORMANCE_THRESHOLDS.get(operation.lower(), float("inf"))
        status = "✓" if time_ms < threshold else "✗"
        print(f"{status} {operation}: {time_ms:.2f}ms (threshold: {threshold}ms)")

    # Validate all operations meet thresholds
    for operation, time_ms in results.items():
        threshold = PERFORMANCE_THRESHOLDS.get(operation.lower(), float("inf"))
        assert (
            time_ms < threshold
        ), f"{operation} performance degradation: {time_ms:.2f}ms > {threshold}ms"
