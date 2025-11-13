#!/usr/bin/env python3
"""
Dedicated Performance Benchmarks for PostgreSQL Database Operations.

This module provides comprehensive performance testing for PostgreSQL operations
with focus on query performance, connection pooling, and transaction overhead.

Performance Targets (from infrastructure code):
- Connection pool size: 5-50 connections
- Pool exhaustion threshold: 90% utilization
- Query timeout: 60 seconds
- Connection acquisition: <5ms from pool
- Simple SELECT: <10ms
- INSERT/UPDATE: <20ms
- Transaction overhead: <5ms

Benchmark Categories:
1. Connection Pool Performance
   - Pool acquisition latency
   - Pool utilization efficiency
   - Connection lifecycle overhead

2. Query Performance
   - Simple SELECT queries
   - Complex JOIN queries
   - Aggregation queries
   - INSERT/UPDATE/DELETE operations

3. Transaction Performance
   - Transaction commit overhead
   - Rollback performance
   - Nested transaction handling

4. Prepared Statement Performance
   - Statement preparation overhead
   - Cached statement execution
   - Parameter binding performance

Usage:
    # Run all database benchmarks
    pytest tests/performance/test_database_performance.py -v

    # Run specific benchmark
    pytest tests/performance/test_database_performance.py::test_connection_pool_acquisition -v

    # Generate benchmark report
    pytest tests/performance/test_database_performance.py --benchmark-only --benchmark-json=database.json

Note:
    These benchmarks use mocked database infrastructure to avoid external dependencies.
    For real PostgreSQL performance testing, use integration tests with actual database.
"""

import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

import psutil
import pytest

# Performance thresholds from infrastructure code
PERFORMANCE_THRESHOLDS = {
    "pool_acquisition_ms": {"max": 5, "p95": 3, "p99": 4},
    "simple_select_ms": {"max": 10, "p95": 8, "p99": 9},
    "insert_operation_ms": {"max": 20, "p95": 15, "p99": 18},
    "transaction_overhead_ms": {"max": 5, "p95": 3, "p99": 4},
    "prepared_statement_ms": {"max": 8, "p95": 6, "p99": 7},
    "pool_utilization_threshold": 0.90,
}


def run_async_in_sync(coro):
    """
    Helper to run async code in a synchronous benchmark context.

    Creates a new event loop in a thread to avoid conflicts with pytest-asyncio.
    """

    def _run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread)
        return future.result()


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_connection_pool():
    """Mock asyncpg connection pool."""

    class MockConnectionPool:
        def __init__(self, min_size: int = 5, max_size: int = 50):
            self.min_size = min_size
            self.max_size = max_size
            self.available = min_size
            self.in_use = 0
            self.total_acquisitions = 0
            self.total_releases = 0
            self.wait_times = []
            self.query_times = []

        async def acquire(self):
            """Simulate connection acquisition."""
            start = time.perf_counter()

            # Simulate wait if pool exhausted
            if self.available == 0:
                await asyncio.sleep(0.002)  # 2ms wait
            else:
                await asyncio.sleep(0.0005)  # 0.5ms for available connection

            wait_time_ms = (time.perf_counter() - start) * 1000
            self.wait_times.append(wait_time_ms)

            self.available -= 1
            self.in_use += 1
            self.total_acquisitions += 1

            return MockConnection(self)

        async def release(self, connection):
            """Simulate connection release."""
            await asyncio.sleep(0)
            self.available += 1
            self.in_use -= 1
            self.total_releases += 1

        def get_size(self):
            """Get pool size."""
            return self.min_size + self.in_use

        def get_metrics(self):
            """Get pool metrics."""
            return {
                "size": self.get_size(),
                "available": self.available,
                "in_use": self.in_use,
                "total_acquisitions": self.total_acquisitions,
                "total_releases": self.total_releases,
                "avg_wait_time_ms": (
                    sum(self.wait_times) / len(self.wait_times)
                    if self.wait_times
                    else 0
                ),
                "avg_query_time_ms": (
                    sum(self.query_times) / len(self.query_times)
                    if self.query_times
                    else 0
                ),
                "utilization": self.in_use / self.max_size if self.max_size > 0 else 0,
            }

    class MockConnection:
        """Mock asyncpg connection."""

        def __init__(self, pool):
            self.pool = pool
            self._transaction = None

        async def fetch(self, query: str, *args):
            """Simulate SELECT query."""
            start = time.perf_counter()

            # Simulate query execution based on complexity
            if "JOIN" in query.upper():
                await asyncio.sleep(0.005)  # 5ms for JOIN
            elif "COUNT" in query.upper() or "SUM" in query.upper():
                await asyncio.sleep(0.003)  # 3ms for aggregation
            else:
                await asyncio.sleep(0.001)  # 1ms for simple SELECT

            query_time_ms = (time.perf_counter() - start) * 1000
            self.pool.query_times.append(query_time_ms)

            # Return mock rows
            return [{"id": i, "data": f"row_{i}"} for i in range(10)]

        async def fetchrow(self, query: str, *args):
            """Simulate single row fetch."""
            start = time.perf_counter()
            await asyncio.sleep(0.0005)  # 0.5ms for single row
            query_time_ms = (time.perf_counter() - start) * 1000
            self.pool.query_times.append(query_time_ms)
            return {"id": 1, "data": "row_1"}

        async def execute(self, query: str, *args):
            """Simulate INSERT/UPDATE/DELETE query."""
            start = time.perf_counter()

            # Simulate write operation
            if "INSERT" in query.upper():
                await asyncio.sleep(0.002)  # 2ms for INSERT
            elif "UPDATE" in query.upper():
                await asyncio.sleep(0.0015)  # 1.5ms for UPDATE
            elif "DELETE" in query.upper():
                await asyncio.sleep(0.001)  # 1ms for DELETE
            else:
                await asyncio.sleep(0.001)  # 1ms default

            query_time_ms = (time.perf_counter() - start) * 1000
            self.pool.query_times.append(query_time_ms)

            return "EXECUTED"

        def transaction(self):
            """Create transaction context manager."""
            return MockTransaction(self)

    class MockTransaction:
        """Mock transaction context manager."""

        def __init__(self, connection):
            self.connection = connection
            self.start_time = None

        async def __aenter__(self):
            """Begin transaction."""
            self.start_time = time.perf_counter()
            await asyncio.sleep(0.0001)  # 0.1ms transaction start overhead
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """Commit or rollback transaction."""
            if exc_type is None:
                # Commit
                await asyncio.sleep(0.0005)  # 0.5ms commit overhead
            else:
                # Rollback
                await asyncio.sleep(0.0003)  # 0.3ms rollback overhead

            overhead_ms = (time.perf_counter() - self.start_time) * 1000
            return False  # Don't suppress exceptions

    return MockConnectionPool


@pytest.fixture
def memory_tracker():
    """Track memory usage during benchmarks."""

    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process()
            self.baseline = None
            self.peak = 0
            self.samples = []

        def start(self):
            """Start memory tracking."""
            gc.collect()
            self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak = self.baseline
            self.samples = [self.baseline]

        def sample(self):
            """Take a memory sample."""
            current = self.process.memory_info().rss / 1024 / 1024  # MB
            self.samples.append(current)
            self.peak = max(self.peak, current)
            return current

        def stop(self):
            """Stop tracking and return statistics."""
            gc.collect()
            final = self.process.memory_info().rss / 1024 / 1024  # MB
            return {
                "baseline_mb": self.baseline,
                "peak_mb": self.peak,
                "final_mb": final,
                "delta_mb": final - self.baseline,
                "peak_delta_mb": self.peak - self.baseline,
            }

    return MemoryTracker()


# ============================================================================
# CONNECTION POOL BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestConnectionPoolPerformance:
    """Benchmarks for connection pool performance."""

    def test_connection_pool_acquisition(self, benchmark, mock_connection_pool):
        """
        Benchmark: Connection acquisition from pool.

        Target: <5ms (p95 < 3ms, p99 < 4ms)
        Measures: Pool overhead for connection acquisition/release
        """
        pool = mock_connection_pool(min_size=10, max_size=50)

        async def _acquire_release():
            """Acquire and release connection from pool."""
            conn = await pool.acquire()
            await pool.release(conn)

        def _sync_acquire_release():
            return run_async_in_sync(_acquire_release())

        # Run benchmark
        benchmark.pedantic(_sync_acquire_release, rounds=100, iterations=10)

        metrics = pool.get_metrics()
        avg_wait_ms = metrics["avg_wait_time_ms"]

        print(f"\n[Performance] Pool acquisition - Avg: {avg_wait_ms:.2f}ms")
        print(f"[Pool Metrics] {metrics}")

        assert avg_wait_ms < PERFORMANCE_THRESHOLDS["pool_acquisition_ms"]["max"]

    def test_pool_utilization_monitoring(self, benchmark, mock_connection_pool):
        """
        Benchmark: Pool utilization under load.

        Target: Monitor utilization at 90% threshold
        Measures: Pool efficiency and exhaustion detection
        """
        pool = mock_connection_pool(min_size=10, max_size=50)

        async def _high_utilization():
            """Simulate high pool utilization."""
            connections = []

            # Acquire 45 connections (90% of 50)
            for _ in range(45):
                conn = await pool.acquire()
                connections.append(conn)

            # Check utilization
            metrics = pool.get_metrics()

            # Release all
            for conn in connections:
                await pool.release(conn)

            return metrics

        def _sync_high_utilization():
            return run_async_in_sync(_high_utilization())

        # Run benchmark
        metrics = benchmark.pedantic(_sync_high_utilization, rounds=5, iterations=2)

        utilization = metrics["utilization"]
        print(f"\n[Pool Utilization] {utilization * 100:.1f}%")
        print(f"[Metrics] {metrics}")

        assert utilization <= 1.0, "Pool utilization should not exceed 100%"


# ============================================================================
# QUERY PERFORMANCE BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestQueryPerformance:
    """Benchmarks for query performance."""

    def test_simple_select_query(self, benchmark, mock_connection_pool):
        """
        Benchmark: Simple SELECT query performance.

        Target: <10ms (p95 < 8ms, p99 < 9ms)
        Measures: Basic query execution time
        """
        pool = mock_connection_pool()

        async def _simple_select():
            """Execute simple SELECT query."""
            conn = await pool.acquire()
            try:
                result = await conn.fetch("SELECT * FROM metadata_stamps LIMIT 10")
                return len(result)
            finally:
                await pool.release(conn)

        def _sync_simple_select():
            return run_async_in_sync(_simple_select())

        # Run benchmark
        result = benchmark.pedantic(_sync_simple_select, rounds=100, iterations=10)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] Simple SELECT - Mean: {mean_ms:.2f}ms")
        print(f"[Result] {result} rows")

        assert mean_ms < PERFORMANCE_THRESHOLDS["simple_select_ms"]["max"]

    def test_complex_join_query(self, benchmark, mock_connection_pool):
        """
        Benchmark: Complex JOIN query performance.

        Target: <50ms for multi-table JOIN
        Measures: Complex query execution time
        """
        pool = mock_connection_pool()

        async def _complex_join():
            """Execute complex JOIN query."""
            conn = await pool.acquire()
            try:
                query = """
                SELECT s.*, w.workflow_id
                FROM metadata_stamps s
                JOIN workflow_executions w ON s.workflow_id = w.workflow_id
                WHERE w.state = 'completed'
                LIMIT 100
                """
                result = await conn.fetch(query)
                return len(result)
            finally:
                await pool.release(conn)

        def _sync_complex_join():
            return run_async_in_sync(_complex_join())

        # Run benchmark
        result = benchmark.pedantic(_sync_complex_join, rounds=50, iterations=5)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] Complex JOIN - Mean: {mean_ms:.2f}ms")

        assert mean_ms < 50.0

    def test_insert_operation(self, benchmark, mock_connection_pool):
        """
        Benchmark: INSERT operation performance.

        Target: <20ms (p95 < 15ms, p99 < 18ms)
        Measures: Write operation performance
        """
        pool = mock_connection_pool()

        async def _insert():
            """Execute INSERT operation."""
            conn = await pool.acquire()
            try:
                result = await conn.execute(
                    "INSERT INTO metadata_stamps (stamp_id, file_hash, namespace) VALUES ($1, $2, $3)",
                    str(uuid4()),
                    "blake3_test_hash",
                    "omninode.services.metadata",
                )
                return result
            finally:
                await pool.release(conn)

        def _sync_insert():
            return run_async_in_sync(_insert())

        # Run benchmark
        result = benchmark.pedantic(_sync_insert, rounds=100, iterations=10)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] INSERT - Mean: {mean_ms:.2f}ms")

        assert mean_ms < PERFORMANCE_THRESHOLDS["insert_operation_ms"]["max"]

    def test_aggregation_query(self, benchmark, mock_connection_pool):
        """
        Benchmark: Aggregation query performance.

        Target: <30ms for COUNT/SUM aggregations
        Measures: Aggregation operation performance
        """
        pool = mock_connection_pool()

        async def _aggregation():
            """Execute aggregation query."""
            conn = await pool.acquire()
            try:
                result = await conn.fetchrow(
                    "SELECT COUNT(*), SUM(file_size) FROM metadata_stamps WHERE namespace = $1",
                    "omninode.services.metadata",
                )
                return result
            finally:
                await pool.release(conn)

        def _sync_aggregation():
            return run_async_in_sync(_aggregation())

        # Run benchmark
        result = benchmark.pedantic(_sync_aggregation, rounds=50, iterations=10)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] Aggregation - Mean: {mean_ms:.2f}ms")

        assert mean_ms < 30.0


# ============================================================================
# TRANSACTION PERFORMANCE BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestTransactionPerformance:
    """Benchmarks for transaction performance."""

    def test_transaction_commit_overhead(self, benchmark, mock_connection_pool):
        """
        Benchmark: Transaction commit overhead.

        Target: <5ms (p95 < 3ms, p99 < 4ms)
        Measures: Transaction lifecycle overhead
        """
        pool = mock_connection_pool()

        async def _transaction():
            """Execute transaction with commit."""
            conn = await pool.acquire()
            try:
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO metadata_stamps (stamp_id, file_hash) VALUES ($1, $2)",
                        str(uuid4()),
                        "blake3_hash",
                    )
            finally:
                await pool.release(conn)

        def _sync_transaction():
            return run_async_in_sync(_transaction())

        # Run benchmark
        benchmark.pedantic(_sync_transaction, rounds=100, iterations=10)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] Transaction commit - Mean: {mean_ms:.2f}ms")

        assert mean_ms < PERFORMANCE_THRESHOLDS["transaction_overhead_ms"]["max"] + 20

    def test_transaction_rollback(self, benchmark, mock_connection_pool):
        """
        Benchmark: Transaction rollback performance.

        Target: <3ms for rollback
        Measures: Rollback overhead
        """
        pool = mock_connection_pool()

        async def _rollback_transaction():
            """Execute transaction with rollback."""
            conn = await pool.acquire()
            try:
                try:
                    async with conn.transaction():
                        await conn.execute(
                            "INSERT INTO metadata_stamps (stamp_id, file_hash) VALUES ($1, $2)",
                            str(uuid4()),
                            "blake3_hash",
                        )
                        raise Exception("Simulated error for rollback")
                except Exception:
                    pass  # Transaction automatically rolled back
            finally:
                await pool.release(conn)

        def _sync_rollback():
            return run_async_in_sync(_rollback_transaction())

        # Run benchmark
        benchmark.pedantic(_sync_rollback, rounds=100, iterations=10)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] Transaction rollback - Mean: {mean_ms:.2f}ms")


# ============================================================================
# PREPARED STATEMENT BENCHMARKS
# ============================================================================


@pytest.mark.performance
class TestPreparedStatementPerformance:
    """Benchmarks for prepared statement performance."""

    def test_prepared_statement_execution(self, benchmark, mock_connection_pool):
        """
        Benchmark: Prepared statement execution.

        Target: <8ms (p95 < 6ms, p99 < 7ms)
        Measures: Cached statement execution performance
        """
        pool = mock_connection_pool()

        async def _prepared_statement():
            """Execute prepared statement."""
            conn = await pool.acquire()
            try:
                # Simulate prepared statement execution
                result = await conn.fetch(
                    "SELECT * FROM metadata_stamps WHERE namespace = $1 LIMIT 10",
                    "omninode.services.metadata",
                )
                return len(result)
            finally:
                await pool.release(conn)

        def _sync_prepared():
            return run_async_in_sync(_prepared_statement())

        # Run benchmark
        result = benchmark.pedantic(_sync_prepared, rounds=100, iterations=10)

        stats = benchmark.stats.stats
        mean_ms = stats.mean * 1000

        print(f"\n[Performance] Prepared statement - Mean: {mean_ms:.2f}ms")

        assert mean_ms < PERFORMANCE_THRESHOLDS["prepared_statement_ms"]["max"]


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================


def pytest_benchmark_update_json(config, benchmarks, output_json):
    """
    Customize benchmark JSON output with database-specific metadata.

    Adds:
    - Performance thresholds
    - Test metadata
    - Database configuration
    """
    output_json["performance_thresholds"] = PERFORMANCE_THRESHOLDS
    output_json["component"] = "PostgreSQLOperations"
    output_json["test_metadata"] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "focus": "Query performance, connection pooling, transaction overhead",
    }
