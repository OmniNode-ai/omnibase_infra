"""
Integration tests for PostgresConnectionManager.

Tests real PostgreSQL database operations using testcontainers for isolated
testing environment. Validates connection pooling, query execution, transactions,
and health monitoring with actual database.

Requirements:
- Docker must be running for testcontainers
- PostgreSQL container will be automatically started and stopped
- Tests use isolated database to prevent interference

Test Coverage:
- Real connection pool initialization with PostgreSQL
- Actual query execution (SELECT, INSERT, UPDATE, DELETE)
- Transaction ACID compliance
- Connection acquisition and release
- Health monitoring with real database
- Pool statistics and metrics
- Error handling with database failures
"""

import asyncio

import pytest

try:
    from testcontainers.postgres import PostgresContainer

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    PostgresContainer = None  # type: ignore

from omninode_bridge.infrastructure.postgres_connection_manager import (
    ModelPostgresConfig,
    PostgresConnectionManager,
)

# Skip all tests if testcontainers not available
pytestmark = pytest.mark.skipif(
    not TESTCONTAINERS_AVAILABLE,
    reason="testcontainers not installed - required for integration tests",
)


@pytest.fixture(scope="module")
def postgres_container():
    """
    Start PostgreSQL container for integration tests.

    Yields:
        PostgresContainer instance with running database
    """
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers not available")

    container = PostgresContainer("postgres:16-alpine")
    container.start()

    yield container

    container.stop()


@pytest.fixture
async def postgres_config(postgres_container) -> ModelPostgresConfig:
    """
    Create PostgreSQL configuration from container.

    Args:
        postgres_container: Running PostgreSQL container

    Returns:
        ModelPostgresConfig with container connection details
    """
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
    """
    Create and initialize connection manager with real database.

    Args:
        postgres_config: PostgreSQL configuration from container

    Yields:
        Initialized PostgresConnectionManager instance
    """
    manager = PostgresConnectionManager(postgres_config)
    await manager.initialize()

    yield manager

    await manager.close()


@pytest.fixture
async def test_table(connection_manager):
    """
    Create test table for integration tests.

    Args:
        connection_manager: Initialized connection manager

    Yields:
        Table name for testing
    """
    table_name = "test_users"

    # Create test table
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(100) UNIQUE,
        age INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    await connection_manager.execute_query(create_table_query)

    yield table_name

    # Cleanup
    await connection_manager.execute_query(f"DROP TABLE IF EXISTS {table_name}")


# Integration Tests


@pytest.mark.integration
@pytest.mark.asyncio
class TestPostgresConnectionManagerIntegration:
    """Integration test suite for PostgresConnectionManager."""

    async def test_pool_initialization_with_real_database(self, postgres_config):
        """Test connection pool initialization with real PostgreSQL."""
        manager = PostgresConnectionManager(postgres_config)

        await manager.initialize()

        assert manager.is_initialized
        assert manager.pool is not None
        assert manager.pool_size > 0

        await manager.close()

    async def test_health_check_with_real_database(self, connection_manager):
        """Test health check with real database connection."""
        is_healthy = await connection_manager.health_check()

        assert is_healthy is True

    async def test_connection_acquisition_and_release(self, connection_manager):
        """Test connection acquisition and release cycle."""
        initial_free = connection_manager.pool_free

        async with connection_manager.acquire_connection() as conn:
            # Connection should be checked out
            assert conn is not None
            current_free = connection_manager.pool_free
            assert current_free < initial_free

        # Connection should be released
        final_free = connection_manager.pool_free
        assert final_free == initial_free

    async def test_multiple_concurrent_connections(self, connection_manager):
        """Test acquiring multiple concurrent connections."""
        connections = []

        async def acquire_connection():
            async with connection_manager.acquire_connection() as conn:
                connections.append(conn)
                await asyncio.sleep(0.1)

        # Acquire 5 concurrent connections
        await asyncio.gather(*[acquire_connection() for _ in range(5)])

        # All connections should have been acquired
        assert len(connections) == 5

        # All connections should be released
        assert connection_manager.pool_free > 0

    async def test_execute_select_query(self, connection_manager, test_table):
        """Test SELECT query execution with real database."""
        # Insert test data
        await connection_manager.execute_query(
            f"INSERT INTO {test_table} (name, email, age) VALUES ($1, $2, $3)",
            "Alice",
            "alice@example.com",
            30,
        )

        # Execute SELECT query
        results = await connection_manager.execute_query(
            f"SELECT * FROM {test_table} WHERE name = $1", "Alice"
        )

        assert len(results) == 1
        assert results[0]["name"] == "Alice"
        assert results[0]["email"] == "alice@example.com"
        assert results[0]["age"] == 30

    async def test_execute_insert_query(self, connection_manager, test_table):
        """Test INSERT query execution with real database."""
        result = await connection_manager.execute_query(
            f"INSERT INTO {test_table} (name, email, age) VALUES ($1, $2, $3)",
            "Bob",
            "bob@example.com",
            25,
        )

        assert result == "INSERT 0 1"

        # Verify insertion
        rows = await connection_manager.execute_query(
            f"SELECT * FROM {test_table} WHERE name = $1", "Bob"
        )
        assert len(rows) == 1
        assert rows[0]["name"] == "Bob"

    async def test_execute_update_query(self, connection_manager, test_table):
        """Test UPDATE query execution with real database."""
        # Insert initial data
        await connection_manager.execute_query(
            f"INSERT INTO {test_table} (name, email, age) VALUES ($1, $2, $3)",
            "Charlie",
            "charlie@example.com",
            28,
        )

        # Update data
        result = await connection_manager.execute_query(
            f"UPDATE {test_table} SET age = $1 WHERE name = $2", 29, "Charlie"
        )

        assert "UPDATE" in result

        # Verify update
        rows = await connection_manager.execute_query(
            f"SELECT age FROM {test_table} WHERE name = $1", "Charlie"
        )
        assert rows[0]["age"] == 29

    async def test_execute_delete_query(self, connection_manager, test_table):
        """Test DELETE query execution with real database."""
        # Insert test data
        await connection_manager.execute_query(
            f"INSERT INTO {test_table} (name, email) VALUES ($1, $2)",
            "David",
            "david@example.com",
        )

        # Delete data
        result = await connection_manager.execute_query(
            f"DELETE FROM {test_table} WHERE name = $1", "David"
        )

        assert "DELETE" in result

        # Verify deletion
        rows = await connection_manager.execute_query(
            f"SELECT * FROM {test_table} WHERE name = $1", "David"
        )
        assert len(rows) == 0

    async def test_transaction_commit(self, connection_manager, test_table):
        """Test transaction commit with real database."""
        async with connection_manager.transaction() as conn:
            # Insert data within transaction
            await conn.execute(
                f"INSERT INTO {test_table} (name, email) VALUES ($1, $2)",
                "Eve",
                "eve@example.com",
            )

        # Verify data was committed
        rows = await connection_manager.execute_query(
            f"SELECT * FROM {test_table} WHERE name = $1", "Eve"
        )
        assert len(rows) == 1
        assert rows[0]["name"] == "Eve"

    async def test_transaction_rollback(self, connection_manager, test_table):
        """Test transaction rollback on error."""
        try:
            async with connection_manager.transaction() as conn:
                # Insert valid data
                await conn.execute(
                    f"INSERT INTO {test_table} (name, email) VALUES ($1, $2)",
                    "Frank",
                    "frank@example.com",
                )

                # Force error with duplicate email
                await conn.execute(
                    f"INSERT INTO {test_table} (name, email) VALUES ($1, $2)",
                    "Frank2",
                    "frank@example.com",  # Duplicate
                )
        except Exception:
            pass  # Expected error

        # Verify data was rolled back
        rows = await connection_manager.execute_query(
            f"SELECT * FROM {test_table} WHERE name = $1", "Frank"
        )
        assert len(rows) == 0

    async def test_transaction_isolation_serializable(
        self, connection_manager, test_table
    ):
        """Test serializable transaction isolation."""
        # Insert initial data
        await connection_manager.execute_query(
            f"INSERT INTO {test_table} (name, age) VALUES ($1, $2)", "Grace", 30
        )

        async with connection_manager.transaction(isolation="serializable") as conn:
            # Read data
            result = await conn.fetchrow(
                f"SELECT age FROM {test_table} WHERE name = $1", "Grace"
            )
            assert result["age"] == 30

            # Update data
            await conn.execute(
                f"UPDATE {test_table} SET age = $1 WHERE name = $2", 31, "Grace"
            )

        # Verify update
        rows = await connection_manager.execute_query(
            f"SELECT age FROM {test_table} WHERE name = $1", "Grace"
        )
        assert rows[0]["age"] == 31

    async def test_metrics_collection_with_real_queries(
        self, connection_manager, test_table
    ):
        """Test metrics collection with real query execution."""
        initial_metrics_count = len(connection_manager._query_metrics)

        # Execute multiple queries
        await connection_manager.execute_query(
            f"INSERT INTO {test_table} (name) VALUES ($1)", "Henry"
        )
        await connection_manager.execute_query(
            f"SELECT * FROM {test_table} WHERE name = $1", "Henry"
        )
        await connection_manager.execute_query(
            f"UPDATE {test_table} SET age = $1 WHERE name = $2", 40, "Henry"
        )

        # Verify metrics were collected
        assert len(connection_manager._query_metrics) == initial_metrics_count + 3

        # Get performance summary
        summary = connection_manager.get_performance_summary()
        assert summary["total_queries"] >= 3
        assert "SELECT" in summary["query_types"]
        assert "INSERT" in summary["query_types"]
        assert "UPDATE" in summary["query_types"]

    async def test_pool_statistics_tracking(self, connection_manager):
        """Test pool statistics tracking with real operations."""
        stats = connection_manager.get_pool_stats()

        assert stats["initialized"] is True
        assert stats["pool_size"] > 0
        assert stats["pool_free"] >= 0
        assert stats["pool_max"] == connection_manager.config.max_connections
        assert stats["checked_out"] >= 0
        assert stats["checked_in"] >= 0

    async def test_schema_search_path_configuration(self, connection_manager):
        """Test schema search_path is properly configured."""
        async with connection_manager.acquire_connection() as conn:
            # Check current search_path
            result = await conn.fetchval("SHOW search_path")

            assert connection_manager.config.schema in result
            assert "public" in result

    async def test_connection_pool_max_size_limit(self, postgres_config):
        """Test connection pool respects max_connections limit."""
        # Create manager with small pool
        small_config = ModelPostgresConfig(
            host=postgres_config.host,
            port=postgres_config.port,
            database=postgres_config.database,
            user=postgres_config.user,
            password=postgres_config.password,
            min_connections=1,
            max_connections=5,
        )

        manager = PostgresConnectionManager(small_config)
        await manager.initialize()

        try:
            # Pool should not exceed max_connections
            assert manager.pool_size <= 2

            # Acquire connections up to max
            async with manager.acquire_connection() as conn1:
                async with manager.acquire_connection() as conn2:
                    # Both connections acquired
                    assert conn1 is not None
                    assert conn2 is not None

        finally:
            await manager.close()

    async def test_query_timeout(self, connection_manager):
        """Test query timeout enforcement."""
        # This query should complete quickly
        result = await connection_manager.execute_query(
            "SELECT 1", timeout=1.0  # 1 second timeout
        )

        assert result is not None

    async def test_concurrent_query_execution(self, connection_manager, test_table):
        """Test concurrent query execution."""

        async def execute_insert(name: str):
            return await connection_manager.execute_query(
                f"INSERT INTO {test_table} (name) VALUES ($1)", name
            )

        # Execute 10 concurrent inserts
        results = await asyncio.gather(*[execute_insert(f"User{i}") for i in range(10)])

        # All inserts should succeed
        assert len(results) == 10
        assert all("INSERT" in str(r) for r in results)

        # Verify all rows were inserted
        rows = await connection_manager.execute_query(f"SELECT * FROM {test_table}")
        user_rows = [r for r in rows if r["name"].startswith("User")]
        assert len(user_rows) == 10

    async def test_pool_cleanup_on_close(self, postgres_config):
        """Test proper pool cleanup when closing connection manager."""
        manager = PostgresConnectionManager(postgres_config)
        await manager.initialize()

        assert manager.is_initialized
        assert manager.pool is not None

        await manager.close()

        assert not manager.is_initialized
        assert manager.pool is None
        assert manager.pool_size == 0
        assert manager.pool_free == 0

    async def test_error_handling_invalid_query(self, connection_manager):
        """Test error handling for invalid SQL query."""
        from omnibase_core import EnumCoreErrorCode, ModelOnexError

        with pytest.raises(ModelOnexError) as exc_info:
            await connection_manager.execute_query("INVALID SQL QUERY")

        assert exc_info.value.error_code == EnumCoreErrorCode.DATABASE_OPERATION_ERROR

    async def test_error_handling_connection_failure(self, postgres_config):
        """Test error handling for connection failure."""
        from omnibase_core import EnumCoreErrorCode, ModelOnexError

        # Create manager with invalid port
        bad_config = ModelPostgresConfig(
            host=postgres_config.host,
            port=9999,  # Invalid port
            database=postgres_config.database,
            user=postgres_config.user,
            password=postgres_config.password,
        )

        manager = PostgresConnectionManager(bad_config)

        with pytest.raises(ModelOnexError) as exc_info:
            await manager.initialize()

        assert exc_info.value.error_code == EnumCoreErrorCode.DATABASE_OPERATION_ERROR
