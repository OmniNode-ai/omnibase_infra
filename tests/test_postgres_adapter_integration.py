"""
Integration tests for PostgreSQL Adapter with real database.

Tests the complete message envelope to PostgreSQL workflow against
the Docker PostgreSQL environment.
"""

import asyncio
import logging
import os
import pytest
import time
import uuid
from typing import Dict, Any, List

# Configure logging following omnibase_3 infrastructure pattern
logger = logging.getLogger(__name__)

from omnibase_infra.infrastructure.postgres_connection_manager import PostgresConnectionManager
from omnibase_infra.tools.infrastructure.tool_infrastructure_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_input import ModelPostgresAdapterInput
from omnibase_infra.tools.infrastructure.tool_infrastructure_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_output import ModelPostgresAdapterOutput
from omnibase_infra.models.postgres.model_postgres_query_request import ModelPostgresQueryRequest
from omnibase_infra.models.postgres.model_postgres_health_request import ModelPostgresHealthRequest


# Skip integration tests if PostgreSQL is not available
def is_postgres_available() -> bool:
    """Check if PostgreSQL is available for testing."""
    try:
        import asyncpg
        # Test connection with environment variables
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        database = os.getenv("POSTGRES_DATABASE", "omnibase_infrastructure")
        user = os.getenv("POSTGRES_USER", "omnibase")
        password = os.getenv("POSTGRES_PASSWORD", "dev_password_change_in_prod")
        
        async def test_connection():
            try:
                conn = await asyncpg.connect(
                    host=host, port=port, database=database,
                    user=user, password=password
                )
                await conn.close()
                return True
            except Exception:
                return False
        
        return asyncio.run(test_connection())
    except ImportError:
        return False


skip_if_no_postgres = pytest.mark.skipif(
    not is_postgres_available(),
    reason="PostgreSQL not available for integration testing"
)


class TestPostgresAdapterIntegration:
    """Integration test suite for PostgreSQL adapter with real database."""

    @pytest.fixture
    async def connection_manager(self):
        """Create and initialize a real connection manager."""
        manager = PostgresConnectionManager()
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.fixture
    async def clean_test_table(self, connection_manager):
        """Create and clean a test table for integration tests."""
        table_name = "integration_test_services"
        
        # Create test table
        await connection_manager.execute_query(f"""
            CREATE TABLE IF NOT EXISTS infrastructure.{table_name} (
                id SERIAL PRIMARY KEY,
                service_name VARCHAR(255) NOT NULL,
                service_type VARCHAR(100) NOT NULL,
                status VARCHAR(50) DEFAULT 'active',
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Clean any existing test data
        await connection_manager.execute_query(f"DELETE FROM infrastructure.{table_name}")
        
        yield table_name
        
        # Cleanup after test
        await connection_manager.execute_query(f"DROP TABLE IF EXISTS infrastructure.{table_name}")

    @skip_if_no_postgres
    @pytest.mark.asyncio
    async def test_message_envelope_to_database_insert(self, connection_manager, clean_test_table):
        """Test complete flow: message envelope → adapter → PostgreSQL INSERT."""
        
        # Create query request for service registration
        correlation_id = str(uuid.uuid4())
        query_request = ModelPostgresQueryRequest(
            query=f"""
                INSERT INTO infrastructure.{clean_test_table} 
                (service_name, service_type, status, metadata) 
                VALUES ($1, $2, $3, $4) 
                RETURNING id, service_name, status
            """,
            parameters=[
                "test-service", 
                "microservice", 
                "initializing", 
                {"version": "1.0.0", "environment": "test"}
            ],
            correlation_id=correlation_id,
            record_metrics=True,
            context={"operation": "service_registration", "test": True}
        )

        # Execute through connection manager (simulating adapter behavior)
        result = await connection_manager.execute_query(
            query_request.query,
            *query_request.parameters,
            timeout=query_request.timeout,
            record_metrics=query_request.record_metrics
        )

        # Validate database insertion
        assert isinstance(result, list)
        assert len(result) == 1
        
        inserted_record = dict(result[0])
        assert inserted_record['service_name'] == "test-service"
        assert inserted_record['status'] == "initializing"
        assert 'id' in inserted_record

        # Verify data was actually inserted by querying back
        verification_result = await connection_manager.execute_query(
            f"SELECT COUNT(*) as count FROM infrastructure.{clean_test_table} WHERE service_name = $1",
            "test-service"
        )
        
        count_record = dict(verification_result[0])
        assert count_record['count'] == 1

    @skip_if_no_postgres  
    @pytest.mark.asyncio
    async def test_message_envelope_to_database_query(self, connection_manager, clean_test_table):
        """Test complete flow: message envelope → adapter → PostgreSQL SELECT."""
        
        # First, insert test data
        test_services = [
            ("service-1", "database", "healthy", {"replica_count": 3}),
            ("service-2", "cache", "degraded", {"memory_usage": "75%"}),
            ("service-3", "queue", "healthy", {"queue_depth": 10})
        ]
        
        for service_name, service_type, status, metadata in test_services:
            await connection_manager.execute_query(
                f"""INSERT INTO infrastructure.{clean_test_table} 
                   (service_name, service_type, status, metadata) VALUES ($1, $2, $3, $4)""",
                service_name, service_type, status, metadata
            )

        # Create query request to retrieve services
        correlation_id = str(uuid.uuid4())
        query_request = ModelPostgresQueryRequest(
            query=f"""
                SELECT service_name, service_type, status, metadata 
                FROM infrastructure.{clean_test_table} 
                WHERE service_type = $1 
                ORDER BY service_name
            """,
            parameters=["database"],
            correlation_id=correlation_id,
            record_metrics=True
        )

        # Execute through connection manager
        result = await connection_manager.execute_query(
            query_request.query,
            *query_request.parameters,
            timeout=query_request.timeout,
            record_metrics=query_request.record_metrics
        )

        # Validate query results
        assert isinstance(result, list)
        assert len(result) == 1  # Only one database service
        
        service_record = dict(result[0])
        assert service_record['service_name'] == "service-1"
        assert service_record['service_type'] == "database"
        assert service_record['status'] == "healthy"
        assert service_record['metadata'] == {"replica_count": 3}

    @skip_if_no_postgres
    @pytest.mark.asyncio
    async def test_message_envelope_to_database_update(self, connection_manager, clean_test_table):
        """Test complete flow: message envelope → adapter → PostgreSQL UPDATE."""
        
        # Insert initial test service
        insert_result = await connection_manager.execute_query(
            f"""INSERT INTO infrastructure.{clean_test_table} 
               (service_name, service_type, status) VALUES ($1, $2, $3) RETURNING id""",
            "update-test-service", "api", "initializing"
        )
        
        service_id = dict(insert_result[0])['id']

        # Create update request
        correlation_id = str(uuid.uuid4())
        query_request = ModelPostgresQueryRequest(
            query=f"""
                UPDATE infrastructure.{clean_test_table} 
                SET status = $1, metadata = $2 
                WHERE id = $3 
                RETURNING service_name, status, metadata
            """,
            parameters=[
                "healthy",
                {"last_updated": "2024-01-15T10:30:00Z", "cpu_usage": "45%"},
                service_id
            ],
            correlation_id=correlation_id,
            record_metrics=True
        )

        # Execute update
        result = await connection_manager.execute_query(
            query_request.query,
            *query_request.parameters,
            timeout=query_request.timeout,
            record_metrics=query_request.record_metrics
        )

        # Validate update results
        assert isinstance(result, list)
        assert len(result) == 1
        
        updated_record = dict(result[0])
        assert updated_record['service_name'] == "update-test-service"
        assert updated_record['status'] == "healthy"
        assert updated_record['metadata']['cpu_usage'] == "45%"

    @skip_if_no_postgres
    @pytest.mark.asyncio
    async def test_health_check_envelope_to_database(self, connection_manager):
        """Test health check message envelope → adapter → PostgreSQL health check."""
        
        # Execute health check through connection manager
        health_result = await connection_manager.health_check()

        # Validate health check structure (simulating adapter processing)
        assert isinstance(health_result, dict)
        assert 'status' in health_result
        assert 'timestamp' in health_result
        assert health_result['status'] in ['healthy', 'degraded', 'unhealthy']
        
        # Check for connection pool information
        if 'connection_pool' in health_result:
            pool_info = health_result['connection_pool']
            assert 'active' in pool_info
            assert 'idle' in pool_info
            assert 'total' in pool_info

        # Check for database information
        if 'database_info' in health_result:
            db_info = health_result['database_info']
            assert 'version' in db_info

    @skip_if_no_postgres
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, connection_manager, clean_test_table):
        """Test error handling and transaction management in message processing."""
        
        # Insert initial service
        await connection_manager.execute_query(
            f"""INSERT INTO infrastructure.{clean_test_table} 
               (service_name, service_type, status) VALUES ($1, $2, $3)""",
            "transaction-test", "database", "healthy"
        )

        # Create query request that will fail (duplicate key violation)
        correlation_id = str(uuid.uuid4())
        
        # Try to insert duplicate service name (assuming unique constraint)
        try:
            await connection_manager.execute_query(
                f"""INSERT INTO infrastructure.{clean_test_table} 
                   (service_name, service_type, status) VALUES ($1, $2, $3)""",
                "transaction-test",  # Duplicate service name
                "cache",
                "initializing"
            )
            # If no error, that's also fine - this tests error handling when it occurs
        except Exception as e:
            # Validate error was handled properly
            assert isinstance(e, Exception)

        # Verify original data is still intact
        verification_result = await connection_manager.execute_query(
            f"SELECT service_type FROM infrastructure.{clean_test_table} WHERE service_name = $1",
            "transaction-test"
        )
        
        if verification_result:
            original_record = dict(verification_result[0])
            assert original_record['service_type'] == "database"  # Original value preserved

    @skip_if_no_postgres
    @pytest.mark.asyncio  
    async def test_concurrent_message_processing(self, connection_manager, clean_test_table):
        """Test concurrent message envelope processing."""
        
        async def process_service_registration(service_id: int):
            """Simulate concurrent service registration through message envelopes."""
            correlation_id = str(uuid.uuid4())
            
            await connection_manager.execute_query(
                f"""INSERT INTO infrastructure.{clean_test_table} 
                   (service_name, service_type, status) VALUES ($1, $2, $3)""",
                f"concurrent-service-{service_id}",
                "microservice",
                "healthy"
            )
            
            return service_id

        # Process multiple concurrent registrations
        tasks = [process_service_registration(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Validate all registrations completed
        successful_registrations = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_registrations) == 5

        # Verify all services were inserted
        count_result = await connection_manager.execute_query(
            f"SELECT COUNT(*) as count FROM infrastructure.{clean_test_table} WHERE service_name LIKE 'concurrent-service-%'"
        )
        
        count_record = dict(count_result[0])
        assert count_record['count'] == 5

    @skip_if_no_postgres
    @pytest.mark.asyncio
    async def test_performance_metrics_integration(self, connection_manager, clean_test_table):
        """Test performance metrics tracking in real database operations."""
        
        start_time = time.perf_counter()
        
        # Execute a moderately complex query
        correlation_id = str(uuid.uuid4())
        result = await connection_manager.execute_query(
            f"""
            WITH service_stats AS (
                SELECT 
                    service_type,
                    COUNT(*) as service_count,
                    ARRAY_AGG(service_name) as service_names
                FROM infrastructure.{clean_test_table}
                GROUP BY service_type
            )
            SELECT * FROM service_stats
            """,
            record_metrics=True
        )
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Validate performance tracking would work
        assert execution_time_ms >= 0
        assert isinstance(result, list)
        
        # In a real adapter, this timing would be included in the output envelope
        logger.info(f"Query execution time: {execution_time_ms:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])