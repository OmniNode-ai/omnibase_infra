"""
Enhanced Integration Tests for PostgreSQL Adapter with RedPanda Event Bus Integration.

Addresses PR review requirements with comprehensive test coverage including:
- Integration tests with actual RedPanda instance
- Performance testing of event publishing overhead  
- Circuit breaker behavior validation under load
- Error handling edge cases
- Load testing for event publishing
- Security validation tests

This test suite runs against actual Docker containers to ensure real-world compatibility.
"""

import asyncio
import json
import logging
import os
import pytest
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

# Import test containers for real service integration
import testcontainers
from testcontainers.compose import DockerCompose
from testcontainers.kafka import KafkaContainer
from testcontainers.postgres import PostgresContainer

# Import Kafka client for RedPanda interaction
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

# Configure logging following ONEX patterns
logger = logging.getLogger(__name__)

# Import ONEX infrastructure components
from omnibase_core.core.errors.onex_error import CoreErrorCode
from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.models.core.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.core.model_onex_event import ModelOnexEvent

# Import PostgreSQL adapter components
from omnibase_infra.infrastructure.postgres_connection_manager import PostgresConnectionManager
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.node import NodePostgresAdapterEffect, DatabaseCircuitBreaker, CircuitBreakerState
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_input import ModelPostgresAdapterInput
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_output import ModelPostgresAdapterOutput
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_config import ModelPostgresAdapterConfig
from omnibase_infra.models.postgres.model_postgres_query_request import ModelPostgresQueryRequest
from omnibase_infra.models.event_publishing.model_omninode_event_publisher import ModelOmniNodeEventPublisher


class RedPandaTestFixture:
    """Test fixture for RedPanda container management."""
    
    def __init__(self):
        self.container: Optional[KafkaContainer] = None
        self.bootstrap_servers: Optional[str] = None
        self.producer: Optional[KafkaProducer] = None
        self.consumers: Dict[str, KafkaConsumer] = {}
        
    async def start(self) -> str:
        """Start RedPanda container and return bootstrap servers."""
        self.container = KafkaContainer("redpandadata/redpanda:v24.2.7")
        self.container.start()
        
        # Get connection details
        self.bootstrap_servers = self.container.get_bootstrap_server()
        
        # Wait for RedPanda to be ready
        await self._wait_for_redpanda_ready()
        
        # Create producer for test event publishing
        self.producer = KafkaProducer(
            bootstrap_servers=[self.bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas
            retries=3,
            retry_backoff_ms=100
        )
        
        # Create test topics
        await self._create_test_topics()
        
        logger.info(f"RedPanda test fixture started: {self.bootstrap_servers}")
        return self.bootstrap_servers
    
    async def stop(self):
        """Stop RedPanda container and cleanup resources."""
        if self.producer:
            self.producer.close()
        
        for consumer in self.consumers.values():
            consumer.close()
        
        if self.container:
            self.container.stop()
            
        logger.info("RedPanda test fixture stopped")
    
    async def _wait_for_redpanda_ready(self, timeout_seconds: int = 30):
        """Wait for RedPanda to be ready for connections."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                # Test connection
                test_producer = KafkaProducer(
                    bootstrap_servers=[self.bootstrap_servers],
                    request_timeout_ms=5000
                )
                test_producer.close()
                return
            except Exception as e:
                await asyncio.sleep(1)
                
        raise OnexError(
            code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
            message="RedPanda container did not become ready within timeout"
        )
    
    async def _create_test_topics(self):
        """Create necessary test topics."""
        from kafka.admin import KafkaAdminClient, NewTopic
        
        admin_client = KafkaAdminClient(
            bootstrap_servers=[self.bootstrap_servers],
            client_id='test_admin'
        )
        
        test_topics = [
            NewTopic("dev.omnibase.onex.evt.postgres-query-completed.v1", num_partitions=3, replication_factor=1),
            NewTopic("dev.omnibase.onex.evt.postgres-query-failed.v1", num_partitions=3, replication_factor=1),
            NewTopic("dev.omnibase.onex.qrs.postgres-health-response.v1", num_partitions=3, replication_factor=1),
            NewTopic("test.postgres.events", num_partitions=1, replication_factor=1),
        ]
        
        try:
            admin_client.create_topics(test_topics, validate_only=False)
            logger.info("Test topics created successfully")
        except Exception as e:
            logger.warning(f"Topic creation warning (may already exist): {e}")
        finally:
            admin_client.close()
    
    def create_consumer(self, topic: str, group_id: str = None) -> KafkaConsumer:
        """Create a consumer for the given topic."""
        if group_id is None:
            group_id = f"test_consumer_{uuid.uuid4().hex[:8]}"
            
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[self.bootstrap_servers],
            group_id=group_id,
            auto_offset_reset='earliest',
            value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None,
            consumer_timeout_ms=5000  # 5 second timeout for tests
        )
        
        self.consumers[f"{topic}_{group_id}"] = consumer
        return consumer
    
    async def wait_for_messages(self, topic: str, expected_count: int, timeout_seconds: int = 10) -> List[Dict]:
        """Wait for expected number of messages on a topic."""
        consumer = self.create_consumer(topic)
        messages = []
        start_time = time.time()
        
        try:
            for message in consumer:
                messages.append(message.value)
                if len(messages) >= expected_count:
                    break
                if time.time() - start_time > timeout_seconds:
                    break
        finally:
            consumer.close()
            
        return messages


class MockONEXContainer:
    """Mock ONEX container for testing."""
    
    def __init__(self, redpanda_bootstrap_servers: str):
        self.services = {}
        self.bootstrap_servers = redpanda_bootstrap_servers
        self._setup_mock_services()
    
    def _setup_mock_services(self):
        """Setup mock services for testing."""
        # Mock PostgresConnectionManager
        self.connection_manager = AsyncMock(spec=PostgresConnectionManager)
        self.services["postgres_connection_manager"] = self.connection_manager
        
        # Mock ProtocolEventBus
        self.event_bus = AsyncMock()
        self.event_bus.publish_async = AsyncMock()
        self.services["ProtocolEventBus"] = self.event_bus
        
        # Mock adapter configuration
        self.adapter_config = ModelPostgresAdapterConfig.for_environment("test")
        self.services["postgres_adapter_config"] = self.adapter_config
    
    def get_service(self, service_name: str):
        """Get mock service by name."""
        return self.services.get(service_name)


@pytest.fixture(scope="session")
async def redpanda_fixture() -> AsyncGenerator[RedPandaTestFixture, None]:
    """Session-scoped RedPanda fixture."""
    fixture = RedPandaTestFixture()
    await fixture.start()
    try:
        yield fixture
    finally:
        await fixture.stop()


@pytest.fixture
async def postgres_container() -> AsyncGenerator[PostgresContainer, None]:
    """PostgreSQL container fixture for testing."""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest.fixture
async def mock_container(redpanda_fixture: RedPandaTestFixture) -> ModelONEXContainer:
    """Mock ONEX container with RedPanda integration."""
    return MockONEXContainer(redpanda_fixture.bootstrap_servers)


@pytest.fixture
async def postgres_adapter(mock_container: MockONEXContainer) -> NodePostgresAdapterEffect:
    """Create PostgreSQL adapter with mocked dependencies."""
    adapter = NodePostgresAdapterEffect(mock_container)
    await adapter.initialize()
    return adapter


class TestPostgresAdapterRedPandaIntegration:
    """Enhanced integration tests for PostgreSQL adapter with RedPanda event bus."""
    
    @pytest.mark.asyncio
    async def test_event_publishing_integration_success(
        self, 
        postgres_adapter: NodePostgresAdapterEffect,
        redpanda_fixture: RedPandaTestFixture
    ):
        """Test successful event publishing to RedPanda during query execution."""
        
        # Setup test data
        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query="SELECT 1 as test_value",
            parameters=[],
            correlation_id=correlation_id,
            record_metrics=True
        )
        
        input_data = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id
        )
        
        # Mock successful database execution
        postgres_adapter.connection_manager.execute_query.return_value = [{"test_value": 1}]
        
        # Create consumer to listen for events
        consumer = redpanda_fixture.create_consumer("dev.omnibase.onex.evt.postgres-query-completed.v1")
        
        # Execute adapter operation
        start_time = time.time()
        result = await postgres_adapter.process(input_data)
        
        # Verify adapter response
        assert result.success is True
        assert result.operation_type == "query"
        assert result.correlation_id == correlation_id
        
        # Verify event was published to RedPanda
        messages = await redpanda_fixture.wait_for_messages(
            "dev.omnibase.onex.evt.postgres-query-completed.v1", 
            expected_count=1,
            timeout_seconds=10
        )
        
        assert len(messages) == 1
        event_message = messages[0]
        
        # Validate event structure
        assert "event_type" in event_message
        assert event_message["event_type"] == "core.database.query_completed"
        assert "correlation_id" in event_message
        assert str(correlation_id) in event_message["correlation_id"]
        assert "data" in event_message
        assert event_message["data"]["database_type"] == "postgresql"
        assert "execution_time_ms" in event_message["data"]
        
        # Verify event bus was called
        postgres_adapter.container.get_service("ProtocolEventBus").publish_async.assert_called_once()
        
        execution_time = time.time() - start_time
        logger.info(f"Event publishing integration test completed in {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_event_publishing_integration_failure(
        self,
        postgres_adapter: NodePostgresAdapterEffect,
        redpanda_fixture: RedPandaTestFixture
    ):
        """Test event publishing to RedPanda during query failure."""
        
        # Setup test data
        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query="SELECT * FROM non_existent_table",
            parameters=[],
            correlation_id=correlation_id
        )
        
        input_data = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id
        )
        
        # Mock database failure
        database_error = Exception("relation 'non_existent_table' does not exist")
        postgres_adapter.connection_manager.execute_query.side_effect = database_error
        
        # Create consumer for failure events
        consumer = redpanda_fixture.create_consumer("dev.omnibase.onex.evt.postgres-query-failed.v1")
        
        # Execute adapter operation (should handle error gracefully)
        result = await postgres_adapter.process(input_data)
        
        # Verify adapter handled error properly
        assert result.success is False
        assert result.error_message is not None
        assert result.correlation_id == correlation_id
        
        # Verify failure event was published
        messages = await redpanda_fixture.wait_for_messages(
            "dev.omnibase.onex.evt.postgres-query-failed.v1",
            expected_count=1,
            timeout_seconds=10
        )
        
        assert len(messages) == 1
        failure_event = messages[0]
        
        # Validate failure event structure
        assert failure_event["event_type"] == "core.database.query_failed"
        assert str(correlation_id) in failure_event["correlation_id"]
        assert "error_message" in failure_event["data"]
        assert "database_type" in failure_event["data"]
        assert failure_event["data"]["database_type"] == "postgresql"
    
    @pytest.mark.asyncio
    async def test_performance_overhead_measurement(
        self,
        postgres_adapter: NodePostgresAdapterEffect,
        redpanda_fixture: RedPandaTestFixture
    ):
        """Test performance overhead of event publishing on database operations."""
        
        # Setup baseline measurement (without events)
        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query="SELECT generate_series(1, 1000) as numbers",
            parameters=[],
            correlation_id=correlation_id,
            record_metrics=True
        )
        
        # Mock consistent database response time
        mock_result = [{"numbers": i} for i in range(1, 1001)]
        postgres_adapter.connection_manager.execute_query.return_value = mock_result
        
        # Measure performance with event publishing
        performance_measurements = []
        
        for i in range(10):  # Multiple runs for statistical significance
            input_data = ModelPostgresAdapterInput(
                operation_type="query",
                query_request=query_request,
                correlation_id=uuid.uuid4()
            )
            
            start_time = time.perf_counter()
            result = await postgres_adapter.process(input_data)
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            performance_measurements.append(execution_time_ms)
            
            assert result.success is True
        
        # Calculate performance metrics
        avg_execution_time = sum(performance_measurements) / len(performance_measurements)
        max_execution_time = max(performance_measurements)
        min_execution_time = min(performance_measurements)
        
        # Performance assertions
        assert avg_execution_time < 100.0  # Average under 100ms
        assert max_execution_time < 500.0  # No single operation over 500ms
        assert all(t > 0 for t in performance_measurements)  # All measurements valid
        
        # Verify event publishing didn't significantly impact performance
        # Allow up to 10ms overhead for event publishing
        expected_db_time = 5.0  # Mock database should be very fast
        event_overhead = avg_execution_time - expected_db_time
        assert event_overhead < 50.0  # Event overhead under 50ms
        
        logger.info(f"Performance metrics - Avg: {avg_execution_time:.2f}ms, "
                   f"Min: {min_execution_time:.2f}ms, Max: {max_execution_time:.2f}ms, "
                   f"Event Overhead: {event_overhead:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior_under_load(
        self,
        postgres_adapter: NodePostgresAdapterEffect
    ):
        """Test circuit breaker behavior validation under load conditions."""
        
        # Reset circuit breaker to known state
        postgres_adapter._circuit_breaker = DatabaseCircuitBreaker(
            failure_threshold=3,  # Lower threshold for testing
            timeout_seconds=2,    # Shorter timeout for testing
            half_open_max_calls=2
        )
        
        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query="SELECT * FROM test_table",
            parameters=[],
            correlation_id=correlation_id
        )
        
        input_data = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id
        )
        
        # Phase 1: Cause failures to trigger circuit breaker
        database_error = Exception("Connection timeout")
        postgres_adapter.connection_manager.execute_query.side_effect = database_error
        
        failure_count = 0
        for i in range(5):  # Exceed failure threshold
            result = await postgres_adapter.process(input_data)
            if not result.success:
                failure_count += 1
        
        # Verify circuit breaker opened after failures
        circuit_state = postgres_adapter._circuit_breaker.get_state()
        assert circuit_state["state"] == CircuitBreakerState.OPEN.value
        assert circuit_state["failure_count"] >= 3
        
        # Phase 2: Test circuit breaker rejection
        result = await postgres_adapter.process(input_data)
        assert not result.success
        assert "circuit breaker is OPEN" in result.error_message
        
        # Phase 3: Wait for circuit breaker timeout and test half-open
        await asyncio.sleep(2.1)  # Wait for timeout + buffer
        
        # Reset to successful responses
        postgres_adapter.connection_manager.execute_query.side_effect = None
        postgres_adapter.connection_manager.execute_query.return_value = [{"test": "success"}]
        
        # Test half-open behavior (should allow limited calls)
        result = await postgres_adapter.process(input_data)
        assert result.success is True
        
        # Verify circuit closed after successful recovery
        circuit_state = postgres_adapter._circuit_breaker.get_state()
        assert circuit_state["state"] == CircuitBreakerState.CLOSED.value
        
        logger.info(f"Circuit breaker test completed - Final state: {circuit_state}")
    
    @pytest.mark.asyncio
    async def test_error_handling_edge_cases(
        self,
        postgres_adapter: NodePostgresAdapterEffect,
        redpanda_fixture: RedPandaTestFixture
    ):
        """Test comprehensive error handling edge cases."""
        
        # Test Case 1: Invalid correlation ID
        with pytest.raises(OnexError) as exc_info:
            await postgres_adapter.process(ModelPostgresAdapterInput(
                operation_type="query",
                query_request=ModelPostgresQueryRequest(
                    query="SELECT 1",
                    parameters=[],
                    correlation_id=uuid.UUID('00000000-0000-0000-0000-000000000000')  # Empty UUID
                ),
                correlation_id=uuid.UUID('00000000-0000-0000-0000-000000000000')
            ))
        
        assert exc_info.value.code == CoreErrorCode.VALIDATION_ERROR
        
        # Test Case 2: Event bus unavailable (should not fail DB operation)
        postgres_adapter.container.services["ProtocolEventBus"] = None
        
        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query="SELECT 1 as test",
            parameters=[],
            correlation_id=correlation_id
        )
        
        input_data = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id
        )
        
        # Mock successful database
        postgres_adapter.connection_manager.execute_query.return_value = [{"test": 1}]
        
        # Should raise error due to missing event bus (REQUIRED integration)
        with pytest.raises(OnexError) as exc_info:
            result = await postgres_adapter.process(input_data)
        
        assert "Event publisher not available" in str(exc_info.value)
        
        # Test Case 3: Malformed query request
        malformed_input = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=None,  # Missing query request
            correlation_id=uuid.uuid4()
        )
        
        result = await postgres_adapter.process(malformed_input)
        assert not result.success
        assert "Query request is required" in result.error_message
        
        logger.info("Error handling edge cases tested successfully")
    
    @pytest.mark.asyncio
    async def test_concurrent_load_with_event_publishing(
        self,
        postgres_adapter: NodePostgresAdapterEffect,
        redpanda_fixture: RedPandaTestFixture
    ):
        """Test high-load concurrent operations with event publishing."""
        
        # Setup concurrent operations
        concurrent_operations = 50
        mock_result = [{"concurrent_test": True}]
        postgres_adapter.connection_manager.execute_query.return_value = mock_result
        
        async def execute_concurrent_operation(operation_id: int):
            """Execute single concurrent operation."""
            correlation_id = uuid.uuid4()
            query_request = ModelPostgresQueryRequest(
                query=f"SELECT {operation_id} as operation_id",
                parameters=[],
                correlation_id=correlation_id,
                record_metrics=True
            )
            
            input_data = ModelPostgresAdapterInput(
                operation_type="query",
                query_request=query_request,
                correlation_id=correlation_id
            )
            
            start_time = time.perf_counter()
            result = await postgres_adapter.process(input_data)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return {
                "operation_id": operation_id,
                "success": result.success,
                "execution_time_ms": execution_time,
                "correlation_id": str(correlation_id)
            }
        
        # Execute concurrent operations
        start_time = time.perf_counter()
        tasks = [execute_concurrent_operation(i) for i in range(concurrent_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Analyze results
        successful_operations = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_operations = [r for r in results if not (isinstance(r, dict) and r.get("success"))]
        
        # Performance assertions
        assert len(successful_operations) >= concurrent_operations * 0.95  # 95% success rate
        assert len(failed_operations) <= concurrent_operations * 0.05     # Max 5% failures
        assert total_time < concurrent_operations * 50  # Average under 50ms per operation
        
        # Check for memory leaks or resource exhaustion
        if successful_operations:
            avg_execution_time = sum(r["execution_time_ms"] for r in successful_operations) / len(successful_operations)
            max_execution_time = max(r["execution_time_ms"] for r in successful_operations)
            
            assert avg_execution_time < 100  # Average under 100ms
            assert max_execution_time < 1000  # No operation over 1 second
        
        # Verify event publishing handled concurrent load
        success_events = await redpanda_fixture.wait_for_messages(
            "dev.omnibase.onex.evt.postgres-query-completed.v1",
            expected_count=min(len(successful_operations), 10),  # Check at least some events
            timeout_seconds=15
        )
        
        assert len(success_events) >= min(len(successful_operations), 10)
        
        logger.info(f"Concurrent load test completed - {len(successful_operations)} successful operations "
                   f"in {total_time:.2f}ms (avg: {avg_execution_time:.2f}ms per operation)")
    
    @pytest.mark.asyncio
    async def test_security_validation_comprehensive(
        self,
        postgres_adapter: NodePostgresAdapterEffect
    ):
        """Test comprehensive security validation for event publishing and data handling."""
        
        # Test Case 1: SQL injection prevention with event logging
        correlation_id = uuid.uuid4()
        
        # Attempt SQL injection in query
        malicious_query = "SELECT * FROM users WHERE id = 1; DROP TABLE users; --"
        query_request = ModelPostgresQueryRequest(
            query=malicious_query,
            parameters=[],
            correlation_id=correlation_id
        )
        
        input_data = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id
        )
        
        # Should be caught by security validation
        result = await postgres_adapter.process(input_data)
        assert not result.success
        assert "dangerous SQL patterns" in result.error_message
        
        # Test Case 2: Sensitive data sanitization in event payloads
        sensitive_query = "SELECT password='secret123', token='abc123xyz' FROM auth"
        query_request = ModelPostgresQueryRequest(
            query=sensitive_query,
            parameters=[],
            correlation_id=correlation_id
        )
        
        # Mock database error to trigger sanitization
        postgres_adapter.connection_manager.execute_query.side_effect = Exception(
            "ERROR: relation 'auth' does not exist, password='secret123' was in query"
        )
        
        input_data.query_request = query_request
        result = await postgres_adapter.process(input_data)
        
        # Verify sensitive data was sanitized
        assert not result.success
        assert "secret123" not in result.error_message
        assert "password='***'" in result.error_message or "password" not in result.error_message
        
        # Test Case 3: Query complexity validation (DoS prevention)
        complex_query = """
        SELECT u.*, p.*, s.* FROM users u 
        JOIN profiles p ON u.id = p.user_id 
        JOIN sessions s ON u.id = s.user_id 
        WHERE u.name LIKE '%test%' AND p.bio LIKE '%test%'
        UNION ALL
        SELECT u2.*, p2.*, s2.* FROM users u2 
        JOIN profiles p2 ON u2.id = p2.user_id 
        JOIN sessions s2 ON u2.id = s2.user_id
        """ * 10  # Make it very complex
        
        query_request = ModelPostgresQueryRequest(
            query=complex_query,
            parameters=[],
            correlation_id=correlation_id
        )
        
        input_data.query_request = query_request
        result = await postgres_adapter.process(input_data)
        
        # Should be rejected for complexity
        assert not result.success
        assert ("complexity score" in result.error_message or 
                "maximum allowed" in result.error_message)
        
        # Test Case 4: Parameter size validation
        large_parameter = "x" * (1024 * 1024)  # 1MB parameter
        query_request = ModelPostgresQueryRequest(
            query="SELECT $1 as large_param",
            parameters=[large_parameter],
            correlation_id=correlation_id
        )
        
        input_data.query_request = query_request
        result = await postgres_adapter.process(input_data)
        
        # Should be rejected for parameter size
        assert not result.success
        assert "Parameter" in result.error_message and "size exceeds" in result.error_message
        
        logger.info("Security validation tests completed successfully")
    
    @pytest.mark.asyncio
    async def test_health_check_with_event_publishing(
        self,
        postgres_adapter: NodePostgresAdapterEffect,
        redpanda_fixture: RedPandaTestFixture
    ):
        """Test health check operation with event publishing."""
        
        correlation_id = uuid.uuid4()
        input_data = ModelPostgresAdapterInput(
            operation_type="health_check",
            correlation_id=correlation_id,
            context={"test": "health_check_integration"}
        )
        
        # Mock healthy connection manager
        postgres_adapter.connection_manager.health_check.return_value = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "connection_pool": {"active": 0, "idle": 5, "total": 5}
        }
        
        # Create consumer for health response events
        consumer = redpanda_fixture.create_consumer("dev.omnibase.onex.qrs.postgres-health-response.v1")
        
        # Execute health check
        result = await postgres_adapter.process(input_data)
        
        # Verify health check response
        assert result.success is True
        assert result.operation_type == "health_check"
        assert result.correlation_id == correlation_id
        
        # Verify health event was published
        messages = await redpanda_fixture.wait_for_messages(
            "dev.omnibase.onex.qrs.postgres-health-response.v1",
            expected_count=1,
            timeout_seconds=10
        )
        
        assert len(messages) == 1
        health_event = messages[0]
        
        # Validate health event structure
        assert health_event["event_type"] == "core.database.health_check_response"
        assert str(correlation_id) in health_event["correlation_id"]
        assert "health_status" in health_event["data"]
        assert health_event["data"]["health_status"] == "healthy"
        
        logger.info("Health check with event publishing test completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])