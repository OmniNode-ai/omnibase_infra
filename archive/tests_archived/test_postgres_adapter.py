"""
Comprehensive tests for PostgreSQL Adapter Tool.

Tests the event envelope to PostgreSQL message conversion functionality
and validates the message bus bridge pattern implementation.
"""

import time
import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from omnibase_core.core.onex_container import ModelONEXContainer

from omnibase_infra.models.postgres.model_postgres_query_request import (
    ModelPostgresQueryRequest,
)
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_input import (
    ModelPostgresAdapterInput,
)
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_output import (
    ModelPostgresAdapterOutput,
)
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.node import (
    NodePostgresAdapterEffect,
)


class TestPostgresAdapter:
    """Test suite for PostgreSQL adapter message conversion functionality."""

    @pytest.fixture
    def container(self):
        """Create a basic ModelONEXContainer for testing."""
        return ModelONEXContainer()

    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager."""
        manager = AsyncMock()

        # Mock successful query execution
        manager.execute_query.return_value = [{"id": 1, "name": "test_service"}]

        # Mock health check
        manager.health_check.return_value = {
            "status": "healthy",
            "timestamp": time.time(),
            "connection_pool": {
                "active": 5,
                "idle": 15,
                "total": 20,
            },
            "database_info": {
                "version": "PostgreSQL 15.4",
            },
            "errors": [],
        }

        return manager

    @pytest.fixture
    def adapter_with_mock(self, container, mock_connection_manager):
        """Create adapter with mocked connection manager."""
        # For testing purposes, we'll mock the container to avoid service resolution issues
        mock_container = Mock(spec=ModelONEXContainer)

        with patch(
            "omnibase_infra.infrastructure.postgres_connection_manager.PostgresConnectionManager",
        ) as mock_manager_class:
            mock_manager_class.return_value = mock_connection_manager

            adapter = NodePostgresAdapterEffect(mock_container)
            adapter._connection_manager = mock_connection_manager

            return adapter

    def test_adapter_initialization(self, container):
        """Test adapter can be initialized with container."""
        # We'll test just the class structure since full initialization requires event bus setup
        assert NodePostgresAdapterEffect is not None

        # Test that the class has the expected attributes without instantiation
        assert hasattr(NodePostgresAdapterEffect, "process")
        assert hasattr(NodePostgresAdapterEffect, "initialize")
        assert hasattr(NodePostgresAdapterEffect, "cleanup")

    @pytest.mark.asyncio
    async def test_query_message_envelope_conversion(self, adapter_with_mock):
        """Test converting event envelope with query request to PostgreSQL operation."""

        # Create event envelope with query request
        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query="SELECT * FROM infrastructure.service_registry WHERE service_type = $1",
            parameters=["database"],
            correlation_id=correlation_id,
            context={"operation": "list_services"},
        )

        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id,
            context={"source": "event_bus"},
        )

        # Process the envelope through adapter
        result = await adapter_with_mock.process(input_envelope)

        # Validate conversion to PostgreSQL operation
        assert isinstance(result, ModelPostgresAdapterOutput)
        assert result.operation_type == "query"
        assert result.success is True
        assert result.correlation_id == correlation_id
        assert result.query_response is not None

        # Verify the connection manager was called with correct parameters
        adapter_with_mock.connection_manager.execute_query.assert_called_once_with(
            "SELECT * FROM infrastructure.service_registry WHERE service_type = $1",
            "database",  # parameters unpacked
            timeout=query_request.timeout,
            record_metrics=query_request.record_metrics,
        )

    @pytest.mark.asyncio
    async def test_mixin_health_check_functionality(self, adapter_with_mock):
        """Test MixinHealthCheck integration for PostgreSQL adapter health monitoring."""

        # Test that the adapter has proper health check methods from mixin
        assert hasattr(adapter_with_mock, "health_check")
        assert hasattr(adapter_with_mock, "health_check_async")
        assert hasattr(adapter_with_mock, "get_health_checks")

        # Test get_health_checks returns PostgreSQL-specific checks
        health_checks = adapter_with_mock.get_health_checks()
        assert len(health_checks) == 2
        assert callable(health_checks[0])  # database connectivity check
        assert callable(health_checks[1])  # connection pool check

        # Test synchronous health check
        health_result = adapter_with_mock.health_check()
        assert hasattr(health_result, "status")
        assert hasattr(health_result, "message")
        assert hasattr(health_result, "timestamp")

        # Test that health check aggregates multiple checks
        # The mixin should run both database connectivity and connection pool checks
        assert health_result.status is not None
        assert health_result.message is not None

        # Test asynchronous health check
        async_health_result = await adapter_with_mock.health_check_async()
        assert hasattr(async_health_result, "status")
        assert hasattr(async_health_result, "message")
        assert hasattr(async_health_result, "timestamp")

    @pytest.mark.asyncio
    async def test_invalid_operation_type(self, adapter_with_mock):
        """Test handling of invalid operation types in message envelope."""

        # Create event envelope with invalid operation
        correlation_id = uuid.uuid4()
        input_envelope = ModelPostgresAdapterInput(
            operation_type="invalid_operation",
            correlation_id=correlation_id,
            context={"source": "test"},
        )

        # Process the envelope through adapter
        result = await adapter_with_mock.process(input_envelope)

        # Validate error handling
        assert isinstance(result, ModelPostgresAdapterOutput)
        assert result.operation_type == "invalid_operation"
        assert result.success is False
        assert result.error_message is not None
        assert "Unsupported operation type" in result.error_message
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_missing_query_request(self, adapter_with_mock):
        """Test handling of query operation without query request."""

        # Create event envelope with query operation but no query request
        correlation_id = uuid.uuid4()
        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            # query_request=None  # Missing required field
            correlation_id=correlation_id,
        )

        # Process the envelope through adapter
        result = await adapter_with_mock.process(input_envelope)

        # Validate error handling
        assert isinstance(result, ModelPostgresAdapterOutput)
        assert result.operation_type == "query"
        assert result.success is False
        assert result.error_message is not None
        assert "Query request is required" in result.error_message

    @pytest.mark.asyncio
    async def test_database_error_handling(self, adapter_with_mock):
        """Test handling of database errors during query execution."""

        # Configure mock to raise database error
        adapter_with_mock.connection_manager.execute_query.side_effect = Exception(
            "Connection timeout",
        )

        # Create valid query request
        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query="SELECT * FROM infrastructure.service_registry",
            parameters=[],
            correlation_id=correlation_id,
        )

        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id,
        )

        # Process the envelope through adapter
        result = await adapter_with_mock.process(input_envelope)

        # Validate error handling maintains envelope structure
        assert isinstance(result, ModelPostgresAdapterOutput)
        assert result.operation_type == "query"
        assert result.success is False
        assert result.error_message == "Connection timeout"
        assert result.correlation_id == correlation_id

        # Verify query response contains error information
        assert result.query_response is not None
        assert result.query_response.success is False
        assert result.query_response.error_message == "Connection timeout"

    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, adapter_with_mock):
        """Test that adapter tracks performance metrics for envelope processing."""

        # Create query request
        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query="SELECT COUNT(*) FROM infrastructure.service_registry",
            parameters=[],
            correlation_id=correlation_id,
            record_metrics=True,
        )

        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id,
        )

        # Process the envelope
        result = await adapter_with_mock.process(input_envelope)

        # Validate performance tracking
        assert result.execution_time_ms is not None
        assert result.execution_time_ms > 0
        assert result.timestamp is not None

        # Verify query response also has execution time
        assert result.query_response.execution_time_ms is not None
        assert result.query_response.execution_time_ms > 0

    def test_envelope_structure_preservation(self):
        """Test that adapter maintains event envelope structure patterns."""

        # Test input envelope structure
        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            correlation_id="test-123",
            timestamp=time.time(),
            context={"source": "event_bus", "user": "system"},
        )

        # Validate input envelope has required fields for message bus integration
        assert hasattr(input_envelope, "operation_type")
        assert hasattr(input_envelope, "correlation_id")
        assert hasattr(input_envelope, "context")
        assert hasattr(input_envelope, "timestamp")

        # Test output envelope structure
        output_envelope = ModelPostgresAdapterOutput(
            operation_type="query",
            success=True,
            correlation_id="test-123",
            timestamp=time.time(),
            execution_time_ms=100.0,
        )

        # Validate output envelope has required fields for message bus response
        assert hasattr(output_envelope, "operation_type")
        assert hasattr(output_envelope, "success")
        assert hasattr(output_envelope, "correlation_id")
        assert hasattr(output_envelope, "timestamp")
        assert hasattr(output_envelope, "execution_time_ms")
        assert hasattr(output_envelope, "context")

    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, adapter_with_mock):
        """Test that adapter prevents SQL injection attacks."""

        # Test various SQL injection attempts
        malicious_queries = [
            "SELECT * FROM users WHERE id = 1; DROP TABLE users; --",
            "SELECT * FROM users WHERE name = 'admin' OR '1'='1'",
            "SELECT * FROM users UNION SELECT password FROM admin_users",
            "'; DELETE FROM users; --",
        ]

        correlation_id = uuid.uuid4()

        for malicious_query in malicious_queries:
            # Create query request with potentially malicious SQL
            query_request = ModelPostgresQueryRequest(
                query=malicious_query,
                parameters=[],
                correlation_id=correlation_id,
            )

            input_envelope = ModelPostgresAdapterInput(
                operation_type="query",
                query_request=query_request,
                correlation_id=correlation_id,
                timestamp=time.time(),
            )

            # Process through adapter - should handle safely
            result = await adapter_with_mock.process(input_envelope)

            # Verify result structure (adapter should process without crashing)
            assert isinstance(result, ModelPostgresAdapterOutput)
            assert result.correlation_id == correlation_id

            # Error handling should sanitize any database error messages
            if not result.success and result.error_message:
                # Ensure error message doesn't contain sensitive schema information
                assert "password" not in result.error_message.lower()
                assert "admin" not in result.error_message.lower()
                assert "DROP" not in result.error_message.upper()

    def test_error_message_sanitization(self, adapter_with_mock):
        """Test error message sanitization prevents information leakage."""

        # Test various sensitive error messages
        sensitive_errors = [
            "connection failed to postgresql://user:password123@host:5432/db",
            'schema "secret_schema" does not exist',
            'table "admin_passwords" not found',
            "authentication failed for user admin with password secret123",
        ]

        for sensitive_error in sensitive_errors:
            sanitized = adapter_with_mock._sanitize_error_message(sensitive_error)

            # Verify sensitive information is masked
            assert "password123" not in sanitized
            assert "secret_schema" not in sanitized
            assert "admin_passwords" not in sanitized
            assert "secret123" not in sanitized

            # Verify sanitized message still provides useful information
            assert len(sanitized) > 0
            assert sanitized != sensitive_error

    @pytest.mark.asyncio
    async def test_complex_query_with_parameters(self, adapter_with_mock):
        """Test complex queries with multiple parameters through message envelope."""

        # Mock complex query result
        adapter_with_mock.connection_manager.execute_query.return_value = [
            {"service_id": 1, "service_name": "postgres", "status": "healthy"},
            {"service_id": 2, "service_name": "redis", "status": "healthy"},
        ]

        # Create complex query request
        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query="""
                INSERT INTO infrastructure.service_registry
                (service_name, service_type, hostname, port, status, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, service_name, status
            """,
            parameters=[
                "new_service",
                "microservice",
                "app-server",
                8080,
                "initializing",
                {"version": "1.0.0", "environment": "development"},
            ],
            correlation_id=correlation_id,
            timeout=30.0,
            record_metrics=True,
            context={"operation": "service_registration", "source": "orchestrator"},
        )

        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id,
            context={"source": "service_mesh"},
        )

        # Process the envelope
        result = await adapter_with_mock.process(input_envelope)

        # Validate successful processing
        assert result.success is True
        assert result.query_response is not None
        assert len(result.query_response.data) == 2
        assert result.query_response.rows_affected == 2

        # Verify all parameters were passed correctly
        adapter_with_mock.connection_manager.execute_query.assert_called_once()
        call_args = adapter_with_mock.connection_manager.execute_query.call_args

        # Check that all 6 parameters were unpacked
        assert len(call_args[0]) == 7  # query + 6 parameters
        assert call_args[0][1] == "new_service"  # First parameter
        assert call_args[0][6] == {
            "version": "1.0.0",
            "environment": "development",
        }  # Last parameter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
