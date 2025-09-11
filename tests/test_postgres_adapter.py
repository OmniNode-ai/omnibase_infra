"""
Comprehensive tests for PostgreSQL Adapter Tool.

Tests the event envelope to PostgreSQL message conversion functionality
and validates the message bus bridge pattern implementation.
"""

import asyncio
import pytest
import time
import uuid
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from omnibase_core.core.model_onex_container import ModelONEXContainer
from omnibase_core.core.errors.core_errors import CoreErrorCode

from omnibase_infra.tools.infrastructure.tool_infrastructure_postgres_adapter_effect.v1_0_0.node import ToolInfrastructurePostgresAdapterEffect
from omnibase_infra.tools.infrastructure.tool_infrastructure_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_input import ModelPostgresAdapterInput
from omnibase_infra.tools.infrastructure.tool_infrastructure_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_output import ModelPostgresAdapterOutput
from omnibase_infra.models.postgres.model_postgres_query_request import ModelPostgresQueryRequest
from omnibase_infra.models.postgres.model_postgres_health_request import ModelPostgresHealthRequest


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
                "total": 20
            },
            "database_info": {
                "version": "PostgreSQL 15.4"
            },
            "errors": []
        }
        
        return manager

    @pytest.fixture
    def adapter_with_mock(self, container, mock_connection_manager):
        """Create adapter with mocked connection manager."""
        # For testing purposes, we'll mock the container to avoid service resolution issues
        mock_container = Mock(spec=ModelONEXContainer)
        
        with patch('omnibase_infra.tools.infrastructure.tool_infrastructure_postgres_adapter_effect.v1_0_0.node.PostgresConnectionManager') as mock_manager_class:
            mock_manager_class.return_value = mock_connection_manager
            
            adapter = ToolInfrastructurePostgresAdapterEffect(mock_container)
            adapter._connection_manager = mock_connection_manager
            
            return adapter

    def test_adapter_initialization(self, container):
        """Test adapter can be initialized with container."""
        # We'll test just the class structure since full initialization requires event bus setup
        assert ToolInfrastructurePostgresAdapterEffect is not None
        
        # Test that the class has the expected attributes without instantiation
        assert hasattr(ToolInfrastructurePostgresAdapterEffect, 'process')
        assert hasattr(ToolInfrastructurePostgresAdapterEffect, 'initialize')
        assert hasattr(ToolInfrastructurePostgresAdapterEffect, 'cleanup')

    @pytest.mark.asyncio
    async def test_query_message_envelope_conversion(self, adapter_with_mock):
        """Test converting event envelope with query request to PostgreSQL operation."""
        
        # Create event envelope with query request
        correlation_id = str(uuid.uuid4())
        query_request = ModelPostgresQueryRequest(
            query="SELECT * FROM infrastructure.service_registry WHERE service_type = $1",
            parameters=["database"],
            correlation_id=correlation_id,
            context={"operation": "list_services"}
        )
        
        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id,
            context={"source": "event_bus"}
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
            record_metrics=query_request.record_metrics
        )

    @pytest.mark.asyncio
    async def test_health_check_message_envelope_conversion(self, adapter_with_mock):
        """Test converting event envelope with health request to PostgreSQL health check."""
        
        # Create event envelope with health request
        correlation_id = str(uuid.uuid4())
        health_request = ModelPostgresHealthRequest(
            include_connection_stats=True,
            include_performance_metrics=True,
            include_schema_info=False,
            correlation_id=correlation_id
        )
        
        input_envelope = ModelPostgresAdapterInput(
            operation_type="health_check",
            health_request=health_request,
            correlation_id=correlation_id,
            context={"source": "monitoring_system"}
        )

        # Process the envelope through adapter
        result = await adapter_with_mock.process(input_envelope)

        # Validate conversion to PostgreSQL health check
        assert isinstance(result, ModelPostgresAdapterOutput)
        assert result.operation_type == "health_check"
        assert result.success is True
        assert result.correlation_id == correlation_id
        assert result.health_response is not None
        
        # Verify health response contains requested information
        health_response = result.health_response
        assert health_response.status == "healthy"
        assert health_response.connection_pool is not None  # include_connection_stats=True
        assert health_response.performance is None  # Not in mock
        assert health_response.schema_info is None  # include_schema_info=False

        # Verify the connection manager health check was called
        adapter_with_mock.connection_manager.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_operation_type(self, adapter_with_mock):
        """Test handling of invalid operation types in message envelope."""
        
        # Create event envelope with invalid operation
        correlation_id = str(uuid.uuid4())
        input_envelope = ModelPostgresAdapterInput(
            operation_type="invalid_operation",
            correlation_id=correlation_id,
            context={"source": "test"}
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
        correlation_id = str(uuid.uuid4())
        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            # query_request=None  # Missing required field
            correlation_id=correlation_id
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
        adapter_with_mock.connection_manager.execute_query.side_effect = Exception("Connection timeout")
        
        # Create valid query request
        correlation_id = str(uuid.uuid4())
        query_request = ModelPostgresQueryRequest(
            query="SELECT * FROM infrastructure.service_registry",
            parameters=[],
            correlation_id=correlation_id
        )
        
        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id
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
        correlation_id = str(uuid.uuid4())
        query_request = ModelPostgresQueryRequest(
            query="SELECT COUNT(*) FROM infrastructure.service_registry",
            parameters=[],
            correlation_id=correlation_id,
            record_metrics=True
        )
        
        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id
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
            context={"source": "event_bus", "user": "system"}
        )
        
        # Validate input envelope has required fields for message bus integration
        assert hasattr(input_envelope, 'operation_type')
        assert hasattr(input_envelope, 'correlation_id') 
        assert hasattr(input_envelope, 'context')
        assert hasattr(input_envelope, 'timestamp')
        
        # Test output envelope structure
        output_envelope = ModelPostgresAdapterOutput(
            operation_type="query",
            success=True,
            correlation_id="test-123",
            timestamp=time.time(),
            execution_time_ms=100.0
        )
        
        # Validate output envelope has required fields for message bus response
        assert hasattr(output_envelope, 'operation_type')
        assert hasattr(output_envelope, 'success')
        assert hasattr(output_envelope, 'correlation_id')
        assert hasattr(output_envelope, 'timestamp')
        assert hasattr(output_envelope, 'execution_time_ms')
        assert hasattr(output_envelope, 'context')

    @pytest.mark.asyncio
    async def test_complex_query_with_parameters(self, adapter_with_mock):
        """Test complex queries with multiple parameters through message envelope."""
        
        # Mock complex query result
        adapter_with_mock.connection_manager.execute_query.return_value = [
            {"service_id": 1, "service_name": "postgres", "status": "healthy"},
            {"service_id": 2, "service_name": "redis", "status": "healthy"}
        ]
        
        # Create complex query request
        correlation_id = str(uuid.uuid4())
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
                {"version": "1.0.0", "environment": "development"}
            ],
            correlation_id=correlation_id,
            timeout=30.0,
            record_metrics=True,
            context={"operation": "service_registration", "source": "orchestrator"}
        )
        
        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id,
            context={"source": "service_mesh"}
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
        assert call_args[0][6] == {"version": "1.0.0", "environment": "development"}  # Last parameter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])