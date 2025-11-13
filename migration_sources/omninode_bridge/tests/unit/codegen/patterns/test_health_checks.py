#!/usr/bin/env python3
"""
Unit tests for Health Check Pattern Generator.

Tests comprehensive health check generation for ONEX v2.0 nodes including:
- Input validation (node_type, dependencies, operations)
- Code generation for different dependency types
- AST compilation verification
- Method name generation
- Edge cases and error handling
"""

import ast

import pytest

from omninode_bridge.codegen.patterns.health_checks import (
    HealthCheckGenerator,
    generate_consul_health_check,
    generate_database_health_check,
    generate_health_check_method,
    generate_http_service_health_check,
    generate_kafka_health_check,
    generate_self_health_check,
)


class TestHealthCheckGenerator:
    """Test suite for HealthCheckGenerator class."""

    def test_generator_initialization(self):
        """Test HealthCheckGenerator initializes correctly."""
        generator = HealthCheckGenerator()
        assert generator is not None
        assert hasattr(generator, "generated_checks")
        assert isinstance(generator.generated_checks, list)
        assert len(generator.generated_checks) == 0

    def test_generate_health_check_method_basic(self):
        """Test basic health check method generation."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["postgres"],
            operations=["read"],
        )

        # Basic structure checks
        assert isinstance(code, str)
        assert len(code) > 100
        assert "health" in code.lower()

        # Should contain self check and postgres check
        assert "_check_node_runtime" in code
        assert "_check_postgres_health" in code
        assert "_register_component_checks" in code

    def test_generate_health_check_postgres(self):
        """Test PostgreSQL health check is included."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodePostgresCRUDEffect",
            dependencies=["postgres"],
            operations=["read", "write"],
        )

        # PostgreSQL-specific content
        assert "_check_postgres_health" in code
        assert "database" in code.lower() or "postgres" in code.lower()
        assert "SELECT 1" in code
        assert "db_client" in code
        assert "query_time_ms" in code
        assert "connection pool" in code.lower() or "pool" in code

        # Should be marked as critical
        assert "critical=True" in code

    def test_generate_health_check_kafka(self):
        """Test Kafka health check is included."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeKafkaConsumerEffect",
            dependencies=["kafka"],
            operations=["consume"],
        )

        # Kafka-specific content
        assert "_check_kafka_health" in code
        assert "kafka" in code.lower()
        assert "kafka_producer" in code or "kafka_client" in code
        assert "bootstrap" in code.lower()

        # Kafka is non-critical
        assert "critical=False" in code

    def test_generate_health_check_consul(self):
        """Test Consul health check is included."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeServiceRegistryEffect",
            dependencies=["consul"],
            operations=[],
        )

        # Consul-specific content
        assert "_check_consul_health" in code
        assert "consul" in code.lower()
        assert "consul_client" in code
        assert "service" in code.lower()
        assert "registered" in code

        # Consul is non-critical
        assert "critical=False" in code

    def test_generate_health_check_multiple_dependencies(self):
        """Test health check with multiple dependencies."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeComplexEffect",
            dependencies=["postgres", "kafka", "consul"],
            operations=["read", "write"],
        )

        # All checks should be present
        assert "_check_node_runtime" in code
        assert "_check_postgres_health" in code
        assert "_check_kafka_health" in code
        assert "_check_consul_health" in code

        # All should be registered
        assert "_register_component_checks" in code
        assert "register_component_check" in code

    def test_invalid_node_type_empty_string(self):
        """Test that empty node_type raises ValueError."""
        generator = HealthCheckGenerator()
        with pytest.raises(ValueError, match="node_type must be a non-empty string"):
            generator.generate_health_check_method("", ["postgres"], [])

    def test_invalid_node_type_none(self):
        """Test that None node_type raises ValueError."""
        generator = HealthCheckGenerator()
        with pytest.raises(ValueError, match="node_type must be a non-empty string"):
            generator.generate_health_check_method(None, ["postgres"], [])

    def test_invalid_node_type_not_string(self):
        """Test that non-string node_type raises ValueError."""
        generator = HealthCheckGenerator()
        with pytest.raises(ValueError, match="node_type must be a non-empty string"):
            generator.generate_health_check_method(123, ["postgres"], [])

    def test_invalid_dependencies_not_list(self):
        """Test that non-list dependencies raises TypeError."""
        generator = HealthCheckGenerator()
        with pytest.raises(TypeError, match="dependencies must be a list"):
            generator.generate_health_check_method("NodeTestEffect", "postgres", [])

    def test_invalid_dependencies_unknown(self):
        """Test that unknown dependencies raise ValueError."""
        generator = HealthCheckGenerator()
        with pytest.raises(ValueError, match="Invalid dependencies"):
            generator.generate_health_check_method(
                "NodeTestEffect",
                ["postgres", "mongodb", "unknown"],
                [],
            )

    def test_invalid_operations_not_list(self):
        """Test that non-list operations raises TypeError."""
        generator = HealthCheckGenerator()
        with pytest.raises(TypeError, match="operations must be a list or None"):
            generator.generate_health_check_method(
                "NodeTestEffect",
                ["postgres"],
                "read",
            )

    def test_operations_none_allowed(self):
        """Test that operations=None is allowed."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            "NodeTestEffect",
            ["postgres"],
            operations=None,
        )
        assert isinstance(code, str)
        assert len(code) > 0

    def test_generated_code_compiles(self):
        """Test that generated code compiles successfully."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["postgres", "kafka"],
            operations=["read", "write"],
        )

        # Wrap in a class to test compilation (generated methods are indented for class context)
        wrapped_code = f"class TestNode:\n{code}"

        # Should parse without SyntaxError
        try:
            ast.parse(wrapped_code)
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")

    def test_generated_code_compiles_all_dependencies(self):
        """Test code compilation with all dependency types."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeAllDepsEffect",
            dependencies=["postgres", "kafka", "consul"],
            operations=["read", "write", "subscribe"],
        )

        # Wrap in a class to test compilation (generated methods are indented for class context)
        wrapped_code = f"class TestNode:\n{code}"

        # Should parse without SyntaxError
        try:
            ast.parse(wrapped_code)
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")

    def test_all_required_methods_generated(self):
        """Test that all expected methods are generated."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["postgres", "kafka", "consul"],
            operations=[],
        )

        # Check for required method signatures
        required_methods = [
            "_register_component_checks",
            "_check_node_runtime",
            "_check_postgres_health",
            "_check_kafka_health",
            "_check_consul_health",
        ]

        for method_name in required_methods:
            assert method_name in code, f"Missing method: {method_name}"

    def test_method_signatures_are_async(self):
        """Test that check methods are async."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["postgres"],
            operations=[],
        )

        # All check methods should be async
        assert "async def _check_node_runtime" in code
        assert "async def _check_postgres_health" in code

    def test_method_return_types(self):
        """Test that methods have correct return type hints."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["postgres"],
            operations=[],
        )

        # Check for proper return type annotation
        assert "-> tuple[HealthStatus, str, dict[str, Any]]" in code
        assert "-> None" in code  # for _register_component_checks

    def test_docstrings_present(self):
        """Test that generated methods have docstrings."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["postgres"],
            operations=[],
        )

        # All methods should have docstrings
        assert '"""' in code
        # Count docstring pairs (should be at least 3 methods with docstrings)
        docstring_count = code.count('"""')
        assert docstring_count >= 6  # 3 methods * 2 (opening and closing)

    def test_error_handling_present(self):
        """Test that error handling is included in generated code."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["postgres"],
            operations=[],
        )

        # Should have try-except blocks
        assert "try:" in code
        assert "except" in code
        assert "Exception" in code

        # Should have error details
        assert "error" in code.lower()
        assert "HealthStatus.UNHEALTHY" in code or "HealthStatus.DEGRADED" in code


class TestSelfHealthCheck:
    """Test suite for self health check generation."""

    def test_generate_self_health_check(self):
        """Test self health check generation."""
        generator = HealthCheckGenerator()
        code = generator.generate_self_health_check()

        assert isinstance(code, str)
        assert len(code) > 100
        assert "_check_node_runtime" in code
        assert "node_id" in code
        assert "container" in code
        assert "uptime" in code

    def test_self_check_includes_memory_monitoring(self):
        """Test that self check includes memory monitoring."""
        generator = HealthCheckGenerator()
        code = generator.generate_self_health_check()

        assert "psutil" in code
        assert "memory" in code.lower()
        assert "memory_mb" in code

    def test_self_check_compiles(self):
        """Test that self check code compiles."""
        generator = HealthCheckGenerator()
        code = generator.generate_self_health_check()

        # Wrap in a class to test compilation (generated methods are indented for class context)
        wrapped_code = f"class TestNode:\n{code}"

        try:
            ast.parse(wrapped_code)
        except SyntaxError as e:
            pytest.fail(f"Self check code has syntax errors: {e}")


class TestDatabaseHealthCheck:
    """Test suite for database health check generation."""

    def test_generate_database_health_check(self):
        """Test database health check generation."""
        generator = HealthCheckGenerator()
        code = generator.generate_database_health_check()

        assert isinstance(code, str)
        assert len(code) > 100
        assert "_check_postgres_health" in code
        assert "db_client" in code
        assert "SELECT 1" in code

    def test_database_check_includes_pool_monitoring(self):
        """Test that database check includes connection pool monitoring."""
        generator = HealthCheckGenerator()
        code = generator.generate_database_health_check()

        assert "pool" in code.lower()
        assert "connection" in code.lower()

    def test_database_check_includes_query_timing(self):
        """Test that database check measures query timing."""
        generator = HealthCheckGenerator()
        code = generator.generate_database_health_check()

        assert "query_time_ms" in code
        assert "time.time()" in code

    def test_database_check_compiles(self):
        """Test that database check code compiles."""
        generator = HealthCheckGenerator()
        code = generator.generate_database_health_check()

        # Wrap in a class to test compilation (generated methods are indented for class context)
        wrapped_code = f"class TestNode:\n{code}"

        try:
            ast.parse(wrapped_code)
        except SyntaxError as e:
            pytest.fail(f"Database check code has syntax errors: {e}")


class TestKafkaHealthCheck:
    """Test suite for Kafka health check generation."""

    def test_generate_kafka_health_check(self):
        """Test Kafka health check generation."""
        generator = HealthCheckGenerator()
        code = generator.generate_kafka_health_check()

        assert isinstance(code, str)
        assert len(code) > 100
        assert "_check_kafka_health" in code
        assert "kafka_producer" in code
        assert "bootstrap" in code.lower()

    def test_kafka_check_includes_connection_status(self):
        """Test that Kafka check includes connection status."""
        generator = HealthCheckGenerator()
        code = generator.generate_kafka_health_check()

        assert "connected" in code
        assert "bootstrap_connected" in code

    def test_kafka_check_compiles(self):
        """Test that Kafka check code compiles."""
        generator = HealthCheckGenerator()
        code = generator.generate_kafka_health_check()

        # Wrap in a class to test compilation (generated methods are indented for class context)
        wrapped_code = f"class TestNode:\n{code}"

        try:
            ast.parse(wrapped_code)
        except SyntaxError as e:
            pytest.fail(f"Kafka check code has syntax errors: {e}")


class TestConsulHealthCheck:
    """Test suite for Consul health check generation."""

    def test_generate_consul_health_check(self):
        """Test Consul health check generation."""
        generator = HealthCheckGenerator()
        code = generator.generate_consul_health_check()

        assert isinstance(code, str)
        assert len(code) > 100
        assert "_check_consul_health" in code
        assert "consul_client" in code
        assert "service" in code.lower()

    def test_consul_check_includes_registration_status(self):
        """Test that Consul check includes registration status."""
        generator = HealthCheckGenerator()
        code = generator.generate_consul_health_check()

        assert "registered" in code
        assert "service" in code.lower()

    def test_consul_check_compiles(self):
        """Test that Consul check code compiles."""
        generator = HealthCheckGenerator()
        code = generator.generate_consul_health_check()

        # Wrap in a class to test compilation (generated methods are indented for class context)
        wrapped_code = f"class TestNode:\n{code}"

        try:
            ast.parse(wrapped_code)
        except SyntaxError as e:
            pytest.fail(f"Consul check code has syntax errors: {e}")


class TestHTTPServiceHealthCheck:
    """Test suite for HTTP service health check generation."""

    def test_generate_http_service_health_check(self):
        """Test HTTP service health check generation."""
        generator = HealthCheckGenerator()
        code = generator.generate_http_service_health_check(
            service_name="test_service",
            service_url="http://test:8080",
        )

        assert isinstance(code, str)
        assert len(code) > 100
        assert "_check_test_service_health" in code
        assert "http://test:8080" in code
        assert "aiohttp" in code

    def test_http_service_check_includes_timing(self):
        """Test that HTTP service check includes request timing."""
        generator = HealthCheckGenerator()
        code = generator.generate_http_service_health_check(
            service_name="api_gateway",
            service_url="https://api.example.com",
        )

        assert "response_time_ms" in code
        assert "time.time()" in code

    def test_http_service_check_compiles(self):
        """Test that HTTP service check code compiles."""
        generator = HealthCheckGenerator()
        code = generator.generate_http_service_health_check(
            service_name="test_service",
            service_url="http://test:8080",
        )

        # Wrap in a class to test compilation (generated methods are indented for class context)
        wrapped_code = f"class TestNode:\n{code}"

        try:
            ast.parse(wrapped_code)
        except SyntaxError as e:
            pytest.fail(f"HTTP service check code has syntax errors: {e}")

    def test_http_service_invalid_name(self):
        """Test that invalid service name raises ValueError."""
        generator = HealthCheckGenerator()
        with pytest.raises(ValueError, match="service_name must be a non-empty string"):
            generator.generate_http_service_health_check("", "http://test:8080")

    def test_http_service_invalid_url(self):
        """Test that invalid service URL raises ValueError."""
        generator = HealthCheckGenerator()
        with pytest.raises(ValueError, match="service_url must start with"):
            generator.generate_http_service_health_check("test_service", "invalid-url")


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_generate_health_check_method_function(self):
        """Test convenience function for health check method generation."""
        code = generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["postgres"],
            operations=["read"],
        )

        assert isinstance(code, str)
        assert len(code) > 100
        assert "_check_postgres_health" in code

    def test_generate_self_health_check_function(self):
        """Test convenience function for self health check."""
        code = generate_self_health_check()

        assert isinstance(code, str)
        assert "_check_node_runtime" in code

    def test_generate_database_health_check_function(self):
        """Test convenience function for database health check."""
        code = generate_database_health_check()

        assert isinstance(code, str)
        assert "_check_postgres_health" in code

    def test_generate_kafka_health_check_function(self):
        """Test convenience function for Kafka health check."""
        code = generate_kafka_health_check()

        assert isinstance(code, str)
        assert "_check_kafka_health" in code

    def test_generate_consul_health_check_function(self):
        """Test convenience function for Consul health check."""
        code = generate_consul_health_check()

        assert isinstance(code, str)
        assert "_check_consul_health" in code

    def test_generate_http_service_health_check_function(self):
        """Test convenience function for HTTP service health check."""
        code = generate_http_service_health_check(
            service_name="test",
            service_url="http://test:8080",
        )

        assert isinstance(code, str)
        assert "_check_test_health" in code


class TestEdgeCases:
    """Test suite for edge cases and special scenarios."""

    def test_empty_dependencies_list(self):
        """Test generation with empty dependencies list."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeMinimalEffect",
            dependencies=[],
            operations=[],
        )

        # Should still have self check
        assert "_check_node_runtime" in code
        assert "_register_component_checks" in code

    def test_database_alias_accepted(self):
        """Test that 'database' is accepted as alias for 'postgres'."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["database"],
            operations=[],
        )

        assert "_check_postgres_health" in code

    def test_url_dependency_accepted(self):
        """Test that URL dependencies are accepted."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["http://service:8080", "https://api.example.com"],
            operations=[],
        )

        # Should generate HTTP service checks
        assert "aiohttp" in code
        assert "health_url" in code

    def test_large_operations_list(self):
        """Test generation with many operations."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeComplexEffect",
            dependencies=["postgres"],
            operations=["read", "write", "update", "delete", "query", "batch"],
        )

        # Should generate without issues
        assert isinstance(code, str)
        assert len(code) > 100

    def test_method_name_sanitization(self):
        """Test that service names are sanitized for method names."""
        generator = HealthCheckGenerator()
        code = generator.generate_http_service_health_check(
            service_name="api-gateway-service",
            service_url="http://api:8080",
        )

        # Should convert hyphens to underscores
        assert "_check_api_gateway_service_health" in code

    def test_register_method_contains_all_dependencies(self):
        """Test that register method includes all dependency checks."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["postgres", "kafka", "consul"],
            operations=[],
        )

        # Extract register method
        register_method = code[code.find("def _register_component_checks") :]
        register_method = register_method[: register_method.find("\n    async def")]

        # Should have all registrations
        assert "node_runtime" in register_method
        assert "postgres" in register_method
        assert "kafka" in register_method
        assert "consul" in register_method

    def test_critical_flags_correct(self):
        """Test that critical flags are set correctly for different dependencies."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["postgres", "kafka", "consul"],
            operations=[],
        )

        # Extract register method
        register_method = code[code.find("def _register_component_checks") :]

        # postgres should be critical
        postgres_section = register_method[
            register_method.find('"postgres"') : register_method.find('"postgres"')
            + 200
        ]
        assert "critical=True" in postgres_section

        # kafka should be non-critical
        kafka_section = register_method[
            register_method.find('"kafka"') : register_method.find('"kafka"') + 200
        ]
        assert "critical=False" in kafka_section

        # consul should be non-critical
        consul_section = register_method[
            register_method.find('"consul"') : register_method.find('"consul"') + 200
        ]
        assert "critical=False" in consul_section

    def test_http_dependency_registration(self):
        """Test that HTTP dependencies are properly registered."""
        generator = HealthCheckGenerator()
        code = generator.generate_health_check_method(
            node_type="NodeTestEffect",
            dependencies=["http://metadata-service:8057", "https://api-gateway:8080"],
            operations=[],
        )

        # Should generate HTTP service check methods with correct names
        assert "_check_metadata_service_health" in code
        assert "_check_api_gateway_health" in code

        # Extract register method
        register_method = code[code.find("def _register_component_checks") :]

        # Should register metadata_service
        assert '"metadata_service"' in register_method
        assert "self._check_metadata_service_health" in register_method

        # Should register api_gateway
        assert '"api_gateway"' in register_method
        assert "self._check_api_gateway_health" in register_method

        # HTTP services should be non-critical
        metadata_section = register_method[
            register_method.find('"metadata_service"') : register_method.find(
                '"metadata_service"'
            )
            + 200
        ]
        assert "critical=False" in metadata_section

    def test_http_dependency_service_name_extraction(self):
        """Test that service names are correctly extracted from URLs."""
        generator = HealthCheckGenerator()

        # Test various URL formats
        test_cases = [
            ("http://service:8080", "service"),
            ("https://api-gateway:443", "api_gateway"),
            ("http://metadata-service:8057/api", "metadata_service"),
            ("https://my-service.example.com:8443", "my_service.example.com"),
        ]

        for url, expected_key in test_cases:
            code = generator.generate_health_check_method(
                node_type="NodeTestEffect",
                dependencies=[url],
                operations=[],
            )

            # Should generate method with correct name
            assert f"_check_{expected_key}_health" in code

            # Should register with correct key
            register_method = code[code.find("def _register_component_checks") :]
            assert f'"{expected_key}"' in register_method
            assert f"self._check_{expected_key}_health" in register_method
