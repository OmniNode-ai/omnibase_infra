"""
Unit tests for Consul integration pattern generator.

Tests cover:
- Basic registration code generation
- Port validation (1-65535)
- Service name validation
- Node type validation
- Health endpoint validation
- AST compilation verification
- Deregistration code generation
- Service discovery code generation
- generate_all_patterns method
- Required imports
- Convenience functions

Part of Phase 2 codegen automation testing.
"""

import ast
import importlib.util
import textwrap
from pathlib import Path

import pytest

# Direct file import (bypassing package __init__.py to avoid omnibase_core dependency)
repo_root = Path(__file__).parent.parent.parent.parent.parent
consul_integration_path = (
    repo_root
    / "src"
    / "omninode_bridge"
    / "codegen"
    / "patterns"
    / "consul_integration.py"
)

spec = importlib.util.spec_from_file_location(
    "consul_integration", consul_integration_path
)
consul_integration = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(consul_integration)  # type: ignore

# Import classes and functions from the loaded module
ConsulPatternGenerator = consul_integration.ConsulPatternGenerator
ConsulRegistrationConfig = consul_integration.ConsulRegistrationConfig
generate_consul_registration = consul_integration.generate_consul_registration
generate_consul_discovery = consul_integration.generate_consul_discovery
generate_consul_deregistration = consul_integration.generate_consul_deregistration


class TestConsulPatternGenerator:
    """Test suite for ConsulPatternGenerator class."""

    def test_init(self):
        """Test generator initialization."""
        generator = ConsulPatternGenerator()
        assert generator is not None
        assert generator._generated_patterns == []

    def test_generate_registration_basic(self):
        """Test basic registration code generation with valid inputs."""
        generator = ConsulPatternGenerator()
        code = generator.generate_registration(
            node_type="effect",
            service_name="test-service",
            port=8000,
            health_endpoint="/health",
        )

        # Verify code contains expected elements
        assert "consul" in code.lower()
        assert "register" in code.lower()
        assert "test-service" in code
        assert "8000" in code
        assert "/health" in code
        assert "_register_with_consul" in code
        assert "async def" in code

        # Verify pattern was tracked
        assert len(generator._generated_patterns) == 1
        assert generator._generated_patterns[0]["type"] == "registration"
        assert generator._generated_patterns[0]["service_name"] == "test-service"
        assert generator._generated_patterns[0]["port"] == 8000

    def test_generate_registration_with_port(self):
        """Test registration with various valid port numbers."""
        generator = ConsulPatternGenerator()

        # Test common ports
        valid_ports = [80, 443, 8000, 8080, 9000, 3000, 5000]
        for port in valid_ports:
            code = generator.generate_registration(
                node_type="effect",
                service_name="test-service",
                port=port,
                health_endpoint="/health",
            )
            assert str(port) in code
            assert "consul" in code.lower()

    def test_generate_registration_all_node_types(self):
        """Test registration with all valid node types."""
        generator = ConsulPatternGenerator()
        valid_node_types = ["effect", "compute", "reducer", "orchestrator"]

        for node_type in valid_node_types:
            code = generator.generate_registration(
                node_type=node_type,
                service_name="test-service",
                port=8000,
                health_endpoint="/health",
            )
            assert node_type in code
            assert "consul" in code.lower()

    def test_generate_registration_with_custom_params(self):
        """Test registration with custom version and domain."""
        generator = ConsulPatternGenerator()
        code = generator.generate_registration(
            node_type="effect",
            service_name="test-service",
            port=8000,
            health_endpoint="/api/health",
            version="2.0.0",
            domain="production",
        )

        assert "2.0.0" in code
        assert "production" in code
        assert "/api/health" in code

    def test_invalid_service_name_raises_error(self):
        """Test that invalid service names raise ValueError."""
        generator = ConsulPatternGenerator()

        # Empty string
        with pytest.raises(ValueError, match="service_name must be a non-empty string"):
            generator.generate_registration(
                node_type="effect",
                service_name="",
                port=8000,
                health_endpoint="/health",
            )

        # None value
        with pytest.raises(ValueError, match="service_name must be a non-empty string"):
            generator.generate_registration(
                node_type="effect",
                service_name=None,
                port=8000,
                health_endpoint="/health",
            )

        # Invalid type (integer)
        with pytest.raises(ValueError, match="service_name must be a non-empty string"):
            generator.generate_registration(
                node_type="effect",
                service_name=12345,  # type: ignore
                port=8000,
                health_endpoint="/health",
            )

    def test_invalid_port_raises_error(self):
        """Test that invalid ports raise appropriate errors."""
        generator = ConsulPatternGenerator()

        # Port too high
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            generator.generate_registration(
                node_type="effect",
                service_name="test-service",
                port=99999,
                health_endpoint="/health",
            )

        # Port too low
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            generator.generate_registration(
                node_type="effect",
                service_name="test-service",
                port=0,
                health_endpoint="/health",
            )

        # Negative port
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            generator.generate_registration(
                node_type="effect",
                service_name="test-service",
                port=-1,
                health_endpoint="/health",
            )

        # Non-integer port
        with pytest.raises(TypeError, match="port must be an integer"):
            generator.generate_registration(
                node_type="effect",
                service_name="test-service",
                port="8000",  # type: ignore
                health_endpoint="/health",
            )

    def test_invalid_node_type_raises_error(self):
        """Test that invalid node types raise ValueError."""
        generator = ConsulPatternGenerator()

        # Empty string
        with pytest.raises(ValueError, match="node_type must be a non-empty string"):
            generator.generate_registration(
                node_type="",
                service_name="test-service",
                port=8000,
                health_endpoint="/health",
            )

        # None value
        with pytest.raises(ValueError, match="node_type must be a non-empty string"):
            generator.generate_registration(
                node_type=None,
                service_name="test-service",
                port=8000,
                health_endpoint="/health",
            )

        # Invalid node type
        with pytest.raises(ValueError, match="Invalid node_type"):
            generator.generate_registration(
                node_type="invalid",
                service_name="test-service",
                port=8000,
                health_endpoint="/health",
            )

    def test_health_endpoint_validation(self):
        """Test that health endpoints must start with '/'."""
        generator = ConsulPatternGenerator()

        # Valid health endpoints
        valid_endpoints = ["/health", "/api/health", "/status", "/healthz"]
        for endpoint in valid_endpoints:
            code = generator.generate_registration(
                node_type="effect",
                service_name="test-service",
                port=8000,
                health_endpoint=endpoint,
            )
            assert endpoint in code

        # Invalid health endpoints (missing leading /)
        with pytest.raises(ValueError, match="health_endpoint must start with '/'"):
            generator.generate_registration(
                node_type="effect",
                service_name="test-service",
                port=8000,
                health_endpoint="health",
            )

        # Non-string health endpoint
        with pytest.raises(TypeError, match="health_endpoint must be a string"):
            generator.generate_registration(
                node_type="effect",
                service_name="test-service",
                port=8000,
                health_endpoint=123,  # type: ignore
            )

    def test_generated_code_compiles(self):
        """Test that generated registration code compiles as valid Python."""
        generator = ConsulPatternGenerator()
        code = generator.generate_registration(
            node_type="effect",
            service_name="test-service",
            port=8000,
            health_endpoint="/health",
        )

        # Should parse without errors (dedent to remove class indentation)
        try:
            ast.parse(textwrap.dedent(code))
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")

    def test_generate_deregistration(self):
        """Test deregistration code generation."""
        generator = ConsulPatternGenerator()
        code = generator.generate_deregistration()

        # Verify code contains expected elements
        assert "deregister" in code.lower()
        assert "_deregister_from_consul" in code
        assert "async def" in code
        assert "consul_service_id" in code
        assert "consul_client" in code

        # Verify pattern was tracked
        assert len(generator._generated_patterns) == 1
        assert generator._generated_patterns[0]["type"] == "deregistration"

        # Verify code compiles (dedent to remove class indentation)
        try:
            ast.parse(textwrap.dedent(code))
        except SyntaxError as e:
            pytest.fail(f"Generated deregistration code has syntax errors: {e}")

    def test_generate_discovery(self):
        """Test service discovery code generation."""
        generator = ConsulPatternGenerator()
        code = generator.generate_discovery()

        # Verify code contains expected elements
        assert "discover" in code.lower()
        assert "_discover_service" in code
        assert "async def" in code
        assert "consul_client" in code
        assert "service_name" in code

        # Verify pattern was tracked
        assert len(generator._generated_patterns) == 1
        assert generator._generated_patterns[0]["type"] == "discovery"

        # Verify code compiles (dedent to remove class indentation)
        try:
            ast.parse(textwrap.dedent(code))
        except SyntaxError as e:
            pytest.fail(f"Generated discovery code has syntax errors: {e}")

    def test_generate_all_patterns(self):
        """Test generate_all_patterns method returns all three patterns."""
        generator = ConsulPatternGenerator()
        patterns = generator.generate_all_patterns(
            node_type="effect",
            service_name="test-service",
            port=8000,
            health_endpoint="/health",
            version="1.0.0",
            domain="default",
        )

        # Verify all patterns are present
        assert "registration" in patterns
        assert "discovery" in patterns
        assert "deregistration" in patterns

        # Verify each pattern is a string
        assert isinstance(patterns["registration"], str)
        assert isinstance(patterns["discovery"], str)
        assert isinstance(patterns["deregistration"], str)

        # Verify registration contains service name and port
        assert "test-service" in patterns["registration"]
        assert "8000" in patterns["registration"]

        # Verify all code compiles (dedent to remove class indentation)
        for pattern_name, code in patterns.items():
            try:
                ast.parse(textwrap.dedent(code))
            except SyntaxError as e:
                pytest.fail(f"{pattern_name} pattern has syntax errors: {e}")

    def test_generate_all_patterns_validation(self):
        """Test that generate_all_patterns validates inputs."""
        generator = ConsulPatternGenerator()

        # Invalid node type
        with pytest.raises(ValueError, match="Invalid node_type"):
            generator.generate_all_patterns(
                node_type="invalid",
                service_name="test-service",
                port=8000,
            )

        # Invalid port
        with pytest.raises(
            ValueError, match="port must be an integer between 1 and 65535"
        ):
            generator.generate_all_patterns(
                node_type="effect",
                service_name="test-service",
                port=99999,
            )

        # Invalid service name
        with pytest.raises(ValueError, match="service_name must be a non-empty string"):
            generator.generate_all_patterns(
                node_type="effect",
                service_name="",
                port=8000,
            )

    def test_get_required_imports(self):
        """Test that required imports are returned correctly."""
        generator = ConsulPatternGenerator()
        imports = generator.get_required_imports()

        # Verify we have imports
        assert len(imports) > 0
        assert isinstance(imports, list)

        # Verify key imports are present
        expected_imports = [
            "from datetime import UTC, datetime",
            "from typing import Optional",
        ]
        for expected_import in expected_imports:
            assert expected_import in imports

    def test_get_generated_patterns(self):
        """Test tracking of generated patterns."""
        generator = ConsulPatternGenerator()

        # Initially empty
        assert generator.get_generated_patterns() == []

        # Generate some patterns
        generator.generate_registration(
            node_type="effect",
            service_name="test-service",
            port=8000,
            health_endpoint="/health",
        )
        generator.generate_discovery()
        generator.generate_deregistration()

        # Verify all were tracked
        patterns = generator.get_generated_patterns()
        assert len(patterns) == 3
        assert patterns[0]["type"] == "registration"
        assert patterns[1]["type"] == "discovery"
        assert patterns[2]["type"] == "deregistration"


class TestConsulRegistrationConfig:
    """Test suite for ConsulRegistrationConfig dataclass."""

    def test_config_creation(self):
        """Test creating a configuration object."""
        config = ConsulRegistrationConfig(
            node_type="effect",
            service_name="test-service",
            port=8000,
            health_endpoint="/health",
            version="1.0.0",
            domain="default",
        )

        assert config.node_type == "effect"
        assert config.service_name == "test-service"
        assert config.port == 8000
        assert config.health_endpoint == "/health"
        assert config.version == "1.0.0"
        assert config.domain == "default"

    def test_config_defaults(self):
        """Test configuration default values."""
        config = ConsulRegistrationConfig(
            node_type="effect",
            service_name="test-service",
            port=8000,
        )

        assert config.health_endpoint == "/health"
        assert config.version == "1.0.0"
        assert config.domain == "default"


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_generate_consul_registration_function(self):
        """Test the generate_consul_registration convenience function."""
        code = generate_consul_registration(
            node_type="effect",
            service_name="test-service",
            port=8000,
            health_endpoint="/health",
        )

        assert "consul" in code.lower()
        assert "test-service" in code
        assert "8000" in code

        # Verify code compiles (dedent to remove class indentation)
        try:
            ast.parse(textwrap.dedent(code))
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")

    def test_generate_consul_discovery_function(self):
        """Test the generate_consul_discovery convenience function."""
        code = generate_consul_discovery()

        assert "discover" in code.lower()
        assert "consul" in code.lower()

        # Verify code compiles (dedent to remove class indentation)
        try:
            ast.parse(textwrap.dedent(code))
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")

    def test_generate_consul_deregistration_function(self):
        """Test the generate_consul_deregistration convenience function."""
        code = generate_consul_deregistration()

        assert "deregister" in code.lower()
        assert "consul" in code.lower()

        # Verify code compiles (dedent to remove class indentation)
        try:
            ast.parse(textwrap.dedent(code))
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_port_boundary_values(self):
        """Test port validation at boundaries."""
        generator = ConsulPatternGenerator()

        # Minimum valid port
        code = generator.generate_registration(
            node_type="effect",
            service_name="test-service",
            port=1,
            health_endpoint="/health",
        )
        assert "1" in code

        # Maximum valid port
        code = generator.generate_registration(
            node_type="effect",
            service_name="test-service",
            port=65535,
            health_endpoint="/health",
        )
        assert "65535" in code

    def test_service_name_with_special_characters(self):
        """Test service names with hyphens and underscores."""
        generator = ConsulPatternGenerator()

        service_names = [
            "test-service",
            "test_service",
            "test-service-123",
            "my_special_service",
        ]

        for service_name in service_names:
            code = generator.generate_registration(
                node_type="effect",
                service_name=service_name,
                port=8000,
                health_endpoint="/health",
            )
            assert service_name in code

    def test_multiple_pattern_generations(self):
        """Test that generator can be used multiple times."""
        generator = ConsulPatternGenerator()

        # Generate multiple patterns
        for i in range(5):
            code = generator.generate_registration(
                node_type="effect",
                service_name=f"test-service-{i}",
                port=8000 + i,
                health_endpoint="/health",
            )
            assert f"test-service-{i}" in code

        # Verify all were tracked
        patterns = generator.get_generated_patterns()
        assert len(patterns) == 5

    def test_node_type_case_insensitive(self):
        """Test that node type validation is case-insensitive."""
        generator = ConsulPatternGenerator()

        # Should work with different cases
        node_types = [
            "Effect",
            "EFFECT",
            "effect",
            "Compute",
            "REDUCER",
            "orchestrator",
        ]

        for node_type in node_types:
            code = generator.generate_registration(
                node_type=node_type,
                service_name="test-service",
                port=8000,
                health_endpoint="/health",
            )
            # The generated code should contain the exact case provided
            assert node_type in code


class TestGeneratedCodeQuality:
    """Test suite for generated code quality and structure."""

    def test_registration_contains_error_handling(self):
        """Test that registration code includes error handling."""
        generator = ConsulPatternGenerator()
        code = generator.generate_registration(
            node_type="effect",
            service_name="test-service",
            port=8000,
            health_endpoint="/health",
        )

        # Should have try/except blocks
        assert "try:" in code
        assert "except" in code
        assert "Exception" in code

        # Should have logging
        assert "emit_log_event" in code

    def test_discovery_contains_caching_logic(self):
        """Test that discovery code includes caching."""
        generator = ConsulPatternGenerator()
        code = generator.generate_discovery()

        # Should have caching logic
        assert "cache" in code.lower()
        assert "timestamp" in code.lower()

    def test_deregistration_is_graceful(self):
        """Test that deregistration handles errors gracefully."""
        generator = ConsulPatternGenerator()
        code = generator.generate_deregistration()

        # Should have error handling
        assert "try:" in code
        assert "except" in code

        # Should check for existence before deregistering
        assert "hasattr" in code

    def test_all_patterns_are_async(self):
        """Test that all generated methods are async."""
        generator = ConsulPatternGenerator()
        patterns = generator.generate_all_patterns(
            node_type="effect",
            service_name="test-service",
            port=8000,
        )

        for pattern_name, code in patterns.items():
            assert "async def" in code, f"{pattern_name} should be async"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
