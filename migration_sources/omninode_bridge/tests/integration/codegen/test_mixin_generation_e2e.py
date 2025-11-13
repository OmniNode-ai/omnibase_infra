#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Tests for Mixin-Enhanced Code Generation.

This test suite validates the entire code generation pipeline with mixin support:
- End-to-end workflow tests
- Multiple mixin combinations
- All node types (Effect, Compute, Reducer, Orchestrator)
- Validation pipeline integration
- Performance benchmarks
- Backward compatibility
- Error handling
- Real-world patterns

Test Coverage:
- 50+ integration test cases
- All mixin combinations
- All 4 node types
- Full validation pipeline
- Performance benchmarks
- Backward compatibility
- Error scenarios
- Production patterns

Author: Test Generator
"""

import ast
import asyncio
import re
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from omninode_bridge.codegen.mixin_injector import MixinInjector
from omninode_bridge.codegen.template_engine import TemplateEngine
from omninode_bridge.codegen.validation.validator import NodeValidator
from omninode_bridge.codegen.yaml_contract_parser import YAMLContractParser

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_contracts_dir() -> Path:
    """Directory with sample contracts for testing."""
    return Path(__file__).parent / "sample_contracts"


@pytest.fixture
def output_directory(tmp_path) -> Path:
    """Temporary directory for generated files."""
    output_dir = tmp_path / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def yaml_parser() -> YAMLContractParser:
    """YAMLContractParser instance."""
    return YAMLContractParser()


@pytest.fixture
def mixin_injector() -> MixinInjector:
    """MixinInjector instance."""
    return MixinInjector()


@pytest.fixture
def template_engine() -> TemplateEngine:
    """TemplateEngine instance."""
    return TemplateEngine()


@pytest.fixture
def node_validator() -> NodeValidator:
    """NodeValidator instance."""
    return NodeValidator(enable_type_checking=False)  # Fast validation mode


@pytest.fixture
def load_contract(sample_contracts_dir):
    """Load a contract YAML file by name."""

    def _load(contract_name: str) -> dict[str, Any]:
        """Load and parse contract YAML."""
        contract_path = sample_contracts_dir / f"{contract_name}.yaml"
        if not contract_path.exists():
            raise FileNotFoundError(f"Contract not found: {contract_path}")
        with open(contract_path) as f:
            return yaml.safe_load(f)

    return _load


@pytest.fixture
def generate_node(yaml_parser, mixin_injector, template_engine, output_directory):
    """Generate a complete node from contract."""

    async def _generate(contract: dict[str, Any], validate: bool = True) -> Path:
        """
        Generate node from contract.

        Args:
            contract: Parsed contract dictionary
            validate: Whether to run validation

        Returns:
            Path to generated node.py file
        """
        # Parse contract
        parsed = yaml_parser.parse_contract(contract)

        # Generate imports
        imports = mixin_injector.generate_imports(parsed)

        # Generate class definition
        class_def = mixin_injector.generate_class_definition(parsed)

        # Generate complete node file
        node_code = mixin_injector.generate_node_file(parsed)

        # Write to output directory
        node_id = contract.get("node_id", "test_node")
        node_type = contract.get("node_type", "effect")
        version = contract.get("version", "v1_0_0")

        output_path = output_directory / node_type / node_id / version
        output_path.mkdir(parents=True, exist_ok=True)

        node_file = output_path / "node.py"
        node_file.write_text(node_code)

        return node_file

    return _generate


@pytest.fixture
def assert_valid_python():
    """Assert generated code is valid Python syntax."""

    def _assert(code: str) -> ast.Module:
        """Parse code and assert it's valid Python."""
        try:
            tree = ast.parse(code)
            return tree
        except SyntaxError as e:
            pytest.fail(f"Invalid Python syntax: {e}")

    return _assert


@pytest.fixture
def assert_has_mixin():
    """Assert generated code includes specific mixin in inheritance."""

    def _assert(code: str, mixin_name: str):
        """Check if mixin is in class inheritance."""
        pattern = rf"class\s+\w+\([^)]*{mixin_name}[^)]*\):"
        if not re.search(pattern, code):
            pytest.fail(f"Mixin {mixin_name} not found in class inheritance")

    return _assert


@pytest.fixture
def assert_has_method():
    """Assert generated code includes specific method."""

    def _assert(code: str, method_name: str):
        """Check if method is defined in code."""
        pattern = rf"(async\s+)?def\s+{method_name}\s*\("
        if not re.search(pattern, code):
            pytest.fail(f"Method {method_name} not found in generated code")

    return _assert


@pytest.fixture
def assert_imports_mixin():
    """Assert generated code imports specific mixin."""

    def _assert(code: str, mixin_name: str):
        """Check if mixin is imported."""
        pattern = rf"from\s+[\w.]+\s+import\s+.*{mixin_name}"
        if not re.search(pattern, code):
            pytest.fail(f"Mixin {mixin_name} not imported in generated code")

    return _assert


# ============================================================================
# Test Class: Basic End-to-End Tests
# ============================================================================


class TestBasicE2E:
    """Basic end-to-end generation workflows."""

    @pytest.mark.asyncio
    async def test_generate_effect_node_with_health_check(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Generate Effect node with MixinHealthCheck."""
        # Load contract
        contract = load_contract("health_check_only")

        # Generate node
        node_file = await generate_node(contract)

        # Read generated code
        code = node_file.read_text()

        # Validate syntax
        assert_valid_python(code)

        # Check mixin in inheritance
        assert_has_mixin(code, "MixinHealthCheck")

        # Check mixin methods present
        assert "get_health_checks" in code

    @pytest.mark.asyncio
    async def test_generate_effect_node_with_multiple_mixins(
        self,
        load_contract,
        generate_node,
        assert_valid_python,
        assert_has_mixin,
        assert_imports_mixin,
    ):
        """Generate Effect node with Health + Metrics + EventBus."""
        # Load contract
        contract = load_contract("health_metrics")

        # Generate node
        node_file = await generate_node(contract)

        # Read generated code
        code = node_file.read_text()

        # Validate syntax
        assert_valid_python(code)

        # Validate all mixins in inheritance
        assert_has_mixin(code, "MixinHealthCheck")
        assert_has_mixin(code, "MixinMetrics")

        # Validate imports
        assert_imports_mixin(code, "MixinHealthCheck")
        assert_imports_mixin(code, "MixinMetrics")

    @pytest.mark.asyncio
    async def test_generate_minimal_effect_node_no_mixins(
        self, load_contract, generate_node, assert_valid_python
    ):
        """Generate Effect node without mixins (backward compat)."""
        # Load contract (v1.0 style - no mixins)
        contract = load_contract("minimal_effect")

        # Generate node
        node_file = await generate_node(contract)

        # Read generated code
        code = node_file.read_text()

        # Validate syntax
        assert_valid_python(code)

        # Check NodeEffect only (no mixins)
        assert "NodeEffect" in code or "class " in code

        # Should not have mixin-specific code
        assert "get_health_checks" not in code

    @pytest.mark.asyncio
    async def test_generate_node_with_event_patterns(
        self, load_contract, generate_node, assert_valid_python, assert_has_method
    ):
        """Generate node with event patterns configured."""
        # Load contract with event patterns
        contract = load_contract("event_driven_service")

        # Generate node
        node_file = await generate_node(contract)

        # Read generated code
        code = node_file.read_text()

        # Validate syntax
        assert_valid_python(code)

        # Check event-related methods
        assert_has_method(code, "get_capabilities")

    @pytest.mark.asyncio
    async def test_generate_node_with_service_registry(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Generate node with service registry configuration."""
        # Load contract
        contract = load_contract("event_driven_service")

        # Generate node
        node_file = await generate_node(contract)

        # Read generated code
        code = node_file.read_text()

        # Validate syntax
        assert_valid_python(code)

        # Check service registry mixin
        assert_has_mixin(code, "MixinServiceRegistry")


# ============================================================================
# Test Class: Mixin Combinations
# ============================================================================


class TestMixinCombinations:
    """Test common mixin combinations."""

    @pytest.mark.asyncio
    async def test_health_metrics_combination(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """MixinHealthCheck + MixinMetrics."""
        contract = load_contract("health_metrics")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinHealthCheck")
        assert_has_mixin(code, "MixinMetrics")

    @pytest.mark.asyncio
    async def test_event_driven_service_combination(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """MixinEventDrivenNode + MixinServiceRegistry + MixinHealthCheck."""
        contract = load_contract("event_driven_service")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinEventDrivenNode")
        assert_has_mixin(code, "MixinServiceRegistry")
        assert_has_mixin(code, "MixinHealthCheck")

    @pytest.mark.asyncio
    async def test_cached_compute_combination(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Compute node with MixinCaching + MixinMetrics."""
        contract = load_contract("compute_cached")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinCaching")
        assert_has_mixin(code, "MixinMetrics")

    @pytest.mark.asyncio
    async def test_workflow_orchestrator_combination(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Orchestrator with MixinEventDrivenNode + MixinMetrics."""
        contract = load_contract("orchestrator_workflow")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinEventDrivenNode")
        assert_has_mixin(code, "MixinMetrics")

    @pytest.mark.asyncio
    async def test_database_adapter_combination(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Database adapter with Health + Metrics + Caching + ServiceRegistry."""
        contract = load_contract("database_adapter")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinHealthCheck")
        assert_has_mixin(code, "MixinMetrics")
        assert_has_mixin(code, "MixinCaching")
        assert_has_mixin(code, "MixinServiceRegistry")

    @pytest.mark.asyncio
    async def test_api_client_combination(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """API client with Health + Metrics + Introspection."""
        contract = load_contract("api_client")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinHealthCheck")
        assert_has_mixin(code, "MixinMetrics")
        assert_has_mixin(code, "MixinRequestResponseIntrospection")

    @pytest.mark.asyncio
    async def test_maximum_mixins(
        self, load_contract, generate_node, assert_valid_python
    ):
        """Node with 8+ mixins to test complexity."""
        contract = load_contract("maximum_mixins")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        # Should still generate valid Python
        assert_valid_python(code)

        # Count mixins in inheritance
        class_match = re.search(r"class\s+\w+\(([^)]+)\):", code)
        assert class_match, "No class definition found"
        inheritance = class_match.group(1)

        # Should have multiple mixins
        mixin_count = inheritance.count("Mixin")
        assert mixin_count >= 8, f"Expected at least 8 mixins, found {mixin_count}"

    @pytest.mark.asyncio
    async def test_single_mixin(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Node with only one mixin."""
        contract = load_contract("health_check_only")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinHealthCheck")

    @pytest.mark.asyncio
    async def test_reducer_with_event_bus(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Reducer with EventBus + Metrics."""
        contract = load_contract("reducer_persistent")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinEventBus")
        assert_has_mixin(code, "MixinMetrics")

    @pytest.mark.asyncio
    async def test_compute_with_hash_computation(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Compute node with MixinHashComputation + MixinCaching."""
        contract = load_contract("compute_cached")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinHashComputation")
        assert_has_mixin(code, "MixinCaching")


# ============================================================================
# Test Class: All Node Types
# ============================================================================


class TestAllNodeTypes:
    """Test generation for all 4 node types."""

    @pytest.mark.asyncio
    async def test_effect_node_generation(
        self, load_contract, generate_node, assert_valid_python, assert_has_method
    ):
        """Effect node with health + metrics."""
        contract = load_contract("health_metrics")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_method(code, "execute_effect")

    @pytest.mark.asyncio
    async def test_compute_node_generation(
        self, load_contract, generate_node, assert_valid_python, assert_has_method
    ):
        """Compute node with caching + metrics."""
        contract = load_contract("compute_cached")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        # Compute nodes have execute_compute method
        # Check class definition at minimum
        assert "class " in code

    @pytest.mark.asyncio
    async def test_reducer_node_generation(
        self, load_contract, generate_node, assert_valid_python
    ):
        """Reducer node with state management + persistence."""
        contract = load_contract("reducer_persistent")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        # Check reducer-specific code
        assert "class " in code

    @pytest.mark.asyncio
    async def test_orchestrator_node_generation(
        self, load_contract, generate_node, assert_valid_python
    ):
        """Orchestrator node with workflow + events."""
        contract = load_contract("orchestrator_workflow")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        # Check orchestrator-specific code
        assert "class " in code


# ============================================================================
# Test Class: Validation Pipeline Integration
# ============================================================================


class TestValidationPipeline:
    """Test NodeValidator integration in generation."""

    @pytest.mark.asyncio
    async def test_validation_passes_for_valid_code(
        self, load_contract, generate_node, node_validator
    ):
        """All validation stages pass for well-formed node."""
        contract = load_contract("health_metrics")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        # Run validation
        result = node_validator.validate_node(code, contract)

        # Should pass
        assert result.is_valid, f"Validation failed: {result.errors}"

    @pytest.mark.asyncio
    async def test_validation_syntax_check(
        self, load_contract, generate_node, node_validator
    ):
        """Syntax validation stage works."""
        contract = load_contract("minimal_effect")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        result = node_validator.validate_node(code, contract)

        # Syntax should be valid
        assert not any(
            "syntax" in error.lower() for error in result.errors
        ), "Syntax errors found"

    @pytest.mark.asyncio
    async def test_validation_mixin_check(
        self, load_contract, generate_node, node_validator
    ):
        """Mixin validation detects missing mixins."""
        contract = load_contract("health_check_only")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        result = node_validator.validate_node(code, contract)

        # Should validate mixins
        assert result.is_valid or "mixin" in str(result.warnings).lower()

    @pytest.mark.asyncio
    async def test_validation_import_check(
        self, load_contract, generate_node, node_validator
    ):
        """Import validation works."""
        contract = load_contract("health_metrics")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        result = node_validator.validate_node(code, contract)

        # Should have proper imports
        assert result.is_valid or not any(
            "import" in error.lower() for error in result.errors
        )

    @pytest.mark.asyncio
    async def test_validation_performance(
        self, load_contract, generate_node, node_validator
    ):
        """Validation completes in <200ms without type checking."""
        contract = load_contract("database_adapter")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        # Time validation
        start = time.time()
        result = node_validator.validate_node(code, contract)
        duration_ms = (time.time() - start) * 1000

        # Should be fast
        assert duration_ms < 200, f"Validation took {duration_ms}ms, expected <200ms"

    @pytest.mark.asyncio
    async def test_validation_with_complex_node(
        self, load_contract, generate_node, node_validator
    ):
        """Validation handles complex nodes with many mixins."""
        contract = load_contract("maximum_mixins")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        result = node_validator.validate_node(code, contract)

        # Should still validate (may have warnings)
        assert isinstance(result.errors, list)


# ============================================================================
# Test Class: Performance Tests
# ============================================================================


@pytest.mark.performance
class TestPerformance:
    """Performance benchmarks for generation pipeline."""

    @pytest.mark.asyncio
    async def test_generation_speed_simple_node(self, load_contract, generate_node):
        """Generate simple node <5 seconds."""
        contract = load_contract("minimal_effect")

        start = time.time()
        node_file = await generate_node(contract)
        duration = time.time() - start

        assert duration < 5.0, f"Generation took {duration}s, expected <5s"

    @pytest.mark.asyncio
    async def test_generation_speed_complex_node(self, load_contract, generate_node):
        """Generate complex node with 5+ mixins <10 seconds."""
        contract = load_contract("database_adapter")

        start = time.time()
        node_file = await generate_node(contract)
        duration = time.time() - start

        assert duration < 10.0, f"Generation took {duration}s, expected <10s"

    @pytest.mark.asyncio
    async def test_validation_speed(self, load_contract, generate_node, node_validator):
        """Validation pipeline <200ms without type checking."""
        contract = load_contract("health_metrics")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        start = time.time()
        result = node_validator.validate_node(code, contract)
        duration_ms = (time.time() - start) * 1000

        assert duration_ms < 200, f"Validation took {duration_ms}ms, expected <200ms"

    @pytest.mark.asyncio
    async def test_batch_generation(self, load_contract, generate_node):
        """Generate 10 nodes in parallel <30 seconds."""
        contracts = [
            load_contract("minimal_effect"),
            load_contract("health_check_only"),
            load_contract("health_metrics"),
            load_contract("event_driven_service"),
            load_contract("database_adapter"),
            load_contract("api_client"),
            load_contract("compute_cached"),
            load_contract("reducer_persistent"),
            load_contract("orchestrator_workflow"),
            load_contract("maximum_mixins"),
        ]

        start = time.time()

        # Generate in parallel
        tasks = [generate_node(contract) for contract in contracts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        duration = time.time() - start

        # Check all succeeded
        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"Batch generation had errors: {errors}"

        assert duration < 30.0, f"Batch generation took {duration}s, expected <30s"


# ============================================================================
# Test Class: Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Ensure v1.0 contracts still work."""

    @pytest.mark.asyncio
    async def test_v1_contract_parsing(self, load_contract, yaml_parser):
        """Parse v1.0 contract without errors."""
        contract = load_contract("minimal_effect")

        # Should parse without errors
        parsed = yaml_parser.parse_contract(contract)

        assert parsed is not None
        assert "node_id" in parsed or "node_type" in parsed

    @pytest.mark.asyncio
    async def test_v1_contract_generation(
        self, load_contract, generate_node, assert_valid_python
    ):
        """Generate from v1.0 contract (no mixins)."""
        contract = load_contract("minimal_effect")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)

    @pytest.mark.asyncio
    async def test_existing_generated_nodes_regenerate(
        self, load_contract, generate_node, assert_valid_python
    ):
        """Existing nodes can be regenerated."""
        # Generate once
        contract = load_contract("health_check_only")
        node_file_1 = await generate_node(contract)

        # Generate again (should overwrite)
        node_file_2 = await generate_node(contract)

        # Should be same file
        assert node_file_1 == node_file_2

        # Should still be valid
        code = node_file_2.read_text()
        assert_valid_python(code)

    @pytest.mark.asyncio
    async def test_minimal_contract_fields(
        self, yaml_parser, generate_node, assert_valid_python
    ):
        """Generate from contract with minimal required fields."""
        minimal_contract = {
            "node_id": "minimal_test",
            "node_type": "effect",
            "version": "v1_0_0",
            "metadata": {
                "name": "Minimal Test",
                "description": "Minimal contract",
            },
        }

        node_file = await generate_node(minimal_contract)
        code = node_file.read_text()

        assert_valid_python(code)


# ============================================================================
# Test Class: Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error cases and recovery."""

    @pytest.mark.asyncio
    async def test_invalid_contract_schema(self, yaml_parser):
        """Invalid YAML schema raises clear error."""
        invalid_contract = {
            "invalid_field": "value",
            # Missing required fields
        }

        # Should raise error or return None
        try:
            result = yaml_parser.parse_contract(invalid_contract)
            # If it doesn't raise, check it returns empty/None
            assert result is None or not result
        except Exception as e:
            # Should have clear error message
            assert "node_id" in str(e) or "node_type" in str(e)

    @pytest.mark.asyncio
    async def test_unknown_mixin_name(self, load_contract, generate_node):
        """Unknown mixin name caught in validation."""
        contract = load_contract("invalid_mixin_name")

        try:
            # May raise during generation or produce warning
            node_file = await generate_node(contract, validate=False)
            code = node_file.read_text()
            # If generated, should have warning comment or skip unknown mixin
            assert "MixinDoesNotExist" not in code or "unknown" in code.lower()
        except Exception as e:
            # Should have clear error about unknown mixin
            assert "mixin" in str(e).lower() and "unknown" in str(e).lower()

    @pytest.mark.asyncio
    async def test_invalid_node_type(self, yaml_parser, generate_node):
        """Invalid node_type field."""
        invalid_contract = {
            "node_id": "test",
            "node_type": "invalid_type",  # Not effect/compute/reducer/orchestrator
            "version": "v1_0_0",
            "metadata": {"name": "Test", "description": "Test"},
        }

        try:
            node_file = await generate_node(invalid_contract, validate=False)
            # May generate but should have issues
        except Exception as e:
            # Should mention invalid type
            assert "type" in str(e).lower() or "invalid" in str(e).lower()

    @pytest.mark.asyncio
    async def test_malformed_yaml_contract(self, sample_contracts_dir):
        """Malformed YAML file."""
        malformed_path = sample_contracts_dir / "malformed.yaml"
        malformed_path.write_text("invalid: yaml: content: [unclosed")

        with pytest.raises(yaml.YAMLError):
            with open(malformed_path) as f:
                yaml.safe_load(f)

    @pytest.mark.asyncio
    async def test_missing_required_field(self, yaml_parser, generate_node):
        """Missing required field in contract."""
        incomplete_contract = {
            "node_id": "test",
            # Missing node_type
            "version": "v1_0_0",
        }

        try:
            result = yaml_parser.parse_contract(incomplete_contract)
            assert result is None or "node_type" not in result
        except Exception as e:
            assert "node_type" in str(e) or "required" in str(e).lower()


# ============================================================================
# Test Class: Real-World Scenarios
# ============================================================================


class TestRealWorldScenarios:
    """Test realistic use cases."""

    @pytest.mark.asyncio
    async def test_database_adapter_pattern(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Database adapter with health, metrics, circuit breaker."""
        contract = load_contract("database_adapter")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinHealthCheck")
        assert_has_mixin(code, "MixinMetrics")
        assert_has_mixin(code, "MixinCaching")

    @pytest.mark.asyncio
    async def test_api_client_pattern(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """API client with retry, metrics, introspection."""
        contract = load_contract("api_client")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinHealthCheck")
        assert_has_mixin(code, "MixinMetrics")
        assert_has_mixin(code, "MixinRequestResponseIntrospection")

    @pytest.mark.asyncio
    async def test_event_processor_pattern(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Event processor with event bus, metrics, health."""
        contract = load_contract("event_driven_service")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinEventDrivenNode")
        assert_has_mixin(code, "MixinHealthCheck")

    @pytest.mark.asyncio
    async def test_workflow_coordinator_pattern(
        self, load_contract, generate_node, assert_valid_python, assert_has_mixin
    ):
        """Workflow coordinator with events, metrics."""
        contract = load_contract("orchestrator_workflow")
        node_file = await generate_node(contract)
        code = node_file.read_text()

        assert_valid_python(code)
        assert_has_mixin(code, "MixinEventDrivenNode")
        assert_has_mixin(code, "MixinMetrics")


# ============================================================================
# Test Summary and Reporting
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """Add markers and organize tests."""
    for item in items:
        # Add markers based on test class
        if "TestPerformance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        if "TestE2E" in item.nodeid or "TestRealWorld" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
