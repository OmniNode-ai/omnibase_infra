#!/usr/bin/env python3
"""
Integration tests for node validation with real-world scenarios.

Tests validation pipeline with:
- Real generated nodes (if available)
- Complex mixin combinations
- Various node types (effect, compute, reducer, orchestrator)
- Performance validation under load
"""

import pytest

from omninode_bridge.codegen.models_contract import (
    ModelEnhancedContract,
    ModelMixinDeclaration,
    ModelVersionInfo,
)
from omninode_bridge.codegen.validation import EnumValidationStage, NodeValidator

# ===== Real-World Node Examples =====


@pytest.fixture
def complex_effect_node_code() -> str:
    """Complex effect node with multiple mixins."""
    return '''
"""
PostgreSQL CRUD Effect Node with Health Check and Introspection.

Production-grade node for PostgreSQL operations with:
- Health monitoring
- Introspection capabilities
- Circuit breaker protection (via NodeEffect)
- Retry policy (via NodeEffect)
"""

from typing import Any, Optional

from omnibase_core.models import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
from omnibase_core.mixins.mixin_introspection import MixinIntrospection


class NodePostgresCrudEffect(NodeEffect, MixinHealthCheck, MixinIntrospection):
    """
    PostgreSQL CRUD operations effect node.

    Provides create, read, update, delete operations for PostgreSQL
    with built-in health monitoring and introspection.

    Attributes:
        connection_string: PostgreSQL connection string
        pool_size: Connection pool size
        timeout_seconds: Query timeout
    """

    def __init__(self, container: ModelContainer):
        """
        Initialize PostgreSQL CRUD effect node.

        Args:
            container: Dependency injection container
        """
        super().__init__(container)
        self.connection_string: Optional[str] = None
        self.pool_size: int = 10
        self.timeout_seconds: int = 30
        self.pool: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool."""
        await super().initialize()

        # Get configuration from container
        config = self.container.get("config")
        self.connection_string = config.get("postgres_connection_string")
        self.pool_size = config.get("postgres_pool_size", 10)
        self.timeout_seconds = config.get("postgres_timeout", 30)

        # Initialize connection pool (pseudo-code)
        # self.pool = await create_pool(self.connection_string, ...)

    async def shutdown(self) -> None:
        """Shutdown PostgreSQL connection pool."""
        if self.pool:
            # await self.pool.close()
            self.pool = None

        await super().shutdown()

    async def execute_effect(
        self,
        operation: str,
        table: str,
        data: Optional[dict[str, Any]] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute CRUD operation.

        Args:
            operation: CRUD operation (create, read, update, delete)
            table: Table name
            data: Data for create/update operations
            filters: Filters for read/update/delete operations

        Returns:
            Operation result

        Raises:
            ValueError: If operation is invalid
            DatabaseError: If database operation fails
        """
        if operation not in ("create", "read", "update", "delete"):
            raise ValueError(f"Invalid operation: {operation}")

        # Validate inputs
        if not table:
            raise ValueError("Table name is required")

        # Execute operation (pseudo-code)
        result = {
            "operation": operation,
            "table": table,
            "status": "success",
            "rows_affected": 0,
        }

        return result

    async def health_check(self) -> dict[str, Any]:
        """
        Check PostgreSQL connection health.

        Returns:
            Health status
        """
        try:
            # Simple query to check connection
            # await self.pool.execute("SELECT 1")
            return {"status": "healthy", "details": "Database connection OK"}
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": f"Database connection failed: {e!s}",
            }

    def get_introspection_data(self) -> dict[str, Any]:
        """
        Get introspection data.

        Returns:
            Node introspection information
        """
        return {
            "node_name": "NodePostgresCrudEffect",
            "node_type": "effect",
            "connection_string": "***REDACTED***",
            "pool_size": self.pool_size,
            "timeout_seconds": self.timeout_seconds,
            "pool_active": self.pool is not None,
        }
'''


@pytest.fixture
def compute_node_code() -> str:
    """Compute node example."""
    return '''
"""Data Transformer Compute Node."""

from typing import Any

from omnibase_core.models import ModelContainer
from omnibase_core.nodes.node_compute import NodeCompute


class NodeDataTransformerCompute(NodeCompute):
    """
    Transform data between formats.

    Pure compute node for data transformations.
    """

    def __init__(self, container: ModelContainer):
        """Initialize transformer."""
        super().__init__(container)
        self.transformation_rules: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize transformation rules."""
        await super().initialize()
        config = self.container.get("config")
        self.transformation_rules = config.get("transformation_rules", {})

    async def execute_compute(
        self, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute data transformation.

        Args:
            input_data: Input data to transform

        Returns:
            Transformed data
        """
        transformed = {}

        for key, value in input_data.items():
            # Apply transformation rules
            if key in self.transformation_rules:
                rule = self.transformation_rules[key]
                transformed[rule["target_field"]] = self._apply_transformation(
                    value, rule
                )
            else:
                transformed[key] = value

        return transformed

    def _apply_transformation(self, value: Any, rule: dict[str, Any]) -> Any:
        """Apply transformation rule to value."""
        # Implementation here
        return value
'''


@pytest.fixture
def complex_contract() -> ModelEnhancedContract:
    """Complex contract with multiple mixins."""
    return ModelEnhancedContract(
        name="NodePostgresCrudEffect",
        version=ModelVersionInfo(major=1, minor=2, patch=3),
        node_type="effect",
        description="PostgreSQL CRUD operations",
        schema_version="v2.0.0",
        mixins=[
            ModelMixinDeclaration(
                name="MixinHealthCheck",
                enabled=True,
                import_path="omnibase_core.mixins.mixin_health_check",
                config={"interval_seconds": 30},
            ),
            ModelMixinDeclaration(
                name="MixinIntrospection",
                enabled=True,
                import_path="omnibase_core.mixins.mixin_introspection",
                config={"expose_sensitive": False},
            ),
        ],
    )


# ===== Integration Tests =====


@pytest.mark.asyncio
async def test_validate_complex_effect_node(
    complex_effect_node_code: str, complex_contract: ModelEnhancedContract
):
    """Test validation of complex effect node with multiple mixins."""
    validator = NodeValidator(
        enable_type_checking=False,  # Skip for speed
        enable_security_scan=True,
    )

    results = await validator.validate_generated_node(
        complex_effect_node_code, complex_contract
    )

    # All stages should pass
    assert all(
        r.passed for r in results
    ), f"Failed stages: {[r for r in results if not r.passed]}"

    # Verify all expected stages ran
    stage_names = {r.stage for r in results}
    assert EnumValidationStage.SYNTAX in stage_names
    assert EnumValidationStage.AST in stage_names
    assert EnumValidationStage.IMPORTS in stage_names
    assert EnumValidationStage.ONEX_COMPLIANCE in stage_names
    assert EnumValidationStage.SECURITY in stage_names

    # Check performance
    total_time = sum(r.execution_time_ms for r in results)
    assert total_time < 200  # Should be fast


@pytest.mark.asyncio
async def test_validate_compute_node(compute_node_code: str):
    """Test validation of compute node."""
    validator = NodeValidator(enable_security_scan=True)

    contract = ModelEnhancedContract(
        name="NodeDataTransformerCompute",
        version=ModelVersionInfo(),
        node_type="compute",
        description="Data transformer",
    )

    results = await validator.validate_generated_node(compute_node_code, contract)

    # Should pass all validations
    assert all(r.passed for r in results)


@pytest.mark.asyncio
async def test_validate_node_with_disabled_mixin():
    """Test validation with disabled mixin."""
    code = """
from omnibase_core.nodes.node_effect import NodeEffect

class NodeTestEffect(NodeEffect):
    def __init__(self, container):
        super().__init__(container)

    async def initialize(self):
        await super().initialize()

    async def execute_effect(self):
        return {}
"""

    contract = ModelEnhancedContract(
        name="NodeTestEffect",
        version=ModelVersionInfo(),
        node_type="effect",
        description="Test",
        mixins=[
            ModelMixinDeclaration(
                name="MixinHealthCheck",
                enabled=False,  # Disabled
            ),
        ],
    )

    validator = NodeValidator()
    results = await validator.validate_generated_node(code, contract)

    # Should pass (disabled mixin not required)
    onex_result = next(
        (r for r in results if r.stage == EnumValidationStage.ONEX_COMPLIANCE), None
    )
    assert onex_result is not None
    assert onex_result.passed is True


@pytest.mark.asyncio
async def test_validate_all_node_types():
    """Test validation works for all node types."""
    node_types = ["effect", "compute", "reducer", "orchestrator"]

    validator = NodeValidator(enable_security_scan=False)

    for node_type in node_types:
        code = f"""
from omnibase_core.nodes.node_{node_type} import Node{node_type.capitalize()}

class NodeTest{node_type.capitalize()}(Node{node_type.capitalize()}):
    def __init__(self, container):
        super().__init__(container)

    async def initialize(self):
        await super().initialize()

    async def execute_{node_type}(self):
        return {"{}"}
"""

        contract = ModelEnhancedContract(
            name=f"NodeTest{node_type.capitalize()}",
            version=ModelVersionInfo(),
            node_type=node_type,
            description=f"Test {node_type}",
        )

        results = await validator.validate_generated_node(code, contract)

        # All should pass for all node types
        assert all(r.passed for r in results), f"Failed for node_type={node_type}"


@pytest.mark.asyncio
async def test_validation_error_suggestions_are_helpful():
    """Test that validation errors include helpful suggestions."""
    code = """
from omnibase_core.nodes.node_effect import NodeEffect

class NodeTestEffect(NodeEffect):
    pass  # Missing everything
"""

    contract = ModelEnhancedContract(
        name="NodeTestEffect",
        version=ModelVersionInfo(),
        node_type="effect",
        description="Test",
    )

    validator = NodeValidator()
    results = await validator.validate_generated_node(code, contract)

    # Should have AST errors
    ast_result = next((r for r in results if r.stage == EnumValidationStage.AST), None)
    assert ast_result is not None
    assert not ast_result.passed

    # Should have suggestions
    assert len(ast_result.suggestions) > 0
    suggestions_text = " ".join(ast_result.suggestions).lower()
    assert "add" in suggestions_text or "implement" in suggestions_text


@pytest.mark.asyncio
async def test_validation_with_multiple_security_issues():
    """Test validation catches multiple security issues."""
    code = """
from omnibase_core.nodes.node_effect import NodeEffect
import os
import pickle

class NodeInsecureEffect(NodeEffect):
    def __init__(self, container):
        super().__init__(container)
        self.api_key = "sk-1234567890"  # Hardcoded secret
        self.password = "admin123"  # Another secret

    async def execute_effect(self, command, data):
        # Multiple security issues
        result = eval(command)  # Dangerous
        os.system(f"process {command}")  # Dangerous
        serialized = pickle.dumps(data)  # Suspicious
        return result
"""

    contract = ModelEnhancedContract(
        name="NodeInsecureEffect",
        version=ModelVersionInfo(),
        node_type="effect",
        description="Insecure node",
    )

    validator = NodeValidator(enable_security_scan=True)
    results = await validator.validate_generated_node(code, contract)

    security_result = next(
        (r for r in results if r.stage == EnumValidationStage.SECURITY), None
    )
    assert security_result is not None
    assert not security_result.passed

    # Should catch multiple issues
    assert len(security_result.errors) >= 3  # eval, os.system, secrets
    assert len(security_result.warnings) >= 1  # pickle

    # Check specific issues caught
    errors_text = " ".join(security_result.errors).lower()
    assert "eval" in errors_text
    assert "system" in errors_text
    assert "secret" in errors_text or "api_key" in errors_text


@pytest.mark.asyncio
@pytest.mark.performance
async def test_validation_performance_under_load():
    """Test validation performance with realistic node sizes."""
    # Generate a realistic-sized node (100+ lines)
    code = '''
"""Complex production node."""

from typing import Any, Optional
from omnibase_core.models import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck


class NodeComplexEffect(NodeEffect, MixinHealthCheck):
    """Complex production node with many methods."""

    def __init__(self, container: ModelContainer):
        super().__init__(container)
        self.config: dict[str, Any] = {}
        self.state: dict[str, Any] = {}
        self.metrics: dict[str, int] = {}

    async def initialize(self) -> None:
        await super().initialize()
        self.config = self.container.get("config", {})
        self.state = {"initialized": True}
        self.metrics = {"calls": 0, "errors": 0}

    async def shutdown(self) -> None:
        self.state = {}
        await super().shutdown()

    async def execute_effect(self, **kwargs) -> dict[str, Any]:
        """Execute complex operation."""
        self.metrics["calls"] += 1

        try:
            result = await self._process_operation(kwargs)
            return result
        except Exception as e:
            self.metrics["errors"] += 1
            raise

    async def _process_operation(self, params: dict) -> dict[str, Any]:
        """Process operation step 1."""
        validated = await self._validate_params(params)
        transformed = await self._transform_data(validated)
        result = await self._execute_core_logic(transformed)
        return result

    async def _validate_params(self, params: dict) -> dict[str, Any]:
        """Validate parameters."""
        return params  # Simplified

    async def _transform_data(self, data: dict) -> dict[str, Any]:
        """Transform data."""
        return data  # Simplified

    async def _execute_core_logic(self, data: dict) -> dict[str, Any]:
        """Execute core business logic."""
        return {"status": "success", "data": data}

    async def health_check(self) -> dict[str, Any]:
        """Check health."""
        return {
            "status": "healthy",
            "metrics": self.metrics,
            "state": self.state,
        }

    def _helper_method_1(self) -> None:
        """Helper method 1."""
        pass

    def _helper_method_2(self) -> None:
        """Helper method 2."""
        pass

    def _helper_method_3(self) -> None:
        """Helper method 3."""
        pass
'''

    contract = ModelEnhancedContract(
        name="NodeComplexEffect",
        version=ModelVersionInfo(),
        node_type="effect",
        description="Complex node",
        mixins=[
            ModelMixinDeclaration(
                name="MixinHealthCheck",
                enabled=True,
            )
        ],
    )

    validator = NodeValidator(enable_type_checking=False)

    # Run validation 10 times to test consistency
    times = []
    for _ in range(10):
        results = await validator.validate_generated_node(code, contract)
        total_time = sum(r.execution_time_ms for r in results)
        times.append(total_time)

        # All should pass
        assert all(r.passed for r in results)

    # Check performance consistency
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    assert avg_time < 200  # Average should be fast
    assert max_time < 300  # Even worst case should be reasonable
    assert min_time > 0  # Should take some time

    # Variance should be low (consistent performance)
    variance = max_time - min_time
    assert variance < 100  # Should be consistent


@pytest.mark.asyncio
async def test_validation_report_formatting():
    """Test that validation results format nicely for display."""
    code = """
from omnibase_core.nodes.node_effect import NodeEffect

class NodeTestEffect(NodeEffect):
    def __init__(self, container):
        pass  # Missing super().__init__

    async def execute_effect(self):
        eval("dangerous")  # Security issue
        return {}
"""

    contract = ModelEnhancedContract(
        name="NodeTestEffect",
        version=ModelVersionInfo(),
        node_type="effect",
        description="Test",
    )

    validator = NodeValidator()
    results = await validator.validate_generated_node(code, contract)

    # Generate report
    report_lines = []
    for result in results:
        result_str = str(result)
        report_lines.append(result_str)

    report = "\n".join(report_lines)

    # Report should be readable
    assert "PASSED" in report or "FAILED" in report
    assert "ms" in report  # Execution times
    assert len(report) > 100  # Should have substantial content
