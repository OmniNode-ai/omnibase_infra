#!/usr/bin/env python3
"""
Unit tests for NodeValidator.

Tests each validation stage independently with valid and invalid code samples.
"""

import pytest

from omninode_bridge.codegen.models_contract import (
    ModelEnhancedContract,
    ModelMixinDeclaration,
    ModelVersionInfo,
)
from omninode_bridge.codegen.validation import EnumValidationStage, NodeValidator

# ===== Test Fixtures =====


@pytest.fixture
def valid_node_code() -> str:
    """Valid ONEX v2.0 node code."""
    return '''
"""Valid ONEX v2.0 Effect Node."""

from omnibase_core.models import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck


class NodeTestEffect(NodeEffect, MixinHealthCheck):
    """Test effect node with health check mixin."""

    def __init__(self, container: ModelContainer):
        """Initialize node."""
        super().__init__(container)
        self.service_name = "test_service"

    async def initialize(self) -> None:
        """Initialize node resources."""
        await super().initialize()
        # Custom initialization here

    async def shutdown(self) -> None:
        """Shutdown node resources."""
        await super().shutdown()
        # Custom shutdown here

    async def execute_effect(self, **kwargs) -> dict:
        """Execute effect operation."""
        return {"status": "success"}
'''


@pytest.fixture
def syntax_error_code() -> str:
    """Code with syntax error."""
    return '''
def broken_function(
    """Missing closing parenthesis."""
    pass
'''


@pytest.fixture
def missing_methods_code() -> str:
    """Code with missing required methods."""
    return '''
from omnibase_core.models import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeTestEffect(NodeEffect):
    """Node missing required methods."""

    def __init__(self, container: ModelContainer):
        """Initialize node."""
        super().__init__(container)

    # Missing: initialize, shutdown, execute_effect
'''


@pytest.fixture
def security_issue_code() -> str:
    """Code with security issues."""
    return '''
from omnibase_core.nodes.node_effect import NodeEffect
import os


class NodeTestEffect(NodeEffect):
    """Node with security issues."""

    def __init__(self, container):
        super().__init__(container)
        self.api_key = "hardcoded_secret_key_12345"  # Security issue

    async def execute_effect(self, command: str):
        """Execute with security issues."""
        # Dangerous: eval
        result = eval(command)

        # Dangerous: os.system
        os.system(f"echo {command}")

        return result
'''


@pytest.fixture
def missing_super_calls_code() -> str:
    """Code missing super() calls."""
    return '''
from omnibase_core.models import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeTestEffect(NodeEffect):
    """Node missing super() calls."""

    def __init__(self, container: ModelContainer):
        """Initialize without super().__init__."""
        self.service_name = "test"
        # Missing: super().__init__(container)

    async def initialize(self) -> None:
        """Initialize without super().initialize()."""
        pass  # Missing: await super().initialize()

    async def execute_effect(self):
        """Execute effect."""
        return {}
'''


@pytest.fixture
def missing_mixin_code() -> str:
    """Code missing declared mixin."""
    return '''
from omnibase_core.models import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect
# Missing: from omnibase_core.mixins.mixin_health_check import MixinHealthCheck


class NodeTestEffect(NodeEffect):
    """Node declaring mixin but not inheriting it."""

    def __init__(self, container: ModelContainer):
        super().__init__(container)

    async def initialize(self) -> None:
        await super().initialize()

    async def execute_effect(self):
        return {}
'''


@pytest.fixture
def sample_contract() -> ModelEnhancedContract:
    """Sample contract for testing."""
    return ModelEnhancedContract(
        name="NodeTestEffect",
        version=ModelVersionInfo(major=1, minor=0, patch=0),
        node_type="effect",
        description="Test effect node",
        schema_version="v2.0.0",
        mixins=[
            ModelMixinDeclaration(
                name="MixinHealthCheck",
                enabled=True,
                import_path="omnibase_core.mixins.mixin_health_check",
            )
        ],
    )


@pytest.fixture
def validator() -> NodeValidator:
    """Validator instance with default settings."""
    return NodeValidator(
        enable_type_checking=False,  # Disable for faster tests
        enable_security_scan=True,
    )


# ===== Syntax Validation Tests =====


@pytest.mark.asyncio
async def test_validate_syntax_valid_code(
    validator: NodeValidator, valid_node_code: str
):
    """Test syntax validation with valid code."""
    result = await validator._validate_syntax(valid_node_code)

    assert result.stage == EnumValidationStage.SYNTAX
    assert result.passed is True
    assert len(result.errors) == 0
    assert result.execution_time_ms < 10  # Should be fast (<10ms)


@pytest.mark.asyncio
async def test_validate_syntax_invalid_code(
    validator: NodeValidator, syntax_error_code: str
):
    """Test syntax validation with syntax error."""
    result = await validator._validate_syntax(syntax_error_code)

    assert result.stage == EnumValidationStage.SYNTAX
    assert result.passed is False
    assert len(result.errors) > 0
    assert any(
        "syntax" in err.lower() or "expected" in err.lower() for err in result.errors
    )


@pytest.mark.asyncio
async def test_validate_syntax_indentation_error(validator: NodeValidator):
    """Test syntax validation with indentation error."""
    code = """
def test():
pass  # Wrong indentation
"""
    result = await validator._validate_syntax(code)

    assert result.passed is False
    assert len(result.errors) > 0


# ===== AST Validation Tests =====


@pytest.mark.asyncio
async def test_validate_ast_valid_code(
    validator: NodeValidator,
    valid_node_code: str,
    sample_contract: ModelEnhancedContract,
):
    """Test AST validation with valid code."""
    result = await validator._validate_ast(valid_node_code, sample_contract)

    assert result.stage == EnumValidationStage.AST
    assert result.passed is True
    assert len(result.errors) == 0
    assert result.execution_time_ms < 20  # Should be fast (<20ms)


@pytest.mark.asyncio
async def test_validate_ast_missing_methods(
    validator: NodeValidator,
    missing_methods_code: str,
    sample_contract: ModelEnhancedContract,
):
    """Test AST validation with missing methods."""
    result = await validator._validate_ast(missing_methods_code, sample_contract)

    assert result.stage == EnumValidationStage.AST
    assert result.passed is False
    assert len(result.errors) >= 2  # Missing initialize and execute_effect
    assert any("initialize" in err for err in result.errors)
    assert any("execute_effect" in err for err in result.errors)


@pytest.mark.asyncio
async def test_validate_ast_non_async_initialize(
    validator: NodeValidator, sample_contract: ModelEnhancedContract
):
    """Test AST validation with non-async initialize method."""
    code = """
from omnibase_core.nodes.node_effect import NodeEffect

class NodeTestEffect(NodeEffect):
    def __init__(self, container):
        super().__init__(container)

    def initialize(self):  # Should be async
        pass

    async def execute_effect(self):
        return {}
"""
    result = await validator._validate_ast(code, sample_contract)

    assert result.passed is False
    assert any(
        "async" in err.lower() and "initialize" in err.lower() for err in result.errors
    )


# ===== Import Validation Tests =====


@pytest.mark.asyncio
async def test_validate_imports_valid_code(
    validator: NodeValidator, valid_node_code: str
):
    """Test import validation with valid imports."""
    result = await validator._validate_imports(valid_node_code)

    assert result.stage == EnumValidationStage.IMPORTS
    assert result.passed is True
    assert result.execution_time_ms < 50  # Should be reasonably fast


@pytest.mark.asyncio
async def test_validate_imports_omnibase_core_allowed(validator: NodeValidator):
    """Test that omnibase_core imports are allowed even if not installed."""
    code = """
from omnibase_core.models import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
"""
    result = await validator._validate_imports(code)

    # Should pass even if omnibase_core not installed
    assert result.passed is True


@pytest.mark.asyncio
async def test_validate_imports_standard_library(validator: NodeValidator):
    """Test import validation with standard library imports."""
    code = """
import os
import sys
from pathlib import Path
from typing import Optional
"""
    result = await validator._validate_imports(code)

    assert result.passed is True
    assert len(result.errors) == 0


# ===== ONEX Compliance Tests =====


@pytest.mark.asyncio
async def test_validate_onex_compliance_valid_code(
    validator: NodeValidator,
    valid_node_code: str,
    sample_contract: ModelEnhancedContract,
):
    """Test ONEX compliance with valid code."""
    result = await validator._validate_onex_compliance(valid_node_code, sample_contract)

    assert result.stage == EnumValidationStage.ONEX_COMPLIANCE
    assert result.passed is True
    assert len(result.errors) == 0


@pytest.mark.asyncio
async def test_validate_onex_compliance_missing_super_init(
    validator: NodeValidator,
    missing_super_calls_code: str,
    sample_contract: ModelEnhancedContract,
):
    """Test ONEX compliance with missing super().__init__."""
    result = await validator._validate_onex_compliance(
        missing_super_calls_code, sample_contract
    )

    assert result.passed is False
    assert any("super().__init__" in err for err in result.errors)


@pytest.mark.asyncio
async def test_validate_onex_compliance_missing_mixin(
    validator: NodeValidator,
    missing_mixin_code: str,
    sample_contract: ModelEnhancedContract,
):
    """Test ONEX compliance with missing mixin."""
    result = await validator._validate_onex_compliance(
        missing_mixin_code, sample_contract
    )

    assert result.passed is False
    assert any("MixinHealthCheck" in err for err in result.errors)


@pytest.mark.asyncio
async def test_validate_onex_compliance_no_mixins(validator: NodeValidator):
    """Test ONEX compliance with no mixins (should pass)."""
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
        mixins=[],  # No mixins declared
    )

    result = await validator._validate_onex_compliance(code, contract)

    # Should pass if no mixins declared
    assert result.passed is True


# ===== Security Validation Tests =====


@pytest.mark.asyncio
async def test_validate_security_valid_code(
    validator: NodeValidator, valid_node_code: str
):
    """Test security validation with safe code."""
    result = await validator._validate_security(valid_node_code)

    assert result.stage == EnumValidationStage.SECURITY
    assert result.passed is True
    assert len(result.errors) == 0


@pytest.mark.asyncio
async def test_validate_security_dangerous_patterns(
    validator: NodeValidator, security_issue_code: str
):
    """Test security validation with dangerous patterns."""
    result = await validator._validate_security(security_issue_code)

    assert result.stage == EnumValidationStage.SECURITY
    assert result.passed is False
    assert len(result.errors) >= 3  # eval, os.system, hardcoded secret

    error_text = " ".join(result.errors).lower()
    assert "eval" in error_text
    assert "system" in error_text or "os.system" in error_text
    assert (
        "secret" in error_text or "api_key" in error_text or "hardcoded" in error_text
    )


@pytest.mark.asyncio
async def test_validate_security_exec_pattern(validator: NodeValidator):
    """Test security validation detects exec()."""
    code = """
def dangerous():
    exec("print('hello')")  # Dangerous
"""
    result = await validator._validate_security(code)

    assert result.passed is False
    assert any("exec" in err.lower() for err in result.errors)


@pytest.mark.asyncio
async def test_validate_security_subprocess_shell(validator: NodeValidator):
    """Test security validation detects subprocess with shell=True."""
    code = """
import subprocess

def risky_command(cmd):
    subprocess.run(cmd, shell=True)  # Risky
"""
    result = await validator._validate_security(code)

    assert result.passed is False
    assert any("shell" in err.lower() for err in result.errors)


@pytest.mark.asyncio
async def test_validate_security_pickle_warning(validator: NodeValidator):
    """Test security validation warns about pickle usage."""
    code = """
import pickle

def serialize(data):
    return pickle.dumps(data)
"""
    result = await validator._validate_security(code)

    # pickle is suspicious, not critical
    assert len(result.warnings) > 0
    assert any("pickle" in warn.lower() for warn in result.warnings)


# ===== Integration Tests =====


@pytest.mark.asyncio
async def test_validate_generated_node_valid_code(
    validator: NodeValidator,
    valid_node_code: str,
    sample_contract: ModelEnhancedContract,
):
    """Test full validation pipeline with valid code."""
    results = await validator.validate_generated_node(valid_node_code, sample_contract)

    # Should have results for all enabled stages
    assert len(results) >= 5  # syntax, ast, imports, onex, security

    # All stages should pass
    assert all(r.passed for r in results)

    # Check performance
    total_time = sum(r.execution_time_ms for r in results)
    assert total_time < 200  # Should be fast without type checking


@pytest.mark.asyncio
async def test_validate_generated_node_syntax_error_stops_pipeline(
    validator: NodeValidator,
    syntax_error_code: str,
    sample_contract: ModelEnhancedContract,
):
    """Test that syntax error stops further AST-based validation."""
    results = await validator.validate_generated_node(
        syntax_error_code, sample_contract
    )

    # Should only have syntax result (pipeline stops on syntax error)
    assert len(results) == 1
    assert results[0].stage == EnumValidationStage.SYNTAX
    assert results[0].passed is False


@pytest.mark.asyncio
async def test_validate_generated_node_multiple_errors(
    validator: NodeValidator, sample_contract: ModelEnhancedContract
):
    """Test validation with multiple error types."""
    code = """
from omnibase_core.nodes.node_effect import NodeEffect

class NodeTestEffect(NodeEffect):
    def __init__(self, container):
        # Missing super().__init__
        self.password = "hardcoded123"  # Security issue

    # Missing initialize
    # Missing execute_effect

    def dangerous_method(self, cmd):
        eval(cmd)  # Security issue
"""

    results = await validator.validate_generated_node(code, sample_contract)

    # Should have multiple failed stages
    failed_results = [r for r in results if not r.passed]
    assert len(failed_results) >= 2

    # Check specific failures
    stage_names = {r.stage for r in failed_results}
    assert EnumValidationStage.AST in stage_names  # Missing methods
    assert EnumValidationStage.SECURITY in stage_names  # Security issues


# ===== Performance Tests =====


@pytest.mark.asyncio
async def test_validation_performance_targets(
    validator: NodeValidator,
    valid_node_code: str,
    sample_contract: ModelEnhancedContract,
):
    """Test that validation meets performance targets."""
    results = await validator.validate_generated_node(valid_node_code, sample_contract)

    # Check individual stage performance
    for result in results:
        if result.stage == EnumValidationStage.SYNTAX:
            assert result.execution_time_ms < 10
        elif result.stage == EnumValidationStage.AST:
            assert result.execution_time_ms < 20
        elif result.stage == EnumValidationStage.IMPORTS:
            assert result.execution_time_ms < 50
        elif (
            result.stage == EnumValidationStage.ONEX_COMPLIANCE
            or result.stage == EnumValidationStage.SECURITY
        ):
            assert result.execution_time_ms < 100

    # Total time without type checking should be fast
    total_time = sum(r.execution_time_ms for r in results)
    assert total_time < 200


@pytest.mark.asyncio
async def test_validation_result_str_formatting(valid_node_code: str):
    """Test ModelValidationResult string formatting."""
    validator = NodeValidator()
    results = await validator.validate_generated_node(
        valid_node_code,
        ModelEnhancedContract(
            name="NodeTestEffect",
            version=ModelVersionInfo(),
            node_type="effect",
            description="Test",
        ),
    )

    # Test string representation
    for result in results:
        result_str = str(result)
        assert result.stage.value in result_str
        assert "PASSED" in result_str or "FAILED" in result_str
        assert "ms" in result_str  # Execution time


# ===== Edge Cases =====


@pytest.mark.asyncio
async def test_validate_empty_code(
    validator: NodeValidator, sample_contract: ModelEnhancedContract
):
    """Test validation with empty code."""
    results = await validator.validate_generated_node("", sample_contract)

    # Syntax should pass (empty code is valid Python)
    # AST should fail (no class found)
    ast_result = next((r for r in results if r.stage == EnumValidationStage.AST), None)
    assert ast_result is not None
    assert ast_result.passed is False


@pytest.mark.asyncio
async def test_validate_minimal_valid_code(validator: NodeValidator):
    """Test validation with minimal valid code."""
    code = """
from omnibase_core.nodes.node_effect import NodeEffect

class NodeMinimalEffect(NodeEffect):
    def __init__(self, container):
        super().__init__(container)

    async def initialize(self):
        await super().initialize()

    async def execute_effect(self):
        return {}
"""
    contract = ModelEnhancedContract(
        name="NodeMinimalEffect",
        version=ModelVersionInfo(),
        node_type="effect",
        description="Minimal node",
    )

    results = await validator.validate_generated_node(code, contract)

    # Should pass all stages
    assert all(r.passed for r in results)


@pytest.mark.asyncio
async def test_validator_with_type_checking_disabled():
    """Test validator with type checking disabled."""
    validator = NodeValidator(enable_type_checking=False)
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
    )

    results = await validator.validate_generated_node(code, contract)

    # Should not include type checking stage
    stage_names = {r.stage for r in results}
    assert EnumValidationStage.TYPE_CHECKING not in stage_names


@pytest.mark.asyncio
async def test_validator_with_security_scan_disabled():
    """Test validator with security scan disabled."""
    validator = NodeValidator(enable_security_scan=False)
    code = """
from omnibase_core.nodes.node_effect import NodeEffect

class NodeTestEffect(NodeEffect):
    def __init__(self, container):
        super().__init__(container)
        eval("dangerous")  # Would normally fail security

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
    )

    results = await validator.validate_generated_node(code, contract)

    # Should not include security stage
    stage_names = {r.stage for r in results}
    assert EnumValidationStage.SECURITY not in stage_names
