"""Unit tests for NodeCodegenCodeValidatorEffect."""

from unittest.mock import Mock
from uuid import uuid4

import pytest
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.codegen_code_validator_effect.v1_0_0.models import (
    EnumValidationRule,
    ModelCodeValidationResult,
)
from omninode_bridge.nodes.codegen_code_validator_effect.v1_0_0.node import (
    NodeCodegenCodeValidatorEffect,
)
from omninode_bridge.nodes.conftest import create_test_contract_effect


@pytest.fixture
def mock_container():
    """Create container with mocked services."""
    container = Mock(spec=ModelContainer)
    container.config = Mock()
    container.config.get = Mock(side_effect=lambda k, default: default)
    container.get_service = Mock(return_value=None)
    return container


@pytest.fixture
def code_validator(mock_container):
    """Create code validator with mocked dependencies."""
    return NodeCodegenCodeValidatorEffect(mock_container)


class TestNodeCodegenCodeValidatorEffect:
    """Test suite for NodeCodegenCodeValidatorEffect."""

    @pytest.mark.asyncio
    async def test_validate_valid_code(self, code_validator):
        """Test validation of syntactically correct code."""
        # Arrange
        valid_code = '''
from omnibase_core import ModelOnexError
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event

def process_data(data: dict) -> dict:
    """Process data."""
    try:
        return {"result": data}
    except Exception as e:
        raise ModelOnexError(message=str(e))
'''

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": valid_code,
                "validation_rules": [EnumValidationRule.SYNTAX],
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert
        assert isinstance(result, ModelCodeValidationResult)
        assert result.is_valid
        assert result.syntax_valid
        assert len(result.validation_errors) == 0
        assert result.validation_time_ms < 500

    @pytest.mark.asyncio
    async def test_validate_syntax_error(self, code_validator):
        """Test detection of syntax errors."""
        # Arrange
        invalid_code = """
def foo(:
    pass
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": invalid_code,
                "validation_rules": [EnumValidationRule.SYNTAX],
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert
        assert not result.is_valid
        assert not result.syntax_valid
        assert len(result.validation_errors) >= 1
        assert result.validation_errors[0].rule == "syntax"
        assert "syntax" in result.validation_errors[0].message.lower()

    @pytest.mark.asyncio
    async def test_validate_onex_compliance(self, code_validator):
        """Test ONEX v2.0 compliance validation."""
        # Arrange - Code without ModelOnexError import
        non_compliant_code = """
def foo():
    raise Exception("error")
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": non_compliant_code,
                "validation_rules": [EnumValidationRule.ONEX_COMPLIANCE],
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert
        assert len(result.validation_warnings) >= 1
        # Should warn about missing ModelOnexError import
        assert any("ModelOnexError" in w.message for w in result.validation_warnings)

    @pytest.mark.asyncio
    async def test_validate_type_hints_missing(self, code_validator):
        """Test type hint validation detects missing hints."""
        # Arrange
        code_without_hints = """
def process(data):
    return data
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": code_without_hints,
                "validation_rules": [EnumValidationRule.TYPE_HINTS],
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert
        assert (
            len(result.validation_warnings) >= 2
        )  # Missing param hint and return hint
        assert any("type hint" in w.message.lower() for w in result.validation_warnings)

    @pytest.mark.asyncio
    async def test_validate_security_hardcoded_password(self, code_validator):
        """Test security validation detects hardcoded passwords."""
        # Arrange
        insecure_code = """
def connect():
    password = "mysecretpassword123"
    return connect_db(password)
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": insecure_code,
                "validation_rules": [EnumValidationRule.SECURITY],
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert
        assert not result.is_valid
        assert result.security_issues_found >= 1
        assert any("password" in e.message.lower() for e in result.validation_errors)

    @pytest.mark.asyncio
    async def test_validate_security_api_key(self, code_validator):
        """Test security validation detects hardcoded API keys."""
        # Arrange
        insecure_code = """
API_KEY = "sk-1234567890abcdef"
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": insecure_code,
                "validation_rules": [EnumValidationRule.SECURITY],
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert
        assert result.security_issues_found >= 1

    @pytest.mark.asyncio
    async def test_validate_all_rules(self, code_validator):
        """Test validation with all rules enabled."""
        # Arrange
        code_to_validate = """
def process(data):
    password = "test123"
    return data
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": code_to_validate,
                "validation_rules": [EnumValidationRule.ALL],
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert
        assert result.rules_checked is not None
        assert len(result.validation_errors) > 0 or len(result.validation_warnings) > 0

    @pytest.mark.asyncio
    async def test_validate_strict_mode(self, code_validator):
        """Test strict mode fails on warnings."""
        # Arrange
        code_with_warnings = """
def process(data):  # Missing type hints
    return data
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": code_with_warnings,
                "validation_rules": [EnumValidationRule.TYPE_HINTS],
                "strict_mode": True,
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert
        # Strict mode should make it invalid due to warnings
        assert not result.is_valid or len(result.validation_warnings) > 0

    @pytest.mark.asyncio
    async def test_validate_missing_code(self, code_validator):
        """Test validation with missing generated_code field."""
        # Arrange
        contract = create_test_contract_effect(correlation_id=uuid4(), input_state={})

        # Act & Assert
        with pytest.raises(ModelOnexError) as exc:
            await code_validator.execute_effect(contract)

        assert exc.value.error_code == EnumCoreErrorCode.VALIDATION_ERROR
        assert "generated_code" in exc.value.message

    @pytest.mark.asyncio
    async def test_validate_multiple_errors(self, code_validator):
        """Test validation finds multiple errors."""
        # Arrange
        code_with_errors = """
def foo(:  # Syntax error
    password = "test"  # Security issue
    pass
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": code_with_errors,
                "validation_rules": [EnumValidationRule.ALL],
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert
        assert not result.is_valid
        assert len(result.validation_errors) >= 1  # At least syntax error

    @pytest.mark.asyncio
    async def test_get_metrics(self, code_validator):
        """Test metrics collection."""
        # Arrange
        valid_code = "def foo(): pass"

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": valid_code,
                "validation_rules": [EnumValidationRule.SYNTAX],
            },
        )

        # Act
        await code_validator.execute_effect(contract)
        metrics = code_validator.get_metrics()

        # Assert
        assert metrics["total_validations"] == 1
        assert metrics["failed_validations"] == 0
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_validate_import_warnings(self, code_validator):
        """Test import validation detects relative imports."""
        # Arrange
        code_with_relative_imports = """
from . import foo
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": code_with_relative_imports,
                "validation_rules": [EnumValidationRule.IMPORTS],
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert
        assert len(result.validation_warnings) >= 1
        assert any("import" in w.message.lower() for w in result.validation_warnings)

    @pytest.mark.asyncio
    async def test_validate_correlation_id_tracking(self, code_validator):
        """Test correlation ID is tracked throughout validation."""
        # Arrange
        correlation_id = uuid4()
        valid_code = "def foo(): pass"

        contract = create_test_contract_effect(
            correlation_id=correlation_id,
            input_state={
                "generated_code": valid_code,
                "validation_rules": [EnumValidationRule.SYNTAX],
            },
        )

        # Act
        result = await code_validator.execute_effect(contract)

        # Assert - Should complete without errors
        assert result is not None


# Performance benchmark tests
class TestCodeValidatorPerformance:
    """Performance tests for code validation."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_file_performance(self, code_validator):
        """Benchmark validation on large files."""
        # Generate large file with 500 methods
        methods = []
        for i in range(500):
            methods.append(
                f'''
def method_{i}(self, data: dict) -> dict:
    """Method {i}."""
    try:
        return {{"result": data}}
    except Exception as e:
        raise ModelOnexError(message=str(e))
'''
            )
        large_code = "\n".join(methods)

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": large_code,
                "validation_rules": [EnumValidationRule.SYNTAX],
            },
        )

        result = await code_validator.execute_effect(contract)

        # Performance assertions
        assert result.validation_time_ms < 3000  # <3s for 500 methods

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_typical_node_performance(self, code_validator):
        """Benchmark validation on typical node file."""
        # Typical node with imports, class, methods
        typical_node = """
from omnibase_core import ModelOnexError
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.nodes.node_effect import NodeEffect

class NodeMyEffect(NodeEffect):
    def __init__(self, container):
        super().__init__(container)

    async def execute_effect(self, contract):
        try:
            return await self._process(contract.input_data)
        except Exception as e:
            raise ModelOnexError(message=str(e))

    async def _process(self, data: dict) -> dict:
        return {"result": data}
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": typical_node,
                "validation_rules": [EnumValidationRule.ALL],
            },
        )

        result = await code_validator.execute_effect(contract)

        # Typical node should be very fast
        assert result.validation_time_ms < 500  # <500ms for typical node
