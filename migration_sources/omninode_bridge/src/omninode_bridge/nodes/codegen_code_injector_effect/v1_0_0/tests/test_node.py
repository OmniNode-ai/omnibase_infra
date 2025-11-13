"""Unit tests for NodeCodegenCodeInjectorEffect."""

from unittest.mock import Mock
from uuid import uuid4

import pytest
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.codegen_code_injector_effect.v1_0_0.models import (
    ModelCodeInjectionResult,
)
from omninode_bridge.nodes.codegen_code_injector_effect.v1_0_0.node import (
    NodeCodegenCodeInjectorEffect,
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
def code_injector(mock_container):
    """Create code injector with mocked dependencies."""
    return NodeCodegenCodeInjectorEffect(mock_container)


class TestNodeCodegenCodeInjectorEffect:
    """Test suite for NodeCodegenCodeInjectorEffect."""

    @pytest.mark.asyncio
    async def test_inject_single_method(self, code_injector):
        """Test injection into a single method stub."""
        # Arrange
        source_code = '''
def foo():
    """Do something."""
    # IMPLEMENTATION REQUIRED
    pass
'''

        injection_requests = [
            {
                "method_name": "foo",
                "line_number": 2,
                "generated_code": "    return 42",
                "preserve_signature": True,
                "preserve_docstring": True,
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": source_code,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert isinstance(result, ModelCodeInjectionResult)
        assert result.success
        assert result.injections_performed == 1
        assert "return 42" in result.modified_source
        assert '"""Do something."""' in result.modified_source
        assert result.injection_time_ms < 200

    @pytest.mark.asyncio
    async def test_inject_multiple_methods(self, code_injector):
        """Test injection into multiple methods."""
        # Arrange
        source_code = '''
class MyNode:
    def method1(self):
        """Method 1."""
        pass

    def method2(self):
        """Method 2."""
        pass
'''

        injection_requests = [
            {
                "method_name": "method1",
                "line_number": 3,
                "generated_code": "        return 'result1'",
            },
            {
                "method_name": "method2",
                "line_number": 7,
                "generated_code": "        return 'result2'",
            },
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": source_code,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert result.success
        assert result.injections_performed == 2
        assert "return 'result1'" in result.modified_source
        assert "return 'result2'" in result.modified_source
        assert len(result.methods_modified) == 2

    @pytest.mark.asyncio
    async def test_inject_async_method(self, code_injector):
        """Test injection into async method."""
        # Arrange
        source_code = '''
async def execute_effect(self, contract):
    """Execute the effect."""
    pass
'''

        injection_requests = [
            {
                "method_name": "execute_effect",
                "line_number": 2,
                "generated_code": "    result = await self._process(contract)\n    return result",
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": source_code,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert result.success
        assert "await self._process(contract)" in result.modified_source
        assert "async def execute_effect" in result.modified_source

    @pytest.mark.asyncio
    async def test_preserve_decorators(self, code_injector):
        """Test that decorators are preserved."""
        # Arrange
        source_code = '''
class MyNode:
    @property
    def my_property(self):
        """Property docstring."""
        pass
'''

        injection_requests = [
            {
                "method_name": "my_property",
                "line_number": 3,
                "generated_code": "        return self._value",
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": source_code,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert result.success
        assert "@property" in result.modified_source
        assert "return self._value" in result.modified_source

    @pytest.mark.asyncio
    async def test_preserve_signature_and_docstring(self, code_injector):
        """Test signature and docstring preservation."""
        # Arrange
        source_code = '''
def process(self, data: dict) -> dict:
    """
    Process the data.

    Args:
        data: Input data

    Returns:
        Processed data
    """
    pass
'''

        injection_requests = [
            {
                "method_name": "process",
                "line_number": 2,
                "generated_code": "    return {'result': data}",
                "preserve_signature": True,
                "preserve_docstring": True,
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": source_code,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert result.success
        assert "def process(self, data: dict) -> dict:" in result.modified_source
        assert "Process the data" in result.modified_source
        assert "Args:" in result.modified_source

    @pytest.mark.asyncio
    async def test_method_not_found_error(self, code_injector):
        """Test error when method not found."""
        # Arrange
        source_code = """
def foo():
    pass
"""

        injection_requests = [
            {
                "method_name": "bar",  # Method doesn't exist
                "line_number": 2,
                "generated_code": "    return 42",
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": source_code,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert not result.success
        assert result.injections_performed == 0
        assert len(result.injection_errors) == 1
        assert result.injection_errors[0].error_type == "method_not_found"
        assert "bar" in result.injection_errors[0].message

    @pytest.mark.asyncio
    async def test_syntax_error_in_source(self, code_injector):
        """Test handling of syntax errors in source code."""
        # Arrange
        source_code = "def foo(: invalid"  # Invalid syntax

        injection_requests = [
            {
                "method_name": "foo",
                "line_number": 1,
                "generated_code": "    return 42",
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": source_code,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert not result.success
        assert len(result.injection_errors) >= 1
        assert result.injection_errors[0].error_type == "syntax_error"

    @pytest.mark.asyncio
    async def test_missing_source_code(self, code_injector):
        """Test error when source_code is missing."""
        # Arrange
        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"injection_requests": []}
        )

        # Act & Assert
        with pytest.raises(ModelOnexError) as exc:
            await code_injector.execute_effect(contract)

        assert exc.value.error_code == EnumCoreErrorCode.VALIDATION_ERROR
        assert "source_code" in exc.value.message

    @pytest.mark.asyncio
    async def test_missing_injection_requests(self, code_injector):
        """Test error when injection_requests is missing."""
        # Arrange
        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"source_code": "def foo(): pass"}
        )

        # Act & Assert
        with pytest.raises(ModelOnexError) as exc:
            await code_injector.execute_effect(contract)

        assert exc.value.error_code == EnumCoreErrorCode.VALIDATION_ERROR
        assert "injection_requests" in exc.value.message

    @pytest.mark.asyncio
    async def test_indentation_preservation(self, code_injector):
        """Test that indentation is correctly preserved."""
        # Arrange
        source_code = '''
class MyClass:
    def method(self):
        """Method."""
        pass
'''

        injection_requests = [
            {
                "method_name": "method",
                "line_number": 3,
                "generated_code": "        result = self._compute()\n        return result",
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": source_code,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert result.success
        # Check indentation is correct (8 spaces for class method body)
        assert "        result = self._compute()" in result.modified_source

    @pytest.mark.asyncio
    async def test_get_metrics(self, code_injector):
        """Test metrics collection."""
        # Arrange
        source_code = "def foo(): pass"
        injection_requests = [
            {"method_name": "foo", "line_number": 1, "generated_code": "    return 42"}
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": source_code,
                "injection_requests": injection_requests,
            },
        )

        # Act
        await code_injector.execute_effect(contract)
        metrics = code_injector.get_metrics()

        # Assert
        assert metrics["total_injections"] == 1
        assert metrics["successful_injections"] == 1
        assert metrics["failed_injections"] == 0
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_file_lines_tracking(self, code_injector):
        """Test that file line counts are tracked correctly."""
        # Arrange
        source_code = """
def foo():
    pass

def bar():
    pass
"""

        injection_requests = [
            {
                "method_name": "foo",
                "line_number": 2,
                "generated_code": "    return 'foo result'",
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": source_code,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert result.file_lines_before > 0
        assert result.file_lines_after > 0
        # After injection might have different line count
        assert result.file_lines_after >= result.file_lines_before - 2


# Performance benchmark tests
class TestCodeInjectorPerformance:
    """Performance tests for code injection."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_file_performance(self, code_injector):
        """Benchmark injection on large files."""
        # Generate large file with 100 methods
        methods = []
        for i in range(100):
            methods.append(
                f'''
def method_{i}(self):
    """Method {i}."""
    pass
'''
            )
        large_file = "\n".join(methods)

        # Create injection requests for 10 methods
        injection_requests = [
            {
                "method_name": f"method_{i}",
                "line_number": i * 5 + 2,
                "generated_code": f"    return {i}",
            }
            for i in range(0, 10)
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": large_file,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert result.success
        assert result.injections_performed == 10
        assert result.injection_time_ms < 1000  # <1s for 10 injections in large file

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_typical_node_performance(self, code_injector):
        """Benchmark injection on typical node file."""
        # Typical node with 3 stub methods
        typical_node = '''
from omnibase_core import ModelOnexError
from omnibase_core.nodes.node_effect import NodeEffect

class NodeMyEffect(NodeEffect):
    def __init__(self, container):
        super().__init__(container)

    async def execute_effect(self, contract):
        """Execute the effect."""
        pass

    async def _process(self, data: dict) -> dict:
        """Process data."""
        pass

    def _validate(self, input_data: dict) -> bool:
        """Validate input."""
        pass
'''

        injection_requests = [
            {
                "method_name": "execute_effect",
                "line_number": 9,
                "generated_code": "        result = await self._process(contract.input_data)\n        return result",
            },
            {
                "method_name": "_process",
                "line_number": 13,
                "generated_code": "        return {'result': data}",
            },
            {
                "method_name": "_validate",
                "line_number": 17,
                "generated_code": "        return input_data is not None",
            },
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "source_code": typical_node,
                "injection_requests": injection_requests,
            },
        )

        # Act
        result = await code_injector.execute_effect(contract)

        # Assert
        assert result.success
        assert result.injections_performed == 3
        assert result.injection_time_ms < 200  # <200ms for typical node
