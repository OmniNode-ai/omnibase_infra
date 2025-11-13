"""Unit tests for NodeCodegenStubExtractorEffect."""

from unittest.mock import Mock
from uuid import uuid4

import pytest
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.codegen_stub_extractor_effect.v1_0_0.models import (
    ModelStubExtractionResult,
)
from omninode_bridge.nodes.codegen_stub_extractor_effect.v1_0_0.node import (
    NodeCodegenStubExtractorEffect,
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
def stub_extractor(mock_container):
    """Create stub extractor with mocked dependencies."""
    return NodeCodegenStubExtractorEffect(mock_container)


class TestNodeCodegenStubExtractorEffect:
    """Test suite for NodeCodegenStubExtractorEffect."""

    @pytest.mark.asyncio
    async def test_extract_stubs_success(self, stub_extractor):
        """Test successful stub extraction."""
        # Arrange
        node_file_content = '''
def foo():
    """Do something."""
    # IMPLEMENTATION REQUIRED
    pass  # Stub

def bar():
    """Already implemented."""
    return 42
'''

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "node_file_content": node_file_content,
                "extraction_patterns": ["# IMPLEMENTATION REQUIRED"],
            },
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert
        assert isinstance(result, ModelStubExtractionResult)
        assert result.total_stubs_found == 1
        assert len(result.stubs) == 1
        assert result.stubs[0].name == "foo"
        assert result.extraction_time_ms < 100  # Performance check
        assert result.file_lines == 10

    @pytest.mark.asyncio
    async def test_extract_multiple_stubs(self, stub_extractor):
        """Test extraction of multiple stubs."""
        # Arrange
        node_file_content = """
class MyNode:
    def method1(self):
        # IMPLEMENTATION REQUIRED
        pass

    def method2(self):
        # IMPLEMENTATION REQUIRED
        pass

    def method3(self):
        return "implemented"
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": node_file_content}
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert
        assert result.total_stubs_found == 2
        assert len(result.stubs) == 2
        assert result.stubs[0].name == "method1"
        assert result.stubs[1].name == "method2"
        assert result.stubs[0].context == "class MyNode"

    @pytest.mark.asyncio
    async def test_extract_async_stubs(self, stub_extractor):
        """Test extraction of async method stubs."""
        # Arrange
        node_file_content = '''
async def execute_effect(self, contract):
    """Execute effect."""
    # IMPLEMENTATION REQUIRED
    pass
'''

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": node_file_content}
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert
        assert result.total_stubs_found == 1
        assert "async def execute_effect" in result.stubs[0].signature

    @pytest.mark.asyncio
    async def test_extract_stubs_with_type_hints(self, stub_extractor):
        """Test extraction preserves type hints."""
        # Arrange
        node_file_content = '''
async def process(self, data: dict[str, Any]) -> ModelResult:
    """Process data."""
    # IMPLEMENTATION REQUIRED
    pass
'''

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": node_file_content}
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert
        assert result.total_stubs_found == 1
        assert "dict[str, Any]" in result.stubs[0].signature
        assert "ModelResult" in result.stubs[0].signature

    @pytest.mark.asyncio
    async def test_extract_stubs_empty_file(self, stub_extractor):
        """Test extraction from file with no stubs."""
        # Arrange
        node_file_content = """
def foo():
    return 42

def bar():
    return "hello"
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": node_file_content}
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert
        assert result.total_stubs_found == 0
        assert len(result.stubs) == 0

    @pytest.mark.asyncio
    async def test_extract_stubs_invalid_syntax(self, stub_extractor):
        """Test extraction from file with syntax errors."""
        # Arrange
        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"node_file_content": "def foo(: invalid"},
        )

        # Act & Assert
        with pytest.raises(ModelOnexError) as exc:
            await stub_extractor.execute_effect(contract)

        assert exc.value.error_code == EnumCoreErrorCode.VALIDATION_ERROR
        assert "syntax error" in exc.value.message.lower()

    @pytest.mark.asyncio
    async def test_extract_stubs_missing_content(self, stub_extractor):
        """Test extraction with missing node_file_content."""
        # Arrange
        contract = create_test_contract_effect(correlation_id=uuid4(), input_state={})

        # Act & Assert
        with pytest.raises(ModelOnexError) as exc:
            await stub_extractor.execute_effect(contract)

        assert exc.value.error_code == EnumCoreErrorCode.VALIDATION_ERROR
        assert "node_file_content" in exc.value.message

    @pytest.mark.asyncio
    async def test_extract_stubs_custom_patterns(self, stub_extractor):
        """Test extraction with custom stub patterns."""
        # Arrange
        node_file_content = """
def foo():
    # TODO: implement
    pass
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "node_file_content": node_file_content,
                "extraction_patterns": ["# TODO: implement"],
            },
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert
        assert result.total_stubs_found == 1
        assert result.stubs[0].stub_marker == "# TODO: implement"

    @pytest.mark.asyncio
    async def test_extract_stubs_with_docstrings(self, stub_extractor):
        """Test extraction preserves docstrings."""
        # Arrange
        node_file_content = '''
def process_data(self):
    """
    Process the data.

    Returns:
        dict: Processed data
    """
    # IMPLEMENTATION REQUIRED
    pass
'''

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": node_file_content}
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert
        assert result.total_stubs_found == 1
        assert "Process the data" in result.stubs[0].docstring
        assert "Returns:" in result.stubs[0].docstring

    @pytest.mark.asyncio
    async def test_extract_stubs_performance(self, stub_extractor):
        """Test extraction performance with large file."""
        # Arrange - Generate large file with 100 stub methods
        methods = []
        for i in range(100):
            methods.append(
                f'''
def method_{i}(self):
    """Method {i}."""
    # IMPLEMENTATION REQUIRED
    pass
'''
            )
        large_file = "\n".join(methods)

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": large_file}
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert
        assert result.total_stubs_found == 100
        assert result.extraction_time_ms < 1000  # <1s for 100 stubs

    @pytest.mark.asyncio
    async def test_get_metrics(self, stub_extractor):
        """Test metrics collection."""
        # Arrange
        node_file_content = """
def foo():
    # IMPLEMENTATION REQUIRED
    pass

def bar():
    # IMPLEMENTATION REQUIRED
    pass
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": node_file_content}
        )

        # Act
        await stub_extractor.execute_effect(contract)
        metrics = stub_extractor.get_metrics()

        # Assert
        assert metrics["total_extractions"] == 1
        assert metrics["total_stubs_found"] == 2
        assert metrics["failed_extractions"] == 0
        assert metrics["success_rate"] == 1.0
        assert metrics["avg_stubs_per_file"] == 2.0

    @pytest.mark.asyncio
    async def test_extract_stubs_with_pass_stub_marker(self, stub_extractor):
        """Test extraction with 'pass # Stub' marker."""
        # Arrange
        node_file_content = """
def execute():
    pass  # Stub
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": node_file_content}
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert
        assert result.total_stubs_found == 1
        assert result.stubs[0].name == "execute"

    @pytest.mark.asyncio
    async def test_extract_stubs_line_numbers(self, stub_extractor):
        """Test that line numbers are correctly extracted."""
        # Arrange
        node_file_content = """# Line 1
# Line 2
def foo():  # Line 3
    # IMPLEMENTATION REQUIRED
    pass

def bar():  # Line 7
    # IMPLEMENTATION REQUIRED
    pass
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": node_file_content}
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert
        assert result.stubs[0].line_number == 3
        assert result.stubs[1].line_number == 7

    @pytest.mark.asyncio
    async def test_extract_stubs_correlation_id_tracking(self, stub_extractor):
        """Test correlation ID is tracked throughout extraction."""
        # Arrange
        correlation_id = uuid4()
        node_file_content = "def foo(): # IMPLEMENTATION REQUIRED\n    pass"

        contract = create_test_contract_effect(
            correlation_id=correlation_id,
            input_state={"node_file_content": node_file_content},
        )

        # Act
        result = await stub_extractor.execute_effect(contract)

        # Assert - Should complete without errors
        assert result is not None


# Performance benchmark tests
class TestStubExtractorPerformance:
    """Performance tests for stub extraction."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_file_performance(self, stub_extractor):
        """Benchmark extraction on large files."""
        # Generate 1000 methods
        methods = [
            f"def method_{i}(): # IMPLEMENTATION REQUIRED\n    pass\n"
            for i in range(1000)
        ]
        large_file = "\n".join(methods)

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": large_file}
        )

        result = await stub_extractor.execute_effect(contract)

        # Performance assertions
        assert result.total_stubs_found == 1000
        assert result.extraction_time_ms < 15000  # <15s for 1000 methods

        # Calculate throughput
        methods_per_second = 1000 / (result.extraction_time_ms / 1000)
        assert methods_per_second > 60  # >60 methods/sec

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_typical_node_performance(self, stub_extractor):
        """Benchmark extraction on typical node file."""
        # Typical node with 5-10 methods, 2-3 stubs
        node_file = """
class NodeMyEffect(NodeEffect):
    def __init__(self, container):
        super().__init__(container)

    async def execute_effect(self, contract):
        # IMPLEMENTATION REQUIRED
        pass

    async def _process_data(self, data):
        # IMPLEMENTATION REQUIRED
        pass

    def _validate_input(self, input_data):
        return input_data is not None

    async def _store_result(self, result):
        # IMPLEMENTATION REQUIRED
        pass
"""

        contract = create_test_contract_effect(
            correlation_id=uuid4(), input_state={"node_file_content": node_file}
        )

        result = await stub_extractor.execute_effect(contract)

        # Typical node should be very fast
        assert result.extraction_time_ms < 50  # <50ms for typical node
        assert result.total_stubs_found == 3
