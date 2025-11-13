"""
Integration tests for node-based code generator workflow.

Tests the complete workflow using all 5 nodes:
1. NodeCodegenStubExtractorEffect - Extract method stubs
2. NodeCodegenCodeValidatorEffect - Validate generated code
3. NodeCodegenCodeInjectorEffect - Inject validated code
4. NodeCodegenStoreEffect - Persist artifacts
5. NodeCodegenMetricsReducer - Aggregate metrics
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from omnibase_core.enums.enum_node_type import EnumNodeType
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.contracts.model_contract_reducer import ModelContractReducer
from omnibase_core.models.contracts.model_io_operation_config import (
    ModelIOOperationConfig,
)
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.codegen_code_injector_effect.v1_0_0.node import (
    NodeCodegenCodeInjectorEffect,
)
from omninode_bridge.nodes.codegen_code_validator_effect.v1_0_0.models.enum_validation_rule import (
    EnumValidationRule,
)
from omninode_bridge.nodes.codegen_code_validator_effect.v1_0_0.node import (
    NodeCodegenCodeValidatorEffect,
)
from omninode_bridge.nodes.codegen_metrics_reducer.v1_0_0.node import (
    NodeCodegenMetricsReducer,
)
from omninode_bridge.nodes.codegen_store_effect.v1_0_0.node import (
    NodeCodegenStoreEffect,
)
from omninode_bridge.nodes.codegen_stub_extractor_effect.v1_0_0.node import (
    NodeCodegenStubExtractorEffect,
)

# Sample node file with stubs
SAMPLE_NODE_WITH_STUBS = '''
from omnibase_core import ModelOnexError
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect


class NodeMyEffect(NodeEffect):
    """My custom effect node."""

    def __init__(self, container):
        super().__init__(container)

    async def execute_effect(self, contract: ModelContractEffect):
        """Execute the effect."""
        # IMPLEMENTATION REQUIRED
        pass

    async def _process_data(self, data: dict) -> dict:
        """Process the data."""
        # IMPLEMENTATION REQUIRED
        pass

    def _validate_input(self, input_data: dict) -> bool:
        """Validate input data."""
        # IMPLEMENTATION REQUIRED
        pass
'''


# Generated implementation code
GENERATED_IMPLEMENTATIONS = {
    "execute_effect": """        result = await self._process_data(contract.input_state)
        if not self._validate_input(result):
            raise ModelOnexError(message="Validation failed")
        return result""",
    "_process_data": """        processed = {"result": data.get("value", 0) * 2}
        return processed""",
    "_validate_input": """        return input_data is not None and isinstance(input_data, dict)""",
}


def create_test_contract(correlation_id, input_state, node_name="test_node"):
    """
    Helper to create a properly initialized ModelContractEffect for testing.

    Provides all required fields with sensible defaults for test scenarios.
    """
    return ModelContractEffect(
        name=node_name,
        version={"major": 1, "minor": 0, "patch": 0},
        description=f"Test contract for {node_name}",
        node_type=EnumNodeType.EFFECT,
        input_model="dict",
        output_model="dict",
        io_operations=[ModelIOOperationConfig(operation_type="READ")],
        correlation_id=correlation_id,
        input_state=input_state,
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_container(temp_dir):
    """Create mock container with test configuration."""
    container = Mock(spec=ModelContainer)
    container.config = Mock()
    container.config.get = Mock(
        side_effect=lambda k, default: (
            temp_dir if k == "codegen_output_dir" else default
        )
    )
    container.get_service = Mock(return_value=None)
    return container


@pytest.fixture
def stub_extractor(mock_container):
    """Create stub extractor node."""
    return NodeCodegenStubExtractorEffect(mock_container)


@pytest.fixture
def code_validator(mock_container):
    """Create code validator node."""
    return NodeCodegenCodeValidatorEffect(mock_container)


@pytest.fixture
def code_injector(mock_container):
    """Create code injector node."""
    return NodeCodegenCodeInjectorEffect(mock_container)


@pytest.fixture
def store_effect(mock_container):
    """Create store effect node."""
    return NodeCodegenStoreEffect(mock_container)


@pytest.fixture
def metrics_reducer(mock_container):
    """Create metrics reducer node."""
    # Create mock kafka client
    mock_kafka_client = AsyncMock()
    mock_kafka_client.publish_event = AsyncMock()

    # Configure mock container
    def get_service_side_effect(service_name):
        if service_name == "kafka_client":
            return mock_kafka_client
        return None

    mock_container.get_service = Mock(side_effect=get_service_side_effect)
    mock_container.config.get = Mock(
        side_effect=lambda k, default: (
            False if k == "consul_enable_registration" else default
        )
    )
    return NodeCodegenMetricsReducer(mock_container)


class TestNodeBasedCodeGeneratorIntegration:
    """Integration tests for complete node-based code generator workflow."""

    @pytest.mark.asyncio
    async def test_complete_workflow_single_node(
        self,
        stub_extractor,
        code_validator,
        code_injector,
        store_effect,
        temp_dir,
    ):
        """Test complete workflow: extract → validate → inject → store."""
        correlation_id = uuid4()

        # Step 1: Extract stubs from generated node file
        extract_contract = create_test_contract(
            correlation_id=correlation_id,
            input_state={
                "node_file_content": SAMPLE_NODE_WITH_STUBS,
                "extraction_patterns": ["# IMPLEMENTATION REQUIRED"],
            },
            node_name="stub_extractor",
        )

        extraction_result = await stub_extractor.execute_effect(extract_contract)

        assert extraction_result.success
        assert extraction_result.total_stubs_found == 3
        assert len(extraction_result.stubs) == 3

        # Step 2: Validate generated implementations
        validation_results = []
        for stub in extraction_result.stubs:
            generated_code = GENERATED_IMPLEMENTATIONS.get(stub.name)
            if generated_code:
                validate_contract = create_test_contract(
                    correlation_id=correlation_id,
                    input_state={
                        "generated_code": generated_code,
                        "validation_rules": [
                            EnumValidationRule.SYNTAX,
                            EnumValidationRule.SECURITY,
                        ],
                    },
                    node_name="code_validator",
                )

                validation_result = await code_validator.execute_effect(
                    validate_contract
                )
                validation_results.append(validation_result)

        # All validations should pass
        assert all(r.is_valid for r in validation_results)

        # Step 3: Inject validated code back into node file
        injection_requests = [
            {
                "method_name": stub.name,
                "line_number": stub.line_number,
                "generated_code": GENERATED_IMPLEMENTATIONS[stub.name],
                "preserve_signature": True,
                "preserve_docstring": True,
            }
            for stub in extraction_result.stubs
            if stub.name in GENERATED_IMPLEMENTATIONS
        ]

        inject_contract = create_test_contract(
            correlation_id=correlation_id,
            input_state={
                "source_code": SAMPLE_NODE_WITH_STUBS,
                "injection_requests": injection_requests,
            },
            node_name="code_injector",
        )

        injection_result = await code_injector.execute_effect(inject_contract)

        assert injection_result.success
        assert injection_result.injections_performed == 3
        assert "# IMPLEMENTATION REQUIRED" not in injection_result.modified_source

        # Step 4: Store the modified node file
        storage_contract = create_test_contract(
            correlation_id=correlation_id,
            input_state={
                "storage_requests": [
                    {
                        "file_path": "my_node.py",
                        "content": injection_result.modified_source,
                        "artifact_type": "node_file",
                        "create_directories": True,
                    }
                ],
                "base_directory": temp_dir,
            },
            node_name="store_effect",
        )

        storage_result = await store_effect.execute_effect(storage_contract)

        assert storage_result.success
        assert storage_result.artifacts_stored == 1
        assert len(storage_result.stored_files) == 1

        # Verify file was created
        stored_file = Path(temp_dir) / "my_node.py"
        assert stored_file.exists()
        stored_content = stored_file.read_text()
        assert "# IMPLEMENTATION REQUIRED" not in stored_content
        assert "return processed" in stored_content  # From _process_data
        assert "isinstance(input_data, dict)" in stored_content  # From _validate_input

    @pytest.mark.asyncio
    async def test_workflow_with_validation_failure(
        self,
        stub_extractor,
        code_validator,
        temp_dir,
    ):
        """Test workflow handles validation failures gracefully."""
        correlation_id = uuid4()

        # Extract stubs
        extract_contract = create_test_contract(
            correlation_id=correlation_id,
            input_state={
                "node_file_content": SAMPLE_NODE_WITH_STUBS,
                "extraction_patterns": ["# IMPLEMENTATION REQUIRED"],
            },
        )

        extraction_result = await stub_extractor.execute_effect(extract_contract)
        assert extraction_result.success

        # Validate insecure code (should fail)
        insecure_code = """
        password = "hardcoded_password123"
        return {"auth": password}
        """

        validate_contract = create_test_contract(
            correlation_id=correlation_id,
            input_state={
                "generated_code": insecure_code,
                "validation_rules": [EnumValidationRule.SECURITY],
            },
        )

        validation_result = await code_validator.execute_effect(validate_contract)

        # Should fail validation due to hardcoded password
        assert not validation_result.is_valid
        assert validation_result.security_issues_found > 0

    @pytest.mark.asyncio
    async def test_workflow_performance_targets(
        self,
        stub_extractor,
        code_validator,
        code_injector,
        store_effect,
        temp_dir,
    ):
        """Test that complete workflow meets performance targets."""
        correlation_id = uuid4()

        # Extract stubs (<100ms target)
        extract_contract = create_test_contract(
            correlation_id=correlation_id,
            input_state={
                "node_file_content": SAMPLE_NODE_WITH_STUBS,
                "extraction_patterns": ["# IMPLEMENTATION REQUIRED"],
            },
        )
        extraction_result = await stub_extractor.execute_effect(extract_contract)
        assert extraction_result.extraction_time_ms < 100

        # Validate code (<500ms target)
        validate_contract = create_test_contract(
            correlation_id=correlation_id,
            input_state={
                "generated_code": GENERATED_IMPLEMENTATIONS["execute_effect"],
                "validation_rules": [EnumValidationRule.ALL],
            },
        )
        validation_result = await code_validator.execute_effect(validate_contract)
        assert validation_result.validation_time_ms < 500

        # Inject code (<200ms target)
        inject_contract = create_test_contract(
            correlation_id=correlation_id,
            input_state={
                "source_code": SAMPLE_NODE_WITH_STUBS,
                "injection_requests": [
                    {
                        "method_name": "execute_effect",
                        "line_number": 14,
                        "generated_code": GENERATED_IMPLEMENTATIONS["execute_effect"],
                    }
                ],
            },
        )
        injection_result = await code_injector.execute_effect(inject_contract)
        assert injection_result.injection_time_ms < 200

        # Store artifact (<1s target)
        storage_contract = create_test_contract(
            correlation_id=correlation_id,
            input_state={
                "storage_requests": [
                    {
                        "file_path": "my_node.py",
                        "content": injection_result.modified_source,
                    }
                ],
                "base_directory": temp_dir,
            },
        )
        storage_result = await store_effect.execute_effect(storage_contract)
        assert storage_result.storage_time_ms < 1000

    @pytest.mark.asyncio
    async def test_metrics_aggregation_integration(self, metrics_reducer):
        """Test metrics reducer aggregates workflow events."""
        from datetime import UTC, datetime

        from omninode_bridge.events.models.codegen_events import (
            ModelEventCodegenCompleted,
            ModelEventCodegenStageCompleted,
        )

        # Create workflow events
        workflow_id = uuid4()
        events = [
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                workflow_id=workflow_id,
                stage_name="extraction",
                stage_number=1,
                duration_seconds=0.075,
                success=True,
                timestamp=datetime.now(UTC),
            ),
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                workflow_id=workflow_id,
                stage_name="validation",
                stage_number=2,
                duration_seconds=0.420,
                success=True,
                timestamp=datetime.now(UTC),
            ),
            ModelEventCodegenStageCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                workflow_id=workflow_id,
                stage_name="injection",
                stage_number=3,
                duration_seconds=0.150,
                success=True,
                timestamp=datetime.now(UTC),
            ),
            ModelEventCodegenCompleted(
                correlation_id=uuid4(),
                event_id=uuid4(),
                workflow_id=workflow_id,
                total_duration_seconds=0.645,
                generated_files=["node.py", "test_node.py"],
                node_type="effect",
                service_name="test_service",
                quality_score=0.95,
                primary_model="gemini-2.5-flash",
                total_tokens=1500,
                total_cost_usd=0.05,
                contract_yaml="contract.yaml",
                node_module="node.py",
                models=[],
                enums=[],
                tests=["test_node.py"],
                timestamp=datetime.now(UTC),
            ),
        ]

        # Create async iterator for events
        async def event_stream():
            for event in events:
                yield event

        # Mock contract
        contract = Mock(spec=ModelContractReducer)
        contract.correlation_id = uuid4()
        contract.input_stream = event_stream()
        contract.input_state = {}

        # Mock publish_event_intent to avoid Kafka I/O
        from unittest.mock import patch

        with patch.object(metrics_reducer, "publish_event_intent", AsyncMock()):
            # Act
            metrics_state = await metrics_reducer.execute_reduction(contract)

            # Assert
            assert metrics_state.events_processed == len(events)
            assert metrics_state.aggregation_duration_ms < 100  # Should be fast
            assert metrics_state.items_per_second > 10  # Basic throughput check


class TestContractBasedTesting:
    """Demonstrate contract-based testing with mock vs real services."""

    @pytest.mark.asyncio
    async def test_contract_with_mock_container(self, stub_extractor):
        """Test using mock container (unit test)."""
        contract = create_test_contract(
            correlation_id=uuid4(),
            input_state={
                "node_file_content": "def foo(): pass",
                "extraction_patterns": ["pass"],
            },
        )

        result = await stub_extractor.execute_effect(contract)

        assert result is not None
        # Mock container works for unit testing

    @pytest.mark.asyncio
    async def test_same_contract_different_dependencies(self, temp_dir):
        """Show same contract works with different container configurations."""
        # Configuration 1: Mock container
        mock_container = Mock(spec=ModelContainer)
        mock_container.config = Mock()
        mock_container.config.get = Mock(side_effect=lambda k, default: default)

        node1 = NodeCodegenStubExtractorEffect(mock_container)

        # Configuration 2: Container with custom patterns
        custom_container = Mock(spec=ModelContainer)
        custom_container.config = Mock()

        def custom_config_get(k, default):
            if k == "stub_extraction_patterns":
                return ["# TODO", "# FIXME"]
            return default

        custom_container.config.get = Mock(side_effect=custom_config_get)

        node2 = NodeCodegenStubExtractorEffect(custom_container)

        # Same contract, different dependencies
        contract = create_test_contract(
            correlation_id=uuid4(),
            input_state={
                "node_file_content": "def foo(): # TODO implement\n    pass",
                "extraction_patterns": None,  # Use container defaults
            },
        )

        result1 = await node1.execute_effect(contract)
        result2 = await node2.execute_effect(contract)

        # Both work, potentially with different behavior based on container config
        assert result1 is not None
        assert result2 is not None
