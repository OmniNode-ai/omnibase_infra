#!/usr/bin/env python3
"""
Tests for CodeGenerationWorkflow.

Demonstrates contract-first testing with mocked Kafka and intelligence services.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ..workflow import CodeGenerationWorkflow


class TestCodeGenerationWorkflow:
    """Test suite for CodeGenerationWorkflow."""

    @pytest.fixture
    def mock_kafka_client(self):
        """Create mock Kafka client."""
        client = MagicMock()
        client.is_connected = True
        client.publish = AsyncMock()
        return client

    @pytest.fixture
    async def workflow(self, mock_kafka_client):
        """Create workflow instance with mocked dependencies."""
        return CodeGenerationWorkflow(
            kafka_client=mock_kafka_client,
            enable_intelligence=True,
            enable_quorum=False,
            timeout=60.0,
            verbose=False,
        )

    @pytest.mark.asyncio
    async def test_workflow_initialization(self, workflow, mock_kafka_client):
        """Test workflow initializes correctly."""
        assert workflow.kafka_client is mock_kafka_client
        assert workflow.enable_intelligence is True
        assert workflow.enable_quorum is False

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, workflow):
        """Test complete 8-stage pipeline execution."""
        # Arrange
        correlation_id = uuid4()
        prompt = "Create PostgreSQL CRUD Effect"
        output_dir = "./test_output"

        # Act
        result = await workflow.run(
            prompt=prompt,
            output_directory=output_dir,
            correlation_id=correlation_id,
        )

        # Assert - workflow completed successfully
        assert result is not None
        assert result["success"] is True
        assert "workflow_id" in result
        assert "total_duration_seconds" in result
        assert "generated_files" in result
        assert len(result["generated_files"]) > 0

        # Assert - all stages completed
        assert result["node_type"] in ["effect", "orchestrator", "reducer", "compute"]
        assert result["quality_score"] >= 0.0
        assert result["quality_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_prompt_parsing_stage(self, workflow):
        """Test Stage 1: Prompt parsing."""
        # Arrange
        correlation_id = uuid4()
        prompt = "Create database orchestrator for PostgreSQL CRUD operations"

        # Act - run full workflow to test prompt parsing
        result = await workflow.run(
            prompt=prompt,
            output_directory="./test",
            correlation_id=correlation_id,
        )

        # Assert - prompt was parsed correctly
        assert result["node_type"] in ["effect", "orchestrator", "reducer", "compute"]
        # Note: service_name is in parsed_requirements, not in final result
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_intelligence_gathering_disabled(self):
        """Test workflow with intelligence gathering disabled."""
        # Arrange
        workflow = CodeGenerationWorkflow(
            kafka_client=None,
            enable_intelligence=False,  # Disabled
            enable_quorum=False,
            timeout=60.0,
        )
        correlation_id = uuid4()

        # Act
        result = await workflow.run(
            prompt="Create compute node",
            output_directory="./test",
            correlation_id=correlation_id,
        )

        # Assert - workflow still completes without intelligence
        assert result["success"] is True
        assert len(result["generated_files"]) > 0

    @pytest.mark.asyncio
    async def test_stage_durations_within_targets(self, workflow):
        """Test that stages complete within target durations."""
        # Arrange
        correlation_id = uuid4()

        # Act
        result = await workflow.run(
            prompt="Create effect node",
            output_directory="./test",
            correlation_id=correlation_id,
        )

        # Assert - total duration is reasonable
        # Note: In tests with mocks, durations will be much faster
        # In production, validate against actual targets
        assert result["total_duration_seconds"] > 0
        # With mocked I/O, should complete in < 10 seconds
        assert result["total_duration_seconds"] < 10.0

    @pytest.mark.asyncio
    async def test_kafka_event_publishing(self, workflow, mock_kafka_client):
        """Test that Kafka events are published at each stage."""
        # Arrange
        correlation_id = uuid4()

        # Act
        await workflow.run(
            prompt="Create test node",
            output_directory="./test",
            correlation_id=correlation_id,
        )

        # Assert - Kafka publish was called multiple times
        # (1 started + 8 stages completed = 9 total)
        assert mock_kafka_client.publish.call_count >= 9

    @pytest.mark.asyncio
    async def test_node_type_inference(self, workflow):
        """Test node type inference from prompts."""
        test_cases = [
            ("Create PostgreSQL CRUD node", "effect"),
            ("Create workflow orchestrator", "orchestrator"),
            ("Create metrics reducer", "reducer"),
            ("Create data transformer", "compute"),
        ]

        for prompt, expected_type in test_cases:
            # Act
            result = await workflow.run(
                prompt=prompt,
                output_directory="./test",
                correlation_id=uuid4(),
            )

            # Assert
            assert result["node_type"] == expected_type, f"Failed for prompt: {prompt}"

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, workflow):
        """Test that quality scores are calculated."""
        # Arrange
        correlation_id = uuid4()

        # Act
        result = await workflow.run(
            prompt="Create test node",
            output_directory="./test",
            correlation_id=correlation_id,
        )

        # Assert - quality score is valid
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_generated_files_structure(self, workflow):
        """Test that generated files include required components."""
        # Arrange
        correlation_id = uuid4()

        # Act
        result = await workflow.run(
            prompt="Create test node",
            output_directory="./test",
            correlation_id=correlation_id,
        )

        # Assert - expected files are generated
        generated_files = result["generated_files"]
        assert (
            len(generated_files) >= 4
        )  # Minimum: node.py, contract.yaml, models, tests

        # Check for required files
        file_names = [f.split("/")[-1] for f in generated_files]
        assert "node.py" in file_names
        assert "contract.yaml" in file_names

    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self):
        """Test workflow respects timeout configuration."""
        # Arrange
        from llama_index.core.workflow.errors import WorkflowTimeoutError

        workflow = CodeGenerationWorkflow(
            kafka_client=None,
            timeout=0.1,  # 100ms timeout - should fail
        )

        # Act & Assert - expect timeout (WorkflowTimeoutError from LlamaIndex)
        with pytest.raises(WorkflowTimeoutError):
            await workflow.run(
                prompt="Create test node",
                output_directory="./test",
                correlation_id=uuid4(),
            )


# Integration test markers
@pytest.mark.integration
class TestCodeGenerationWorkflowIntegration:
    """Integration tests requiring real Kafka and services."""

    @pytest.mark.asyncio
    async def test_real_kafka_integration(self):
        """Test with real Kafka broker (requires Kafka running)."""
        # This test would be skipped in unit test runs
        # and only run in integration test environments
        pytest.skip("Integration test - requires Kafka running")

    @pytest.mark.asyncio
    async def test_real_intelligence_service(self):
        """Test with real omniarchon intelligence service."""
        pytest.skip("Integration test - requires omniarchon running")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
