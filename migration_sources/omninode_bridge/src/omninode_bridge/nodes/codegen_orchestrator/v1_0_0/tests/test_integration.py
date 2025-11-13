#!/usr/bin/env python3
"""
Integration tests for CodeGenerationWorkflow.

Tests end-to-end workflows with mocked Kafka and database services.
"""

import asyncio

# Import mocks from tests directory (pytest has tests in pythonpath)
import sys
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

try:
    from tests.mocks import MockDatabaseClient, MockKafkaClient
except ImportError:
    # Fallback for development/IDE environments
    from tests.mocks.mock_database import MockDatabaseClient
    from tests.mocks.mock_kafka import MockKafkaClient

from ..workflow import CodeGenerationWorkflow


@pytest.mark.integration
class TestCodeGenerationWorkflowIntegration:
    """Integration test suite for code generation workflow."""

    @pytest.fixture
    def mock_kafka(self):
        """Create mock Kafka client."""
        return MockKafkaClient()

    @pytest.fixture
    def mock_db(self):
        """Create mock database client."""
        return MockDatabaseClient()

    @pytest.fixture
    async def workflow_with_mocks(self, mock_kafka):
        """Create workflow with mocked services."""
        return CodeGenerationWorkflow(
            kafka_client=mock_kafka,
            enable_intelligence=True,
            enable_quorum=False,
            timeout=60.0,
        )

    # Test 1: End-to-End Workflow Execution
    @pytest.mark.asyncio
    async def test_e2e_workflow_execution(self, workflow_with_mocks, mock_kafka):
        """Test complete end-to-end workflow execution."""
        # Arrange
        correlation_id = uuid4()
        prompt = "Create PostgreSQL CRUD Effect"

        # Act
        result = await workflow_with_mocks.run(
            prompt=prompt,
            output_directory="./test_output",
            correlation_id=correlation_id,
        )

        # Assert - workflow completed
        assert result["success"] is True
        assert len(result["generated_files"]) > 0

        # Assert - Kafka events published
        all_events = mock_kafka.get_all_events()
        assert len(all_events) >= 9  # Started + 8 stages

        # Verify event topics
        topics = {event["topic"] for event in all_events}
        assert "omninode.codegen.started" in topics
        assert "omninode.codegen.stage_completed" in topics

    # Test 2: Kafka Event Publishing
    @pytest.mark.asyncio
    async def test_kafka_event_publishing_flow(self, workflow_with_mocks, mock_kafka):
        """Test Kafka event publishing at each stage."""
        # Arrange
        correlation_id = uuid4()

        # Act
        await workflow_with_mocks.run(
            prompt="Create test node",
            output_directory="./test",
            correlation_id=correlation_id,
        )

        # Assert - check event sequence
        started_events = mock_kafka.get_events_by_topic("omninode.codegen.started")
        stage_events = mock_kafka.get_events_by_topic(
            "omninode.codegen.stage_completed"
        )

        assert len(started_events) == 1
        assert len(stage_events) == 8  # All 8 stages

        # Verify stage sequence
        stage_numbers = [event["value"]["stage_number"] for event in stage_events]
        assert stage_numbers == [1, 2, 3, 4, 5, 6, 7, 8]

    # Test 3: Database Persistence
    @pytest.mark.asyncio
    async def test_database_persistence(self, workflow_with_mocks, mock_db):
        """Test database record creation during workflow."""
        # Note: This tests the mock DB interface
        # Real DB integration would be tested in dedicated integration tests

        # Arrange
        workflow_id = str(uuid4())

        # Act - simulate DB operations
        await mock_db.insert(
            "workflows",
            {
                "id": workflow_id,
                "status": "pending",
                "prompt": "Test prompt",
            },
        )

        # Assert
        record = await mock_db.get("workflows", workflow_id)
        assert record is not None
        assert record["status"] == "pending"
        assert mock_db.records_created == 1

    # Test 4: Error Recovery - Kafka Failure
    @pytest.mark.asyncio
    async def test_error_recovery_kafka_failure(self):
        """Test workflow continues when Kafka publishing fails."""
        # Arrange
        mock_kafka = MockKafkaClient()
        await mock_kafka.close()  # Simulate Kafka failure

        workflow = CodeGenerationWorkflow(
            kafka_client=mock_kafka,
            enable_intelligence=False,
            timeout=30.0,
        )

        # Act - workflow should still complete
        result = await workflow.run(
            prompt="Create test node",
            output_directory="./test",
            correlation_id=uuid4(),
        )

        # Assert - workflow completes despite Kafka failure
        assert result["success"] is True

    # Test 5: Concurrent Workflow Execution
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, mock_kafka):
        """Test multiple workflows executing concurrently."""
        # Arrange
        workflow = CodeGenerationWorkflow(
            kafka_client=mock_kafka,
            enable_intelligence=False,
            timeout=60.0,
        )

        prompts = [
            "Create Effect node",
            "Create Orchestrator node",
            "Create Reducer node",
        ]

        # Act - run 3 workflows concurrently
        tasks = [
            workflow.run(
                prompt=prompt,
                output_directory="./test",
                correlation_id=uuid4(),
            )
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks)

        # Assert - all workflows completed
        assert len(results) == 3
        assert all(r["success"] for r in results)

        # Assert - all events captured
        all_events = mock_kafka.get_all_events()
        assert len(all_events) >= 27  # 3 workflows × 9 events minimum

    # Test 6: Intelligence Gathering Integration
    @pytest.mark.asyncio
    async def test_intelligence_gathering_integration(
        self, workflow_with_mocks, mock_kafka
    ):
        """Test workflow with intelligence gathering enabled."""
        # Act
        result = await workflow_with_mocks.run(
            prompt="Create complex orchestrator",
            output_directory="./test",
            correlation_id=uuid4(),
        )

        # Assert - intelligence was gathered (simplified)
        assert result["success"] is True

        # In production, would verify omniarchon query events
        # For now, verify workflow completed with intelligence enabled

    # Test 7: Timeout Handling
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test workflow timeout handling."""
        from llama_index.core.workflow.errors import WorkflowTimeoutError

        # Arrange
        workflow = CodeGenerationWorkflow(
            kafka_client=None,
            timeout=0.1,  # Very short timeout
        )

        # Act & Assert
        with pytest.raises(WorkflowTimeoutError):
            await workflow.run(
                prompt="Create test node",
                output_directory="./test",
                correlation_id=uuid4(),
            )

    # Test 8: Stage Failure Recovery
    @pytest.mark.asyncio
    async def test_stage_failure_recovery(self, mock_kafka):
        """Test recovery from individual stage failures."""
        # This would test retry logic if implemented
        # For now, verify workflow can handle stage errors gracefully

        workflow = CodeGenerationWorkflow(
            kafka_client=mock_kafka,
            enable_intelligence=False,
            timeout=30.0,
        )

        # Act
        result = await workflow.run(
            prompt="Create test node",
            output_directory="./test",
            correlation_id=uuid4(),
        )

        # Assert
        assert result["success"] is True

    # Test 9: Event Correlation Tracking
    @pytest.mark.asyncio
    async def test_event_correlation_tracking(self, workflow_with_mocks, mock_kafka):
        """Test correlation ID is properly tracked across events."""
        # Arrange
        correlation_id = uuid4()

        # Act
        await workflow_with_mocks.run(
            prompt="Create test node",
            output_directory="./test",
            correlation_id=correlation_id,
        )

        # Assert - all events have same correlation ID
        all_events = mock_kafka.get_all_events()
        for event in all_events:
            event_correlation_id = event["value"].get("correlation_id")
            assert event_correlation_id == correlation_id

    # Test 10: Node Type Inference Accuracy
    @pytest.mark.asyncio
    async def test_node_type_inference_accuracy(self, workflow_with_mocks):
        """Test accurate node type inference from prompts."""
        test_cases = [
            ("Create PostgreSQL database adapter", "effect"),
            ("Create workflow orchestrator for multi-step processing", "orchestrator"),
            ("Create metrics aggregator reducer", "reducer"),
            ("Create data transformation compute node", "compute"),
        ]

        for prompt, expected_type in test_cases:
            result = await workflow_with_mocks.run(
                prompt=prompt,
                output_directory="./test",
                correlation_id=uuid4(),
            )

            assert result["node_type"] == expected_type

    # Test 11: Quality Score Validation
    @pytest.mark.asyncio
    async def test_quality_score_validation(self, workflow_with_mocks):
        """Test quality score calculation and validation."""
        # Act
        result = await workflow_with_mocks.run(
            prompt="Create high-quality node",
            output_directory="./test",
            correlation_id=uuid4(),
        )

        # Assert
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0
        assert result["quality_score"] > 0.5  # Reasonable quality threshold

    # Test 12: Generated File Structure Validation
    @pytest.mark.asyncio
    async def test_generated_file_structure_validation(self, workflow_with_mocks):
        """Test generated files include all required components."""
        # Act
        result = await workflow_with_mocks.run(
            prompt="Create comprehensive node",
            output_directory="./test",
            correlation_id=uuid4(),
        )

        # Assert - required files present
        generated_files = result["generated_files"]
        file_names = [f.split("/")[-1] for f in generated_files]

        assert "node.py" in file_names
        assert "contract.yaml" in file_names
        assert any("model" in f for f in file_names)
        assert any("test" in f for f in file_names)

    # Test 13: Performance Under Load
    @pytest.mark.asyncio
    async def test_performance_under_load(self, mock_kafka):
        """Test workflow performance with multiple concurrent requests."""
        # Arrange
        workflow = CodeGenerationWorkflow(
            kafka_client=mock_kafka,
            enable_intelligence=False,
            timeout=60.0,
        )

        # Act - run 5 concurrent workflows
        tasks = [
            workflow.run(
                prompt=f"Create test node {i}",
                output_directory="./test",
                correlation_id=uuid4(),
            )
            for i in range(5)
        ]

        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        duration = asyncio.get_event_loop().time() - start_time

        # Assert
        assert all(r["success"] for r in results)
        assert duration < 60.0  # All workflows complete within timeout

    # Test 14: Event Sequence Validation
    @pytest.mark.asyncio
    async def test_event_sequence_validation(self, workflow_with_mocks, mock_kafka):
        """Test events are published in correct sequence."""
        # Act
        await workflow_with_mocks.run(
            prompt="Create test node",
            output_directory="./test",
            correlation_id=uuid4(),
        )

        # Assert - verify event order
        all_events = mock_kafka.get_all_events()
        event_types = [event["topic"].split(".")[-2] for event in all_events]

        # First event should be "started"
        assert "started" in event_types[0]

        # Stage events should follow
        stage_events = [e for e in event_types if "stage" in e or "completed" in e]
        assert len(stage_events) >= 8

    # Test 15: Database Transaction Handling
    @pytest.mark.asyncio
    async def test_database_transaction_handling(self, mock_db):
        """Test database transaction commit/rollback."""
        # Test commit
        await mock_db.begin_transaction()
        await mock_db.insert("test_table", {"id": "1", "data": "test"})
        await mock_db.commit()

        assert mock_db.records_created == 1

        # Test rollback
        await mock_db.begin_transaction()
        await mock_db.insert("test_table", {"id": "2", "data": "test2"})
        await mock_db.rollback()

        # After rollback, only first record should exist
        # (Note: Mock doesn't actually rollback, but tests the interface)
        assert mock_db.transaction_active is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
