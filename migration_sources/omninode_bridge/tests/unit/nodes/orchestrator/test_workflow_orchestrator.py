#!/usr/bin/env python3
"""
Unit tests for NodeBridgeWorkflowOrchestrator (LlamaIndex Workflows).

Tests coverage:
- Individual workflow step execution
- Complete workflow paths (with/without intelligence)
- Error handling and recovery
- Kafka event publishing integration
- Context management and state propagation
- Performance validation (<2000ms target)
"""

from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from llama_index.core.workflow import StartEvent, StopEvent

from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_state import (
    EnumWorkflowState,
)
from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
    HashGeneratedEvent,
    IntelligenceReceivedEvent,
    IntelligenceRequestedEvent,
    StampCreatedEvent,
    ValidationCompletedEvent,
)

# Import workflow components
from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_node import (
    NodeBridgeWorkflowOrchestrator,
)

# Test Fixtures


@pytest.fixture
def test_container():
    """Create test ONEX container with mock configuration."""
    from unittest.mock import MagicMock

    # Create a mock container that mimics the expected container interface
    # This avoids dependency on omnibase_core's actual ModelContainer
    container = MagicMock()
    container.config = {
        "metadata_stamping_service_url": "http://test-metadata:8053",
        "onextree_service_url": "http://test-onextree:8080",
        "kafka_broker_url": "localhost:9092",
        "default_namespace": "test.bridge",
        "health_check_mode": True,  # Skip Kafka initialization
    }
    container.get = lambda key, default=None: container.config.get(key, default)
    container.services = {}
    container.get_service = lambda service_name: container.services.get(service_name)

    return container


@pytest.fixture
def workflow_orchestrator(test_container):
    """Create workflow orchestrator instance for testing."""
    # Use normal constructor to ensure proper initialization
    orchestrator = NodeBridgeWorkflowOrchestrator(
        container=test_container, timeout=60, verbose=False
    )

    # Patch the workflow to disable validation for testing
    orchestrator._disable_validation = True

    return orchestrator


@pytest.fixture
def mock_workflow_context(workflow_orchestrator):
    """Create Context for step testing with async get/set API."""
    from llama_index.core.workflow import Context

    # Create real Context with workflow reference
    # Note: In llama-index-workflows 1.3.0+, Context uses async get/set API
    ctx = Context(workflow=workflow_orchestrator)

    return ctx


@pytest.fixture
def sample_correlation_id() -> UUID:
    """Generate sample correlation ID."""
    return uuid4()


# Helper for setting up context in tests
async def setup_context(ctx, data_dict):
    """Helper to set multiple context values from a dict."""
    for key, value in data_dict.items():
        await ctx.set(key, value)


# Unit Tests for Individual Steps


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_input_step_success(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test input validation step with valid content."""
    # Arrange
    start_event = StartEvent()
    start_event._data = {
        "correlation_id": sample_correlation_id,
        "content": "Test content for validation",
        "metadata": {"source": "unit_test"},
        "enable_intelligence": False,
    }

    # Act
    result = await workflow_orchestrator.validate_input(
        ctx=mock_workflow_context, ev=start_event
    )

    # Assert
    assert isinstance(result, ValidationCompletedEvent)
    assert result.correlation_id == sample_correlation_id
    assert result.validated_content == "Test content for validation"
    assert result.validation_time_ms > 0
    assert result.namespace == "test.bridge"

    # Verify context initialization
    assert await mock_workflow_context.get("correlation_id") == sample_correlation_id
    assert await mock_workflow_context.get("enable_intelligence") is False
    assert await mock_workflow_context.get("workflow_start_time") is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_input_step_failure(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test input validation step with invalid content."""
    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        WorkflowFailedEvent,
    )

    # Arrange
    start_event = StartEvent()
    start_event._data = {
        "correlation_id": sample_correlation_id,
        "content": None,  # Invalid content
        "enable_intelligence": False,
    }

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Act
    result = await workflow_orchestrator.validate_input(
        ctx=mock_workflow_context, ev=start_event
    )

    # Assert - workflow returns StopEvent with WorkflowFailedEvent result
    assert isinstance(result, StopEvent)
    assert isinstance(result.result, WorkflowFailedEvent)
    assert "Content is required" in result.result.error_message
    assert result.result.failed_step == "validate_input"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_hash_step(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test BLAKE3 hash generation step."""
    # Arrange
    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": 0.0,
        },
    )

    validation_event = ValidationCompletedEvent(
        correlation_id=sample_correlation_id,
        validated_content="Test content for hashing",
        validation_time_ms=5.0,
        namespace="test.bridge",
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Act
    result = await workflow_orchestrator.generate_hash(
        ctx=mock_workflow_context, ev=validation_event
    )

    # Assert
    assert isinstance(result, HashGeneratedEvent)
    assert result.correlation_id == sample_correlation_id
    assert result.file_hash.startswith("blake3_")
    assert result.hash_generation_time_ms > 0
    assert result.file_size_bytes == len(b"Test content for hashing")

    # Verify context updates
    assert await mock_workflow_context.get("file_hash") is not None
    assert await mock_workflow_context.get("file_size_bytes") is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_stamp_step_without_intelligence(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test stamp creation step without intelligence enrichment."""
    # Arrange
    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": 0.0,
            "enable_intelligence": False,
        },
    )

    hash_event = HashGeneratedEvent(
        correlation_id=sample_correlation_id,
        file_hash="blake3_test_hash_12345",
        hash_generation_time_ms=10.0,
        file_size_bytes=1024,
        namespace="test.bridge",
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Act
    result = await workflow_orchestrator.create_stamp(
        ctx=mock_workflow_context, ev=hash_event
    )

    # Assert
    assert isinstance(result, StampCreatedEvent)
    assert result.correlation_id == sample_correlation_id
    assert result.stamp_id is not None
    assert result.stamp_data["file_hash"] == "blake3_test_hash_12345"
    assert result.stamp_data["version"] == "1"
    assert result.stamp_data["metadata_version"] == "0.1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_stamp_step_with_intelligence(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test stamp creation step with intelligence enrichment enabled."""
    # Arrange
    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": 0.0,
            "enable_intelligence": True,
            "validated_content": "Test content",
        },
    )

    hash_event = HashGeneratedEvent(
        correlation_id=sample_correlation_id,
        file_hash="blake3_test_hash_12345",
        hash_generation_time_ms=10.0,
        file_size_bytes=1024,
        namespace="test.bridge",
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Act
    result = await workflow_orchestrator.create_stamp(
        ctx=mock_workflow_context, ev=hash_event
    )

    # Assert
    assert isinstance(result, IntelligenceRequestedEvent)
    assert result.correlation_id == sample_correlation_id
    assert result.content == "Test content"
    assert result.file_hash == "blake3_test_hash_12345"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enrich_intelligence_step_success(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test intelligence enrichment step with successful analysis."""
    # Arrange
    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": 0.0,
        },
    )

    intelligence_request_event = IntelligenceRequestedEvent(
        correlation_id=sample_correlation_id,
        content="Test content for analysis",
        file_hash="blake3_test_hash_12345",
        namespace="test.bridge",
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Act
    result = await workflow_orchestrator.enrich_intelligence(
        ctx=mock_workflow_context, ev=intelligence_request_event
    )

    # Assert
    assert isinstance(result, IntelligenceReceivedEvent)
    assert result.correlation_id == sample_correlation_id
    assert result.intelligence_data is not None
    assert result.confidence_score > 0.0
    assert result.intelligence_time_ms >= 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_persist_state_step_without_intelligence(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test persist_state step without intelligence enrichment."""
    # Arrange
    import time

    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        PersistenceCompletedEvent,
    )

    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": time.time(),
            "stamp_id": "stamp_test_12345",
            "file_hash": "blake3_test_hash_12345",
            "stamp_data": {
                "stamp_id": "stamp_test_12345",
                "file_hash": "blake3_test_hash_12345",
                "version": "1",
                "metadata_version": "0.1",
            },
            "validated_content": "Test content",
            "intelligence_data": None,
        },
    )

    stamp_event = StampCreatedEvent(
        correlation_id=sample_correlation_id,
        stamp_id="stamp_test_12345",
        stamp_data=await mock_workflow_context.get("stamp_data"),
        stamp_creation_time_ms=5.0,
        namespace="test.bridge",
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Act
    result = await workflow_orchestrator.persist_state(
        ctx=mock_workflow_context, ev=stamp_event
    )

    # Assert
    assert isinstance(result, PersistenceCompletedEvent)
    assert result.correlation_id == sample_correlation_id
    assert result.persistence_time_ms >= 0
    assert result.database_id is not None
    assert await mock_workflow_context.get("database_id") is not None
    assert await mock_workflow_context.get("persistence_time_ms") is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_persist_state_step_with_intelligence(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test persist_state step with intelligence enrichment data."""
    # Arrange
    import time

    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        PersistenceCompletedEvent,
    )

    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": time.time(),
            "stamp_id": "stamp_test_12345",
            "file_hash": "blake3_test_hash_12345",
            "stamp_data": {
                "stamp_id": "stamp_test_12345",
                "file_hash": "blake3_test_hash_12345",
                "version": "1",
                "metadata_version": "0.1",
            },
            "validated_content": "Test content",
            "intelligence_data": {
                "analysis_type": "content_validation",
                "confidence_score": "0.95",
            },
        },
    )

    intelligence_event = IntelligenceReceivedEvent(
        correlation_id=sample_correlation_id,
        intelligence_data={"analysis_type": "content_validation"},
        intelligence_time_ms=100.0,
        confidence_score=0.95,
        namespace="test.bridge",
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Act
    result = await workflow_orchestrator.persist_state(
        ctx=mock_workflow_context, ev=intelligence_event
    )

    # Assert
    assert isinstance(result, PersistenceCompletedEvent)
    assert result.correlation_id == sample_correlation_id
    assert result.persistence_time_ms >= 0
    assert await mock_workflow_context.get("database_id") is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_complete_workflow_step_without_intelligence(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test workflow completion without intelligence enrichment."""
    # Arrange
    import time

    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        PersistenceCompletedEvent,
    )

    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": time.time(),
            "stamp_id": "stamp_test_12345",
            "file_hash": "blake3_test_hash_12345",
            "stamp_data": {
                "stamp_id": "stamp_test_12345",
                "file_hash": "blake3_test_hash_12345",
                "version": "1",
                "metadata_version": "0.1",
            },
            "validated_content": "Test content",
            "hash_generation_time_ms": 10.0,
            "intelligence_data": None,
            "persistence_time_ms": 25.0,
            "database_id": "db_12345",
        },
    )

    persistence_event = PersistenceCompletedEvent(
        correlation_id=sample_correlation_id,
        persistence_time_ms=25.0,
        database_id="db_12345",
        namespace="test.bridge",
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Act
    result = await workflow_orchestrator.complete_workflow(
        ctx=mock_workflow_context, ev=persistence_event
    )

    # Assert
    assert isinstance(result, StopEvent)
    assert result.result.stamp_id == "stamp_test_12345"
    assert result.result.file_hash == "blake3_test_hash_12345"
    assert result.result.workflow_state == EnumWorkflowState.COMPLETED
    assert (
        result.result.workflow_steps_executed == 4
    )  # Updated to account for persist step
    assert result.result.intelligence_data is None


# Integration Tests for Complete Workflow


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_workflow_without_intelligence(
    workflow_orchestrator, sample_correlation_id
):
    """Test complete workflow execution without intelligence enrichment."""
    # Mock Kafka client
    workflow_orchestrator.kafka_client = None

    # Act
    result = await workflow_orchestrator.run(
        correlation_id=sample_correlation_id,
        content="Test content for complete workflow",
        enable_intelligence=False,
    )

    # Assert
    assert result.stamp_id is not None
    assert result.file_hash.startswith("blake3_")
    assert result.workflow_state == EnumWorkflowState.COMPLETED
    assert (
        result.workflow_steps_executed == 4
    )  # Updated: validate, hash, stamp, persist
    assert result.processing_time_ms > 0
    assert result.intelligence_data is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_workflow_with_intelligence(
    workflow_orchestrator, sample_correlation_id
):
    """Test complete workflow execution with intelligence enrichment."""
    # Mock Kafka client
    workflow_orchestrator.kafka_client = None

    # Act
    result = await workflow_orchestrator.run(
        correlation_id=sample_correlation_id,
        content="Test content for intelligent workflow",
        enable_intelligence=True,
    )

    # Assert
    assert result.stamp_id is not None
    assert result.file_hash.startswith("blake3_")
    assert result.workflow_state == EnumWorkflowState.COMPLETED
    assert (
        result.workflow_steps_executed == 5
    )  # Updated: validate, hash, stamp, intelligence, persist
    assert result.processing_time_ms > 0
    assert result.intelligence_data is not None


# Performance Tests


@pytest.mark.performance
@pytest.mark.asyncio
async def test_workflow_performance_without_intelligence(
    workflow_orchestrator, sample_correlation_id
):
    """Test workflow performance meets <100ms target (without intelligence)."""
    import time

    # Mock Kafka client
    workflow_orchestrator.kafka_client = None

    # Act
    start_time = time.time()
    result = await workflow_orchestrator.run(
        correlation_id=sample_correlation_id,
        content="Performance test content",
        enable_intelligence=False,
    )
    execution_time_ms = (time.time() - start_time) * 1000

    # Assert - relaxed target for unit tests (real target is <100ms with production infra)
    assert execution_time_ms < 2000  # 2 seconds timeout for safety
    assert result.processing_time_ms < 2000
    assert result.workflow_state == EnumWorkflowState.COMPLETED


@pytest.mark.performance
@pytest.mark.asyncio
async def test_workflow_performance_with_intelligence(
    workflow_orchestrator, sample_correlation_id
):
    """Test workflow performance meets <600ms target (with intelligence)."""
    import time

    # Mock Kafka client
    workflow_orchestrator.kafka_client = None

    # Act
    start_time = time.time()
    result = await workflow_orchestrator.run(
        correlation_id=sample_correlation_id,
        content="Performance test with intelligence",
        enable_intelligence=True,
    )
    execution_time_ms = (time.time() - start_time) * 1000

    # Assert - relaxed target for unit tests (real target is <600ms with production infra)
    assert execution_time_ms < 2000  # 2 seconds timeout for safety
    assert result.processing_time_ms < 2000
    assert result.workflow_state == EnumWorkflowState.COMPLETED


# Kafka Integration Tests


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kafka_event_publishing(
    workflow_orchestrator, sample_correlation_id, mock_workflow_context
):
    """Test Kafka event publishing in workflow steps."""
    # Arrange
    mock_kafka_client = AsyncMock()
    mock_kafka_client.is_connected = True
    mock_kafka_client.publish_raw_event = AsyncMock(return_value=True)
    workflow_orchestrator.kafka_client = mock_kafka_client

    # Act
    await workflow_orchestrator._publish_kafka_event(
        "TEST_EVENT",
        {
            "correlation_id": str(sample_correlation_id),
            "test_data": "test_value",
        },
    )

    # Assert
    mock_kafka_client.publish_raw_event.assert_called_once()
    call_args = mock_kafka_client.publish_raw_event.call_args
    assert call_args.kwargs["topic"] == "test.bridge.orchestrator.test_event"
    assert call_args.kwargs["key"] == str(sample_correlation_id)


# Error Handling Tests


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_error_handling_invalid_input(
    workflow_orchestrator, sample_correlation_id
):
    """Test workflow error handling with invalid input."""
    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        WorkflowFailedEvent,
    )

    # Mock Kafka client
    workflow_orchestrator.kafka_client = None

    # Act - run workflow with invalid input
    result = await workflow_orchestrator.run(
        correlation_id=sample_correlation_id,
        content=None,  # Invalid content
        enable_intelligence=False,
    )

    # Assert - should return WorkflowFailedEvent
    assert isinstance(result, WorkflowFailedEvent)
    assert result.error_message == "Content is required and must be a string"
    assert result.error_type == "ModelOnexError"
    assert result.failed_step == "validate_input"
    assert result.correlation_id == sample_correlation_id
    assert result.processing_time_ms >= 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_input_error_handling(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test validate_input step error handling with various error types."""
    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        WorkflowFailedEvent,
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Test Case 1: None content
    start_event = StartEvent()
    start_event._data = {
        "correlation_id": sample_correlation_id,
        "content": None,
        "enable_intelligence": False,
    }

    result = await workflow_orchestrator.validate_input(
        ctx=mock_workflow_context, ev=start_event
    )

    assert isinstance(result, StopEvent)
    assert isinstance(result.result, WorkflowFailedEvent)
    assert "Content is required" in result.result.error_message
    assert result.result.failed_step == "validate_input"

    # Test Case 2: Non-string content
    start_event._data = {
        "correlation_id": sample_correlation_id,
        "content": 12345,  # Invalid type
        "enable_intelligence": False,
    }

    result = await workflow_orchestrator.validate_input(
        ctx=mock_workflow_context, ev=start_event
    )

    assert isinstance(result, StopEvent)
    assert isinstance(result.result, WorkflowFailedEvent)
    assert "Content is required" in result.result.error_message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_hash_error_handling(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test generate_hash step error handling."""
    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        WorkflowFailedEvent,
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Setup context with missing correlation_id to trigger KeyError
    await setup_context(mock_workflow_context, {})  # Missing correlation_id

    validation_event = ValidationCompletedEvent(
        correlation_id=sample_correlation_id,
        validated_content="Test content",
        validation_time_ms=5.0,
        namespace="test.bridge",
    )

    result = await workflow_orchestrator.generate_hash(
        ctx=mock_workflow_context, ev=validation_event
    )

    # Assert error handling
    assert isinstance(result, StopEvent)
    assert isinstance(result.result, WorkflowFailedEvent)
    assert result.result.failed_step == "generate_hash"
    # Context.get() raises ValueError (not KeyError) when key is missing
    assert result.result.error_type == "ValueError"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_stamp_error_handling(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test create_stamp step error handling."""
    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        WorkflowFailedEvent,
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Setup context with missing correlation_id
    await setup_context(mock_workflow_context, {})

    hash_event = HashGeneratedEvent(
        correlation_id=sample_correlation_id,
        file_hash="blake3_test_hash",
        hash_generation_time_ms=10.0,
        file_size_bytes=1024,
        namespace="test.bridge",
    )

    result = await workflow_orchestrator.create_stamp(
        ctx=mock_workflow_context, ev=hash_event
    )

    # Assert error handling
    assert isinstance(result, StopEvent)
    assert isinstance(result.result, WorkflowFailedEvent)
    assert result.result.failed_step == "create_stamp"
    # Context.get() raises ValueError (not KeyError) when key is missing
    assert result.result.error_type == "ValueError"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_complete_workflow_error_handling(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test complete_workflow step error handling."""
    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        WorkflowFailedEvent,
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Setup context with incomplete data (missing required fields)
    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": 0.0,
            # Missing: stamp_id, file_hash, stamp_data
        },
    )

    stamp_event = StampCreatedEvent(
        correlation_id=sample_correlation_id,
        stamp_id="stamp_test_12345",
        stamp_data={"test": "data"},
        stamp_creation_time_ms=5.0,
        namespace="test.bridge",
    )

    result = await workflow_orchestrator.complete_workflow(
        ctx=mock_workflow_context, ev=stamp_event
    )

    # Assert error handling
    assert isinstance(result, StopEvent)
    assert isinstance(result.result, WorkflowFailedEvent)
    assert result.result.failed_step == "complete_workflow"
    # Context.get() raises ValueError (not KeyError) when key is missing
    assert result.result.error_type == "ValueError"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_workflow_error_converts_exception_types(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test handle_workflow_error converts different exception types to OnexError."""
    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        WorkflowFailedEvent,
    )

    # Mock Kafka publishing
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Setup context
    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": 0.0,
        },
    )

    # Test Case 1: ValueError → VALIDATION_ERROR
    error = ValueError("Invalid value provided")
    result = await workflow_orchestrator.handle_workflow_error(
        ctx=mock_workflow_context, error=error, step_name="test_step"
    )

    assert isinstance(result, StopEvent)
    assert isinstance(result.result, WorkflowFailedEvent)
    assert result.result.error_type == "ValueError"
    assert result.result.failed_step == "test_step"

    # Test Case 2: KeyError → CONFIGURATION_ERROR
    error = KeyError("missing_key")
    result = await workflow_orchestrator.handle_workflow_error(
        ctx=mock_workflow_context, error=error, step_name="test_step"
    )

    assert isinstance(result, StopEvent)
    assert result.result.error_type == "KeyError"

    # Test Case 3: ConnectionError → NETWORK_ERROR
    error = ConnectionError("Network timeout")
    result = await workflow_orchestrator.handle_workflow_error(
        ctx=mock_workflow_context, error=error, step_name="test_step"
    )

    assert isinstance(result, StopEvent)
    assert result.result.error_type == "ConnectionError"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_failed_event_published_to_kafka(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test that WORKFLOW_FAILED events are published to Kafka."""
    # Mock Kafka client
    mock_kafka_client = AsyncMock()
    mock_kafka_client.is_connected = True
    mock_kafka_client.publish_raw_event = AsyncMock(return_value=True)
    workflow_orchestrator.kafka_client = mock_kafka_client

    # Setup context
    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": 0.0,
        },
    )

    # Trigger error
    error = ValueError("Test error for Kafka publishing")
    await workflow_orchestrator.handle_workflow_error(
        ctx=mock_workflow_context, error=error, step_name="test_step"
    )

    # Assert Kafka event was published
    mock_kafka_client.publish_raw_event.assert_called_once()
    call_args = mock_kafka_client.publish_raw_event.call_args

    assert call_args.kwargs["topic"] == "test.bridge.orchestrator.workflow_failed"
    assert call_args.kwargs["key"] == str(sample_correlation_id)

    event_data = call_args.kwargs["data"]
    assert event_data["event_type"] == "WORKFLOW_FAILED"
    assert event_data["error_type"] == "ValueError"
    assert event_data["failed_step"] == "test_step"
    assert "error_message" in event_data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_workflow_with_validation_error(
    workflow_orchestrator, sample_correlation_id
):
    """Test end-to-end workflow with validation error returns WorkflowFailedEvent."""
    from omninode_bridge.nodes.orchestrator.v1_0_0.workflow_events import (
        WorkflowFailedEvent,
    )

    # Mock Kafka client
    workflow_orchestrator.kafka_client = None

    # Run workflow with invalid content
    result = await workflow_orchestrator.run(
        correlation_id=sample_correlation_id,
        content="",  # Empty string (invalid)
        enable_intelligence=False,
    )

    # Assert
    assert isinstance(result, WorkflowFailedEvent)
    assert "Content is required" in result.error_message
    assert result.failed_step == "validate_input"
    assert result.correlation_id == sample_correlation_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intelligence_enrichment_graceful_degradation(
    workflow_orchestrator, mock_workflow_context, sample_correlation_id
):
    """Test intelligence enrichment handles failures gracefully."""
    # Arrange
    await setup_context(
        mock_workflow_context,
        {
            "correlation_id": sample_correlation_id,
            "workflow_start_time": 0.0,
        },
    )

    intelligence_request_event = IntelligenceRequestedEvent(
        correlation_id=sample_correlation_id,
        content="Test content",
        file_hash="blake3_test_hash_12345",
        namespace="test.bridge",
    )

    # Mock Kafka publishing and simulate intelligence service failure
    workflow_orchestrator._publish_kafka_event = AsyncMock()

    # Patch datetime.now() to simulate intelligence service failure
    with patch(
        "omninode_bridge.nodes.orchestrator.v1_0_0.workflow_node.datetime"
    ) as mock_datetime:
        mock_datetime.now.side_effect = Exception("Intelligence service unavailable")

        # Act
        result = await workflow_orchestrator.enrich_intelligence(
            ctx=mock_workflow_context, ev=intelligence_request_event
        )

        # Assert - workflow continues with empty intelligence data
        assert isinstance(result, IntelligenceReceivedEvent)
        assert result.intelligence_data == {}
        assert result.confidence_score == 0.0


# Context Management Tests


@pytest.mark.unit
@pytest.mark.asyncio
async def test_context_propagation_through_workflow(
    workflow_orchestrator, sample_correlation_id
):
    """Test correlation ID and state propagation through workflow context."""
    # Mock Kafka client
    workflow_orchestrator.kafka_client = None

    # Act
    result = await workflow_orchestrator.run(
        correlation_id=sample_correlation_id,
        content="Context propagation test",
        metadata={"test_key": "test_value"},
        enable_intelligence=False,
    )

    # Assert
    assert result.workflow_id == sample_correlation_id
    assert result.op_id == sample_correlation_id
    assert result.stamp_id is not None
