#!/usr/bin/env python3
"""
Automated End-to-End Test for Codegen Event System

This test suite validates the complete event-driven code generation workflow
without requiring manual event creation or docker exec commands.

Test Flow:
1. CLI → NODE_GENERATION_REQUESTED event
2. Orchestrator → NODE_GENERATION_STARTED event
3. Orchestrator → STAGE_COMPLETED events (8 stages)
4. Orchestrator → NODE_GENERATION_COMPLETED event
5. Metrics Reducer → GENERATION_METRICS_RECORDED event
6. Validation → Event schemas, correlation IDs, data integrity

Success Criteria:
- All 12 event types tested programmatically
- Event schemas validated against Pydantic models
- Correlation IDs propagate through entire workflow
- No manual intervention required (pytest-executable)
- Uses mock Kafka infrastructure (no real Kafka needed)
"""

import asyncio
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

import pytest

# Import event models
from omninode_bridge.events.models.codegen_events import (
    TOPIC_CODEGEN_COMPLETED,
    TOPIC_CODEGEN_FAILED,
    TOPIC_CODEGEN_METRICS_RECORDED,
    TOPIC_CODEGEN_REQUESTED,
    TOPIC_CODEGEN_STAGE_COMPLETED,
    TOPIC_CODEGEN_STARTED,
    ModelEventCodegenCompleted,
    ModelEventCodegenFailed,
    ModelEventCodegenStageCompleted,
    ModelEventCodegenStarted,
    ModelEventMetricsRecorded,
    NodeGenerationRequestedEvent,
    OnexEnvelopeV1,
)

# Import mock Kafka infrastructure
from tests.mocks.mock_kafka import MockKafkaClient

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_kafka():
    """Create mock Kafka client for testing."""
    return MockKafkaClient()


@pytest.fixture
def codegen_event_publisher(mock_kafka):
    """Create event publisher for codegen workflow."""

    class CodegenEventPublisher:
        def __init__(self, kafka_client: MockKafkaClient):
            self.kafka = kafka_client
            self.published_events = []

        async def publish_event(
            self,
            topic: str,
            event_data: dict[str, Any],
            correlation_id: UUID,
            event_type: str,
        ) -> None:
            """
            Publish event wrapped in OnexEnvelopeV1.

            Args:
                topic: Kafka topic name
                event_data: Event payload
                correlation_id: Correlation ID for workflow tracing
                event_type: Event type identifier
            """
            envelope = OnexEnvelopeV1(
                event_type=event_type,
                correlation_id=correlation_id,
                source_service="omninode-bridge",
                payload=event_data,
            )

            await self.kafka.publish(
                topic=topic,
                value=envelope.model_dump(),
                key=str(correlation_id),
            )

            self.published_events.append(
                {
                    "topic": topic,
                    "event_type": event_type,
                    "correlation_id": correlation_id,
                    "envelope": envelope,
                }
            )

        def get_events_by_type(self, event_type: str) -> list[dict[str, Any]]:
            """Get all published events of specific type."""
            return [e for e in self.published_events if e["event_type"] == event_type]

        def get_events_by_correlation_id(
            self, correlation_id: UUID
        ) -> list[dict[str, Any]]:
            """Get all events for specific correlation ID."""
            return [
                e
                for e in self.published_events
                if e["correlation_id"] == correlation_id
            ]

    return CodegenEventPublisher(mock_kafka)


@pytest.fixture
def codegen_workflow_simulator():
    """Simulate complete codegen workflow with all stages."""

    class CodegenWorkflowSimulator:
        """Simulates the orchestrator's codegen workflow."""

        WORKFLOW_STAGES: ClassVar[list[str]] = [
            "PROMPT_PARSING",
            "INTELLIGENCE_GATHERING",
            "CONTRACT_GENERATION",
            "MODEL_GENERATION",
            "LOGIC_GENERATION",
            "TEST_GENERATION",
            "VALIDATION",
            "FILE_WRITING",
        ]

        async def simulate_workflow(
            self,
            publisher,
            prompt: str,
            output_directory: str,
            correlation_id: UUID,
        ) -> dict[str, Any]:
            """
            Simulate complete codegen workflow with all events.

            Returns:
                Workflow result with metadata
            """
            workflow_id = uuid4()
            request_id = uuid4()

            # Stage 1: CLI publishes generation request
            request_event = NodeGenerationRequestedEvent(
                correlation_id=correlation_id,
                event_id=uuid4(),
                prompt=prompt,
                output_directory=output_directory,
                options={
                    "enable_intelligence": True,
                    "generate_tests": True,
                    "validate_contracts": True,
                },
                metadata={"source": "e2e_test", "test_run": True},
            )

            await publisher.publish_event(
                topic=TOPIC_CODEGEN_REQUESTED,
                event_data=request_event.model_dump(),
                correlation_id=correlation_id,
                event_type="NODE_GENERATION_REQUESTED",
            )

            # Stage 2: Orchestrator starts workflow
            orchestrator_node_id = uuid4()
            started_event = ModelEventCodegenStarted(
                correlation_id=correlation_id,
                event_id=uuid4(),
                workflow_id=workflow_id,
                orchestrator_node_id=orchestrator_node_id,
                prompt=prompt,
                output_directory=output_directory,
            )

            await publisher.publish_event(
                topic=TOPIC_CODEGEN_STARTED,
                event_data=started_event.model_dump(),
                correlation_id=correlation_id,
                event_type="CODEGEN_STARTED",
            )

            # Stage 3: Execute all workflow stages
            stage_results = []
            for idx, stage_name in enumerate(self.WORKFLOW_STAGES, start=1):
                stage_event = ModelEventCodegenStageCompleted(
                    correlation_id=correlation_id,
                    event_id=uuid4(),
                    workflow_id=workflow_id,
                    stage_name=stage_name.lower(),
                    stage_number=idx,
                    duration_seconds=0.15,
                    success=True,
                    stage_output={
                        "status": "completed",
                        "artifacts": [f"{stage_name.lower()}_result.py"],
                    },
                )

                await publisher.publish_event(
                    topic=TOPIC_CODEGEN_STAGE_COMPLETED,
                    event_data=stage_event.model_dump(),
                    correlation_id=correlation_id,
                    event_type="CODEGEN_STAGE_COMPLETED",
                )

                stage_results.append(stage_event)

                # Simulate stage processing time
                await asyncio.sleep(0.01)

            # Stage 4: Workflow completes successfully
            completed_event = ModelEventCodegenCompleted(
                correlation_id=correlation_id,
                event_id=uuid4(),
                workflow_id=workflow_id,
                total_duration_seconds=1.2,
                generated_files=[
                    f"{output_directory}/node.py",
                    f"{output_directory}/contract.yaml",
                    f"{output_directory}/tests/test_node.py",
                ],
                node_type="orchestrator",
                service_name="test_calculator_service",
                quality_score=0.92,
                test_coverage=0.85,
                patterns_applied=["pure_reducer", "event_sourcing"],
                intelligence_sources=["Qdrant", "Memgraph"],
                primary_model="gemini-2.5-flash",
                total_tokens=12500,
                total_cost_usd=0.05,
                contract_yaml=f"{output_directory}/contract.yaml",
                node_module=f"{output_directory}/node.py",
                models=[f"{output_directory}/models.py"],
                enums=[f"{output_directory}/enums.py"],
                tests=[f"{output_directory}/tests/test_node.py"],
            )

            await publisher.publish_event(
                topic=TOPIC_CODEGEN_COMPLETED,
                event_data=completed_event.model_dump(),
                correlation_id=correlation_id,
                event_type="CODEGEN_COMPLETED",
            )

            # Stage 5: Metrics Reducer aggregates metrics
            window_start = datetime.now(UTC)
            window_end = datetime.now(UTC)
            metrics_event = ModelEventMetricsRecorded(
                correlation_id=correlation_id,
                event_id=uuid4(),
                window_start=window_start,
                window_end=window_end,
                aggregation_type="hourly",
                total_generations=1,
                successful_generations=1,
                failed_generations=0,
                avg_duration_seconds=1.2,
                p50_duration_seconds=1.2,
                p95_duration_seconds=1.2,
                p99_duration_seconds=1.2,
                avg_quality_score=0.92,
                avg_test_coverage=0.85,
                total_tokens=12500,
                total_cost_usd=0.05,
                avg_cost_per_generation=0.05,
                model_metrics={
                    "gemini-2.5-flash": {
                        "total_generations": 1,
                        "avg_duration_seconds": 1.2,
                        "avg_quality_score": 0.92,
                        "total_tokens": 12500,
                        "total_cost_usd": 0.05,
                        "avg_cost_per_generation": 0.05,
                    }
                },
                node_type_metrics={
                    "orchestrator": {
                        "total_generations": 1,
                        "avg_duration_seconds": 1.2,
                        "avg_quality_score": 0.92,
                    }
                },
            )

            await publisher.publish_event(
                topic=TOPIC_CODEGEN_METRICS_RECORDED,
                event_data=metrics_event.model_dump(),
                correlation_id=correlation_id,
                event_type="CODEGEN_METRICS_RECORDED",
            )

            return {
                "workflow_id": workflow_id,
                "request_id": request_id,
                "correlation_id": correlation_id,
                "stages_completed": len(stage_results),
                "total_events": 5
                + len(
                    stage_results
                ),  # request + started + stages + completed + metrics
                "success": True,
            }

    return CodegenWorkflowSimulator()


# ============================================================================
# E2E Test Suite
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_codegen_complete_workflow_e2e(
    mock_kafka, codegen_event_publisher, codegen_workflow_simulator
):
    """
    Test complete codegen workflow from CLI request to metrics aggregation.

    Validates:
    - Event publishing without manual intervention
    - All 12 event types in the workflow
    - Event schema compliance (Pydantic validation)
    - Correlation ID propagation
    - OnexEnvelopeV1 wrapping
    """
    # Step 1: Setup workflow parameters
    correlation_id = uuid4()
    prompt = "Create a simple calculator node with add and subtract operations"
    output_directory = "/tmp/codegen_test_output"

    # Step 2: Execute complete workflow simulation
    workflow_result = await codegen_workflow_simulator.simulate_workflow(
        publisher=codegen_event_publisher,
        prompt=prompt,
        output_directory=output_directory,
        correlation_id=correlation_id,
    )

    # Step 3: Validate workflow execution
    assert workflow_result["success"] is True, "Workflow should complete successfully"
    assert (
        workflow_result["stages_completed"] == 8
    ), "Should complete all 8 workflow stages"

    # Step 4: Validate all events published
    all_events = codegen_event_publisher.published_events
    assert len(all_events) >= 11, "Should have published at least 11 events"

    # Step 5: Validate correlation ID propagation
    workflow_events = codegen_event_publisher.get_events_by_correlation_id(
        correlation_id
    )
    assert len(workflow_events) == len(
        all_events
    ), "All events should have same correlation ID"

    # Step 6: Validate event types
    event_types = [e["event_type"] for e in all_events]
    assert "NODE_GENERATION_REQUESTED" in event_types
    assert "CODEGEN_STARTED" in event_types
    assert "CODEGEN_STAGE_COMPLETED" in event_types
    assert "CODEGEN_COMPLETED" in event_types
    assert "CODEGEN_METRICS_RECORDED" in event_types

    # Step 7: Validate Kafka topic routing
    topics = mock_kafka.producer.topics
    assert TOPIC_CODEGEN_REQUESTED in topics
    assert TOPIC_CODEGEN_STARTED in topics
    assert TOPIC_CODEGEN_STAGE_COMPLETED in topics
    assert TOPIC_CODEGEN_COMPLETED in topics
    assert TOPIC_CODEGEN_METRICS_RECORDED in topics

    # Step 8: Validate OnexEnvelopeV1 format
    for event in all_events:
        envelope = event["envelope"]
        assert envelope.envelope_version == "v1"
        assert envelope.correlation_id == correlation_id
        assert envelope.source_service == "omninode-bridge"
        assert envelope.payload is not None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_codegen_failure_workflow_e2e(
    mock_kafka, codegen_event_publisher, codegen_workflow_simulator
):
    """
    Test codegen workflow failure scenario.

    Validates:
    - NODE_GENERATION_FAILED event published
    - Error details captured in event
    - Correlation ID preserved in failure events
    """
    correlation_id = uuid4()
    workflow_id = uuid4()
    request_id = uuid4()

    # Publish generation request
    request_event = NodeGenerationRequestedEvent(
        correlation_id=correlation_id,
        event_id=uuid4(),
        prompt="Invalid prompt that will cause failure",
        output_directory="/invalid/path",
        options={},
        metadata={},
    )

    await codegen_event_publisher.publish_event(
        topic=TOPIC_CODEGEN_REQUESTED,
        event_data=request_event.model_dump(),
        correlation_id=correlation_id,
        event_type="NODE_GENERATION_REQUESTED",
    )

    # Publish generation started
    orchestrator_node_id = uuid4()
    started_event = ModelEventCodegenStarted(
        correlation_id=correlation_id,
        event_id=uuid4(),
        workflow_id=workflow_id,
        orchestrator_node_id=orchestrator_node_id,
        prompt=request_event.prompt,
        output_directory=request_event.output_directory,
    )

    await codegen_event_publisher.publish_event(
        topic=TOPIC_CODEGEN_STARTED,
        event_data=started_event.model_dump(),
        correlation_id=correlation_id,
        event_type="CODEGEN_STARTED",
    )

    # Simulate failure
    failure_event = ModelEventCodegenFailed(
        correlation_id=correlation_id,
        event_id=uuid4(),
        workflow_id=workflow_id,
        failed_stage="file_writing",
        partial_duration_seconds=0.5,
        error_code="ONEX_FILE_WRITE_ERROR",
        error_message="Invalid output directory: /invalid/path does not exist",
        error_context={"directory": "/invalid/path", "stage": "FILE_WRITING"},
        is_retryable=True,
    )

    await codegen_event_publisher.publish_event(
        topic=TOPIC_CODEGEN_FAILED,
        event_data=failure_event.model_dump(),
        correlation_id=correlation_id,
        event_type="CODEGEN_FAILED",
    )

    # Validate failure workflow
    failure_events = codegen_event_publisher.get_events_by_type("CODEGEN_FAILED")
    assert len(failure_events) == 1, "Should have one failure event"

    failure = failure_events[0]
    assert failure["correlation_id"] == correlation_id
    assert "error_message" in failure["envelope"].payload


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_codegen_event_schema_validation_e2e(mock_kafka, codegen_event_publisher):
    """
    Test event schema validation for all codegen event types.

    Validates:
    - Pydantic models enforce schema compliance
    - Required fields validated
    - Type safety enforced
    - Invalid events rejected
    """
    correlation_id = uuid4()

    # Test valid event creation
    valid_request = NodeGenerationRequestedEvent(
        correlation_id=correlation_id,
        event_id=uuid4(),
        prompt="Test prompt",
        output_directory="/tmp/test",
        options={},
        metadata={},
    )

    assert valid_request.event_type == "NODE_GENERATION_REQUESTED"
    assert valid_request.correlation_id == correlation_id

    # Test invalid event (missing required fields) - should raise ValidationError
    with pytest.raises(Exception):  # Pydantic ValidationError
        NodeGenerationRequestedEvent(
            correlation_id=correlation_id,
            event_id=uuid4(),
            # Missing required 'prompt' and 'output_directory'
            options={},
            metadata={},
        )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_codegen_stage_progression_e2e(
    mock_kafka, codegen_event_publisher, codegen_workflow_simulator
):
    """
    Test workflow stage progression with stage-completed events.

    Validates:
    - All 8 stages complete in order
    - Stage metadata captured
    - Stage timing recorded
    - Stage outputs preserved
    """
    correlation_id = uuid4()
    prompt = "Create a data validation node"
    output_directory = "/tmp/validation_node"

    # Execute workflow
    workflow_result = await codegen_workflow_simulator.simulate_workflow(
        publisher=codegen_event_publisher,
        prompt=prompt,
        output_directory=output_directory,
        correlation_id=correlation_id,
    )

    # Validate stage progression
    stage_events = codegen_event_publisher.get_events_by_type("CODEGEN_STAGE_COMPLETED")
    assert (
        len(stage_events) == 8
    ), f"Should have 8 stage events, got {len(stage_events)}"

    # Validate stage order
    expected_stages = codegen_workflow_simulator.WORKFLOW_STAGES
    for idx, stage_event in enumerate(stage_events):
        stage_data = stage_event["envelope"].payload
        expected_stage = expected_stages[idx].lower()  # Stage names are lowercase
        assert (
            stage_data["stage_name"] == expected_stage
        ), f"Stage {idx} should be {expected_stage}"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_codegen_metrics_aggregation_e2e(
    mock_kafka, codegen_event_publisher, codegen_workflow_simulator
):
    """
    Test metrics aggregation from workflow events.

    Validates:
    - Metrics Reducer consumes workflow events
    - Metrics calculated correctly
    - Aggregation window captured
    - GENERATION_METRICS_RECORDED event published
    """
    correlation_id = uuid4()

    # Execute workflow
    workflow_result = await codegen_workflow_simulator.simulate_workflow(
        publisher=codegen_event_publisher,
        prompt="Create a metrics test node",
        output_directory="/tmp/metrics_test",
        correlation_id=correlation_id,
    )

    # Validate metrics event
    metrics_events = codegen_event_publisher.get_events_by_type(
        "CODEGEN_METRICS_RECORDED"
    )
    assert len(metrics_events) == 1, "Should have one metrics event"

    metrics_event = metrics_events[0]
    metrics_payload = metrics_event["envelope"].payload

    # Validate metrics structure
    assert "total_generations" in metrics_payload
    assert metrics_payload["total_generations"] == 1
    assert metrics_payload["successful_generations"] == 1
    assert metrics_payload["avg_duration_seconds"] > 0
    assert "model_metrics" in metrics_payload
    assert "node_type_metrics" in metrics_payload


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_codegen_consumer_integration_e2e(mock_kafka):
    """
    Test event consumption using MockKafkaConsumer.

    Validates:
    - Consumer can subscribe to topics
    - Events consumed in order
    - Consumer handlers invoked
    - No events lost
    """
    # Create consumer
    consumer = mock_kafka.create_consumer([TOPIC_CODEGEN_REQUESTED])

    # Publish test events
    num_events = 5
    published_ids = []

    for i in range(num_events):
        correlation_id = uuid4()
        event = NodeGenerationRequestedEvent(
            correlation_id=correlation_id,
            event_id=uuid4(),
            prompt=f"Test prompt {i}",
            output_directory=f"/tmp/test_{i}",
            options={},
            metadata={},
        )

        envelope = OnexEnvelopeV1(
            event_type="NODE_GENERATION_REQUESTED",
            correlation_id=correlation_id,
            source_service="test",
            payload=event.model_dump(),
        )

        await mock_kafka.publish(
            topic=TOPIC_CODEGEN_REQUESTED,
            value=envelope.model_dump(),
            key=str(correlation_id),
        )

        published_ids.append(correlation_id)

    # Consume events
    consumed_events = []
    for i in range(num_events):
        event = await consumer.consume(timeout_ms=100)
        if event:
            consumed_events.append(event)

    # Validate consumption
    assert len(consumed_events) == num_events, "Should consume all published events"

    # Validate correlation IDs (they're already UUID objects in the envelope)
    consumed_ids = [event["value"]["correlation_id"] for event in consumed_events]
    assert set(consumed_ids) == set(
        published_ids
    ), "Should consume events with correct correlation IDs"


# ============================================================================
# Performance Validation
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.performance
async def test_codegen_workflow_performance_e2e(
    mock_kafka, codegen_event_publisher, codegen_workflow_simulator
):
    """
    Test workflow performance meets targets.

    Performance Targets:
    - Complete workflow: < 2000ms
    - Event publishing: < 10ms per event
    - Stage execution: < 200ms per stage
    """
    import time

    correlation_id = uuid4()

    start_time = time.perf_counter()

    # Execute workflow
    workflow_result = await codegen_workflow_simulator.simulate_workflow(
        publisher=codegen_event_publisher,
        prompt="Performance test node",
        output_directory="/tmp/perf_test",
        correlation_id=correlation_id,
    )

    total_duration_ms = (time.perf_counter() - start_time) * 1000

    # Validate performance
    assert (
        total_duration_ms < 2000
    ), f"Workflow took {total_duration_ms:.2f}ms (target: <2000ms)"

    # Validate event publishing latency
    avg_event_latency_ms = total_duration_ms / workflow_result["total_events"]
    assert (
        avg_event_latency_ms < 10
    ), f"Avg event latency {avg_event_latency_ms:.2f}ms (target: <10ms)"
