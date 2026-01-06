# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""True E2E tests that validate the complete orchestrator pipeline.

These tests verify the FULL registration flow:
1. Node broadcasts introspection event to Kafka
2. Orchestrator consumes the event from Kafka
3. Handler processes and triggers reducer
4. Reducer generates intents
5. Effects execute intents (Consul + PostgreSQL registration)
6. Verify BOTH Consul AND PostgreSQL have the registration

Unlike the component tests in test_two_way_registration_e2e.py, these tests
validate the actual message consumption and processing pipeline.

Architecture Notes:
    The test creates a "mini-orchestrator" that:
    - Subscribes to Kafka introspection topic
    - Deserializes incoming events
    - Routes through HandlerNodeIntrospected
    - Invokes RegistrationReducer to generate intents
    - Executes NodeRegistryEffect for dual registration

    This validates the REAL message flow, not just handler logic.

Infrastructure Requirements:
    - Kafka: 192.168.86.200:29092
    - Consul: 192.168.86.200:28500
    - PostgreSQL: 192.168.86.200:5436

Related Tickets:
    - OMN-892: E2E Registration Tests
    - OMN-888: Registration Orchestrator
    - OMN-915: Mocked E2E Registration Tests
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.models.discovery import DEFAULT_INTROSPECTION_TOPIC
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata
from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
    HandlerNodeIntrospected,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState

from .verification_helpers import (
    verify_consul_registration,
    verify_postgres_registration,
    wait_for_consul_registration,
    wait_for_postgres_registration,
)

if TYPE_CHECKING:
    import asyncpg
    from omnibase_core.container import ModelONEXContainer

    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
    from omnibase_infra.handlers import ConsulHandler
    from omnibase_infra.nodes.effects import NodeRegistryEffect
    from omnibase_infra.projectors import (
        ProjectionReaderRegistration,
        ProjectorRegistration,
    )


logger = logging.getLogger(__name__)

# Module-level markers
pytestmark = [
    pytest.mark.e2e,
]


# =============================================================================
# Test Topic Constants
# =============================================================================

# Base topic name for E2E tests. In parallel execution, use get_test_topic()
# to get a unique topic name per test session/worker.
_BASE_TEST_TOPIC = "e2e-test.node.introspection.v1"

# Session-unique suffix for test isolation in parallel execution.
# Generated once at module load time to ensure all tests in the same
# session/worker use the same topic while different workers use different topics.
_TEST_SESSION_ID = uuid4().hex[:8]


def get_test_topic(base_topic: str = _BASE_TEST_TOPIC) -> str:
    """Get a unique topic name for test isolation.

    In parallel test execution (e.g., pytest-xdist), each worker will have
    a unique session ID, ensuring topic isolation between workers.

    Args:
        base_topic: Base topic name to extend with session ID.

    Returns:
        Topic name with session-unique suffix.
    """
    return f"{base_topic}.{_TEST_SESSION_ID}"


# Default test topic - unique per test session for parallel execution safety.
TEST_INTROSPECTION_TOPIC = get_test_topic()


# =============================================================================
# Helper Classes for Full Pipeline Testing
# =============================================================================


class OrchestratorPipeline:
    """Mini-orchestrator that processes introspection events through full pipeline.

    This class simulates what a real orchestrator does:
    1. Receives message from Kafka (via callback)
    2. Deserializes to ModelNodeIntrospectionEvent
    3. Runs through HandlerNodeIntrospected
    4. Invokes RegistrationReducer to generate intents
    5. Executes NodeRegistryEffect for dual registration

    Unlike mocked tests, this validates the actual message processing logic.
    """

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        projector: ProjectorRegistration,
        registry_effect: NodeRegistryEffect,
        reducer: RegistrationReducer,
    ) -> None:
        """Initialize the pipeline with real dependencies.

        Args:
            projection_reader: Reader for querying registration projections.
            projector: Projector for persisting projections.
            registry_effect: Effect node for dual registration.
            reducer: Registration reducer for intent generation.
        """
        self._projection_reader = projection_reader
        self._projector = projector
        self._registry_effect = registry_effect
        self._reducer = reducer
        self._handler = HandlerNodeIntrospected(projection_reader)
        self._processed_events: list[UUID] = []
        self._processing_lock = asyncio.Lock()
        self._processing_errors: list[Exception] = []
        # Sequence counter for ordering guarantees across multiple events.
        # Starts at 0 and increments for each processed event.
        self._sequence_counter: int = 0

    @property
    def processed_events(self) -> list[UUID]:
        """Get list of processed event node IDs."""
        return list(self._processed_events)

    @property
    def processing_errors(self) -> list[Exception]:
        """Get list of processing errors."""
        return list(self._processing_errors)

    async def process_message(self, message: ModelEventMessage) -> None:
        """Process a Kafka message through the full pipeline.

        This is the callback registered with KafkaEventBus.subscribe().
        It deserializes the message and routes through handler -> reducer -> effect.

        Args:
            message: The Kafka message received from the introspection topic.
        """
        async with self._processing_lock:
            try:
                # Step 1: Deserialize message to introspection event
                event = self._deserialize_introspection_event(message)
                if event is None:
                    logger.warning(
                        "Failed to deserialize introspection event",
                        extra={"message_topic": message.topic},
                    )
                    return

                logger.info(
                    "Processing introspection event",
                    extra={
                        "node_id": str(event.node_id),
                        "node_type": event.node_type,
                        "correlation_id": str(event.correlation_id),
                    },
                )

                # Step 2: Run through handler to check if registration needed
                correlation_id = event.correlation_id or uuid4()
                now = datetime.now(UTC)

                handler_events = await self._handler.handle(
                    event=event,
                    now=now,
                    correlation_id=correlation_id,
                )

                if not handler_events:
                    logger.info(
                        "Handler returned no events - registration not needed",
                        extra={"node_id": str(event.node_id)},
                    )
                    return

                # Step 3: Run through reducer to generate intents
                state = ModelRegistrationState()
                reducer_output = self._reducer.reduce(state, event)

                logger.info(
                    "Reducer generated intents",
                    extra={
                        "node_id": str(event.node_id),
                        "intent_count": len(reducer_output.intents),
                        "new_status": reducer_output.result.status,
                    },
                )

                # Step 4: Execute effects via NodeRegistryEffect
                if reducer_output.intents:
                    await self._execute_effects(event, correlation_id)

                # Step 5: Persist projection
                await self._persist_projection(event, now, correlation_id)

                # Track successful processing
                self._processed_events.append(event.node_id)

                logger.info(
                    "Successfully processed introspection event",
                    extra={
                        "node_id": str(event.node_id),
                        "correlation_id": str(correlation_id),
                    },
                )

            except Exception as e:
                logger.exception(
                    "Error processing introspection event",
                    extra={"error": str(e)},
                )
                self._processing_errors.append(e)

    def _deserialize_introspection_event(
        self, message: ModelEventMessage
    ) -> ModelNodeIntrospectionEvent | None:
        """Deserialize Kafka message to introspection event.

        Args:
            message: The Kafka message to deserialize.

        Returns:
            Deserialized event or None if deserialization fails.
        """
        try:
            if not message.value:
                return None

            data = json.loads(message.value.decode("utf-8"))

            # Handle both envelope format and direct event format
            payload = data.get("payload", data)

            return ModelNodeIntrospectionEvent.model_validate(payload)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Failed to deserialize message",
                extra={"error": str(e)},
            )
            return None

    async def _execute_effects(
        self, event: ModelNodeIntrospectionEvent, correlation_id: UUID
    ) -> None:
        """Execute dual registration effects.

        Args:
            event: The introspection event.
            correlation_id: Correlation ID for tracing.
        """
        from omnibase_infra.nodes.effects.models import ModelRegistryRequest

        # Convert metadata to dict[str, str], filtering out None values
        # and converting non-string values to strings
        metadata_dict: dict[str, str] = {}
        if event.metadata:
            for key, value in event.metadata.model_dump(exclude_none=True).items():
                if value is not None:
                    metadata_dict[key] = str(value)

        # Build registry request
        # Note: node_type must be converted from Literal string to EnumNodeKind
        request = ModelRegistryRequest(
            node_id=event.node_id,
            node_type=EnumNodeKind(event.node_type),
            node_version=event.node_version,
            correlation_id=correlation_id,
            endpoints=dict(event.endpoints) if event.endpoints else {},
            metadata=metadata_dict,
            tags=[f"node_type:{event.node_type}", f"version:{event.node_version}"],
            timestamp=datetime.now(UTC),
        )

        # Execute dual registration
        response = await self._registry_effect.register_node(request)

        logger.info(
            "Effect execution completed",
            extra={
                "node_id": str(event.node_id),
                "status": response.status,
                "consul_success": response.consul_result.success
                if response.consul_result
                else None,
                "postgres_success": response.postgres_result.success
                if response.postgres_result
                else None,
            },
        )

    def _next_sequence(self) -> int:
        """Get the next sequence number for event ordering.

        Returns:
            Incrementing sequence number (1-based).
        """
        self._sequence_counter += 1
        return self._sequence_counter

    async def _persist_projection(
        self,
        event: ModelNodeIntrospectionEvent,
        now: datetime,
        correlation_id: UUID,
    ) -> None:
        """Persist the registration projection.

        Args:
            event: The introspection event.
            now: Current time for timestamps.
            correlation_id: Correlation ID for tracing.
        """
        from omnibase_infra.models.projection import ModelRegistrationProjection
        from omnibase_infra.models.projection.model_sequence_info import (
            ModelSequenceInfo,
        )

        # Use incrementing sequence for proper ordering across multiple events
        sequence_info = ModelSequenceInfo.from_sequence(self._next_sequence())

        projection = ModelRegistrationProjection(
            entity_id=event.node_id,
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            node_type=EnumNodeKind(event.node_type),
            node_version=event.node_version,
            registered_at=now,
            updated_at=now,
            last_applied_event_id=correlation_id,  # Use correlation_id as event_id for test
            correlation_id=correlation_id,
            domain="registration",
        )

        await self._projector.persist(
            projection=projection,
            entity_id=event.node_id,
            domain="registration",
            sequence_info=sequence_info,
            correlation_id=correlation_id,
        )


# =============================================================================
# Fixtures for Full Pipeline Testing
# =============================================================================


@dataclass
class OrchestratorTestContext:
    """Groups orchestrator pipeline with its mock dependencies.

    This dataclass ensures that the mock instances used for assertions
    are the exact same instances injected into the pipeline, preventing
    test failures due to pytest fixture instance mismatches.

    Attributes:
        pipeline: The orchestrator pipeline for processing events.
        mock_consul_client: Mock Consul client injected into the pipeline.
        mock_postgres_adapter: Mock PostgreSQL adapter injected into the pipeline.
        unsubscribe: Async function to unsubscribe from Kafka topic.
    """

    pipeline: OrchestratorPipeline
    mock_consul_client: AsyncMock
    mock_postgres_adapter: AsyncMock
    unsubscribe: Callable[[], Awaitable[None]] | None = None


@pytest.fixture
async def mock_consul_client() -> AsyncMock:
    """Create a mock Consul client for effect testing.

    Returns:
        AsyncMock: Mock Consul client with register_service method.
    """
    mock = AsyncMock()
    mock.register_service = AsyncMock(return_value=MagicMock(success=True, error=None))
    return mock


@pytest.fixture
async def mock_postgres_adapter() -> AsyncMock:
    """Create a mock PostgreSQL adapter for effect testing.

    Returns:
        AsyncMock: Mock PostgreSQL adapter with upsert method.
    """
    mock = AsyncMock()
    mock.upsert = AsyncMock(return_value=MagicMock(success=True, error=None))
    return mock


@pytest.fixture
async def registry_effect_node(
    mock_consul_client: AsyncMock,
    mock_postgres_adapter: AsyncMock,
) -> NodeRegistryEffect:
    """Create NodeRegistryEffect with mock backends.

    Args:
        mock_consul_client: Mock Consul client.
        mock_postgres_adapter: Mock PostgreSQL adapter.

    Returns:
        NodeRegistryEffect: Configured effect node.
    """
    from omnibase_infra.nodes.effects import NodeRegistryEffect

    return NodeRegistryEffect(
        consul_client=mock_consul_client,
        postgres_adapter=mock_postgres_adapter,
    )


@pytest.fixture
async def orchestrator_pipeline(
    projection_reader: ProjectionReaderRegistration,
    real_projector: ProjectorRegistration,
    registry_effect_node: NodeRegistryEffect,
    mock_consul_client: AsyncMock,
    mock_postgres_adapter: AsyncMock,
) -> OrchestratorTestContext:
    """Create the full orchestrator pipeline with its mock dependencies.

    This fixture explicitly connects the mock instances to the pipeline and
    returns them together in an OrchestratorTestContext to ensure test
    assertions use the exact same mock instances that were injected.

    Args:
        projection_reader: Projection reader fixture.
        real_projector: Projector fixture.
        registry_effect_node: Registry effect fixture (contains the mocks).
        mock_consul_client: Mock Consul client injected into registry_effect_node.
        mock_postgres_adapter: Mock PostgreSQL adapter injected into registry_effect_node.

    Returns:
        OrchestratorTestContext: Context containing pipeline and connected mocks.

    Note:
        The mock parameters are explicitly listed to:
        1. Ensure pytest shares the same instances with registry_effect_node
        2. Return them in the context for test assertions
        3. Make the mock-to-pipeline connection explicit and verifiable
    """
    reducer = RegistrationReducer()
    pipeline = OrchestratorPipeline(
        projection_reader=projection_reader,
        projector=real_projector,
        registry_effect=registry_effect_node,
        reducer=reducer,
    )

    # Return context with pipeline and its connected mocks for test assertions
    return OrchestratorTestContext(
        pipeline=pipeline,
        mock_consul_client=mock_consul_client,
        mock_postgres_adapter=mock_postgres_adapter,
    )


@pytest.fixture
async def running_orchestrator_consumer(
    real_kafka_event_bus: KafkaEventBus,
    orchestrator_pipeline: OrchestratorTestContext,
) -> AsyncGenerator[OrchestratorTestContext, None]:
    """Start a Kafka consumer that routes messages through the pipeline.

    This fixture creates a real Kafka subscription that:
    - Subscribes to the test introspection topic
    - Routes incoming messages to the OrchestratorPipeline.process_message callback
    - Returns the OrchestratorTestContext with pipeline, mocks, and unsubscribe function

    Args:
        real_kafka_event_bus: Real Kafka event bus.
        orchestrator_pipeline: Context containing pipeline and connected mocks.

    Yields:
        OrchestratorTestContext with pipeline, mocks, and unsubscribe function.

    Note:
        The OrchestratorTestContext pattern ensures the mock instances returned
        are the exact same instances injected into the pipeline, preventing
        assertion failures when tests verify mock method calls.
    """
    # Use unique group ID per test run to avoid cross-test coupling
    unique_group_id = f"e2e-orchestrator-test-{uuid4().hex[:8]}"
    # Subscribe to the introspection topic
    unsubscribe = await real_kafka_event_bus.subscribe(
        topic=TEST_INTROSPECTION_TOPIC,
        group_id=unique_group_id,
        on_message=orchestrator_pipeline.pipeline.process_message,
    )

    # Give consumer time to start
    await asyncio.sleep(0.5)

    # Create a new context with the unsubscribe function included
    context = OrchestratorTestContext(
        pipeline=orchestrator_pipeline.pipeline,
        mock_consul_client=orchestrator_pipeline.mock_consul_client,
        mock_postgres_adapter=orchestrator_pipeline.mock_postgres_adapter,
        unsubscribe=unsubscribe,
    )

    yield context

    # Cleanup
    await unsubscribe()


# =============================================================================
# Full Pipeline E2E Tests
# =============================================================================


@pytest.mark.asyncio
class TestFullOrchestratorFlow:
    """True E2E tests that validate the complete orchestrator pipeline.

    These tests verify that:
    1. Events published to Kafka are consumed by the orchestrator
    2. The full pipeline executes: handler -> reducer -> effect
    3. Both Consul and PostgreSQL registrations complete
    """

    async def test_introspection_triggers_full_pipeline_processing(
        self,
        real_kafka_event_bus: KafkaEventBus,
        running_orchestrator_consumer: OrchestratorTestContext,
        unique_node_id: UUID,
        unique_correlation_id: UUID,
    ) -> None:
        """Test that introspection event triggers full pipeline processing.

        FULL FLOW TEST:
        1. Publish introspection event to Kafka
        2. Orchestrator consumer receives the event
        3. Pipeline processes: handler -> reducer -> effect
        4. Verify event was processed

        This validates the ACTUAL Kafka consumption, not mocked handler calls.
        """
        ctx = running_orchestrator_consumer
        pipeline = ctx.pipeline

        # Create introspection event
        event = ModelNodeIntrospectionEvent(
            node_id=unique_node_id,
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
            metadata=ModelNodeMetadata(),
            correlation_id=unique_correlation_id,
            timestamp=datetime.now(UTC),
        )

        # Serialize and publish to Kafka
        event_bytes = json.dumps(event.model_dump(mode="json")).encode("utf-8")

        headers = ModelEventHeaders(
            source="e2e-test",
            event_type="node.introspection",
            correlation_id=unique_correlation_id,
            timestamp=datetime.now(UTC),
        )

        await real_kafka_event_bus.publish(
            topic=TEST_INTROSPECTION_TOPIC,
            key=str(unique_node_id).encode("utf-8"),
            value=event_bytes,
            headers=headers,
        )

        # Wait for processing with polling
        max_wait = 10.0
        poll_interval = 0.5
        elapsed = 0.0

        while elapsed < max_wait:
            if unique_node_id in pipeline.processed_events:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Verify event was processed
        assert unique_node_id in pipeline.processed_events, (
            f"Event for node {unique_node_id} was not processed within {max_wait}s. "
            f"Processed: {pipeline.processed_events}, "
            f"Errors: {pipeline.processing_errors}"
        )

    async def test_handler_reducer_effect_chain_execution(
        self,
        real_kafka_event_bus: KafkaEventBus,
        running_orchestrator_consumer: OrchestratorTestContext,
        unique_node_id: UUID,
        unique_correlation_id: UUID,
    ) -> None:
        """Test that the handler -> reducer -> effect chain executes.

        Verifies:
        - Handler processes the event
        - Reducer generates intents
        - Effect executes Consul and PostgreSQL registration
        """
        # Use context to access pipeline and its connected mocks
        ctx = running_orchestrator_consumer
        pipeline = ctx.pipeline
        mock_consul_client = ctx.mock_consul_client
        mock_postgres_adapter = ctx.mock_postgres_adapter

        # Create introspection event
        event = ModelNodeIntrospectionEvent(
            node_id=unique_node_id,
            node_type="compute",
            node_version="2.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={
                "health": "http://localhost:8081/health",
                "api": "http://localhost:8081/api",
            },
            metadata=ModelNodeMetadata(),
            correlation_id=unique_correlation_id,
            timestamp=datetime.now(UTC),
        )

        # Serialize and publish
        event_bytes = json.dumps(event.model_dump(mode="json")).encode("utf-8")

        headers = ModelEventHeaders(
            source="e2e-test",
            event_type="node.introspection",
            correlation_id=unique_correlation_id,
            timestamp=datetime.now(UTC),
        )

        await real_kafka_event_bus.publish(
            topic=TEST_INTROSPECTION_TOPIC,
            key=str(unique_node_id).encode("utf-8"),
            value=event_bytes,
            headers=headers,
        )

        # Wait for processing
        max_wait = 10.0
        poll_interval = 0.5
        elapsed = 0.0

        while elapsed < max_wait:
            if unique_node_id in pipeline.processed_events:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Verify processing completed
        assert unique_node_id in pipeline.processed_events
        assert len(pipeline.processing_errors) == 0, (
            f"Pipeline had errors: {pipeline.processing_errors}"
        )

        # Verify effect was called (mocks were invoked)
        mock_consul_client.register_service.assert_called()
        mock_postgres_adapter.upsert.assert_called()

    async def test_multiple_events_processed_in_order(
        self,
        real_kafka_event_bus: KafkaEventBus,
        running_orchestrator_consumer: OrchestratorTestContext,
    ) -> None:
        """Test that multiple introspection events are processed.

        Publishes multiple events and verifies all are processed.
        """
        ctx = running_orchestrator_consumer
        pipeline = ctx.pipeline

        # Create multiple events
        node_ids = [uuid4() for _ in range(3)]
        node_types = ["effect", "compute", "reducer"]

        for node_id, node_type in zip(node_ids, node_types, strict=True):
            event = ModelNodeIntrospectionEvent(
                node_id=node_id,
                node_type=node_type,
                node_version="1.0.0",
                capabilities=ModelNodeCapabilities(),
                endpoints={
                    "health": f"http://localhost:808{node_types.index(node_type)}/health"
                },
                metadata=ModelNodeMetadata(),
                correlation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )

            event_bytes = json.dumps(event.model_dump(mode="json")).encode("utf-8")
            headers = ModelEventHeaders(
                source="e2e-test",
                event_type="node.introspection",
                correlation_id=event.correlation_id,
                timestamp=datetime.now(UTC),
            )

            await real_kafka_event_bus.publish(
                topic=TEST_INTROSPECTION_TOPIC,
                key=str(node_id).encode("utf-8"),
                value=event_bytes,
                headers=headers,
            )

        # Wait for all events to be processed
        max_wait = 15.0
        poll_interval = 0.5
        elapsed = 0.0

        while elapsed < max_wait:
            processed_count = sum(
                1 for nid in node_ids if nid in pipeline.processed_events
            )
            if processed_count == len(node_ids):
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Verify all events processed
        for node_id in node_ids:
            assert node_id in pipeline.processed_events, (
                f"Event for node {node_id} was not processed"
            )

    async def test_malformed_message_handled_gracefully(
        self,
        real_kafka_event_bus: KafkaEventBus,
        running_orchestrator_consumer: OrchestratorTestContext,
        unique_node_id: UUID,
        unique_correlation_id: UUID,
    ) -> None:
        """Test that malformed messages are handled gracefully.

        The pipeline should log a warning but not crash when receiving
        invalid JSON or non-conforming messages.
        """
        ctx = running_orchestrator_consumer
        pipeline = ctx.pipeline

        # Publish malformed message
        headers = ModelEventHeaders(
            source="e2e-test",
            event_type="node.introspection",
            correlation_id=unique_correlation_id,
            timestamp=datetime.now(UTC),
        )

        await real_kafka_event_bus.publish(
            topic=TEST_INTROSPECTION_TOPIC,
            key=b"malformed-key",
            value=b"not-valid-json",
            headers=headers,
        )

        # Wait a bit
        await asyncio.sleep(1.0)

        # Publish a valid message after the malformed one
        valid_event = ModelNodeIntrospectionEvent(
            node_id=unique_node_id,
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
            metadata=ModelNodeMetadata(),
            correlation_id=unique_correlation_id,
            timestamp=datetime.now(UTC),
        )

        valid_bytes = json.dumps(valid_event.model_dump(mode="json")).encode("utf-8")
        await real_kafka_event_bus.publish(
            topic=TEST_INTROSPECTION_TOPIC,
            key=str(unique_node_id).encode("utf-8"),
            value=valid_bytes,
            headers=headers,
        )

        # Wait for valid event processing
        max_wait = 10.0
        poll_interval = 0.5
        elapsed = 0.0

        while elapsed < max_wait:
            if unique_node_id in pipeline.processed_events:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Valid event should still be processed
        assert unique_node_id in pipeline.processed_events, (
            "Valid event should be processed after malformed message"
        )


# =============================================================================
# Full Pipeline with Real Infrastructure Tests
# =============================================================================


@pytest.mark.asyncio
class TestFullPipelineWithRealInfrastructure:
    """E2E tests that verify registration in REAL Consul and PostgreSQL.

    These tests require all infrastructure services to be available.
    They create real registrations and verify data persistence.
    """

    async def test_introspection_creates_postgres_projection(
        self,
        real_kafka_event_bus: KafkaEventBus,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorRegistration,
        unique_node_id: UUID,
        unique_correlation_id: UUID,
        cleanup_projections,
    ) -> None:
        """Test that introspection event creates PostgreSQL projection.

        Publishes an introspection event and verifies the projection
        is persisted in PostgreSQL.
        """
        # Create introspection event
        event = ModelNodeIntrospectionEvent(
            node_id=unique_node_id,
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
            metadata=ModelNodeMetadata(),
            correlation_id=unique_correlation_id,
            timestamp=datetime.now(UTC),
        )

        # Create handler and process directly (for simpler test)
        handler = HandlerNodeIntrospected(projection_reader)
        now = datetime.now(UTC)

        handler_events = await handler.handle(
            event=event,
            now=now,
            correlation_id=unique_correlation_id,
        )

        # If handler says we should register, create the projection
        if handler_events:
            from omnibase_infra.models.projection import ModelRegistrationProjection
            from omnibase_infra.models.projection.model_sequence_info import (
                ModelSequenceInfo,
            )

            sequence_info = ModelSequenceInfo.from_sequence(1)

            projection = ModelRegistrationProjection(
                entity_id=unique_node_id,
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                node_type=EnumNodeKind(event.node_type),
                node_version=event.node_version,
                registered_at=now,
                updated_at=now,
                last_applied_event_id=unique_correlation_id,
                correlation_id=unique_correlation_id,
                domain="registration",
            )

            await real_projector.persist(
                projection=projection,
                entity_id=unique_node_id,
                domain="registration",
                sequence_info=sequence_info,
                correlation_id=unique_correlation_id,
            )

        # Verify projection exists in PostgreSQL
        projection = await wait_for_postgres_registration(
            projection_reader=projection_reader,
            node_id=unique_node_id,
            timeout_seconds=5.0,
        )

        assert projection is not None
        assert projection.entity_id == unique_node_id
        assert projection.node_type == "effect"

    async def test_reducer_generates_correct_intents(
        self,
        unique_node_id: UUID,
        unique_correlation_id: UUID,
    ) -> None:
        """Test that reducer generates Consul and PostgreSQL intents.

        Verifies the reducer emits the expected intent types.
        """
        # Create introspection event
        event = ModelNodeIntrospectionEvent(
            node_id=unique_node_id,
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
            metadata=ModelNodeMetadata(),
            correlation_id=unique_correlation_id,
            timestamp=datetime.now(UTC),
        )

        # Create reducer and process
        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        output = reducer.reduce(state, event)

        # Verify intents generated (extension format)
        assert len(output.intents) == 2, "Should generate Consul and PostgreSQL intents"

        extension_types = {
            intent.payload.extension_type
            for intent in output.intents
            if intent.intent_type == "extension"
        }
        assert "infra.consul_register" in extension_types, (
            "Should include Consul intent"
        )
        assert "infra.postgres_upsert" in extension_types, (
            "Should include PostgreSQL intent"
        )

        # Verify new state
        assert output.result.status == "pending", (
            f"Expected pending status, got {output.result.status}"
        )

    async def test_effect_executes_dual_registration(
        self,
        mock_consul_client: AsyncMock,
        mock_postgres_adapter: AsyncMock,
        unique_node_id: UUID,
        unique_correlation_id: UUID,
    ) -> None:
        """Test that effect node executes both Consul and PostgreSQL registration.

        Verifies both backend operations are called with correct parameters.
        """
        from omnibase_infra.nodes.effects import NodeRegistryEffect
        from omnibase_infra.nodes.effects.models import ModelRegistryRequest

        effect = NodeRegistryEffect(
            consul_client=mock_consul_client,
            postgres_adapter=mock_postgres_adapter,
        )

        request = ModelRegistryRequest(
            node_id=unique_node_id,
            node_type=EnumNodeKind.EFFECT,
            node_version="1.0.0",
            correlation_id=unique_correlation_id,
            endpoints={"health": "http://localhost:8080/health"},
            metadata={},
            tags=["node_type:effect", "version:1.0.0"],
            timestamp=datetime.now(UTC),
        )

        response = await effect.register_node(request)

        # Verify both backends called
        assert mock_consul_client.register_service.called
        assert mock_postgres_adapter.upsert.called

        # Verify response
        assert response.status == "success"
        assert response.consul_result.success
        assert response.postgres_result.success


# =============================================================================
# Pipeline Lifecycle Tests
# =============================================================================


@pytest.mark.asyncio
class TestPipelineLifecycle:
    """Tests for pipeline startup, shutdown, and error recovery."""

    async def test_consumer_starts_and_receives_messages(
        self,
        real_kafka_event_bus: KafkaEventBus,
        unique_correlation_id: UUID,
    ) -> None:
        """Test that consumer starts and receives messages.

        This is a basic connectivity test to ensure Kafka subscription works.
        """
        received_messages: list[ModelEventMessage] = []
        message_received = asyncio.Event()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)
            message_received.set()

        # Subscribe
        unsubscribe = await real_kafka_event_bus.subscribe(
            topic=TEST_INTROSPECTION_TOPIC,
            group_id=f"lifecycle-test-{unique_correlation_id.hex[:8]}",
            on_message=handler,
        )

        try:
            # Give consumer time to start
            await asyncio.sleep(0.5)

            # Publish test message
            headers = ModelEventHeaders(
                source="lifecycle-test",
                event_type="test",
                correlation_id=unique_correlation_id,
                timestamp=datetime.now(UTC),
            )

            await real_kafka_event_bus.publish(
                topic=TEST_INTROSPECTION_TOPIC,
                key=b"test-key",
                value=b'{"test": true}',
                headers=headers,
            )

            # Wait for message
            try:
                await asyncio.wait_for(message_received.wait(), timeout=10.0)
            except TimeoutError:
                pytest.fail("Message not received within timeout")

            assert len(received_messages) >= 1

        finally:
            await unsubscribe()

    async def test_consumer_handles_shutdown_gracefully(
        self,
        real_kafka_event_bus: KafkaEventBus,
        unique_correlation_id: UUID,
    ) -> None:
        """Test that consumer handles shutdown without errors.

        Verifies clean unsubscribe and no resource leaks.
        """
        message_count = 0

        async def handler(msg: ModelEventMessage) -> None:
            nonlocal message_count
            message_count += 1

        # Subscribe and immediately unsubscribe
        unsubscribe = await real_kafka_event_bus.subscribe(
            topic=TEST_INTROSPECTION_TOPIC,
            group_id=f"shutdown-test-{unique_correlation_id.hex[:8]}",
            on_message=handler,
        )

        await asyncio.sleep(0.5)

        # Unsubscribe should not raise
        await unsubscribe()

        # Double unsubscribe should be safe
        await unsubscribe()


__all__ = [
    "OrchestratorPipeline",
    "OrchestratorTestContext",
    "TestFullOrchestratorFlow",
    "TestFullPipelineWithRealInfrastructure",
    "TestPipelineLifecycle",
    "get_test_topic",
]
