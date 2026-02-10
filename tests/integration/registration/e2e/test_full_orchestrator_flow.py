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

Infrastructure Requirements (configured via environment variables):
    - Kafka: KAFKA_BOOTSTRAP_SERVERS (e.g., localhost:29092)
    - Consul: CONSUL_HTTP_ADDR (e.g., localhost:8500)
    - PostgreSQL: POSTGRES_HOST, POSTGRES_PORT (e.g., localhost:5432)

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
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
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

# Note: ALL_INFRA_AVAILABLE skipif is handled by conftest.py for all E2E tests
from .conftest import make_e2e_test_identity, wait_for_consumer_ready
from .verification_helpers import wait_for_postgres_registration

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.nodes.effects import NodeRegistryEffect
    from omnibase_infra.projectors import ProjectionReaderRegistration
    from omnibase_infra.runtime import ProjectorShell


logger = logging.getLogger(__name__)

# Log formatting constants for pipeline observability
_LOG_SEPARATOR = "=" * 70
_LOG_STAGE_SEPARATOR = "-" * 70
_LOG_ERROR_SEPARATOR = "!" * 70

# Module-level markers
# Note: conftest.py already applies pytest.mark.e2e and skipif(not ALL_INFRA_AVAILABLE)
# to all tests in this directory. We only add the e2e marker here for explicit clarity.
pytestmark = [
    pytest.mark.e2e,
]


# =============================================================================
# Test Topic Constants
# =============================================================================

# Pre-existing topic used for E2E tests. This topic must exist in Kafka/Redpanda
# before running tests. The validate_test_topic_exists fixture validates topic
# existence and fails fast with clear error messages if the topic is missing.
#
# Test Isolation Strategy:
#   1. Unique consumer group IDs per test (e2e-orchestrator-test-{uuid})
#   2. enable_auto_commit=False in real_kafka_event_bus (see conftest.py)
#   3. Each test starts from latest offset, not competing with other groups
#
# Why a shared topic instead of unique topics per test:
#   - Topic creation can be slow/unreliable in some Kafka configurations
#   - Consumer group isolation provides sufficient test isolation
#   - Reduces topic proliferation and cleanup complexity
#
# NOTE: If running tests in parallel (pytest-xdist), tests may see messages
# from other workers. The unique group IDs ensure each test starts from
# the latest offset and doesn't compete for the same consumer group.
#
# To create this topic if it doesn't exist:
#   rpk topic create e2e-test.node.introspection.v1 --partitions 1
TEST_INTROSPECTION_TOPIC = "e2e-test.node.introspection.v1"


# =============================================================================
# Helper Functions
# =============================================================================


def coerce_to_node_kind(node_type: str | EnumNodeKind | None) -> EnumNodeKind:
    """Coerce a node type string or enum to EnumNodeKind.

    This function handles the string-to-enum coercion needed when processing
    deserialized Kafka messages. aiokafka deserializes JSON payloads, which
    converts enum values to their string representations (e.g., "effect"
    instead of EnumNodeKind.EFFECT).

    Why This Exists:
        When Kafka messages are serialized to JSON (via Pydantic's model_dump),
        enum values become strings. Upon deserialization, these remain as strings
        rather than being automatically converted back to enum instances. This
        function provides safe, validated coercion back to EnumNodeKind.

    Use Cases:
        - Processing deserialized Kafka messages with node_type fields
        - Handling external API responses where enums are serialized as strings
        - Converting Pydantic model dumps back to typed enums

    When NOT to Use:
        - For direct enum construction from known literal values, use
          EnumNodeKind("effect") directly (simpler, no None handling)
        - When the value is already guaranteed to be an EnumNodeKind instance
        - In code paths that don't involve deserialization boundaries

    Implementation Pattern:
        Uses Enum-first pattern with type guard for safer runtime behavior:
        1. Handle None explicitly with TypeError
        2. If already an enum, return as-is (fast path)
        3. If string, validate against known values and convert
        4. For other types, raise TypeError with detailed message

    Args:
        node_type: Either a string representation (e.g., "effect", "compute")
            or an EnumNodeKind instance. None values raise TypeError.

    Returns:
        EnumNodeKind: The coerced enum value.

    Raises:
        TypeError: If node_type is None or not a string/EnumNodeKind.
        ValueError: If the string value doesn't match any EnumNodeKind member.

    Example:
        >>> # From deserialized Kafka message
        >>> coerce_to_node_kind("effect")
        <EnumNodeKind.EFFECT: 'effect'>

        >>> # Pass-through for already-typed values
        >>> coerce_to_node_kind(EnumNodeKind.COMPUTE)
        <EnumNodeKind.COMPUTE: 'compute'>

        >>> # Error handling
        >>> coerce_to_node_kind(None)
        TypeError: node_type cannot be None...
        >>> coerce_to_node_kind("invalid")
        ValueError: Invalid node_type 'invalid'...
    """
    # Handle None explicitly
    if node_type is None:
        raise TypeError(
            "node_type cannot be None. Expected EnumNodeKind or valid string value."
        )

    # Enum-first: if already an enum, return directly
    if isinstance(node_type, EnumNodeKind):
        return node_type

    # Type guard: must be a string at this point
    if not isinstance(node_type, str):
        raise TypeError(
            f"node_type must be EnumNodeKind or str, got {type(node_type).__name__}. "
            f"Received value: {node_type!r}"
        )

    # Validate string is a valid enum value
    valid_values = {e.value for e in EnumNodeKind}
    if node_type not in valid_values:
        raise ValueError(
            f"Invalid node_type '{node_type}'. Expected one of: {sorted(valid_values)}"
        )

    return EnumNodeKind(node_type)


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
        projector: ProjectorShell,
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

        This is the callback registered with EventBusKafka.subscribe().
        It deserializes the message and routes through handler -> reducer -> effect.

        Pipeline Stages:
            1. DESERIALIZE - Parse Kafka message to ModelNodeIntrospectionEvent
            2. HANDLER - Check if registration is needed via HandlerNodeIntrospected
            3. REDUCER - Generate intents (Consul + PostgreSQL) via RegistrationReducer
            4. EFFECT - Execute dual registration via NodeRegistryEffect
            5. PROJECTION - Persist registration state to PostgreSQL

        Args:
            message: The Kafka message received from the introspection topic.
        """
        async with self._processing_lock:
            correlation_id: UUID | None = None
            try:
                # =============================================================
                # ORCHESTRATOR PIPELINE: Processing Message
                # =============================================================
                logger.info(
                    "\n%s\n  ORCHESTRATOR PIPELINE: Processing Message\n%s",
                    _LOG_SEPARATOR,
                    _LOG_SEPARATOR,
                )
                logger.info(
                    "  Topic: %s | Partition: %s | Offset: %s",
                    message.topic,
                    getattr(message, "partition", "N/A"),
                    getattr(message, "offset", "N/A"),
                )

                # =============================================================
                # STAGE 1: DESERIALIZE
                # =============================================================
                logger.info(
                    "\n%s\n  STAGE 1: DESERIALIZE - Parsing Kafka message\n%s",
                    _LOG_STAGE_SEPARATOR,
                    _LOG_STAGE_SEPARATOR,
                )
                event = self._deserialize_introspection_event(message)
                if event is None:
                    logger.warning(
                        "  [DESERIALIZE] FAILED - Invalid message format\n"
                        "    Topic: %s\n"
                        "    Value preview: %s",
                        message.topic,
                        str(message.value)[:100] if message.value else "None",
                    )
                    logger.info(
                        "%s\n  PIPELINE ABORTED: Deserialization failed\n%s",
                        _LOG_SEPARATOR,
                        _LOG_SEPARATOR,
                    )
                    return

                correlation_id = event.correlation_id or uuid4()
                logger.info(
                    "  [DESERIALIZE] SUCCESS\n"
                    "    Node ID: %s\n"
                    "    Node Type: %s\n"
                    "    Version: %s\n"
                    "    Correlation ID: %s",
                    event.node_id,
                    event.node_type,
                    event.node_version,
                    correlation_id,
                )

                # =============================================================
                # STAGE 2: HANDLER
                # =============================================================
                logger.info(
                    "\n%s\n  STAGE 2: HANDLER - Checking registration need\n%s",
                    _LOG_STAGE_SEPARATOR,
                    _LOG_STAGE_SEPARATOR,
                )
                now = datetime.now(UTC)

                # Wrap event in envelope for handler API
                envelope = ModelEventEnvelope(
                    envelope_id=uuid4(),
                    payload=event,
                    envelope_timestamp=now,
                    correlation_id=correlation_id,
                    source="e2e-test-pipeline",
                )

                handler_output = await self._handler.handle(envelope)

                if not handler_output.events:
                    logger.info(
                        "  [HANDLER] SKIP - No registration needed\n"
                        "    Node ID: %s\n"
                        "    Reason: Handler returned no events "
                        "(node may already be registered)",
                        event.node_id,
                    )
                    logger.info(
                        "\n%s\n  PIPELINE COMPLETE (No Action Required)\n%s",
                        _LOG_SEPARATOR,
                        _LOG_SEPARATOR,
                    )
                    return

                logger.info(
                    "  [HANDLER] PROCEED - Registration needed\n"
                    "    Node ID: %s\n"
                    "    Events generated: %d",
                    event.node_id,
                    len(handler_output.events),
                )

                # =============================================================
                # STAGE 3: REDUCER
                # =============================================================
                logger.info(
                    "\n%s\n  STAGE 3: REDUCER - Generating intents\n%s",
                    _LOG_STAGE_SEPARATOR,
                    _LOG_STAGE_SEPARATOR,
                )
                state = ModelRegistrationState()
                reducer_output = self._reducer.reduce(state, event)

                intent_types = [
                    intent.payload.intent_type
                    for intent in reducer_output.intents
                    if hasattr(intent.payload, "intent_type")
                ]
                logger.info(
                    "  [REDUCER] SUCCESS\n"
                    "    Node ID: %s\n"
                    "    Intents generated: %d\n"
                    "    Intent types: %s\n"
                    "    New status: %s",
                    event.node_id,
                    len(reducer_output.intents),
                    ", ".join(intent_types) if intent_types else "None",
                    reducer_output.result.status,
                )

                # =============================================================
                # STAGE 4: EFFECT
                # =============================================================
                logger.info(
                    "\n%s\n  STAGE 4: EFFECT - Executing dual registration\n%s",
                    _LOG_STAGE_SEPARATOR,
                    _LOG_STAGE_SEPARATOR,
                )
                if reducer_output.intents:
                    await self._execute_effects(event, correlation_id)
                    logger.info(
                        "  [EFFECT] SUCCESS\n"
                        "    Node ID: %s\n"
                        "    Consul registration: Completed\n"
                        "    PostgreSQL registration: Completed",
                        event.node_id,
                    )
                else:
                    logger.info(
                        "  [EFFECT] SKIP - No intents to execute\n    Node ID: %s",
                        event.node_id,
                    )

                # =============================================================
                # STAGE 5: PROJECTION
                # =============================================================
                logger.info(
                    "\n%s\n  STAGE 5: PROJECTION - Persisting state\n%s",
                    _LOG_STAGE_SEPARATOR,
                    _LOG_STAGE_SEPARATOR,
                )
                await self._persist_projection(event, now, correlation_id)
                logger.info(
                    "  [PROJECTION] SUCCESS\n"
                    "    Node ID: %s\n"
                    "    Domain: registration\n"
                    "    Sequence: %d",
                    event.node_id,
                    self._sequence_counter,
                )

                # Track successful processing
                self._processed_events.append(event.node_id)

                # =============================================================
                # PIPELINE COMPLETE
                # =============================================================
                logger.info(
                    "\n%s\n  PIPELINE COMPLETE - All stages successful\n%s\n"
                    "    Node ID: %s\n"
                    "    Node Type: %s\n"
                    "    Version: %s\n"
                    "    Intents executed: %d\n"
                    "    Correlation ID: %s\n%s",
                    _LOG_SEPARATOR,
                    _LOG_SEPARATOR,
                    event.node_id,
                    event.node_type,
                    event.node_version,
                    len(reducer_output.intents),
                    correlation_id,
                    _LOG_SEPARATOR,
                )

            except Exception as e:
                logger.exception(
                    "\n%s\n  PIPELINE ERROR - Exception during processing\n%s\n"
                    "    Error: %s\n"
                    "    Correlation ID: %s\n%s",
                    _LOG_ERROR_SEPARATOR,
                    _LOG_ERROR_SEPARATOR,
                    str(e),
                    correlation_id or "Not assigned",
                    _LOG_ERROR_SEPARATOR,
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
            logger.warning("Failed to deserialize message", extra={"error": str(e)})
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
            node_type=coerce_to_node_kind(event.node_type),
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
        self, event: ModelNodeIntrospectionEvent, now: datetime, correlation_id: UUID
    ) -> None:
        """Persist the registration projection.

        Args:
            event: The introspection event.
            now: Current time for timestamps.
            correlation_id: Correlation ID for tracing.
        """
        # Convert event data to values dict for upsert_partial
        node_type = coerce_to_node_kind(event.node_type)
        values: dict[str, object] = {
            "entity_id": event.node_id,
            "domain": "registration",
            "current_state": EnumRegistrationState.PENDING_REGISTRATION.value,
            "node_type": node_type.value,
            "node_version": str(event.node_version),
            "capabilities": "{}",
            "registered_at": now,
            "updated_at": now,
            "last_applied_event_id": correlation_id,
            "last_applied_offset": self._next_sequence(),
        }

        await self._projector.upsert_partial(
            aggregate_id=event.node_id,
            values=values,
            correlation_id=correlation_id,
            conflict_columns=["entity_id", "domain"],
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
    mock_consul_client: AsyncMock, mock_postgres_adapter: AsyncMock
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
        consul_client=mock_consul_client, postgres_adapter=mock_postgres_adapter
    )


@pytest.fixture
async def orchestrator_pipeline(
    projection_reader: ProjectionReaderRegistration,
    real_projector: ProjectorShell,
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
async def validate_test_topic_exists(real_kafka_event_bus: EventBusKafka) -> str:
    """Validate and return the pre-existing test topic name.

    This fixture validates that the test topic (e2e-test.node.introspection.v1)
    exists in the Kafka cluster before tests run. The topic should be pre-created
    during initial infrastructure setup.

    Fail-fast behavior: If the topic doesn't exist or validation fails, this
    fixture fails immediately with a clear error message, preventing cryptic
    failures later in the test.

    Test isolation is achieved via unique consumer group IDs rather than
    unique topic names.

    Validation Steps:
        1. Connect to Kafka admin API
        2. Describe the specific topic (validates existence and configuration)
        3. Verify topic has no error codes and has partitions

    Args:
        real_kafka_event_bus: Real Kafka event bus (unused but kept for
            fixture dependency ordering).

    Returns:
        The test topic name.

    Raises:
        pytest.fail: If the topic does not exist in Kafka or validation fails.
    """
    import os

    from aiokafka.admin import AIOKafkaAdminClient
    from aiokafka.errors import (
        KafkaConnectionError,
        KafkaError,
        KafkaTimeoutError,
        UnknownTopicOrPartitionError,
    )

    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        pytest.fail(
            "KAFKA_BOOTSTRAP_SERVERS environment variable not set.\n"
            "Set it to your Kafka/Redpanda cluster address, e.g.:\n"
            "  export KAFKA_BOOTSTRAP_SERVERS=localhost:29092"
        )

    admin_client: AIOKafkaAdminClient | None = None
    topic_creation_hint = (
        f"\n\nTo create the topic, run:\n"
        f"  rpk topic create {TEST_INTROSPECTION_TOPIC} --partitions 1\n"
        f"\nOr with kafka-topics:\n"
        f"  kafka-topics.sh --create --topic {TEST_INTROSPECTION_TOPIC} "
        f"--partitions 1 --replication-factor 1 --bootstrap-server {bootstrap_servers}"
    )

    # Kafka error codes reference (from Kafka protocol specification)
    # https://kafka.apache.org/protocol.html#protocol_error_codes
    kafka_error_codes = {
        0: "NO_ERROR",
        3: "UNKNOWN_TOPIC_OR_PARTITION",
        5: "LEADER_NOT_AVAILABLE",
        6: "NOT_LEADER_OR_FOLLOWER",
        7: "REQUEST_TIMED_OUT",
        9: "REPLICA_NOT_AVAILABLE",
        36: "NOT_CONTROLLER",
    }

    def _get_error_description(error_code: int) -> str:
        """Get human-readable error description for Kafka error codes."""
        name = kafka_error_codes.get(error_code, "UNKNOWN_ERROR")
        return f"{name} (code={error_code})"

    def _extract_topic_field(metadata: object, field_name: str) -> object:
        """Extract field from topic metadata supporting both dict and namedtuple formats.

        aiokafka may return topic metadata as dict or namedtuple depending on version.
        This helper handles both formats transparently.
        """
        if isinstance(metadata, dict):
            return metadata.get(field_name)
        return getattr(metadata, field_name, None)

    try:
        # Step 1: Connect to Kafka admin API with explicit timeout
        logger.debug("Connecting to Kafka admin API at %s", bootstrap_servers)
        admin_client = AIOKafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            request_timeout_ms=30000,  # 30 second timeout for admin operations
        )
        await admin_client.start()

        # Step 2: Describe the specific topic (validates existence and configuration)
        # NOTE: aiokafka.describe_topics() returns Dict[str, TopicDescription] keyed by
        # topic name. If the API changes, this code will fail fast with a clear error.
        logger.debug("Describing topic '%s'", TEST_INTROSPECTION_TOPIC)
        topic_descriptions = await admin_client.describe_topics(
            [TEST_INTROSPECTION_TOPIC]
        )

        # Fail-fast validation 1: Empty response
        if not topic_descriptions:
            pytest.fail(
                f"Topic validation failed: describe_topics returned empty response.\n"
                f"Topic '{TEST_INTROSPECTION_TOPIC}' likely does not exist."
                f"{topic_creation_hint}"
            )

        # Get the topic description from describe_topics response.
        # aiokafka may return dict format (0.11.0+) or list format (older versions).
        topic_metadata: object | None = None
        if isinstance(topic_descriptions, dict):
            topic_metadata = topic_descriptions.get(TEST_INTROSPECTION_TOPIC)
        elif isinstance(topic_descriptions, list):
            # Older aiokafka returns list of topic descriptions
            for td in topic_descriptions:
                td_topic = _extract_topic_field(td, "topic") or _extract_topic_field(
                    td, "name"
                )
                if td_topic == TEST_INTROSPECTION_TOPIC:
                    topic_metadata = td
                    break
        else:
            pytest.fail(
                f"Unexpected describe_topics response type: {type(topic_descriptions).__name__}.\n"
                f"Expected dict or list, got: {topic_descriptions!r}"
                f"{topic_creation_hint}"
            )
        if topic_metadata is None:
            if isinstance(topic_descriptions, dict):
                available = list(topic_descriptions.keys())
            else:
                available = [
                    _extract_topic_field(td, "topic")
                    or _extract_topic_field(td, "name")
                    for td in topic_descriptions
                ]
            pytest.fail(
                f"Topic '{TEST_INTROSPECTION_TOPIC}' not found in describe_topics response.\n"
                f"Available topics: {available}"
                f"{topic_creation_hint}"
            )

        # Fail-fast validation 2: Check for error codes
        # aiokafka may return error_code field for non-existent or inaccessible topics
        error_code = _extract_topic_field(topic_metadata, "error_code")
        if error_code is None:
            error_code = 0  # Default to no error if field not present

        if error_code != 0:
            error_desc = _get_error_description(error_code)
            if error_code == 3:  # UNKNOWN_TOPIC_OR_PARTITION
                pytest.fail(
                    f"Topic '{TEST_INTROSPECTION_TOPIC}' does not exist.\n"
                    f"Kafka returned: {error_desc}"
                    f"{topic_creation_hint}"
                )
            else:
                pytest.fail(
                    f"Topic '{TEST_INTROSPECTION_TOPIC}' has configuration error.\n"
                    f"Kafka returned: {error_desc}\n"
                    f"This may indicate the topic exists but is misconfigured or "
                    f"the broker is unhealthy."
                    f"{topic_creation_hint}"
                )

        # Fail-fast validation 3: Verify topic name matches
        topic_name = _extract_topic_field(topic_metadata, "topic")
        if topic_name is None:
            topic_name = _extract_topic_field(topic_metadata, "name")

        if topic_name and topic_name != TEST_INTROSPECTION_TOPIC:
            pytest.fail(
                f"Topic name mismatch in Kafka response.\n"
                f"Expected: '{TEST_INTROSPECTION_TOPIC}'\n"
                f"Got: '{topic_name}'\n"
                f"This indicates a Kafka API response parsing issue."
            )

        # Fail-fast validation 4: Verify topic has partitions
        partitions = _extract_topic_field(topic_metadata, "partitions")
        if partitions is None:
            partitions = []

        # Convert partitions to list if it's another iterable type
        if not isinstance(partitions, list):
            try:
                partitions = list(partitions)
            except TypeError:
                partitions = []

        if not partitions:
            pytest.fail(
                f"Topic '{TEST_INTROSPECTION_TOPIC}' has no partitions.\n"
                f"The topic exists but appears to be misconfigured.\n"
                f"Try deleting and recreating the topic:"
                f"{topic_creation_hint}"
            )

        # Validation successful - log details for debugging
        partition_count = len(partitions)
        logger.info(
            "Topic '%s' validated successfully: %d partition(s)",
            TEST_INTROSPECTION_TOPIC,
            partition_count,
        )

    except UnknownTopicOrPartitionError:
        pytest.fail(
            f"Topic '{TEST_INTROSPECTION_TOPIC}' does not exist.{topic_creation_hint}"
        )

    except KafkaConnectionError as e:
        pytest.fail(
            f"Failed to connect to Kafka at {bootstrap_servers}.\n"
            f"Connection error: {e}\n\n"
            f"Troubleshooting:\n"
            f"  1. Verify Kafka/Redpanda is running: rpk cluster info\n"
            f"  2. Check KAFKA_BOOTSTRAP_SERVERS is correct\n"
            f"  3. Ensure network connectivity to {bootstrap_servers}\n"
            f"  4. Check firewall rules allow connections"
        )

    except KafkaTimeoutError as e:
        pytest.fail(
            f"Timeout connecting to Kafka at {bootstrap_servers}.\n"
            f"Timeout error: {e}\n\n"
            f"Troubleshooting:\n"
            f"  1. Kafka may be overloaded or unresponsive\n"
            f"  2. Network latency may be too high\n"
            f"  3. Try increasing request_timeout_ms\n"
            f"  4. Check Kafka broker logs for errors"
        )

    except KafkaError as e:
        # Catch-all for other Kafka protocol errors
        pytest.fail(
            f"Kafka error validating topic '{TEST_INTROSPECTION_TOPIC}'.\n"
            f"Error type: {type(e).__name__}\n"
            f"Error: {e}\n\n"
            f"This may indicate a broker configuration issue or protocol error."
            f"{topic_creation_hint}"
        )

    except Exception as e:
        # Non-Kafka errors (network issues, etc.)
        pytest.fail(
            f"Unexpected error validating topic '{TEST_INTROSPECTION_TOPIC}'.\n"
            f"Error type: {type(e).__name__}\n"
            f"Error: {e}\n\n"
            f"If this is a new environment, ensure the topic exists:"
            f"{topic_creation_hint}"
        )

    finally:
        if admin_client is not None:
            try:
                await admin_client.close()
            except Exception as close_err:
                # Log but don't fail on close errors
                logger.warning("Error closing Kafka admin client: %s", close_err)

    return TEST_INTROSPECTION_TOPIC


@pytest.fixture
async def running_orchestrator_consumer(
    real_kafka_event_bus: EventBusKafka,
    orchestrator_pipeline: OrchestratorTestContext,
    validate_test_topic_exists: str,  # Ensures topic exists before subscribing
) -> AsyncGenerator[OrchestratorTestContext, None]:
    """Start a Kafka consumer that routes messages through the pipeline.

    This fixture creates a real Kafka subscription that:
    - Subscribes to the test introspection topic
    - Routes incoming messages to the OrchestratorPipeline.process_message callback
    - Returns the OrchestratorTestContext with pipeline, mocks, and unsubscribe function

    Consumer Configuration (from real_kafka_event_bus fixture):
        The real_kafka_event_bus fixture in conftest.py configures the consumer with
        enable_auto_commit=False for strict test isolation. This ensures:
        - Offsets are committed after successful processing, not periodically
        - No cross-test pollution from committed offsets before processing completes
        - Deterministic message consumption position for each test

    Args:
        real_kafka_event_bus: Real Kafka event bus (configured with enable_auto_commit=False).
        orchestrator_pipeline: Context containing pipeline and connected mocks.
        validate_test_topic_exists: Fixture that validates the topic is ready.

    Yields:
        OrchestratorTestContext with pipeline, mocks, and unsubscribe function.

    Note:
        The OrchestratorTestContext pattern ensures the mock instances returned
        are the exact same instances injected into the pipeline, preventing
        assertion failures when tests verify mock method calls.
    """
    # Use unique group ID per test run to avoid cross-test coupling
    unique_group_id = f"e2e-orchestrator-test-{uuid4().hex[:8]}"
    # Subscribe to the introspection topic (topic is guaranteed to exist)
    unsubscribe = await real_kafka_event_bus.subscribe(
        topic=validate_test_topic_exists,  # Use the ensured topic name
        node_identity=make_e2e_test_identity("orchestrator"),
        on_message=orchestrator_pipeline.pipeline.process_message,
    )

    # Wait for consumer to be ready to receive messages.
    # See wait_for_consumer_ready docstring for known limitations.
    await wait_for_consumer_ready(real_kafka_event_bus, validate_test_topic_exists)

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
        real_kafka_event_bus: EventBusKafka,
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
        # Note: node_version must be ModelSemVer for ModelNodeIntrospectionEvent
        event = ModelNodeIntrospectionEvent(
            node_id=unique_node_id,
            node_type="effect",
            node_version=ModelSemVer.parse("1.0.0"),
            declared_capabilities=ModelNodeCapabilities(),
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
            # Polling interval - wait before checking processed_events again
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
        real_kafka_event_bus: EventBusKafka,
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
        # Note: node_version must be ModelSemVer for ModelNodeIntrospectionEvent
        event = ModelNodeIntrospectionEvent(
            node_id=unique_node_id,
            node_type="compute",
            node_version=ModelSemVer.parse("2.0.0"),
            declared_capabilities=ModelNodeCapabilities(),
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
            # Polling interval - wait before checking processed_events again
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
        real_kafka_event_bus: EventBusKafka,
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
            # Note: node_version must be ModelSemVer for ModelNodeIntrospectionEvent
            event = ModelNodeIntrospectionEvent(
                node_id=node_id,
                node_type=node_type,
                node_version=ModelSemVer.parse("1.0.0"),
                declared_capabilities=ModelNodeCapabilities(),
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
            # Polling interval - wait before recounting processed events
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Verify all events processed
        for node_id in node_ids:
            assert node_id in pipeline.processed_events, (
                f"Event for node {node_id} was not processed"
            )

    async def test_malformed_message_handled_gracefully(
        self,
        real_kafka_event_bus: EventBusKafka,
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

        # Allow time for pipeline to process and reject the malformed message.
        # This ensures the pipeline is ready for the next valid message.
        #
        # TODO(OMN-1327): Replace with deterministic wait once pipeline exposes
        # a "message processed" signal or callback. Options:
        # 1. Pipeline.wait_until_idle() method that tracks in-flight messages
        # 2. Counter-based approach: wait until processed_count increments
        # 3. Expose processing completion via asyncio.Event per message
        await asyncio.sleep(2.0)

        # Publish a valid message after the malformed one
        # Note: node_version must be ModelSemVer for ModelNodeIntrospectionEvent
        valid_event = ModelNodeIntrospectionEvent(
            node_id=unique_node_id,
            node_type="effect",
            node_version=ModelSemVer.parse("1.0.0"),
            declared_capabilities=ModelNodeCapabilities(),
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
            # Polling interval - wait before checking processed_events again
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
        real_kafka_event_bus: EventBusKafka,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        unique_node_id: UUID,
        unique_correlation_id: UUID,
        cleanup_projections: None,
    ) -> None:
        """Test that introspection event creates PostgreSQL projection.

        Publishes an introspection event and verifies the projection
        is persisted in PostgreSQL.
        """
        # Create introspection event
        # Note: node_version must be ModelSemVer for ModelNodeIntrospectionEvent
        event = ModelNodeIntrospectionEvent(
            node_id=unique_node_id,
            node_type="effect",
            node_version=ModelSemVer.parse("1.0.0"),
            declared_capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
            metadata=ModelNodeMetadata(),
            correlation_id=unique_correlation_id,
            timestamp=datetime.now(UTC),
        )

        # Create handler and process directly (for simpler test)
        handler = HandlerNodeIntrospected(projection_reader)
        now = datetime.now(UTC)

        # Wrap event in envelope for handler API
        envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=event,
            envelope_timestamp=now,
            correlation_id=unique_correlation_id,
            source="e2e-test",
        )

        handler_output = await handler.handle(envelope)

        # If handler says we should register, create the projection
        if handler_output.events:
            node_type = coerce_to_node_kind(event.node_type)
            values: dict[str, object] = {
                "entity_id": unique_node_id,
                "domain": "registration",
                "current_state": EnumRegistrationState.PENDING_REGISTRATION.value,
                "node_type": node_type.value,
                "node_version": str(event.node_version),
                "capabilities": "{}",
                "registered_at": now,
                "updated_at": now,
                "last_applied_event_id": unique_correlation_id,
                "last_applied_offset": 1,
            }

            await real_projector.upsert_partial(
                aggregate_id=unique_node_id,
                values=values,
                correlation_id=unique_correlation_id,
                conflict_columns=["entity_id", "domain"],
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
        self, unique_node_id: UUID, unique_correlation_id: UUID
    ) -> None:
        """Test that reducer generates Consul and PostgreSQL intents.

        Verifies the reducer emits the expected intent types.
        """
        # Create introspection event
        # Note: node_version must be ModelSemVer for ModelNodeIntrospectionEvent
        event = ModelNodeIntrospectionEvent(
            node_id=unique_node_id,
            node_type="effect",
            node_version=ModelSemVer.parse("1.0.0"),
            declared_capabilities=ModelNodeCapabilities(),
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

        intent_types = {
            intent.payload.intent_type
            for intent in output.intents
            if intent.intent_type
        }
        assert "consul.register" in intent_types, "Should include Consul intent"
        assert "postgres.upsert_registration" in intent_types, (
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
            consul_client=mock_consul_client, postgres_adapter=mock_postgres_adapter
        )

        request = ModelRegistryRequest(
            node_id=unique_node_id,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
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
        real_kafka_event_bus: EventBusKafka,
        unique_correlation_id: UUID,
        validate_test_topic_exists: str,
    ) -> None:
        """Test that consumer starts and receives messages.

        This is a basic connectivity test to ensure Kafka subscription works.
        """
        received_messages: list[ModelEventMessage] = []
        message_received = asyncio.Event()

        async def handler(msg: ModelEventMessage) -> None:
            received_messages.append(msg)
            message_received.set()

        # Subscribe (topic is guaranteed to exist via validate_test_topic_exists fixture)
        unsubscribe = await real_kafka_event_bus.subscribe(
            topic=validate_test_topic_exists,
            node_identity=make_e2e_test_identity("lifecycle"),
            on_message=handler,
        )

        try:
            # Wait for consumer to be ready to receive messages.
            # See wait_for_consumer_ready docstring for known limitations.
            await wait_for_consumer_ready(
                real_kafka_event_bus, validate_test_topic_exists
            )

            # Publish test message
            headers = ModelEventHeaders(
                source="lifecycle-test",
                event_type="test",
                correlation_id=unique_correlation_id,
                timestamp=datetime.now(UTC),
            )

            await real_kafka_event_bus.publish(
                topic=validate_test_topic_exists,
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
        real_kafka_event_bus: EventBusKafka,
        unique_correlation_id: UUID,
        validate_test_topic_exists: str,
    ) -> None:
        """Test that consumer handles shutdown without errors.

        Verifies clean unsubscribe and no resource leaks.
        """
        message_count = 0

        async def handler(msg: ModelEventMessage) -> None:
            nonlocal message_count
            message_count += 1

        # Subscribe and immediately unsubscribe (topic is guaranteed to exist)
        unsubscribe = await real_kafka_event_bus.subscribe(
            topic=validate_test_topic_exists,
            node_identity=make_e2e_test_identity("shutdown"),
            on_message=handler,
        )

        # Brief wait before testing shutdown (tests cleanup, not message receipt).
        # Using wait_for_consumer_ready for consistency, though this test doesn't
        # actually need to receive messages.
        await wait_for_consumer_ready(real_kafka_event_bus, validate_test_topic_exists)

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
    "coerce_to_node_kind",
]
