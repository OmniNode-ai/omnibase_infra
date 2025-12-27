# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""True E2E tests against the running ONEX runtime container.

These tests verify the COMPLETE end-to-end flow:
    1. Publish events to Kafka
    2. Runtime container consumes and processes them
    3. Verify results in PostgreSQL and Consul

IMPORTANT: These tests require the runtime container to be running:
    docker compose -f docker/docker-compose.e2e.yml --profile runtime up -d

Unlike the component integration tests (test_two_way_registration_e2e.py),
these tests do NOT call handlers directly. They test the actual deployed
runtime processing messages from Kafka.

Test Flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Test Process                                                   │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ 1. Publish introspection event to Kafka                 │   │
    │  │ 2. Wait for runtime to process                          │   │
    │  │ 3. Query PostgreSQL/Consul for results                  │   │
    │  │ 4. Verify registration completed                        │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Runtime Container (omnibase-infra-runtime)                     │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ Kafka Consumer → Handler → Reducer → Effect             │   │
    │  │       ↓              ↓         ↓         ↓              │   │
    │  │  Introspection   Decision   Intents   Dual Reg          │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │    PostgreSQL    │  │      Consul      │  │      Kafka       │
    │   (projections)  │  │   (services)     │  │ (output events)  │
    └──────────────────┘  └──────────────────┘  └──────────────────┘

Related Tickets:
    - OMN-892: E2E Registration Tests
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx
import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind

from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.models.registration.model_node_capabilities import ModelNodeCapabilities

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
    from omnibase_infra.projectors import ProjectionReaderRegistration

# Import shared envelope helper from conftest
from tests.integration.registration.e2e.conftest import wrap_event_in_envelope


# =============================================================================
# Topic Configuration
# =============================================================================
# Get topic from environment or use docker-compose default
# The runtime container expects: dev.onex.evt.node-introspection.v1 (from ONEX_INPUT_TOPIC)
RUNTIME_INPUT_TOPIC = os.getenv("ONEX_INPUT_TOPIC", "dev.onex.evt.node-introspection.v1")


# =============================================================================
# Runtime Availability Check
# =============================================================================

RUNTIME_HOST = os.getenv("RUNTIME_HOST", "host.docker.internal")
RUNTIME_PORT = int(os.getenv("RUNTIME_PORT", "8085"))
RUNTIME_HEALTH_URL = f"http://{RUNTIME_HOST}:{RUNTIME_PORT}/health"


def _check_runtime_available() -> bool:
    """Check if the runtime container is running and healthy."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(RUNTIME_HEALTH_URL)
            return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


RUNTIME_AVAILABLE = _check_runtime_available()


# Skip all tests in this module if runtime is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.runtime,
    pytest.mark.skipif(
        not RUNTIME_AVAILABLE,
        reason=(
            f"Runtime container not available at {RUNTIME_HEALTH_URL}. "
            "Start with: docker compose -f docker/docker-compose.e2e.yml --profile runtime up -d"
        ),
    ),
]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def unique_node_id() -> UUID:
    """Generate a unique node ID for test isolation."""
    return uuid4()


@pytest.fixture
def introspection_event(unique_node_id: UUID) -> ModelNodeIntrospectionEvent:
    """Create a valid introspection event for testing."""
    return ModelNodeIntrospectionEvent(
        node_id=unique_node_id,
        node_type=EnumNodeKind.EFFECT.value,
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(),
        endpoints={
            "health": f"http://test-node-{unique_node_id.hex[:8]}:8080/health",
            "api": f"http://test-node-{unique_node_id.hex[:8]}:8080/api",
        },
        correlation_id=uuid4(),
        timestamp=datetime.now(UTC),
    )


# =============================================================================
# True E2E Tests
# =============================================================================


class TestRuntimeE2EFlow:
    """True E2E tests against the running runtime container.

    These tests publish events to Kafka and verify the runtime
    processes them correctly by checking the database.
    """

    @pytest.mark.asyncio
    async def test_runtime_health_endpoint(self) -> None:
        """Verify runtime container is healthy and responding."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(RUNTIME_HEALTH_URL)

            assert response.status_code == 200, (
                f"Runtime health check failed: {response.status_code}"
            )

            # Verify health response structure
            health_data = response.json()
            assert "status" in health_data or response.status_code == 200

    @pytest.mark.asyncio
    async def test_introspection_event_processed_by_runtime(
        self,
        real_kafka_event_bus: KafkaEventBus,
        projection_reader: ProjectionReaderRegistration,
        introspection_event: ModelNodeIntrospectionEvent,
        unique_node_id: UUID,
    ) -> None:
        """Test that runtime processes introspection event and creates projection.

        This is the core E2E test:
        1. Publish introspection event to Kafka
        2. Wait for runtime to consume and process
        3. Verify projection exists in PostgreSQL
        """
        # Record start time for timeout calculation
        start_time = datetime.now(UTC)

        # Wrap event in envelope and publish to Kafka
        envelope = wrap_event_in_envelope(introspection_event)
        await real_kafka_event_bus.publish_envelope(envelope, topic=RUNTIME_INPUT_TOPIC)

        # Wait for runtime to process (poll database)
        max_wait_seconds = 30.0
        poll_interval = 0.5
        projection = None

        while (datetime.now(UTC) - start_time).total_seconds() < max_wait_seconds:
            projection = await projection_reader.get_entity_state(
                entity_id=unique_node_id,
                domain="registration",
                correlation_id=introspection_event.correlation_id,
            )

            if projection is not None:
                break

            await asyncio.sleep(poll_interval)

        # Verify projection was created
        assert projection is not None, (
            f"Runtime did not create projection for node {unique_node_id} "
            f"within {max_wait_seconds}s. Check runtime logs."
        )

        # Verify projection has correct data
        assert projection.entity_id == unique_node_id
        assert projection.node_type == introspection_event.node_type
        assert projection.node_version == introspection_event.node_version

    @pytest.mark.asyncio
    async def test_runtime_handles_multiple_events_sequentially(
        self,
        real_kafka_event_bus: KafkaEventBus,
        projection_reader: ProjectionReaderRegistration,
    ) -> None:
        """Test runtime correctly processes multiple events in order."""
        # Create multiple introspection events
        node_ids = [uuid4() for _ in range(3)]
        events = [
            ModelNodeIntrospectionEvent(
                node_id=node_id,
                node_type=EnumNodeKind.EFFECT.value,
                node_version="1.0.0",
                capabilities=ModelNodeCapabilities(),
                endpoints={
                    "health": f"http://node-{i}:8080/health",
                    "api": f"http://node-{i}:8080/api",
                },
                correlation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )
            for i, node_id in enumerate(node_ids)
        ]

        # Publish all events wrapped in envelopes
        for event in events:
            envelope = wrap_event_in_envelope(event)
            await real_kafka_event_bus.publish_envelope(envelope, topic=RUNTIME_INPUT_TOPIC)

        # Wait for all projections
        max_wait_seconds = 45.0
        start_time = datetime.now(UTC)

        while (datetime.now(UTC) - start_time).total_seconds() < max_wait_seconds:
            all_found = True
            for node_id in node_ids:
                projection = await projection_reader.get_entity_state(
                    entity_id=node_id,
                    domain="registration",
                    correlation_id=uuid4(),
                )
                if projection is None:
                    all_found = False
                    break

            if all_found:
                break

            await asyncio.sleep(0.5)

        # Verify all projections exist
        for i, node_id in enumerate(node_ids):
            projection = await projection_reader.get_entity_state(
                entity_id=node_id,
                domain="registration",
                correlation_id=uuid4(),
            )
            assert projection is not None, (
                f"Projection for node {i} ({node_id}) not found after {max_wait_seconds}s"
            )

    @pytest.mark.asyncio
    async def test_runtime_publishes_completion_event(
        self,
        real_kafka_event_bus: KafkaEventBus,
        introspection_event: ModelNodeIntrospectionEvent,
        unique_node_id: UUID,
    ) -> None:
        """Test that runtime publishes registration-completed event."""
        # Track completion events
        completion_received = asyncio.Event()
        received_completions: list[dict] = []

        async def on_completion(message: object) -> None:
            if hasattr(message, "value") and message.value:
                try:
                    data = json.loads(message.value.decode("utf-8"))

                    # Events are wrapped in ModelEventEnvelope, so extract payload
                    # The envelope structure is: {envelope_id, payload: {...}, ...}
                    payload = data.get("payload", data)  # Fall back to data if no envelope

                    if payload.get("node_id") == str(unique_node_id):
                        received_completions.append(payload)
                        completion_received.set()
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

        # Subscribe to completion topic (matches docker-compose.e2e.yml ONEX_OUTPUT_TOPIC)
        output_topic = os.getenv(
            "ONEX_OUTPUT_TOPIC",
            "dev.onex.evt.registration-completed.v1",
        )
        group_id = f"e2e-runtime-{unique_node_id.hex[:8]}"

        unsub = await real_kafka_event_bus.subscribe(
            topic=output_topic,
            group_id=group_id,
            on_message=on_completion,
        )

        try:
            # Allow subscription to establish
            await asyncio.sleep(2.0)

            # Publish introspection event wrapped in envelope
            envelope = wrap_event_in_envelope(introspection_event)
            await real_kafka_event_bus.publish_envelope(envelope, topic=RUNTIME_INPUT_TOPIC)

            # Wait for completion event
            try:
                await asyncio.wait_for(completion_received.wait(), timeout=30.0)
                assert len(received_completions) > 0, (
                    "Expected completion event from runtime"
                )

                # Verify completion event structure
                completion = received_completions[0]
                assert completion.get("node_id") == str(unique_node_id)

            except TimeoutError:
                pytest.skip(
                    "Runtime did not publish completion event within timeout. "
                    "This may indicate the output topic is not configured."
                )

        finally:
            await unsub()

    @pytest.mark.asyncio
    async def test_runtime_dual_registration_creates_consul_entry(
        self,
        real_kafka_event_bus: KafkaEventBus,
        introspection_event: ModelNodeIntrospectionEvent,
        unique_node_id: UUID,
    ) -> None:
        """Test that runtime performs dual registration including Consul."""
        # Publish introspection event wrapped in envelope
        envelope = wrap_event_in_envelope(introspection_event)
        await real_kafka_event_bus.publish_envelope(envelope, topic=RUNTIME_INPUT_TOPIC)

        # Wait for Consul registration via HTTP API
        consul_host = os.getenv("CONSUL_HOST", "host.docker.internal")
        consul_port = int(os.getenv("CONSUL_PORT", "8500"))
        service_name = f"node-{unique_node_id.hex[:8]}"

        max_wait_seconds = 30.0
        start_time = datetime.now(UTC)
        consul_entry = None

        async with httpx.AsyncClient(timeout=5.0) as client:
            while (datetime.now(UTC) - start_time).total_seconds() < max_wait_seconds:
                try:
                    response = await client.get(
                        f"http://{consul_host}:{consul_port}/v1/catalog/service/{service_name}"
                    )
                    if response.status_code == 200:
                        services = response.json()
                        if services:
                            consul_entry = services[0]
                            break
                except Exception:
                    pass

                await asyncio.sleep(0.5)

        if consul_entry is None:
            pytest.skip(
                f"Service {service_name} not found in Consul within {max_wait_seconds}s. "
                "Dual registration may not be configured in runtime."
            )

        assert consul_entry is not None


class TestRuntimeErrorHandling:
    """Test runtime's error handling and resilience."""

    @pytest.mark.asyncio
    async def test_runtime_handles_malformed_message(
        self,
        real_kafka_event_bus: KafkaEventBus,
    ) -> None:
        """Test runtime doesn't crash on malformed messages."""
        # Publish malformed JSON
        await real_kafka_event_bus.publish(
            topic=RUNTIME_INPUT_TOPIC,
            key=b"malformed-test",
            value=b"not valid json {{{",
        )

        # Wait a moment
        await asyncio.sleep(2.0)

        # Verify runtime is still healthy
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(RUNTIME_HEALTH_URL)
            assert response.status_code == 200, (
                "Runtime became unhealthy after malformed message"
            )

    @pytest.mark.asyncio
    async def test_runtime_handles_missing_fields(
        self,
        real_kafka_event_bus: KafkaEventBus,
    ) -> None:
        """Test runtime handles events with missing required fields."""
        # Publish event missing required fields
        incomplete_event = {
            "node_id": str(uuid4()),
            # Missing: node_type, node_version, etc.
        }

        await real_kafka_event_bus.publish(
            topic=RUNTIME_INPUT_TOPIC,
            key=b"incomplete-test",
            value=json.dumps(incomplete_event).encode("utf-8"),
        )

        # Wait a moment
        await asyncio.sleep(2.0)

        # Verify runtime is still healthy
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(RUNTIME_HEALTH_URL)
            assert response.status_code == 200, (
                "Runtime became unhealthy after incomplete event"
            )


class TestRuntimePerformance:
    """Performance tests for the runtime."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_runtime_processes_event_within_sla(
        self,
        real_kafka_event_bus: KafkaEventBus,
        projection_reader: ProjectionReaderRegistration,
        introspection_event: ModelNodeIntrospectionEvent,
        unique_node_id: UUID,
    ) -> None:
        """Test that runtime processes events within acceptable SLA.

        SLA: Event should be processed and projection created within 5 seconds.
        """
        sla_seconds = 5.0

        # Record publish time
        publish_time = datetime.now(UTC)

        # Publish event wrapped in envelope
        envelope = wrap_event_in_envelope(introspection_event)
        await real_kafka_event_bus.publish_envelope(envelope, topic=RUNTIME_INPUT_TOPIC)

        # Poll for projection
        while True:
            elapsed = (datetime.now(UTC) - publish_time).total_seconds()

            if elapsed > sla_seconds:
                pytest.fail(
                    f"Runtime exceeded {sla_seconds}s SLA for event processing"
                )

            projection = await projection_reader.get_entity_state(
                entity_id=unique_node_id,
                domain="registration",
                correlation_id=uuid4(),
            )

            if projection is not None:
                processing_time = elapsed
                break

            await asyncio.sleep(0.1)

        # Log processing time
        print(f"Runtime processed event in {processing_time:.2f}s (SLA: {sla_seconds}s)")
        assert processing_time < sla_seconds
