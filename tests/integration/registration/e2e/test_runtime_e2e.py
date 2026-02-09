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
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx
import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from tests.integration.registration.e2e.conftest import (
    make_e2e_test_identity,
    wait_for_consumer_ready,
    wrap_event_in_envelope,
)

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.projectors import ProjectionReaderRegistration

# Module-level logger
logger = logging.getLogger(__name__)

# =============================================================================
# Topic Configuration
# =============================================================================
# Get topic from environment or use docker-compose default
# The runtime container subscribes to the ONEX 5-segment topic from contract.yaml
RUNTIME_INPUT_TOPIC = os.getenv(
    "ONEX_INPUT_TOPIC", "onex.evt.platform.node-introspection.v1"
)

# =============================================================================
# CI Environment Timing Configuration
# =============================================================================
# CI environments have variable latency due to:
# - Network latency
# - Container warmup
# - Shared runner load
# - Cold connection pools
#
# These timeouts are configurable via environment variables for CI flexibility.

# Single event processing timeout (default: 60s, CI-friendly)
SINGLE_EVENT_TIMEOUT_SECONDS = float(os.getenv("E2E_SINGLE_EVENT_TIMEOUT", "60.0"))

# Multiple event processing timeout (default: 120s for 3 events)
MULTI_EVENT_TIMEOUT_SECONDS = float(os.getenv("E2E_MULTI_EVENT_TIMEOUT", "120.0"))

# SLA target for performance test (default: 15s, soft target)
# Note: 5s was too aggressive for CI. 15s allows for realistic CI latency.
SLA_TARGET_SECONDS = float(os.getenv("E2E_SLA_TARGET", "15.0"))

# Hard SLA limit before test fails (default: 30s)
# Exceeding this indicates a real performance problem, not just CI variance.
SLA_HARD_LIMIT_SECONDS = float(os.getenv("E2E_SLA_HARD_LIMIT", "30.0"))

# Validate SLA threshold relationship at module load time
# This ensures configuration errors are caught early rather than during test execution
assert SLA_HARD_LIMIT_SECONDS > SLA_TARGET_SECONDS, (
    f"E2E_SLA_HARD_LIMIT ({SLA_HARD_LIMIT_SECONDS}s) must be greater than "
    f"E2E_SLA_TARGET ({SLA_TARGET_SECONDS}s). The hard limit catches performance "
    "regressions while the soft target warns about CI variance."
)


# =============================================================================
# Runtime Availability Check
# =============================================================================

RUNTIME_HOST = os.getenv("RUNTIME_HOST", "localhost")
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

# =============================================================================
# Runtime Event Processing Check
# =============================================================================
# The health endpoint only verifies the runtime HTTP server is running.
# It does NOT verify the runtime is actually consuming and processing Kafka events.
#
# Tests that require event processing (publishing to Kafka and waiting for
# database projections) need explicit opt-in via RUNTIME_E2E_PROCESSING_ENABLED.
#
# This prevents long timeout waits (60-120s) when:
# - Runtime is healthy but not connected to Kafka
# - Runtime is healthy but not consuming from the correct topic
# - Runtime is healthy but not writing to the database
#
# Set RUNTIME_E2E_PROCESSING_ENABLED=true to enable these tests when
# you have verified the runtime is fully functional end-to-end.

# Feature flags require explicit opt-in via environment variables.
# Even when runtime is healthy, event processing tests are disabled by
# default to avoid long timeout waits (60-120s) if the runtime pipeline
# isn't fully operational. Set RUNTIME_E2E_*_ENABLED=true after verifying
# the full E2E pipeline works.
_default_enabled = "false"

RUNTIME_PROCESSING_ENABLED = os.getenv(
    "RUNTIME_E2E_PROCESSING_ENABLED", _default_enabled
).lower() in ("true", "1", "yes")

# Skip reason for tests requiring event processing
SKIP_PROCESSING_REASON = (
    "Runtime event processing tests require explicit opt-in. "
    "The runtime health check passed, but event processing verification is disabled. "
    "Set RUNTIME_E2E_PROCESSING_ENABLED=true to enable these tests after verifying: "
    "1) Runtime is consuming from Kafka topic (ONEX_INPUT_TOPIC), "
    "2) Runtime is writing projections to PostgreSQL, "
    "3) Full E2E pipeline is operational. "
    "Without this, tests would wait 60-120s before timing out."
)

# =============================================================================
# Output Event and Consul Registration Feature Flags
# =============================================================================
# These optional features may not be configured in all runtime deployments.
# Tests for these features require explicit opt-in to avoid soft failures.

RUNTIME_OUTPUT_EVENTS_ENABLED = os.getenv(
    "RUNTIME_E2E_OUTPUT_EVENTS_ENABLED", _default_enabled
).lower() in ("true", "1", "yes")

SKIP_OUTPUT_EVENTS_REASON = (
    "Runtime output event tests require explicit opt-in. "
    "Set RUNTIME_E2E_OUTPUT_EVENTS_ENABLED=true to enable these tests after verifying: "
    "1) Runtime is configured to publish completion events to ONEX_OUTPUT_TOPIC, "
    "2) The output topic exists and is accessible. "
    "Without this flag, tests would wait and timeout."
)

RUNTIME_CONSUL_ENABLED = os.getenv(
    "RUNTIME_E2E_CONSUL_ENABLED", _default_enabled
).lower() in (
    "true",
    "1",
    "yes",
)

SKIP_CONSUL_REASON = (
    "Runtime Consul registration tests require explicit opt-in. "
    "Set RUNTIME_E2E_CONSUL_ENABLED=true to enable these tests after verifying: "
    "1) Runtime is configured for dual registration (Consul + PostgreSQL), "
    "2) Consul is accessible at CONSUL_HOST:CONSUL_PORT. "
    "Without this flag, tests would wait and timeout."
)


# Module-level markers
# Note: conftest.py already applies pytest.mark.e2e and skipif(not ALL_INFRA_AVAILABLE)
# to all tests in this directory. We only add runtime-specific markers here:
# - pytest.mark.runtime for categorization
# - skipif(not RUNTIME_AVAILABLE) for the unique runtime container check
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.runtime,
    pytest.mark.skipif(
        not RUNTIME_AVAILABLE,
        reason=(
            "Runtime E2E tests require the runtime container to be running. "
            f"Runtime: MISSING at {RUNTIME_HEALTH_URL}. "
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
        node_version=ModelSemVer.parse("1.0.0"),
        declared_capabilities=ModelNodeCapabilities(),
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

            # Verify health response structure if JSON is returned
            # Note: Some health endpoints return empty body with 200 status
            try:
                health_data = response.json()
                assert "status" in health_data, (
                    "Health response should contain 'status' field"
                )
            except (ValueError, KeyError):
                # Empty body or non-JSON response is acceptable if status was 200
                pass

    @pytest.mark.asyncio
    @pytest.mark.skipif(not RUNTIME_PROCESSING_ENABLED, reason=SKIP_PROCESSING_REASON)
    async def test_introspection_event_processed_by_runtime(
        self,
        real_kafka_event_bus: EventBusKafka,
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
        # Use configurable timeout for CI environments where latency varies
        max_wait_seconds = SINGLE_EVENT_TIMEOUT_SECONDS
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

            # Polling interval - wait before checking database again
            await asyncio.sleep(poll_interval)

        # Verify projection was created
        assert projection is not None, (
            f"Runtime did not create projection for node {unique_node_id} "
            f"within {max_wait_seconds}s. Check runtime logs. "
            f"(Timeout configurable via E2E_SINGLE_EVENT_TIMEOUT env var)"
        )

        # Verify projection has correct data
        assert projection.entity_id == unique_node_id
        assert projection.node_type == introspection_event.node_type
        assert projection.node_version == introspection_event.node_version

    @pytest.mark.asyncio
    @pytest.mark.skipif(not RUNTIME_PROCESSING_ENABLED, reason=SKIP_PROCESSING_REASON)
    async def test_runtime_handles_multiple_events_sequentially(
        self,
        real_kafka_event_bus: EventBusKafka,
        projection_reader: ProjectionReaderRegistration,
    ) -> None:
        """Test runtime correctly processes multiple events in order."""
        # Create multiple introspection events
        node_ids = [uuid4() for _ in range(3)]
        events = [
            ModelNodeIntrospectionEvent(
                node_id=node_id,
                node_type=EnumNodeKind.EFFECT.value,
                node_version=ModelSemVer.parse("1.0.0"),
                declared_capabilities=ModelNodeCapabilities(),
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
            await real_kafka_event_bus.publish_envelope(
                envelope, topic=RUNTIME_INPUT_TOPIC
            )

        # Wait for all projections
        # Use configurable timeout for CI environments where latency varies
        max_wait_seconds = MULTI_EVENT_TIMEOUT_SECONDS
        start_time = datetime.now(UTC)

        while (datetime.now(UTC) - start_time).total_seconds() < max_wait_seconds:
            all_found = True
            for i, node_id in enumerate(node_ids):
                # Preserve original event's correlation_id for tracing
                projection = await projection_reader.get_entity_state(
                    entity_id=node_id,
                    domain="registration",
                    correlation_id=events[i].correlation_id,
                )
                if projection is None:
                    all_found = False
                    break

            if all_found:
                break

            # Polling interval - wait before checking all projections again
            await asyncio.sleep(0.5)

        # Verify all projections exist
        for i, node_id in enumerate(node_ids):
            # Preserve original event's correlation_id for tracing
            projection = await projection_reader.get_entity_state(
                entity_id=node_id,
                domain="registration",
                correlation_id=events[i].correlation_id,
            )
            assert projection is not None, (
                f"Projection for node {i} ({node_id}) not found after {max_wait_seconds}s. "
                f"(Timeout configurable via E2E_MULTI_EVENT_TIMEOUT env var)"
            )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not RUNTIME_OUTPUT_EVENTS_ENABLED, reason=SKIP_OUTPUT_EVENTS_REASON
    )
    async def test_runtime_publishes_completion_event(
        self,
        real_kafka_event_bus: EventBusKafka,
        introspection_event: ModelNodeIntrospectionEvent,
        unique_node_id: UUID,
    ) -> None:
        """Test that runtime publishes registration-completed event.

        This test requires RUNTIME_E2E_OUTPUT_EVENTS_ENABLED=true because output
        event publishing may not be configured in all runtime deployments.
        """
        # Track completion events
        completion_received = asyncio.Event()
        received_completions: list[dict] = []

        async def on_completion(message: object) -> None:
            if hasattr(message, "value") and message.value:
                try:
                    data = json.loads(message.value.decode("utf-8"))

                    # Events are wrapped in ModelEventEnvelope, so extract payload
                    # The envelope structure is: {envelope_id, payload: {...}, ...}
                    payload = data.get(
                        "payload", data
                    )  # Fall back to data if no envelope

                    if payload.get("node_id") == str(unique_node_id):
                        received_completions.append(payload)
                        completion_received.set()
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

        # Subscribe to completion topic (matches docker-compose.e2e.yml ONEX_OUTPUT_TOPIC)
        output_topic = os.getenv(
            "ONEX_OUTPUT_TOPIC", "onex.evt.registration-completed.v1"
        )
        group_id = f"e2e-runtime-{unique_node_id.hex[:8]}"

        unsub = await real_kafka_event_bus.subscribe(
            topic=output_topic,
            node_identity=make_e2e_test_identity("runtime"),
            on_message=on_completion,
        )

        try:
            # Wait for Kafka consumer to be ready before publishing.
            # See wait_for_consumer_ready docstring for known limitations.
            await wait_for_consumer_ready(
                real_kafka_event_bus, output_topic, max_wait=2.0
            )

            # Publish introspection event wrapped in envelope
            envelope = wrap_event_in_envelope(introspection_event)
            await real_kafka_event_bus.publish_envelope(
                envelope, topic=RUNTIME_INPUT_TOPIC
            )

            # Wait for completion event - hard failure on timeout
            await asyncio.wait_for(completion_received.wait(), timeout=30.0)

            assert len(received_completions) > 0, (
                "Expected completion event from runtime"
            )

            # Verify completion event structure
            completion = received_completions[0]
            assert completion.get("node_id") == str(unique_node_id)

        finally:
            await unsub()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not RUNTIME_CONSUL_ENABLED, reason=SKIP_CONSUL_REASON)
    async def test_runtime_dual_registration_creates_consul_entry(
        self,
        real_kafka_event_bus: EventBusKafka,
        introspection_event: ModelNodeIntrospectionEvent,
        unique_node_id: UUID,
    ) -> None:
        """Test that runtime performs dual registration including Consul.

        This test requires RUNTIME_E2E_CONSUL_ENABLED=true because Consul
        dual registration may not be configured in all runtime deployments.
        """
        # Publish introspection event wrapped in envelope
        envelope = wrap_event_in_envelope(introspection_event)
        await real_kafka_event_bus.publish_envelope(envelope, topic=RUNTIME_INPUT_TOPIC)

        # Wait for Consul registration via HTTP API
        consul_host = os.getenv("CONSUL_HOST", "host.docker.internal")
        consul_port = int(os.getenv("CONSUL_PORT", "8500"))
        # Consul service name follows ONEX convention: onex-{node_type}
        # This matches the service_name format used in NodeRegistryEffect._register_consul
        service_name = f"onex-{introspection_event.node_type}"

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
                except (
                    httpx.ConnectError,
                    httpx.TimeoutException,
                    json.JSONDecodeError,
                ):
                    # Connection/timeout errors and JSON decode errors are expected
                    # during polling - Consul may not have the service yet or be
                    # temporarily unavailable
                    pass

                # Polling interval - wait before checking Consul catalog again
                await asyncio.sleep(0.5)

        # Hard failure - if Consul is enabled, registration must succeed
        assert consul_entry is not None, (
            f"Service '{service_name}' not found in Consul within {max_wait_seconds}s. "
            f"Consul: http://{consul_host}:{consul_port}. "
            f"Verify runtime has dual registration enabled and Consul is accessible."
        )


class TestRuntimeErrorHandling:
    """Test runtime's error handling and resilience."""

    @pytest.mark.asyncio
    async def test_runtime_handles_malformed_message(
        self, real_kafka_event_bus: EventBusKafka
    ) -> None:
        """Test runtime doesn't crash on malformed messages."""
        # Publish malformed JSON
        await real_kafka_event_bus.publish(
            topic=RUNTIME_INPUT_TOPIC,
            key=b"malformed-test",
            value=b"not valid json {{{",
        )

        # RATIONALE: This sleep cannot be replaced with deterministic polling.
        # This is a "negative" test verifying the runtime stays healthy after
        # receiving invalid input. There's no observable state change to poll for -
        # the runtime should simply log an error and continue. The 2.0s wait gives
        # enough time for the message to be consumed and (mis)processed.
        # Alternative: Poll health endpoint repeatedly over 2s to catch transient
        # failures, implemented below.
        max_wait = 2.0
        poll_interval = 0.2
        start_time = asyncio.get_running_loop().time()
        health_ok = True

        async with httpx.AsyncClient(timeout=10.0) as client:
            while asyncio.get_running_loop().time() - start_time < max_wait:
                try:
                    response = await client.get(RUNTIME_HEALTH_URL)
                    if response.status_code != 200:
                        health_ok = False
                        break
                except httpx.ConnectError:
                    # Connection refused - runtime crashed or became unavailable
                    health_ok = False
                    break
                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout):
                    # Specific timeout types - runtime may be overloaded but not crashed.
                    # Continue polling as timeout alone doesn't indicate crash.
                    # Note: httpx.WriteTimeout excluded as health GET has no request body.
                    logger.debug(
                        "Health check timed out during polling (expected during load)"
                    )
                except (httpx.ReadError, httpx.ProtocolError) as e:
                    # Network read errors or protocol violations indicate runtime issues
                    logger.warning(
                        "Health check failed with network/protocol error: %s",
                        type(e).__name__,
                    )
                    health_ok = False
                    break
                await asyncio.sleep(poll_interval)

        # Verify runtime remained healthy throughout the wait period
        assert health_ok, "Runtime became unhealthy after malformed message"

    @pytest.mark.asyncio
    async def test_runtime_handles_missing_fields(
        self, real_kafka_event_bus: EventBusKafka
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

        # RATIONALE: This sleep cannot be replaced with deterministic polling.
        # This is a "negative" test verifying the runtime stays healthy after
        # receiving incomplete events. There's no observable state change to poll for -
        # the runtime should validate and reject gracefully. The 2.0s wait gives
        # enough time for the message to be consumed and rejected.
        # Alternative: Poll health endpoint repeatedly over 2s to catch transient
        # failures, implemented below.
        max_wait = 2.0
        poll_interval = 0.2
        start_time = asyncio.get_running_loop().time()
        health_ok = True

        async with httpx.AsyncClient(timeout=10.0) as client:
            while asyncio.get_running_loop().time() - start_time < max_wait:
                try:
                    response = await client.get(RUNTIME_HEALTH_URL)
                    if response.status_code != 200:
                        health_ok = False
                        break
                except httpx.ConnectError:
                    # Connection refused - runtime crashed or became unavailable
                    health_ok = False
                    break
                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout):
                    # Specific timeout types - runtime may be overloaded but not crashed.
                    # Continue polling as timeout alone doesn't indicate crash.
                    # Note: httpx.WriteTimeout excluded as health GET has no request body.
                    logger.debug(
                        "Health check timed out during polling (expected during load)"
                    )
                except (httpx.ReadError, httpx.ProtocolError) as e:
                    # Network read errors or protocol violations indicate runtime issues
                    logger.warning(
                        "Health check failed with network/protocol error: %s",
                        type(e).__name__,
                    )
                    health_ok = False
                    break
                await asyncio.sleep(poll_interval)

        # Verify runtime remained healthy throughout the wait period
        assert health_ok, "Runtime became unhealthy after incomplete event"


class TestRuntimePerformance:
    """Performance tests for the runtime."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not RUNTIME_PROCESSING_ENABLED, reason=SKIP_PROCESSING_REASON)
    async def test_runtime_processes_event_within_sla(
        self,
        real_kafka_event_bus: EventBusKafka,
        projection_reader: ProjectionReaderRegistration,
        introspection_event: ModelNodeIntrospectionEvent,
        unique_node_id: UUID,
    ) -> None:
        """Test that runtime processes events within acceptable SLA.

        This test uses a two-tier SLA approach:
        1. Soft SLA (SLA_TARGET_SECONDS, default 15s): Warn if exceeded
        2. Hard SLA (SLA_HARD_LIMIT_SECONDS, default 30s): Fail if exceeded

        The soft SLA accounts for normal CI variance (network latency, container
        warmup, shared runner load). The hard SLA catches actual performance
        regressions.

        SLA targets are configurable via environment variables:
        - E2E_SLA_TARGET: Soft SLA target (default: 15s)
        - E2E_SLA_HARD_LIMIT: Hard SLA limit (default: 30s)
        """
        soft_sla = SLA_TARGET_SECONDS
        hard_sla = SLA_HARD_LIMIT_SECONDS

        # Record publish time
        publish_time = datetime.now(UTC)

        # Publish event wrapped in envelope
        envelope = wrap_event_in_envelope(introspection_event)
        await real_kafka_event_bus.publish_envelope(envelope, topic=RUNTIME_INPUT_TOPIC)

        # Poll for projection with hard SLA as the absolute limit
        projection = None
        processing_time = 0.0

        while True:
            elapsed = (datetime.now(UTC) - publish_time).total_seconds()

            if elapsed > hard_sla:
                pytest.fail(
                    f"Runtime exceeded hard SLA limit of {hard_sla}s for event processing. "
                    f"This indicates a real performance problem, not just CI variance. "
                    f"(Configurable via E2E_SLA_HARD_LIMIT env var)"
                )

            # Preserve original event's correlation_id for tracing
            projection = await projection_reader.get_entity_state(
                entity_id=unique_node_id,
                domain="registration",
                correlation_id=introspection_event.correlation_id,
            )

            if projection is not None:
                processing_time = elapsed
                break

            # Tight polling interval (0.1s) for accurate SLA measurement
            await asyncio.sleep(0.1)

        # Check soft SLA and warn if exceeded (but don't fail)
        if processing_time > soft_sla:
            logger.warning(
                "Runtime processing time %.2fs exceeded soft SLA target of %ss. "
                "This may indicate CI latency or a performance regression. "
                "(Hard limit is %ss, configurable via E2E_SLA_TARGET env var)",
                processing_time,
                soft_sla,
                hard_sla,
            )

        # Log processing time for debugging (only visible with -v flag)
        logger.info(
            "Runtime processed event",
            extra={
                "processing_time_seconds": processing_time,
                "soft_sla_seconds": soft_sla,
                "hard_sla_seconds": hard_sla,
                "within_soft_sla": processing_time <= soft_sla,
                "margin_to_hard_sla": hard_sla - processing_time,
            },
        )
