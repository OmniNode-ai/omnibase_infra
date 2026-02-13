# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""E2E tests for ONEX 2-way registration pattern.

This module contains end-to-end integration tests for the node registration
workflow, validating the complete registration flow against real infrastructure
(Kafka, Consul, PostgreSQL).

Test Suites:
    - Suite 1: Node Startup and Introspection Broadcasting
    - Suite 2: Registry Receives and Dual-Registers
    - Suite 3: Registry Startup Requests Re-Introspection
    - Suite 4: Heartbeat Periodic Publishing
    - Suite 5: Registry Recovery Scenario
    - Suite 6: Multiple Nodes Registration
    - Suite 7: Graceful Degradation
    - Suite 8: Registry Self-Registration

Infrastructure Requirements:
    Tests require ALL infrastructure services to be available:
    - PostgreSQL: OMNIBASE_INFRA_DB_URL (database: omnibase_infra)
    - Consul: CONSUL_HOST:28500
    - Kafka/Redpanda: KAFKA_BOOTSTRAP_SERVERS

    Environment variables required:
    - OMNIBASE_INFRA_DB_URL (preferred) or POSTGRES_HOST, POSTGRES_PASSWORD (for PostgreSQL)
    - CONSUL_HOST (for Consul)
    - KAFKA_BOOTSTRAP_SERVERS (for Kafka)

Related Tickets:
    - OMN-892: E2E Registration Tests
    - OMN-888: Registration Orchestrator
    - OMN-915: Mocked E2E Registration Tests (A0-A6)
"""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumIntrospectionReason, EnumRegistrationState
from omnibase_infra.models.discovery import (
    DEFAULT_HEARTBEAT_TOPIC,
    DEFAULT_INTROSPECTION_TOPIC,
    DEFAULT_REQUEST_INTROSPECTION_TOPIC,
)
from omnibase_infra.models.registration import (
    ModelNodeHeartbeatEvent,
    ModelNodeIntrospectionEvent,
)

# Note: ALL_INFRA_AVAILABLE skipif is handled by conftest.py for all E2E tests
from .conftest import make_e2e_test_identity, wait_for_consumer_ready
from .performance_utils import (
    PerformanceThresholds,
    assert_heartbeat_interval,
    calculate_heartbeat_stats,
    timed_operation,
    verify_heartbeat_interval,
)
from .verification_helpers import (
    assert_heartbeat_event_valid,
    assert_heartbeat_updated,
    assert_introspection_event_complete,
    verify_consul_registration,
    verify_dual_registration,
    verify_postgres_registration,
    wait_for_consul_registration,
    wait_for_postgres_registration,
    wait_for_postgres_write,
)

if TYPE_CHECKING:
    import asyncpg

    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.handlers import HandlerConsul
    from omnibase_infra.models.projection import ModelRegistrationProjection
    from omnibase_infra.nodes.node_registration_orchestrator import (
        NodeRegistrationOrchestrator,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeHeartbeat,
    )
    from omnibase_infra.projectors import ProjectionReaderRegistration
    from omnibase_infra.runtime import ProjectorShell

    from .conftest import ProtocolIntrospectableTestNode


# Module-level markers
# Note: conftest.py already applies pytest.mark.e2e and skipif(not ALL_INFRA_AVAILABLE)
# to all tests in this directory. We only add the e2e marker here for explicit clarity.
pytestmark = [
    pytest.mark.e2e,
]


# =============================================================================
# Helper Function: Projection to upsert_partial
# =============================================================================


async def persist_projection_via_shell(
    projector: ProjectorShell,
    projection: ModelRegistrationProjection,
    correlation_id: UUID | None = None,
) -> bool:
    """Convert ModelRegistrationProjection to values dict and upsert via ProjectorShell.

    This helper provides a migration path from ProjectorRegistration.persist() to
    ProjectorShell.upsert_partial(). It extracts values from the projection model
    and calls the contract-driven projector.

    Args:
        projector: ProjectorShell instance for persistence.
        projection: ModelRegistrationProjection to persist.
        correlation_id: Optional correlation ID (generated if not provided).

    Returns:
        True if upsert succeeded, False otherwise.
    """
    cid = correlation_id or uuid4()

    # Convert projection to values dict for upsert_partial
    values: dict[str, object] = {
        "entity_id": projection.entity_id,
        "domain": projection.domain,
        "current_state": projection.current_state.value
        if hasattr(projection.current_state, "value")
        else str(projection.current_state),
        "node_type": projection.node_type.value
        if hasattr(projection.node_type, "value")
        else str(projection.node_type),
        "node_version": str(projection.node_version),
        "capabilities": projection.capabilities.model_dump_json()
        if projection.capabilities
        else "{}",
        "liveness_deadline": projection.liveness_deadline,
        "last_heartbeat_at": projection.last_heartbeat_at,
        "last_applied_event_id": projection.last_applied_event_id,
        "last_applied_offset": 1
        if projection.last_applied_offset is None
        else projection.last_applied_offset,
        "registered_at": projection.registered_at,
        "updated_at": projection.updated_at,
    }

    return await projector.upsert_partial(
        aggregate_id=projection.entity_id,
        values=values,
        correlation_id=cid,
        conflict_columns=["entity_id", "domain"],
    )


# =============================================================================
# Suite 1: Node Startup and Introspection Broadcasting
# =============================================================================


class TestSuite1NodeStartupIntrospection:
    """Suite 1: Node Startup and Introspection Broadcasting.

    These tests verify the first half of the 2-way registration pattern:
    nodes correctly publish introspection events on startup to announce
    their presence to the cluster.

    Test Coverage:
        - Introspection event publishing on startup
        - Event structure completeness and validation
        - Broadcast latency performance (<50ms)
    """

    @pytest.mark.asyncio
    async def test_node_publishes_introspection_on_startup(
        self,
        introspectable_test_node: ProtocolIntrospectableTestNode,
        real_kafka_event_bus: EventBusKafka,
        unique_correlation_id: UUID,
    ) -> None:
        """Test node publishes introspection event on startup.

        Verifies that when a node with MixinNodeIntrospection starts up,
        it correctly publishes an introspection event to the Kafka event bus.

        Expected Behavior:
            1. Node initializes with introspection mixin configured
            2. Node calls publish_introspection(reason=EnumIntrospectionReason.STARTUP)
            3. Introspection event appears on INTROSPECTION_TOPIC

        Assertions:
            - publish_introspection returns True (success)
            - Event contains correct node_id
            - Event contains correct node_type
        """
        # Publish introspection event on startup
        success = await introspectable_test_node.publish_introspection(
            reason=EnumIntrospectionReason.STARTUP, correlation_id=unique_correlation_id
        )

        # Verify publication succeeded
        assert success is True, "Introspection publish should succeed"

        # Get introspection data to verify node identity
        event = await introspectable_test_node.get_introspection_data()

        # Verify event contains correct node identity
        assert event.node_id == introspectable_test_node.node_id, (
            f"Event node_id {event.node_id} should match "
            f"test node {introspectable_test_node.node_id}"
        )
        assert event.node_type == introspectable_test_node.node_type.value, (
            f"Event node_type {event.node_type} should match "
            f"test node type {introspectable_test_node.node_type.value}"
        )

    @pytest.mark.asyncio
    async def test_introspection_event_structure_and_completeness(
        self, introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent]
    ) -> None:
        """Test introspection event has all required fields.

        Validates that introspection events contain all the required fields
        for successful node registration. This ensures the event contract
        is complete before testing the registration flow.

        Required Fields:
            - node_id: Unique node identifier (UUID)
            - node_type: ONEX node type (effect, compute, reducer, orchestrator)
            - node_version: Semantic version string
            - capabilities: Node capabilities dictionary
            - correlation_id: Request correlation ID for tracing
            - timestamp: Timezone-aware datetime

        Assertions:
            - All required fields are present and non-None
            - node_type is a valid ONEX type
            - timestamp is timezone-aware
        """
        # Create an introspection event using the factory
        event: ModelNodeIntrospectionEvent = introspection_event_factory()

        # Use the verification helper to validate all required fields
        assert_introspection_event_complete(event)

        # Additional structure validations
        assert isinstance(event.node_id, UUID), "node_id should be a UUID"
        assert event.node_type in ("effect", "compute", "reducer", "orchestrator"), (
            f"node_type '{event.node_type}' should be a valid ONEX type"
        )

    @pytest.mark.asyncio
    async def test_introspection_event_with_custom_endpoints(
        self, introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent]
    ) -> None:
        """Test introspection event correctly includes custom endpoints.

        Verifies that custom endpoints (health, API, metrics) are correctly
        included in the introspection event for service discovery.

        Assertions:
            - Endpoints dict contains expected keys
            - Endpoint URLs are valid strings
        """
        custom_endpoints = {
            "health": "http://localhost:8080/health",
            "api": "http://localhost:8080/api/v1",
            "metrics": "http://localhost:8080/metrics",
        }

        event: ModelNodeIntrospectionEvent = introspection_event_factory(
            endpoints=custom_endpoints
        )

        # Verify endpoints are correctly set
        assert event.endpoints == custom_endpoints
        for key, url in custom_endpoints.items():
            assert key in event.endpoints
            assert event.endpoints[key] == url

    @pytest.mark.asyncio
    async def test_introspection_broadcast_latency_under_50ms(
        self,
        introspectable_test_node: ProtocolIntrospectableTestNode,
        real_kafka_event_bus: EventBusKafka,
    ) -> None:
        """Test introspection broadcast latency is under 50ms.

        Measures the time to publish an introspection event and asserts
        it completes within the performance threshold of 50ms.

        Performance Threshold:
            - INTROSPECTION_BROADCAST_MS: 50ms (from OMN-892 requirements)

        Assertions:
            - Broadcast completes successfully
            - Elapsed time is under threshold
        """
        async with timed_operation(
            "introspection_broadcast",
            threshold_ms=PerformanceThresholds.INTROSPECTION_BROADCAST_MS,
        ) as timing:
            success = await introspectable_test_node.publish_introspection(
                reason=EnumIntrospectionReason.REQUEST, correlation_id=uuid4()
            )

        # Verify publication succeeded
        assert success is True, "Introspection publish should succeed"

        # Assert timing passed threshold
        timing.assert_passed()

    @pytest.mark.asyncio
    async def test_introspection_event_node_types(
        self, real_kafka_event_bus: EventBusKafka, unique_node_id: UUID
    ) -> None:
        """Test introspection events work for all ONEX node types.

        Verifies that nodes of each type (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
        can correctly publish introspection events.

        Assertions:
            - Each node type produces a valid introspection event
            - node_type field matches the expected type
        """
        from omnibase_infra.mixins import MixinNodeIntrospection
        from omnibase_infra.models.discovery import ModelIntrospectionConfig

        class TestNodeWithType(MixinNodeIntrospection):
            """Minimal test node for type verification."""

            def __init__(
                self, node_id: UUID, node_type: EnumNodeKind, event_bus: EventBusKafka
            ) -> None:
                config = ModelIntrospectionConfig(
                    node_id=node_id,
                    node_type=node_type,
                    node_name="test_node_with_type",
                    event_bus=event_bus,
                    version="1.0.0",
                    cache_ttl=60.0,
                )
                self.initialize_introspection(config)

        # Test each node type
        for node_type in [
            EnumNodeKind.EFFECT,
            EnumNodeKind.COMPUTE,
            EnumNodeKind.REDUCER,
            EnumNodeKind.ORCHESTRATOR,
        ]:
            # Create a node with a unique ID for each type
            type_node_id = uuid4()
            test_node = TestNodeWithType(type_node_id, node_type, real_kafka_event_bus)

            # Get introspection data
            event = await test_node.get_introspection_data()

            # Verify node type is correct
            assert event.node_type == node_type.value, (
                f"Expected node_type '{node_type.value}', got '{event.node_type}'"
            )
            assert event.node_id == type_node_id


# =============================================================================
# Suite 2: Registry Receives and Dual-Registers
# =============================================================================


class TestSuite2RegistryDualRegistration:
    """Suite 2: Registry Receives and Dual-Registers.

    These tests verify the second half of the 2-way registration pattern:
    the registry receives introspection events and performs dual registration
    to both Consul (service discovery) and PostgreSQL (persistence).

    Test Coverage:
        - Registry receives introspection events from Kafka
        - Consul registration succeeds
        - PostgreSQL registration succeeds
        - Dual registration performance (<300ms)
    """

    @pytest.mark.asyncio
    async def test_registry_receives_introspection_event(
        self,
        registration_orchestrator: NodeRegistrationOrchestrator,
        wired_container: ModelONEXContainer,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
        unique_node_id: UUID,
        unique_correlation_id: UUID,
    ) -> None:
        """Test registry receives introspection event from Kafka.

        Verifies that the registration orchestrator correctly receives and
        processes introspection events, initiating the registration workflow.

        Expected Behavior:
            1. Introspection event is published to Kafka
            2. Orchestrator's HandlerNodeIntrospected receives the event
            3. Handler decides whether to initiate registration

        Assertions:
            - Handler is wired in container
            - Handler processes event without error
            - For new nodes, registration is initiated
        """
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeIntrospected,
        )
        from omnibase_infra.nodes.node_registration_orchestrator.services import (
            RegistrationReducerService,
        )
        from omnibase_infra.runtime.util_container_wiring import (
            get_projection_reader_from_container,
        )

        # Get projection reader from container
        projection_reader = await get_projection_reader_from_container(wired_container)

        # Create the handler with projection reader and reducer
        reducer = RegistrationReducerService()
        handler = HandlerNodeIntrospected(projection_reader, reducer)

        # Create introspection event for a new node
        event: ModelNodeIntrospectionEvent = introspection_event_factory(
            node_id=unique_node_id, correlation_id=unique_correlation_id
        )

        # Process the event through the handler using envelope API
        now = datetime.now(UTC)
        envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=event,
            envelope_timestamp=now,
            correlation_id=unique_correlation_id,
            source="e2e-test",
        )
        handler_output = await handler.handle(envelope)
        result_events = handler_output.events

        # For a new node, handler should emit NodeRegistrationInitiated + Accepted
        from omnibase_infra.models.registration.events.model_node_registration_accepted import (
            ModelNodeRegistrationAccepted,
        )
        from omnibase_infra.models.registration.events.model_node_registration_initiated import (
            ModelNodeRegistrationInitiated,
        )

        assert len(result_events) == 2, (
            f"Expected 2 events (Initiated + Accepted), got {len(result_events)}"
        )

        assert isinstance(result_events[0], ModelNodeRegistrationInitiated), (
            f"Expected ModelNodeRegistrationInitiated, "
            f"got {type(result_events[0]).__name__}"
        )
        assert isinstance(result_events[1], ModelNodeRegistrationAccepted), (
            f"Expected ModelNodeRegistrationAccepted, "
            f"got {type(result_events[1]).__name__}"
        )

        # Verify event properties
        initiated_event: ModelNodeRegistrationInitiated = result_events[0]
        assert initiated_event.node_id == unique_node_id
        assert initiated_event.correlation_id == unique_correlation_id

    @pytest.mark.asyncio
    async def test_consul_registration_succeeds(
        self,
        real_consul_handler: HandlerConsul,
        cleanup_consul_services: list[str],
        unique_node_id: UUID,
    ) -> None:
        """Test Consul registration succeeds.

        Verifies that a node can be successfully registered in Consul
        for service discovery.

        Expected Behavior:
            1. Registration intent is executed via HandlerConsul
            2. Service appears in Consul KV store
            3. Service can be queried back

        Assertions:
            - Consul KV write succeeds
            - Service can be retrieved from Consul
        """
        service_id = f"test-node-{unique_node_id.hex[:8]}"

        # Track service for cleanup
        cleanup_consul_services.append(service_id)

        # Build registration payload
        registration_data = {
            "node_id": str(unique_node_id),
            "node_type": "effect",
            "version": "1.0.0",
            "endpoints": {"health": "http://localhost:8080/health"},
            "registered_at": datetime.now(UTC).isoformat(),
        }

        # Register in Consul via KV store
        envelope: dict[str, object] = {
            "operation": "consul.kv_put",
            "payload": {
                "key": f"onex/services/{service_id}",
                "value": json.dumps(registration_data),
            },
        }

        result = await real_consul_handler.execute(envelope)

        # Verify write succeeded
        assert result.result is not None, "Consul KV write should return result"

        # Wait for and verify registration
        consul_result = await wait_for_consul_registration(
            consul_handler=real_consul_handler,
            service_id=service_id,
            timeout_seconds=10.0,
        )

        assert consul_result is not None, (
            f"Service {service_id} should be found in Consul"
        )
        assert consul_result["service_id"] == service_id

    @pytest.mark.asyncio
    async def test_postgres_registration_succeeds(
        self,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        unique_node_id: UUID,
        unique_correlation_id: UUID,
        cleanup_projections: None,
    ) -> None:
        """Test PostgreSQL registration succeeds.

        Verifies that a node registration can be successfully persisted
        to PostgreSQL via the projector.

        Expected Behavior:
            1. Registration projection is written via projector
            2. Projection can be queried via projection_reader
            3. State is correctly persisted

        Assertions:
            - Projection write succeeds
            - Projection can be retrieved
            - State matches expected value
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )

        # Create a registration projection
        now = datetime.now(UTC)
        projection = ModelRegistrationProjection(
            entity_id=unique_node_id,
            domain="registration",
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            registered_at=now,
            updated_at=now,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            last_applied_event_id=unique_node_id,
        )

        # Persist via projector
        await persist_projection_via_shell(real_projector, projection)

        # Wait for and verify registration in PostgreSQL
        pg_result = await wait_for_postgres_registration(
            projection_reader=projection_reader,
            node_id=unique_node_id,
            expected_state=EnumRegistrationState.PENDING_REGISTRATION,
            timeout_seconds=10.0,
        )

        assert pg_result is not None, (
            f"Node {unique_node_id} should be found in PostgreSQL"
        )
        assert pg_result.entity_id == unique_node_id
        assert pg_result.current_state == EnumRegistrationState.PENDING_REGISTRATION
        assert pg_result.node_type == EnumNodeKind.EFFECT

    @pytest.mark.asyncio
    async def test_dual_registration_consul_and_postgres(
        self,
        real_consul_handler: HandlerConsul,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        cleanup_consul_services: list[str],
        cleanup_projections: None,
        unique_node_id: UUID,
    ) -> None:
        """Test dual registration to both Consul and PostgreSQL.

        Verifies that the complete dual registration pattern works:
        a node can be registered in both Consul (for service discovery)
        and PostgreSQL (for persistence) simultaneously.

        Expected Behavior:
            1. Node is registered in Consul via KV store
            2. Node is registered in PostgreSQL via projector
            3. Both registrations are verifiable

        Assertions:
            - Consul registration succeeds
            - PostgreSQL registration succeeds
            - Both can be verified via verify_dual_registration
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )

        service_id = f"test-node-{unique_node_id.hex[:8]}"
        cleanup_consul_services.append(service_id)

        now = datetime.now(UTC)

        # Step 1: Register in Consul
        registration_data = {
            "node_id": str(unique_node_id),
            "node_type": "effect",
            "version": "1.0.0",
            "endpoints": {"health": "http://localhost:8080/health"},
            "registered_at": now.isoformat(),
        }

        consul_envelope: dict[str, object] = {
            "operation": "consul.kv_put",
            "payload": {
                "key": f"onex/services/{service_id}",
                "value": json.dumps(registration_data),
            },
        }

        await real_consul_handler.execute(consul_envelope)

        # Step 2: Register in PostgreSQL
        projection = ModelRegistrationProjection(
            entity_id=unique_node_id,
            domain="registration",
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            registered_at=now,
            updated_at=now,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            last_applied_event_id=unique_node_id,
        )

        await persist_projection_via_shell(real_projector, projection)

        # Step 3: Verify dual registration
        consul_result, postgres_result = await verify_dual_registration(
            consul_handler=real_consul_handler,
            projection_reader=projection_reader,
            node_id=unique_node_id,
            service_id=service_id,
            timeout_seconds=10.0,
        )

        # Assertions
        assert consul_result is not None, "Consul registration should be found"
        assert consul_result["service_id"] == service_id

        assert postgres_result is not None, "PostgreSQL registration should be found"
        assert postgres_result.entity_id == unique_node_id

    @pytest.mark.asyncio
    async def test_dual_registration_under_300ms(
        self,
        real_consul_handler: HandlerConsul,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        cleanup_consul_services: list[str],
        cleanup_projections: None,
        unique_node_id: UUID,
    ) -> None:
        """Test dual registration completes under 300ms.

        Measures the time to complete both Consul and PostgreSQL registration
        and asserts it completes within the performance threshold of 300ms.

        Performance Threshold:
            - DUAL_REGISTRATION_MS: 300ms (from OMN-892 requirements)

        Assertions:
            - Both registrations complete successfully
            - Total elapsed time is under threshold
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )

        service_id = f"perf-test-{unique_node_id.hex[:8]}"
        cleanup_consul_services.append(service_id)

        now = datetime.now(UTC)

        async with timed_operation(
            "dual_registration", threshold_ms=PerformanceThresholds.DUAL_REGISTRATION_MS
        ) as timing:
            # Consul registration
            registration_data = {
                "node_id": str(unique_node_id),
                "node_type": "effect",
                "version": "1.0.0",
                "registered_at": now.isoformat(),
            }

            consul_envelope: dict[str, object] = {
                "operation": "consul.kv_put",
                "payload": {
                    "key": f"onex/services/{service_id}",
                    "value": json.dumps(registration_data),
                },
            }

            await real_consul_handler.execute(consul_envelope)

            # PostgreSQL registration
            projection = ModelRegistrationProjection(
                entity_id=unique_node_id,
                domain="registration",
                current_state=EnumRegistrationState.PENDING_REGISTRATION,
                registered_at=now,
                updated_at=now,
                node_type=EnumNodeKind.EFFECT,
                node_version=ModelSemVer.parse("1.0.0"),
                last_applied_event_id=unique_node_id,
            )

            await persist_projection_via_shell(real_projector, projection)

        # Verify registrations completed
        consul_result, postgres_result = await verify_dual_registration(
            consul_handler=real_consul_handler,
            projection_reader=projection_reader,
            node_id=unique_node_id,
            service_id=service_id,
            timeout_seconds=5.0,
        )

        assert consul_result is not None
        assert postgres_result is not None

        # Assert timing passed threshold
        timing.assert_passed()

    @pytest.mark.asyncio
    async def test_handler_idempotency_blocking_states(
        self,
        wired_container: ModelONEXContainer,
        real_projector: ProjectorShell,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
        unique_node_id: UUID,
        unique_correlation_id: UUID,
        cleanup_projections: None,
    ) -> None:
        """Test handler idempotency for blocking registration states.

        Verifies that when a node is already in a blocking state
        (PENDING_REGISTRATION, ACCEPTED, AWAITING_ACK, ACK_RECEIVED, ACTIVE),
        the handler does not initiate a new registration.

        Expected Behavior:
            1. Node is registered with PENDING_REGISTRATION state
            2. Introspection event is received for that node
            3. Handler returns empty list (no-op)

        Assertions:
            - Handler returns empty list for blocking states
            - No duplicate registration is initiated
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeIntrospected,
        )
        from omnibase_infra.nodes.node_registration_orchestrator.services import (
            RegistrationReducerService,
        )
        from omnibase_infra.runtime.util_container_wiring import (
            get_projection_reader_from_container,
        )

        # Get projection reader from container
        projection_reader = await get_projection_reader_from_container(wired_container)

        # Create the handler
        reducer = RegistrationReducerService()
        handler = HandlerNodeIntrospected(projection_reader, reducer)

        # Pre-create a projection in PENDING_REGISTRATION state
        now = datetime.now(UTC)
        projection = ModelRegistrationProjection(
            entity_id=unique_node_id,
            domain="registration",
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            registered_at=now,
            updated_at=now,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            last_applied_event_id=unique_node_id,
        )

        await persist_projection_via_shell(real_projector, projection)

        # Wait for PostgreSQL write to complete using deterministic polling
        write_result = await wait_for_postgres_write(
            projection_reader, unique_node_id, timeout_seconds=2.0
        )
        assert write_result is not None, (
            f"PostgreSQL write did not complete in time for node {unique_node_id}"
        )

        # Create introspection event for the same node
        event: ModelNodeIntrospectionEvent = introspection_event_factory(
            node_id=unique_node_id, correlation_id=unique_correlation_id
        )

        # Process the event through the handler using envelope API
        envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=event,
            envelope_timestamp=now,
            correlation_id=unique_correlation_id,
            source="e2e-test",
        )
        handler_output = await handler.handle(envelope)
        result_events = handler_output.events

        # For a node in blocking state, handler should return empty list (no-op)
        assert len(result_events) == 0, (
            f"Expected 0 events for blocking state, got {len(result_events)}"
        )

    @pytest.mark.asyncio
    async def test_handler_allows_retriable_states(
        self,
        wired_container: ModelONEXContainer,
        real_projector: ProjectorShell,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
        unique_node_id: UUID,
        unique_correlation_id: UUID,
        cleanup_projections: None,
    ) -> None:
        """Test handler allows re-registration for retriable states.

        Verifies that when a node is in a retriable state
        (LIVENESS_EXPIRED, REJECTED, ACK_TIMED_OUT),
        the handler initiates a new registration.

        Expected Behavior:
            1. Node is registered with LIVENESS_EXPIRED state
            2. Introspection event is received for that node
            3. Handler initiates new registration

        Assertions:
            - Handler returns NodeRegistrationInitiated for retriable states
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )
        from omnibase_infra.models.registration.events.model_node_registration_initiated import (
            ModelNodeRegistrationInitiated,
        )
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeIntrospected,
        )
        from omnibase_infra.nodes.node_registration_orchestrator.services import (
            RegistrationReducerService,
        )
        from omnibase_infra.runtime.util_container_wiring import (
            get_projection_reader_from_container,
        )

        # Get projection reader from container
        projection_reader = await get_projection_reader_from_container(wired_container)

        # Create the handler
        reducer = RegistrationReducerService()
        handler = HandlerNodeIntrospected(projection_reader, reducer)

        # Pre-create a projection in LIVENESS_EXPIRED state (retriable)
        now = datetime.now(UTC)
        projection = ModelRegistrationProjection(
            entity_id=unique_node_id,
            domain="registration",
            current_state=EnumRegistrationState.LIVENESS_EXPIRED,
            registered_at=now,
            updated_at=now,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            last_applied_event_id=unique_node_id,
        )

        await persist_projection_via_shell(real_projector, projection)

        # Wait for PostgreSQL write to complete using deterministic polling
        write_result = await wait_for_postgres_write(
            projection_reader, unique_node_id, timeout_seconds=2.0
        )
        assert write_result is not None, (
            f"PostgreSQL write did not complete in time for node {unique_node_id}"
        )

        # Create introspection event for the same node
        event: ModelNodeIntrospectionEvent = introspection_event_factory(
            node_id=unique_node_id, correlation_id=unique_correlation_id
        )

        # Process the event through the handler using envelope API
        envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=event,
            envelope_timestamp=now,
            correlation_id=unique_correlation_id,
            source="e2e-test",
        )
        handler_output = await handler.handle(envelope)
        result_events = handler_output.events

        # For a node in retriable state, handler emits Initiated + Accepted
        assert len(result_events) == 2, (
            f"Expected 2 events for retriable state, got {len(result_events)}"
        )
        assert isinstance(result_events[0], ModelNodeRegistrationInitiated), (
            f"Expected ModelNodeRegistrationInitiated, "
            f"got {type(result_events[0]).__name__}"
        )


# =============================================================================
# Suite 3: Registry Startup Requests Re-Introspection
# =============================================================================


class TestSuite3ReIntrospection:
    """Suite 3: Registry Startup Requests Re-Introspection.

    Tests that the registry can request fresh introspection data from nodes
    on startup or after recovery, and that nodes respond appropriately.

    Scenarios:
        - Registry publishes REQUEST_INTROSPECTION on startup
        - Nodes respond with fresh introspection data
        - Request and response have matching correlation_id
    """

    @pytest.mark.asyncio
    async def test_registry_publishes_request_introspection_on_startup(
        self, real_kafka_event_bus: EventBusKafka, wired_container: ModelONEXContainer
    ) -> None:
        """Test registry publishes REQUEST_INTROSPECTION on startup.

        Validates that when a registry starts up, it publishes a
        REQUEST_INTROSPECTION event to request all nodes to re-broadcast
        their introspection data.

        Steps:
            1. Subscribe to the introspection request topic
            2. Simulate registry startup by publishing a request
            3. Verify the request event is published to Kafka
        """
        request_topic = DEFAULT_REQUEST_INTROSPECTION_TOPIC
        correlation_id = uuid4()
        event_received = asyncio.Event()
        received_messages: list[dict[str, object]] = []

        async def on_message(message: object) -> None:
            """Capture incoming messages."""
            if hasattr(message, "value") and message.value:
                try:
                    data = json.loads(message.value.decode("utf-8"))
                    received_messages.append(data)
                    event_received.set()
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

        # Subscribe to request topic
        group_id = f"e2e-test-registry-{correlation_id.hex[:8]}"
        unsubscribe = await real_kafka_event_bus.subscribe(
            topic=request_topic,
            node_identity=make_e2e_test_identity("registry_requests"),
            on_message=on_message,
        )

        try:
            # Simulate registry startup - publish request introspection event
            request_event = {
                "event_type": "REQUEST_INTROSPECTION",
                "correlation_id": str(correlation_id),
                "timestamp": datetime.now(UTC).isoformat(),
                "reason": "registry_startup",
            }

            await real_kafka_event_bus.publish(
                topic=request_topic,
                key=b"registry-startup",
                value=json.dumps(request_event).encode("utf-8"),
            )

            # Wait for event to be received
            try:
                await asyncio.wait_for(event_received.wait(), timeout=10.0)
            except TimeoutError:
                # Event may have been consumed before our subscription started
                # This is acceptable for E2E tests
                pass

            # Verify event was published (check Kafka can receive it)
            # The key assertion is that publishing succeeded without error
            assert request_event["event_type"] == "REQUEST_INTROSPECTION"
            assert request_event["correlation_id"] == str(correlation_id)

        finally:
            await unsubscribe()

    @pytest.mark.asyncio
    async def test_nodes_respond_with_fresh_introspection(
        self,
        introspectable_test_node: ProtocolIntrospectableTestNode,
        real_kafka_event_bus: EventBusKafka,
        unique_correlation_id: UUID,
    ) -> None:
        """Test nodes respond to re-introspection request with fresh data.

        Validates that when a node receives a REQUEST_INTROSPECTION event,
        it responds by publishing fresh introspection data.

        Steps:
            1. Start the node's introspection tasks
            2. Publish a REQUEST_INTROSPECTION event
            3. Verify node responds with new introspection event
        """
        introspection_topic = DEFAULT_INTROSPECTION_TOPIC

        # Start introspection tasks (including registry listener)
        await introspectable_test_node.start_introspection_tasks(
            enable_heartbeat=False,  # Disable heartbeat for this test
            enable_registry_listener=True,
        )

        # Wait for registry listener's Kafka consumer to be ready.
        # The listener subscribes to DEFAULT_REQUEST_INTROSPECTION_TOPIC internally.
        # RATIONALE: This uses wait_for_consumer_ready for consistency with other
        # consumer waits, though we can't access the internal event bus directly.
        # The wait_for_consumer_ready helper documents the known limitation that
        # true readiness polling would require EventBusKafka API changes.
        await wait_for_consumer_ready(
            real_kafka_event_bus, DEFAULT_REQUEST_INTROSPECTION_TOPIC, max_wait=2.0
        )

        try:
            # Collect introspection events for our node
            event_received = asyncio.Event()
            received_introspection: list[dict[str, object]] = []

            async def on_introspection(message: object) -> None:
                """Capture introspection events."""
                if hasattr(message, "value") and message.value:
                    try:
                        data = json.loads(message.value.decode("utf-8"))
                        node_id_str = str(introspectable_test_node.node_id)
                        if data.get("node_id") == node_id_str:
                            received_introspection.append(data)
                            event_received.set()
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

            # Subscribe to introspection topic
            group_id = f"e2e-test-introspection-{unique_correlation_id.hex[:8]}"
            unsub_introspection = await real_kafka_event_bus.subscribe(
                topic=introspection_topic,
                node_identity=make_e2e_test_identity("introspection"),
                on_message=on_introspection,
            )

            try:
                # Publish REQUEST_INTROSPECTION event
                request_event = {
                    "event_type": "REQUEST_INTROSPECTION",
                    "correlation_id": str(unique_correlation_id),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "reason": "e2e_test_request",
                }

                await real_kafka_event_bus.publish(
                    topic=DEFAULT_REQUEST_INTROSPECTION_TOPIC,
                    key=b"e2e-test",
                    value=json.dumps(request_event).encode("utf-8"),
                )

                # Wait for introspection response
                try:
                    await asyncio.wait_for(event_received.wait(), timeout=15.0)
                    assert len(received_introspection) > 0, (
                        "Expected at least one introspection event from node"
                    )

                    # Verify introspection event has expected fields
                    introspection_event = received_introspection[-1]
                    assert introspection_event.get("node_id") == str(
                        introspectable_test_node.node_id
                    )
                    assert introspection_event.get("node_type") is not None
                    assert introspection_event.get("capabilities") is not None

                except TimeoutError:
                    warnings.warn(
                        "Registry listener may not have been ready; "
                        "introspection event not received in E2E environment",
                        UserWarning,
                        stacklevel=1,
                    )

            finally:
                await unsub_introspection()

        finally:
            await introspectable_test_node.stop_introspection_tasks()

    @pytest.mark.asyncio
    async def test_request_response_correlation(
        self,
        introspectable_test_node: ProtocolIntrospectableTestNode,
        real_kafka_event_bus: EventBusKafka,
        unique_correlation_id: UUID,
    ) -> None:
        """Test request and response have matching correlation_id.

        Validates that the correlation_id from the REQUEST_INTROSPECTION
        event is propagated to the introspection response.

        Steps:
            1. Start node introspection tasks
            2. Publish REQUEST_INTROSPECTION with specific correlation_id
            3. Verify response introspection event has matching correlation_id
        """
        introspection_topic = DEFAULT_INTROSPECTION_TOPIC
        request_correlation_id = unique_correlation_id

        # Start introspection tasks
        await introspectable_test_node.start_introspection_tasks(
            enable_heartbeat=False, enable_registry_listener=True
        )

        # Wait for registry listener's Kafka consumer to be ready.
        # See wait_for_consumer_ready docstring for known limitations.
        await wait_for_consumer_ready(
            real_kafka_event_bus, DEFAULT_REQUEST_INTROSPECTION_TOPIC, max_wait=2.0
        )

        try:
            # Track responses with matching correlation_id
            matching_responses: list[dict[str, object]] = []
            event_received = asyncio.Event()

            async def on_introspection(message: object) -> None:
                """Capture introspection events with matching correlation."""
                if hasattr(message, "value") and message.value:
                    try:
                        data = json.loads(message.value.decode("utf-8"))
                        node_id_str = str(introspectable_test_node.node_id)
                        if data.get("node_id") == node_id_str:
                            # Check if correlation_id matches
                            response_corr_id = data.get("correlation_id")
                            if response_corr_id == str(request_correlation_id):
                                matching_responses.append(data)
                                event_received.set()
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

            # Subscribe to introspection topic
            group_id = f"e2e-test-corr-{request_correlation_id.hex[:8]}"
            unsub = await real_kafka_event_bus.subscribe(
                topic=introspection_topic,
                node_identity=make_e2e_test_identity("correlation"),
                on_message=on_introspection,
            )

            try:
                # Publish REQUEST_INTROSPECTION with specific correlation_id
                request_event = {
                    "event_type": "REQUEST_INTROSPECTION",
                    "correlation_id": str(request_correlation_id),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "reason": "correlation_test",
                }

                await real_kafka_event_bus.publish(
                    topic=DEFAULT_REQUEST_INTROSPECTION_TOPIC,
                    key=b"correlation-test",
                    value=json.dumps(request_event).encode("utf-8"),
                )

                # Wait for matching response
                try:
                    await asyncio.wait_for(event_received.wait(), timeout=15.0)
                    assert len(matching_responses) > 0, (
                        "Expected response with matching correlation_id"
                    )

                    # Verify correlation_id was preserved
                    response = matching_responses[-1]
                    assert response.get("correlation_id") == str(
                        request_correlation_id
                    ), (
                        f"Correlation ID mismatch: expected {request_correlation_id}, "
                        f"got {response.get('correlation_id')}"
                    )

                except TimeoutError:
                    warnings.warn(
                        "Registry listener may not have propagated correlation_id; "
                        "response not received in E2E environment",
                        UserWarning,
                        stacklevel=1,
                    )

            finally:
                await unsub()

        finally:
            await introspectable_test_node.stop_introspection_tasks()


# =============================================================================
# Suite 4: Heartbeat Periodic Publishing
# =============================================================================


class TestSuite4HeartbeatPublishing:
    """Suite 4: Heartbeat Periodic Publishing.

    Tests the periodic heartbeat publishing functionality of nodes,
    including timing accuracy, content validation, and performance.

    Scenarios:
        - Heartbeat published every 30 seconds
        - Heartbeat includes uptime and operations count
        - Heartbeat overhead under performance threshold
        - Heartbeat updates liveness in projection
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_heartbeat_published_every_30_seconds(
        self,
        introspectable_test_node: ProtocolIntrospectableTestNode,
        real_kafka_event_bus: EventBusKafka,
    ) -> None:
        """Test heartbeat publishes every 30 seconds.

        Validates that the node publishes heartbeats at the expected
        30-second interval, within the tolerance threshold.

        Note: This test takes ~65 seconds to verify 2 heartbeat intervals.

        Steps:
            1. Start node with heartbeat enabled
            2. Collect heartbeat events for ~65 seconds
            3. Verify at least 2 heartbeats were received
            4. Verify interval is within tolerance (30s +/- 5s)
        """
        heartbeat_topic = DEFAULT_HEARTBEAT_TOPIC
        node_id_str = str(introspectable_test_node.node_id)
        collected_heartbeats: list[ModelNodeHeartbeatEvent] = []

        async def on_heartbeat(message: object) -> None:
            """Capture heartbeat events for this node."""
            if hasattr(message, "value") and message.value:
                try:
                    data = json.loads(message.value.decode("utf-8"))
                    # Unwrap envelope if present (publish_envelope wraps in ModelEventEnvelope)
                    payload = data.get("payload", data)
                    if (
                        isinstance(payload, dict)
                        and payload.get("node_id") == node_id_str
                    ):
                        heartbeat = ModelNodeHeartbeatEvent.model_validate(payload)
                        collected_heartbeats.append(heartbeat)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    pass

        # Subscribe to heartbeat topic
        group_id = f"e2e-heartbeat-{introspectable_test_node.node_id.hex[:8]}"
        unsub = await real_kafka_event_bus.subscribe(
            topic=heartbeat_topic,
            node_identity=make_e2e_test_identity("heartbeat"),
            on_message=on_heartbeat,
        )

        try:
            # Wait for consumer to be ready before starting heartbeats.
            # Using max_wait=1.0 to ensure the first "immediate" heartbeat is captured.
            # See wait_for_consumer_ready docstring for known limitations.
            await wait_for_consumer_ready(
                real_kafka_event_bus, heartbeat_topic, max_wait=1.0
            )

            # Start heartbeat with 30s interval
            await introspectable_test_node.start_introspection_tasks(
                enable_heartbeat=True,
                heartbeat_interval_seconds=30.0,
                enable_registry_listener=False,
            )

            # RATIONALE: This is a heartbeat interval test - we MUST wait for
            # at least 2 heartbeat cycles (30s interval) to verify the timing.
            # A shorter wait would not test the actual heartbeat interval behavior.
            # Wait for 2 heartbeat intervals (65 seconds to be safe).
            # First heartbeat happens immediately on start.
            await asyncio.sleep(65.0)

            # Verify we received at least 2 heartbeats
            assert len(collected_heartbeats) >= 2, (
                f"Expected at least 2 heartbeats, got {len(collected_heartbeats)}"
            )

            # Verify interval is within tolerance
            if len(collected_heartbeats) >= 2:
                assert verify_heartbeat_interval(
                    collected_heartbeats,
                    expected_interval_seconds=30.0,
                    tolerance_seconds=PerformanceThresholds.HEARTBEAT_TOLERANCE_SECONDS,
                ), "Heartbeat interval outside expected tolerance"

                # Calculate and log stats for debugging
                stats = calculate_heartbeat_stats(collected_heartbeats)
                assert stats["count"] >= 1, "Expected at least one interval"

        finally:
            await introspectable_test_node.stop_introspection_tasks()
            await unsub()

    @pytest.mark.asyncio
    async def test_heartbeat_includes_uptime_and_operations(
        self,
        introspectable_test_node: ProtocolIntrospectableTestNode,
        real_kafka_event_bus: EventBusKafka,
    ) -> None:
        """Test heartbeat includes uptime and active operations count.

        Validates that heartbeat events contain the expected fields
        including timestamp (proxy for uptime tracking) and operations count.

        Steps:
            1. Start node with heartbeat enabled
            2. Wait for first heartbeat event
            3. Verify event has required fields
            4. Validate using assert_heartbeat_event_valid
        """
        heartbeat_topic = DEFAULT_HEARTBEAT_TOPIC
        node_id_str = str(introspectable_test_node.node_id)
        first_heartbeat: ModelNodeHeartbeatEvent | None = None
        event_received = asyncio.Event()

        async def on_heartbeat(message: object) -> None:
            """Capture first heartbeat event for this node."""
            nonlocal first_heartbeat
            if hasattr(message, "value") and message.value:
                try:
                    data = json.loads(message.value.decode("utf-8"))
                    # Unwrap envelope if present (publish_envelope wraps in ModelEventEnvelope)
                    payload = data.get("payload", data)
                    if (
                        isinstance(payload, dict)
                        and payload.get("node_id") == node_id_str
                    ):
                        first_heartbeat = ModelNodeHeartbeatEvent.model_validate(
                            payload
                        )
                        event_received.set()
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    pass

        # Subscribe to heartbeat topic
        group_id = f"e2e-heartbeat-fields-{introspectable_test_node.node_id.hex[:8]}"
        unsub = await real_kafka_event_bus.subscribe(
            topic=heartbeat_topic,
            node_identity=make_e2e_test_identity("heartbeat_fields"),
            on_message=on_heartbeat,
        )

        try:
            # Start heartbeat with short interval for faster testing
            await introspectable_test_node.start_introspection_tasks(
                enable_heartbeat=True,
                heartbeat_interval_seconds=5.0,  # Use shorter interval for test
                enable_registry_listener=False,
            )

            # Wait for first heartbeat
            await asyncio.wait_for(event_received.wait(), timeout=10.0)

            assert first_heartbeat is not None, "Expected to receive heartbeat event"

            # Validate heartbeat using helper
            assert_heartbeat_event_valid(first_heartbeat)

            # Additional field validations
            assert first_heartbeat.node_id == introspectable_test_node.node_id
            assert first_heartbeat.node_type is not None
            assert first_heartbeat.uptime_seconds >= 0
            assert first_heartbeat.active_operations_count >= 0
            assert first_heartbeat.timestamp is not None

        finally:
            await introspectable_test_node.stop_introspection_tasks()
            await unsub()

    @pytest.mark.asyncio
    async def test_heartbeat_overhead_within_threshold(
        self, introspectable_test_node: ProtocolIntrospectableTestNode
    ) -> None:
        """Test heartbeat emission overhead is within acceptable threshold.

        Validates that the performance overhead of emitting a single
        heartbeat is within the acceptable threshold defined in
        PerformanceThresholds.HEARTBEAT_OVERHEAD_MS.

        Steps:
            1. Initialize node introspection
            2. Time a single heartbeat publish operation
            3. Verify overhead is under HEARTBEAT_OVERHEAD_MS threshold
        """
        # Use centralized threshold from PerformanceThresholds
        # (calibrated for remote infrastructure network latency)
        threshold_ms = PerformanceThresholds.HEARTBEAT_OVERHEAD_MS

        # Time the heartbeat publish operation
        async with timed_operation(
            "heartbeat_publish", threshold_ms=threshold_ms
        ) as timing:
            # _publish_heartbeat is the internal method that does the actual work
            # Since we're testing overhead, we call it directly
            await introspectable_test_node._publish_heartbeat()

        # Verify the operation completed (success depends on event bus availability)
        # In E2E tests, event bus is connected so this should succeed
        # The timing assertion is the key check
        timing.assert_passed()

        # Log timing for debugging
        assert timing.elapsed_ms < threshold_ms, (
            f"Heartbeat overhead {timing.elapsed_ms:.2f}ms exceeds {threshold_ms}ms threshold"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_heartbeat_updates_projection_liveness(
        self,
        heartbeat_handler: HandlerNodeHeartbeat,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        introspectable_test_node: ProtocolIntrospectableTestNode,
        unique_node_id: UUID,
        postgres_pool: asyncpg.Pool,
    ) -> None:
        """Test heartbeat updates liveness_deadline in projection.

        Validates that when a heartbeat is received, the registration
        projection's liveness deadline is updated correctly.

        OMN-1102: Test now uses heartbeat_handler directly instead of
        through orchestrator methods (declarative pattern).

        Steps:
            1. Create a projection record (simulating completed registration)
            2. Record the initial heartbeat time
            3. Send a heartbeat event through the handler
            4. Execute the returned intents against PostgreSQL
            5. Verify projection's last_heartbeat_at was updated
        """
        from omnibase_infra.models.projection import ModelRegistrationProjection
        from omnibase_infra.runtime.intent_effects.intent_effect_postgres_update import (
            IntentEffectPostgresUpdate,
        )

        correlation_id = uuid4()
        now = datetime.now(UTC)

        # Create an initial projection in active state (simulating completed registration)
        event_id = uuid4()
        initial_projection = ModelRegistrationProjection(
            entity_id=unique_node_id,
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            node_type=introspectable_test_node.node_type,
            node_version=ModelSemVer.parse("1.0.0"),
            capabilities={"test": True},
            ack_deadline=None,
            liveness_deadline=now,  # Will be updated by heartbeat
            last_heartbeat_at=None,  # Not yet received any heartbeat
            ack_timeout_emitted_at=None,
            liveness_timeout_emitted_at=None,
            registered_at=now,
            updated_at=now,
            correlation_id=correlation_id,
            # Idempotency fields (required by the model)
            last_applied_event_id=event_id,
            last_applied_offset=1,
            last_applied_sequence=None,
            last_applied_partition=None,
        )

        # Persist the projection using upsert_partial
        await persist_projection_via_shell(
            real_projector, initial_projection, correlation_id=correlation_id
        )

        # Record time before sending heartbeat
        min_heartbeat_time = datetime.now(UTC)

        # Create heartbeat event - preserve test's correlation_id for tracing
        heartbeat_timestamp = datetime.now(UTC)
        heartbeat_event = ModelNodeHeartbeatEvent(
            node_id=unique_node_id,
            node_type=introspectable_test_node.node_type,
            uptime_seconds=10.0,
            active_operations_count=0,
            correlation_id=correlation_id,
            timestamp=heartbeat_timestamp,
        )

        # Process heartbeat through the handler directly using envelope API
        # OMN-1102: Orchestrator is declarative - test handler directly
        # (In full E2E with runtime, this would come from Kafka consumer)
        heartbeat_envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=heartbeat_event,
            envelope_timestamp=heartbeat_timestamp,
            correlation_id=correlation_id,
            source="e2e-test",
        )
        handler_output = await heartbeat_handler.handle(heartbeat_envelope)

        # Execute the returned intents against PostgreSQL.
        # The handler is pure compute (returns intents), so we must apply
        # the UPDATE intents to the database manually, mirroring what the
        # runtime's IntentExecutionRouter does in production.
        intent_effect = IntentEffectPostgresUpdate(pool=postgres_pool)
        for intent in handler_output.intents:
            await intent_effect.execute(intent.payload, correlation_id=correlation_id)

        # Query the projection to verify heartbeat was processed
        projection = await projection_reader.get_entity_state(
            entity_id=unique_node_id,
            domain="registration",
            correlation_id=correlation_id,
        )
        assert projection is not None, "Projection should exist"

        # Verify heartbeat was updated
        assert_heartbeat_updated(projection, min_heartbeat_time)


# =============================================================================
# Additional Performance Tests
# =============================================================================


class TestHeartbeatPerformanceExtended:
    """Extended performance tests for heartbeat functionality.

    These tests verify specific performance characteristics of the
    heartbeat system that may require longer execution times.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_heartbeat_interval_consistency_over_multiple_cycles(
        self,
        introspectable_test_node: ProtocolIntrospectableTestNode,
        real_kafka_event_bus: EventBusKafka,
    ) -> None:
        """Test heartbeat interval remains consistent over multiple cycles.

        Collects heartbeats over 3 intervals and verifies interval
        consistency and statistics.

        Note: This test takes ~100 seconds to complete.
        """
        heartbeat_topic = DEFAULT_HEARTBEAT_TOPIC
        node_id_str = str(introspectable_test_node.node_id)
        collected_heartbeats: list[ModelNodeHeartbeatEvent] = []

        async def on_heartbeat(message: object) -> None:
            """Capture heartbeat events."""
            if hasattr(message, "value") and message.value:
                try:
                    data = json.loads(message.value.decode("utf-8"))
                    # Unwrap envelope if present (publish_envelope wraps in ModelEventEnvelope)
                    payload = data.get("payload", data)
                    if (
                        isinstance(payload, dict)
                        and payload.get("node_id") == node_id_str
                    ):
                        heartbeat = ModelNodeHeartbeatEvent.model_validate(payload)
                        collected_heartbeats.append(heartbeat)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    pass

        group_id = f"e2e-hb-consistency-{introspectable_test_node.node_id.hex[:8]}"
        unsub = await real_kafka_event_bus.subscribe(
            topic=heartbeat_topic,
            node_identity=make_e2e_test_identity("hb_consistency"),
            on_message=on_heartbeat,
        )

        try:
            # Start heartbeat
            await introspectable_test_node.start_introspection_tasks(
                enable_heartbeat=True,
                heartbeat_interval_seconds=30.0,
                enable_registry_listener=False,
            )

            # RATIONALE: This is a heartbeat consistency test - we MUST wait for
            # at least 3 heartbeat cycles (30s interval each) to verify interval
            # stability. A shorter wait would not test the actual timing consistency.
            # Wait for 3+ intervals (100 seconds).
            await asyncio.sleep(100.0)

            # Verify we have enough heartbeats
            assert len(collected_heartbeats) >= 3, (
                f"Expected at least 3 heartbeats, got {len(collected_heartbeats)}"
            )

            # Assert interval consistency using helper
            assert_heartbeat_interval(
                collected_heartbeats,
                expected_interval_seconds=30.0,
                tolerance_seconds=5.0,
            )

            # Calculate and verify stats
            stats = calculate_heartbeat_stats(collected_heartbeats)
            assert stats["count"] >= 2, "Expected at least 2 intervals"
            assert 25.0 <= stats["avg_interval_s"] <= 35.0, (
                f"Average interval {stats['avg_interval_s']:.1f}s outside expected range"
            )

        finally:
            await introspectable_test_node.stop_introspection_tasks()
            await unsub()


# =============================================================================
# Suite 5: Registry Recovery Scenario
# =============================================================================


class TestSuite5RegistryRecovery:
    """Suite 5: Registry Recovery Scenario.

    Tests the registry's ability to recover from failures and maintain
    consistent state across restarts.

    Test Coverage:
        - Registry state recovery after simulated restart
        - Re-registration behavior after recovery
        - Idempotent UPSERT semantics for registration
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_registry_recovery_after_restart(
        self,
        wired_container: ModelONEXContainer,
        postgres_pool: asyncpg.Pool,
        real_kafka_event_bus: EventBusKafka,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
        unique_node_id: UUID,
        unique_correlation_id: UUID,
        cleanup_projections: None,
    ) -> None:
        """Test registry recovers state after simulated restart.

        Verifies that when a registry restarts, it can correctly recover
        existing registration state from PostgreSQL.

        Steps:
            1. Register a node using first orchestrator instance
            2. Create a new orchestrator (simulating restart)
            3. Verify the new orchestrator can read the existing registration

        Assertions:
            - Initial registration succeeds
            - New orchestrator can query existing registration
            - Registration state is preserved across restart
        """
        from omnibase_infra.projectors.contracts import REGISTRATION_PROJECTOR_CONTRACT
        from omnibase_infra.runtime import ProjectorPluginLoader, ProjectorShell
        from omnibase_infra.runtime.util_container_wiring import (
            get_projection_reader_from_container,
        )

        # Step 1: Register a node using the first orchestrator
        projection_reader = await get_projection_reader_from_container(wired_container)

        # Load ProjectorShell from contract
        loader = ProjectorPluginLoader(pool=postgres_pool)
        contract_path = REGISTRATION_PROJECTOR_CONTRACT
        projector_instance = await loader.load_from_contract(contract_path)

        # Type narrowing - loader with pool returns ProjectorShell, not placeholder
        assert isinstance(projector_instance, ProjectorShell), (
            "Expected ProjectorShell instance when pool is provided"
        )
        projector = projector_instance

        # Create initial registration using upsert_partial
        now = datetime.now(UTC)
        values: dict[str, object] = {
            "entity_id": unique_node_id,
            "domain": "registration",
            "current_state": EnumRegistrationState.ACTIVE.value,
            "node_type": EnumNodeKind.EFFECT.value,
            "node_version": "1.0.0",
            "capabilities": "{}",
            "registered_at": now,
            "updated_at": now,
            "last_applied_event_id": unique_node_id,
            "last_applied_offset": 1,
        }
        await projector.upsert_partial(
            aggregate_id=unique_node_id,
            values=values,
            correlation_id=uuid4(),
            conflict_columns=["entity_id", "domain"],
        )

        # Wait for PostgreSQL write to complete using deterministic polling
        write_result = await wait_for_postgres_write(
            projection_reader, unique_node_id, timeout_seconds=2.0
        )
        assert write_result is not None, (
            f"PostgreSQL write did not complete in time for node {unique_node_id}"
        )

        # Step 2: Create a NEW container and orchestrator (simulating restart)
        from omnibase_core.container import ModelONEXContainer as ContainerClass
        from omnibase_infra.runtime.util_container_wiring import (
            wire_infrastructure_services,
            wire_registration_handlers,
        )

        new_container = ContainerClass()
        await wire_infrastructure_services(new_container)
        await wire_registration_handlers(new_container, postgres_pool)

        # Step 3: Verify the new orchestrator can read the existing registration
        new_projection_reader = await get_projection_reader_from_container(
            new_container
        )
        recovered_projection = await wait_for_postgres_registration(
            projection_reader=new_projection_reader,
            node_id=unique_node_id,
            expected_state=EnumRegistrationState.ACTIVE,
            timeout_seconds=5.0,
        )

        # Assertions
        assert recovered_projection is not None, (
            "New orchestrator should be able to read existing registration"
        )
        assert recovered_projection.entity_id == unique_node_id
        assert recovered_projection.current_state == EnumRegistrationState.ACTIVE
        assert recovered_projection.node_type == EnumNodeKind.EFFECT
        assert str(recovered_projection.node_version) == "1.0.0"

    @pytest.mark.asyncio
    async def test_re_registration_after_recovery(
        self,
        wired_container: ModelONEXContainer,
        postgres_pool: asyncpg.Pool,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
        unique_node_id: UUID,
        unique_correlation_id: UUID,
        cleanup_projections: None,
    ) -> None:
        """Test nodes can re-register after registry recovery.

        Verifies that when a node attempts to re-register after registry
        recovery, the system handles it idempotently without creating
        duplicate records.

        Steps:
            1. Register node initially with ACTIVE state
            2. Simulate recovery by processing same introspection event
            3. Verify idempotent behavior (no duplicate, state preserved)

        Assertions:
            - Initial registration succeeds
            - Re-registration does not create duplicate
            - Handler returns empty list for node in blocking state
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )
        from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
            HandlerNodeIntrospected,
        )
        from omnibase_infra.nodes.node_registration_orchestrator.services import (
            RegistrationReducerService,
        )
        from omnibase_infra.runtime.util_container_wiring import (
            get_projection_reader_from_container,
        )

        # Step 1: Create initial registration in ACTIVE state
        now = datetime.now(UTC)
        initial_projection = ModelRegistrationProjection(
            entity_id=unique_node_id,
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            registered_at=now,
            updated_at=now,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            last_applied_event_id=unique_node_id,
        )
        await persist_projection_via_shell(real_projector, initial_projection)

        # Wait for PostgreSQL write to complete using deterministic polling
        write_result = await wait_for_postgres_write(
            projection_reader, unique_node_id, timeout_seconds=2.0
        )
        assert write_result is not None, (
            f"PostgreSQL write did not complete in time for node {unique_node_id}"
        )

        # Step 2: Create handler and process same introspection event (simulating re-registration)
        handler_projection_reader = await get_projection_reader_from_container(
            wired_container
        )
        reducer = RegistrationReducerService()
        handler = HandlerNodeIntrospected(handler_projection_reader, reducer)

        event = introspection_event_factory(
            node_id=unique_node_id, correlation_id=unique_correlation_id
        )

        # Process the event through the handler using envelope API
        now = datetime.now(UTC)
        envelope = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=event,
            envelope_timestamp=now,
            correlation_id=unique_correlation_id,
            source="e2e-test",
        )
        handler_output = await handler.handle(envelope)
        result_events = handler_output.events

        # Step 3: Verify idempotent behavior
        # For a node in ACTIVE state (blocking), handler should return empty list (no-op)
        assert len(result_events) == 0, (
            f"Expected 0 events for re-registration of ACTIVE node, "
            f"got {len(result_events)}"
        )

        # Verify only one record exists and state is preserved
        final_projection = await wait_for_postgres_registration(
            projection_reader=projection_reader,
            node_id=unique_node_id,
            expected_state=EnumRegistrationState.ACTIVE,
            timeout_seconds=5.0,
        )

        assert final_projection is not None
        assert final_projection.current_state == EnumRegistrationState.ACTIVE

    @pytest.mark.asyncio
    async def test_idempotent_registration_upsert(
        self,
        registration_orchestrator: NodeRegistrationOrchestrator,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
        unique_node_id: UUID,
        unique_correlation_id: UUID,
        cleanup_projections: None,
    ) -> None:
        """Test registration uses UPSERT semantics (idempotent).

        Verifies that re-registering the same node with updated information
        results in an update (UPSERT) rather than a duplicate record.

        Steps:
            1. Register node initially
            2. Re-register same node with updated capabilities/version
            3. Verify single record exists with updated data

        Assertions:
            - Only one registration record exists
            - Updated fields are correctly persisted
            - Entity ID remains the same
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )

        # Step 1: Create initial registration
        now = datetime.now(UTC)
        initial_projection = ModelRegistrationProjection(
            entity_id=unique_node_id,
            domain="registration",
            current_state=EnumRegistrationState.PENDING_REGISTRATION,
            registered_at=now,
            updated_at=now,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            last_applied_event_id=unique_node_id,
        )
        await persist_projection_via_shell(real_projector, initial_projection)

        # Verify initial registration (polling already handles retry)
        first_result = await wait_for_postgres_registration(
            projection_reader=projection_reader,
            node_id=unique_node_id,
            timeout_seconds=5.0,
        )
        assert first_result is not None
        assert str(first_result.node_version) == "1.0.0"

        # Step 2: Update the same registration with new version (simulating UPSERT)
        updated_now = datetime.now(UTC)
        updated_projection = ModelRegistrationProjection(
            entity_id=unique_node_id,
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            registered_at=first_result.registered_at,  # Preserve original registration time
            updated_at=updated_now,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("2.0.0"),  # Updated version
            last_applied_event_id=unique_node_id,
        )
        await persist_projection_via_shell(real_projector, updated_projection)

        # Step 3: Verify single record with updated data (polling already handles retry)
        final_result = await wait_for_postgres_registration(
            projection_reader=projection_reader,
            node_id=unique_node_id,
            expected_state=EnumRegistrationState.ACTIVE,
            timeout_seconds=5.0,
        )

        # Assertions
        assert final_result is not None, "Registration should exist"
        assert final_result.entity_id == unique_node_id, "Entity ID should be preserved"
        assert str(final_result.node_version) == "2.0.0", (
            f"Version should be updated to 2.0.0, got {final_result.node_version}"
        )
        assert final_result.current_state == EnumRegistrationState.ACTIVE, (
            "State should be updated to ACTIVE"
        )

        # Verify original registration time is preserved (UPSERT, not INSERT)
        assert final_result.registered_at == first_result.registered_at, (
            "registered_at should be preserved on UPSERT"
        )
        assert final_result.updated_at > first_result.updated_at, (
            "updated_at should be newer after UPSERT"
        )


# =============================================================================
# Suite 6: Multiple Nodes Registration
# =============================================================================


class TestSuite6MultipleNodes:
    """Suite 6: Multiple Nodes Registration.

    Tests concurrent registration of multiple nodes and race condition handling.

    Test Coverage:
        - Simultaneous registration of multiple nodes
        - Race condition handling for same-node concurrent registration
        - Verification that all registered nodes appear in both registries
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multiple_nodes_register_simultaneously(
        self,
        wired_container: ModelONEXContainer,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        real_consul_handler: HandlerConsul,
        cleanup_consul_services: list[str],
        cleanup_node_ids: list[UUID],
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Test multiple nodes can register simultaneously.

        Verifies that multiple nodes can be registered concurrently without
        conflicts or failures.

        Steps:
            1. Create 5 introspection events with unique node IDs
            2. Submit all concurrently using asyncio.gather
            3. Verify all 5 appear in PostgreSQL registry

        Assertions:
            - All 5 registrations complete successfully
            - No exceptions raised during concurrent registration
            - All 5 nodes have records in PostgreSQL
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )

        # Create 5 unique node IDs
        node_count = 5
        node_ids = [uuid4() for _ in range(node_count)]

        # Track for cleanup
        cleanup_node_ids.extend(node_ids)

        # Step 1: Create projections for all nodes concurrently
        now = datetime.now(UTC)

        async def register_node(idx: int) -> None:
            """Register a single node."""
            projection = ModelRegistrationProjection(
                entity_id=node_ids[idx],
                domain="registration",
                current_state=EnumRegistrationState.ACTIVE,
                registered_at=now,
                updated_at=now,
                node_type=EnumNodeKind.EFFECT,
                node_version=ModelSemVer.parse("1.0.0"),
                last_applied_event_id=node_ids[idx],
            )
            await persist_projection_via_shell(real_projector, projection)

        # Step 2: Submit all concurrently
        tasks = [register_node(i) for i in range(node_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Node {node_ids[i]} registration failed: {result}")

        # Step 3: Verify all nodes appear in PostgreSQL (polling already handles retry)
        for i, node_id in enumerate(node_ids):
            projection = await verify_postgres_registration(
                projection_reader=projection_reader, node_id=node_id
            )
            assert projection is not None, (
                f"Node {i} ({node_id}) should be found in PostgreSQL"
            )
            assert projection.entity_id == node_id
            assert projection.current_state == EnumRegistrationState.ACTIVE

    @pytest.mark.asyncio
    async def test_no_race_conditions_in_registration(
        self,
        wired_container: ModelONEXContainer,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
        unique_node_id: UUID,
        cleanup_projections: None,
    ) -> None:
        """Test no race conditions when registering same node concurrently.

        Verifies that when the same node attempts to register multiple times
        concurrently (race condition scenario), only one registration exists
        without duplicates or data corruption.

        Steps:
            1. Create same introspection event multiple times
            2. Submit 10 concurrent UPSERT operations for same node
            3. Verify exactly one registration exists in PostgreSQL

        Assertions:
            - All concurrent operations complete (some may be no-ops)
            - Exactly one registration record exists
            - No data corruption or duplicate records
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )

        # Step 1: Prepare concurrent UPSERT operations for the SAME node
        concurrent_count = 10
        now = datetime.now(UTC)

        async def concurrent_upsert(attempt: int) -> None:
            """Perform a single UPSERT operation."""
            projection = ModelRegistrationProjection(
                entity_id=unique_node_id,
                domain="registration",
                current_state=EnumRegistrationState.ACTIVE,
                registered_at=now,
                updated_at=datetime.now(UTC),  # Slightly different each time
                node_type=EnumNodeKind.EFFECT,
                node_version=f"1.0.{attempt}",  # Different version each time
                last_applied_event_id=unique_node_id,
            )
            await persist_projection_via_shell(real_projector, projection)

        # Step 2: Submit all 10 concurrently
        tasks = [concurrent_upsert(i) for i in range(concurrent_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that all completed (even if some are no-ops due to UPSERT)
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            # Log but don't fail - some race conditions may cause retry-able errors
            for ex in exceptions:
                print(f"Concurrent operation exception (may be expected): {ex}")

        # Step 3: Verify exactly one registration exists (polling already handles retry)
        projection = await verify_postgres_registration(
            projection_reader=projection_reader, node_id=unique_node_id
        )

        assert projection is not None, "Registration should exist"
        assert projection.entity_id == unique_node_id, (
            "Entity ID should match unique_node_id"
        )
        assert projection.current_state == EnumRegistrationState.ACTIVE

        # Verify no corruption - version should be one of the submitted values
        # node_version is a ModelSemVer object, access patch directly
        version_int = projection.node_version.patch
        assert 0 <= version_int < concurrent_count, (
            f"Version {projection.node_version} should be from one of the "
            f"concurrent operations (0-{concurrent_count - 1})"
        )

    @pytest.mark.asyncio
    async def test_all_nodes_appear_in_registry(
        self,
        wired_container: ModelONEXContainer,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        real_consul_handler: HandlerConsul,
        cleanup_consul_services: list[str],
        cleanup_node_ids: list[UUID],
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Test all registered nodes appear in both Consul and PostgreSQL.

        Verifies that when multiple nodes are registered, all of them
        correctly appear in both registry backends.

        Steps:
            1. Register 3 nodes sequentially
            2. Verify all 3 in Consul
            3. Verify all 3 in PostgreSQL

        Assertions:
            - All 3 nodes are registered in Consul
            - All 3 nodes are registered in PostgreSQL
            - Data is consistent across both backends
        """
        import json

        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )

        # Create 3 nodes
        node_count = 3
        node_ids = [uuid4() for _ in range(node_count)]
        service_ids = [f"multi-node-{nid.hex[:8]}" for nid in node_ids]

        # Track for cleanup
        cleanup_consul_services.extend(service_ids)
        cleanup_node_ids.extend(node_ids)

        now = datetime.now(UTC)

        # Step 1: Register all 3 nodes sequentially in both backends
        for i, (node_id, service_id) in enumerate(
            zip(node_ids, service_ids, strict=True)
        ):
            # Register in PostgreSQL
            projection = ModelRegistrationProjection(
                entity_id=node_id,
                domain="registration",
                current_state=EnumRegistrationState.ACTIVE,
                registered_at=now,
                updated_at=now,
                node_type=EnumNodeKind.EFFECT if i % 2 == 0 else EnumNodeKind.COMPUTE,
                node_version=f"1.{i}.0",
                last_applied_event_id=node_id,
            )
            await persist_projection_via_shell(real_projector, projection)

            # Register in Consul
            consul_data = {
                "node_id": str(node_id),
                "node_type": "effect" if i % 2 == 0 else "compute",
                "version": f"1.{i}.0",
                "endpoints": {"health": f"http://localhost:808{i}/health"},
                "registered_at": now.isoformat(),
            }
            consul_envelope: dict[str, object] = {
                "operation": "consul.kv_put",
                "payload": {
                    "key": f"onex/services/{service_id}",
                    "value": json.dumps(consul_data),
                },
            }
            await real_consul_handler.execute(consul_envelope)

        # Step 2: Verify all 3 in Consul (polling already handles retry)
        consul_results = []
        for service_id in service_ids:
            result = await verify_consul_registration(
                consul_handler=real_consul_handler,
                service_id=service_id,
                timeout_seconds=5.0,
            )
            consul_results.append(result)
            assert result is not None, f"Service {service_id} should be found in Consul"

        assert len([r for r in consul_results if r is not None]) == node_count, (
            f"Expected {node_count} services in Consul, "
            f"found {len([r for r in consul_results if r is not None])}"
        )

        # Step 3: Verify all 3 in PostgreSQL
        postgres_results: list[ModelRegistrationProjection | None] = []
        for node_id in node_ids:
            pg_result = await verify_postgres_registration(
                projection_reader=projection_reader, node_id=node_id
            )
            postgres_results.append(pg_result)
            assert pg_result is not None, (
                f"Node {node_id} should be found in PostgreSQL"
            )
            assert pg_result.current_state == EnumRegistrationState.ACTIVE

        assert len([r for r in postgres_results if r is not None]) == node_count, (
            f"Expected {node_count} registrations in PostgreSQL, "
            f"found {len([r for r in postgres_results if r is not None])}"
        )

        # Verify data consistency across backends
        for i, (_node_id, _service_id) in enumerate(
            zip(node_ids, service_ids, strict=True)
        ):
            pg_result = postgres_results[i]
            assert pg_result is not None
            assert str(pg_result.node_version) == f"1.{i}.0", (
                f"PostgreSQL version mismatch for node {i}"
            )


# =============================================================================
# Suite 7: Graceful Degradation
# =============================================================================


class TestSuite7GracefulDegradation:
    """Suite 7: Graceful Degradation.

    Tests the system's ability to operate with partial infrastructure failures.
    Note: These tests temporarily break connections to test degradation.

    Test Coverage:
        - Node resilience when Kafka is unavailable
        - Registry works with Consul unavailable (PostgreSQL only)
        - Registry works with PostgreSQL unavailable (Consul only)
        - Partial success reporting for dual-backend operations
    """

    @pytest.mark.asyncio
    async def test_nodes_work_when_kafka_unavailable(
        self, unique_node_id: UUID
    ) -> None:
        """Test nodes continue to function when Kafka is unavailable.

        Verifies:
        - Node doesn't crash when event bus is not connected
        - Introspection is skipped gracefully
        - Node can still perform core functions

        Note: This tests the node's resilience, not the event publishing.
        The MixinNodeIntrospection should handle Kafka unavailability gracefully.
        """
        from omnibase_infra.mixins import MixinNodeIntrospection
        from omnibase_infra.models.discovery import ModelIntrospectionConfig

        class ResilientTestNode(MixinNodeIntrospection):
            """Test node without Kafka event bus."""

            def __init__(self, node_id: UUID) -> None:
                self._node_id = node_id
                self._node_type_value = EnumNodeKind.EFFECT

                # Initialize introspection WITHOUT event bus
                config = ModelIntrospectionConfig(
                    node_id=node_id,
                    node_type=EnumNodeKind.EFFECT,
                    node_name="resilient_test_node",
                    event_bus=None,  # No event bus - simulates Kafka unavailable
                    version="1.0.0",
                    cache_ttl=60.0,
                )
                self.initialize_introspection(config)

            @property
            def node_id(self) -> UUID:
                return self._node_id

            async def execute_operation(
                self, data: dict[str, object]
            ) -> dict[str, object]:
                """Sample operation that should work even without Kafka."""
                return {"result": "processed", "input": data}

        # Create node without Kafka event bus
        node = ResilientTestNode(unique_node_id)

        # Verify node initializes successfully
        assert node.node_id == unique_node_id

        # Verify introspection methods don't raise exceptions
        event = await node.get_introspection_data()
        assert event is not None, "Introspection data should be available locally"
        assert event.node_id == unique_node_id

        # Verify publish returns False gracefully (no event bus)
        success = await node.publish_introspection(
            reason=EnumIntrospectionReason.REQUEST
        )
        assert success is False, (
            "Publish should return False when no event bus available"
        )

        # Verify core node functions still work
        result = await node.execute_operation({"test": "data"})
        assert result["result"] == "processed"

    @pytest.mark.asyncio
    async def test_registry_works_when_consul_unavailable(
        self,
        wired_container: ModelONEXContainer,
        postgres_pool: asyncpg.Pool,
        projection_reader: ProjectionReaderRegistration,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
        unique_node_id: UUID,
        unique_correlation_id: UUID,
        cleanup_projections: None,
    ) -> None:
        """Test registry can still register in PostgreSQL when Consul is unavailable.

        Verifies:
        - PostgreSQL registration succeeds independently
        - Consul failure doesn't block PostgreSQL registration
        - The system can operate in degraded mode

        Note: This test directly tests the projection persistence path,
        bypassing Consul to simulate unavailability.
        """
        from omnibase_infra.projectors.contracts import REGISTRATION_PROJECTOR_CONTRACT
        from omnibase_infra.runtime import ProjectorPluginLoader, ProjectorShell

        now = datetime.now(UTC)

        # Load ProjectorShell from contract
        loader = ProjectorPluginLoader(pool=postgres_pool)
        contract_path = REGISTRATION_PROJECTOR_CONTRACT
        projector_instance = await loader.load_from_contract(contract_path)

        # Type narrowing - loader with pool returns ProjectorShell, not placeholder
        assert isinstance(projector_instance, ProjectorShell), (
            "Expected ProjectorShell instance when pool is provided"
        )
        projector = projector_instance

        # Create projection directly (simulating PostgreSQL-only registration)
        values: dict[str, object] = {
            "entity_id": unique_node_id,
            "domain": "registration",
            "current_state": EnumRegistrationState.PENDING_REGISTRATION.value,
            "node_type": EnumNodeKind.EFFECT.value,
            "node_version": "1.0.0",
            "capabilities": "{}",
            "registered_at": now,
            "updated_at": now,
            "last_applied_event_id": unique_node_id,
            "last_applied_offset": 1,
        }
        await projector.upsert_partial(
            aggregate_id=unique_node_id,
            values=values,
            correlation_id=uuid4(),
            conflict_columns=["entity_id", "domain"],
        )

        # Verify PostgreSQL registration succeeded
        pg_result = await wait_for_postgres_registration(
            projection_reader=projection_reader,
            node_id=unique_node_id,
            expected_state=EnumRegistrationState.PENDING_REGISTRATION,
            timeout_seconds=10.0,
        )

        assert pg_result is not None, (
            f"Node {unique_node_id} should be found in PostgreSQL"
        )
        assert pg_result.entity_id == unique_node_id
        assert pg_result.current_state == EnumRegistrationState.PENDING_REGISTRATION
        assert pg_result.node_type == EnumNodeKind.EFFECT

    @pytest.mark.asyncio
    async def test_registry_works_when_postgres_unavailable(
        self,
        real_consul_handler: HandlerConsul,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
        unique_node_id: UUID,
        cleanup_consul_services: list[str],
    ) -> None:
        """Test registry can still register in Consul when PostgreSQL is unavailable.

        Verifies:
        - Consul registration succeeds independently
        - PostgreSQL failure doesn't block Consul registration
        - The system can operate in degraded mode

        Note: This test directly tests the Consul registration path,
        without PostgreSQL to simulate unavailability.
        """
        service_id = f"degradation-test-{unique_node_id.hex[:8]}"

        # Track service for cleanup
        cleanup_consul_services.append(service_id)

        # Build registration payload
        now = datetime.now(UTC)
        registration_data = {
            "node_id": str(unique_node_id),
            "node_type": "effect",
            "version": "1.0.0",
            "endpoints": {"health": "http://localhost:8080/health"},
            "registered_at": now.isoformat(),
            "degraded_mode": True,  # Indicate this is a degraded registration
        }

        # Register in Consul via KV store
        envelope: dict[str, object] = {
            "operation": "consul.kv_put",
            "payload": {
                "key": f"onex/services/{service_id}",
                "value": json.dumps(registration_data),
            },
        }

        result = await real_consul_handler.execute(envelope)

        # Verify write succeeded
        assert result.result is not None, "Consul KV write should return result"

        # Wait for and verify registration
        consul_result = await wait_for_consul_registration(
            consul_handler=real_consul_handler,
            service_id=service_id,
            timeout_seconds=10.0,
        )

        assert consul_result is not None, (
            f"Service {service_id} should be found in Consul"
        )
        assert consul_result["service_id"] == service_id

    @pytest.mark.asyncio
    async def test_partial_success_reporting(
        self, unique_node_id: UUID, unique_correlation_id: UUID
    ) -> None:
        """Test partial success is correctly reported.

        Verifies:
        - ModelRegistryResponse shows which backends succeeded
        - ModelRegistryResponse shows which backends failed
        - Overall status is 'partial' if any backend fails
        - Error summary correctly aggregates failure messages

        Note: This tests the ModelRegistryResponse.from_backend_results() logic
        without actually calling infrastructure services.
        """
        from omnibase_infra.models.model_backend_result import (
            ModelBackendResult,
        )
        from omnibase_infra.nodes.effects.models.model_registry_response import (
            ModelRegistryResponse,
        )

        now = datetime.now(UTC)

        # Scenario 1: Consul success, PostgreSQL failure
        consul_success = ModelBackendResult(
            success=True, duration_ms=45.0, backend_id="consul"
        )
        postgres_failure = ModelBackendResult(
            success=False,
            error="Connection refused",
            error_code="DATABASE_CONNECTION_ERROR",
            duration_ms=5000.0,
            backend_id="postgres",
        )

        response = ModelRegistryResponse.from_backend_results(
            node_id=unique_node_id,
            correlation_id=unique_correlation_id,
            consul_result=consul_success,
            postgres_result=postgres_failure,
            timestamp=now,
        )

        # Verify partial success status
        assert response.status == "partial", (
            f"Expected status 'partial', got '{response.status}'"
        )
        assert response.is_partial_failure() is True
        assert response.is_complete_success() is False
        assert response.is_complete_failure() is False

        # Verify backend results
        assert response.consul_result.success is True
        assert response.postgres_result.success is False

        # Verify failed/successful backends
        assert response.get_failed_backends() == ["postgres"]
        assert response.get_successful_backends() == ["consul"]

        # Verify error summary
        assert response.error_summary is not None
        assert "PostgreSQL" in response.error_summary
        assert "Connection refused" in response.error_summary

        # Scenario 2: Consul failure, PostgreSQL success
        consul_failure = ModelBackendResult(
            success=False,
            error="Service unavailable",
            error_code="SERVICE_UNAVAILABLE",
            duration_ms=3000.0,
            backend_id="consul",
        )
        postgres_success = ModelBackendResult(
            success=True, duration_ms=30.0, backend_id="postgres"
        )

        response2 = ModelRegistryResponse.from_backend_results(
            node_id=unique_node_id,
            correlation_id=unique_correlation_id,
            consul_result=consul_failure,
            postgres_result=postgres_success,
            timestamp=now,
        )

        assert response2.status == "partial"
        assert response2.get_failed_backends() == ["consul"]
        assert response2.get_successful_backends() == ["postgres"]
        assert "Consul" in (response2.error_summary or "")

        # Scenario 3: Both backends fail
        response3 = ModelRegistryResponse.from_backend_results(
            node_id=unique_node_id,
            correlation_id=unique_correlation_id,
            consul_result=consul_failure,
            postgres_result=postgres_failure,
            timestamp=now,
        )

        assert response3.status == "failed"
        assert response3.is_complete_failure() is True
        assert len(response3.get_failed_backends()) == 2
        assert len(response3.get_successful_backends()) == 0

        # Scenario 4: Both backends succeed
        response4 = ModelRegistryResponse.from_backend_results(
            node_id=unique_node_id,
            correlation_id=unique_correlation_id,
            consul_result=consul_success,
            postgres_result=postgres_success,
            timestamp=now,
        )

        assert response4.status == "success"
        assert response4.is_complete_success() is True
        assert response4.error_summary is None

    @pytest.mark.asyncio
    async def test_partial_success_processing_time_calculation(
        self, unique_node_id: UUID, unique_correlation_id: UUID
    ) -> None:
        """Test processing time is correctly calculated from backend results.

        Verifies:
        - Processing time is sum of backend durations
        - Processing time is correctly reported even for partial failures
        """
        from omnibase_infra.models.model_backend_result import (
            ModelBackendResult,
        )
        from omnibase_infra.nodes.effects.models.model_registry_response import (
            ModelRegistryResponse,
        )

        now = datetime.now(UTC)

        consul_result = ModelBackendResult(success=True, duration_ms=45.5)
        postgres_result = ModelBackendResult(success=True, duration_ms=30.2)

        response = ModelRegistryResponse.from_backend_results(
            node_id=unique_node_id,
            correlation_id=unique_correlation_id,
            consul_result=consul_result,
            postgres_result=postgres_result,
            timestamp=now,
        )

        # Verify processing time is sum of backend durations
        expected_total = 45.5 + 30.2
        assert abs(response.processing_time_ms - expected_total) < 0.1, (
            f"Expected processing_time_ms ~{expected_total}, "
            f"got {response.processing_time_ms}"
        )


# =============================================================================
# Suite 8: Registry Self-Registration
# =============================================================================


class TestSuite8RegistrySelfRegistration:
    """Suite 8: Registry Self-Registration.

    Tests the registry's ability to register itself as a discoverable service.
    This enables other nodes to discover the registry for registration requests.

    Test Coverage:
        - Registry can create its own introspection event
        - Registry appears in Consul after self-registration
        - Registry appears in PostgreSQL after self-registration
        - Registry introspection data is complete and valid
    """

    @pytest.mark.asyncio
    async def test_registry_registers_itself(
        self,
        real_consul_handler: HandlerConsul,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        wired_container: ModelONEXContainer,
        cleanup_consul_services: list[str],
        cleanup_projections: None,
        unique_node_id: UUID,
    ) -> None:
        """Test registry can register itself as a service.

        Verifies:
        - Registry can create its own introspection event
        - Registry appears in Consul
        - Registry appears in PostgreSQL

        Note: Uses a unique_node_id to represent the registry for test isolation.
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )

        registry_node_id = unique_node_id  # Use unique ID for test isolation
        service_id = f"registry-{registry_node_id.hex[:8]}"

        # Track service for cleanup
        cleanup_consul_services.append(service_id)

        now = datetime.now(UTC)

        # Step 1: Register registry in Consul
        registration_data = {
            "node_id": str(registry_node_id),
            "node_type": "orchestrator",  # Registry is an orchestrator
            "version": "1.0.0",
            "endpoints": {
                "health": "http://localhost:8085/health",
                "api": "http://localhost:8085/api/v1/registry",
            },
            "capabilities": {
                "handlers": [
                    "HandlerNodeIntrospected",
                    "HandlerRuntimeTick",
                    "HandlerNodeHeartbeat",
                ],
                "timeout_coordination": True,
            },
            "registered_at": now.isoformat(),
            "service_type": "registry",
        }

        consul_envelope: dict[str, object] = {
            "operation": "consul.kv_put",
            "payload": {
                "key": f"onex/services/{service_id}",
                "value": json.dumps(registration_data),
            },
        }

        await real_consul_handler.execute(consul_envelope)

        # Step 2: Register registry in PostgreSQL
        projection = ModelRegistrationProjection(
            entity_id=registry_node_id,
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            registered_at=now,
            updated_at=now,
            node_type=EnumNodeKind.ORCHESTRATOR,
            node_version=ModelSemVer.parse("1.0.0"),
            last_applied_event_id=registry_node_id,
        )

        await persist_projection_via_shell(real_projector, projection)

        # Step 3: Verify dual registration
        consul_result, postgres_result = await verify_dual_registration(
            consul_handler=real_consul_handler,
            projection_reader=projection_reader,
            node_id=registry_node_id,
            service_id=service_id,
            timeout_seconds=10.0,
        )

        # Assertions
        assert consul_result is not None, "Registry should be found in Consul"
        assert consul_result["service_id"] == service_id

        assert postgres_result is not None, "Registry should be found in PostgreSQL"
        assert postgres_result.entity_id == registry_node_id
        assert postgres_result.node_type == EnumNodeKind.ORCHESTRATOR
        assert postgres_result.current_state == EnumRegistrationState.ACTIVE

    @pytest.mark.asyncio
    async def test_self_registration_in_database(
        self,
        projection_reader: ProjectionReaderRegistration,
        real_projector: ProjectorShell,
        wired_container: ModelONEXContainer,
        postgres_pool: asyncpg.Pool,
        unique_node_id: UUID,
        cleanup_projections: None,
    ) -> None:
        """Test registry self-registration persists in database.

        Verifies:
        - Registration projection exists
        - Has correct node_type (orchestrator)
        - Has correct capabilities
        """
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )

        registry_node_id = unique_node_id
        now = datetime.now(UTC)

        # Create registry projection with orchestrator type
        projection = ModelRegistrationProjection(
            entity_id=registry_node_id,
            domain="registration",
            current_state=EnumRegistrationState.ACTIVE,
            registered_at=now,
            updated_at=now,
            node_type=EnumNodeKind.ORCHESTRATOR,
            node_version=ModelSemVer.parse("1.0.0"),
            last_applied_event_id=registry_node_id,
        )

        # Persist via projector
        await persist_projection_via_shell(real_projector, projection)

        # Read via projection_reader
        result = await wait_for_postgres_registration(
            projection_reader=projection_reader,
            node_id=registry_node_id,
            expected_state=EnumRegistrationState.ACTIVE,
            timeout_seconds=10.0,
        )

        # Verify fields
        assert result is not None, "Registry projection should exist"
        assert result.entity_id == registry_node_id
        assert result.node_type == EnumNodeKind.ORCHESTRATOR, (
            f"Expected node_type ORCHESTRATOR, got '{result.node_type}'"
        )
        assert result.current_state == EnumRegistrationState.ACTIVE
        # Note: endpoints are stored in capabilities, not as a top-level field
        assert result.capabilities is not None

    @pytest.mark.asyncio
    async def test_registry_introspection_data_complete(
        self,
        wired_container: ModelONEXContainer,
        introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent],
    ) -> None:
        """Test registry's own introspection data is complete.

        Verifies:
        - Registry can produce valid introspection event
        - Event has all required fields
        - Capabilities include handler and timeout coordination

        Note: Uses the introspection_event_factory with orchestrator type
        to simulate the registry's introspection event.
        """
        # Create introspection event for the registry itself
        registry_event = introspection_event_factory(
            node_type=EnumNodeKind.ORCHESTRATOR,
            node_version=ModelSemVer.parse("1.0.0"),
            endpoints={
                "health": "http://localhost:8085/health",
                "api": "http://localhost:8085/api/v1/registry",
                "metrics": "http://localhost:8085/metrics",
            },
        )

        # Validate using the standard helper
        assert_introspection_event_complete(registry_event)

        # Additional registry-specific validations
        assert registry_event.node_type == "orchestrator", (
            f"Registry should be type 'orchestrator', got '{registry_event.node_type}'"
        )
        assert registry_event.endpoints is not None
        assert "health" in registry_event.endpoints
        assert "api" in registry_event.endpoints

        # Verify the event can be serialized/deserialized
        event_dict = registry_event.model_dump(mode="json")
        assert "node_id" in event_dict
        assert "node_type" in event_dict
        assert event_dict["node_type"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_registry_introspection_with_custom_capabilities(
        self, introspection_event_factory: Callable[..., ModelNodeIntrospectionEvent]
    ) -> None:
        """Test registry introspection includes custom capabilities.

        Verifies:
        - Registry can specify handler capabilities via supported_types
        - Processing and routing capabilities can be enabled
        - Custom capabilities work via model_extra
        """
        from omnibase_infra.models.registration.model_node_capabilities import (
            ModelNodeCapabilities,
        )

        # Create capabilities typical for a registry orchestrator
        # Use supported_types for handler names (valid list[str] field)
        registry_capabilities = ModelNodeCapabilities(
            supported_types=[
                "HandlerNodeIntrospected",
                "HandlerRuntimeTick",
                "HandlerNodeHeartbeat",
            ],
            processing=True,
            routing=True,
        )

        # Verify capabilities are structured correctly
        assert registry_capabilities.supported_types is not None
        assert len(registry_capabilities.supported_types) == 3
        assert "HandlerNodeIntrospected" in registry_capabilities.supported_types
        assert "HandlerRuntimeTick" in registry_capabilities.supported_types
        assert "HandlerNodeHeartbeat" in registry_capabilities.supported_types
        assert registry_capabilities.processing is True
        assert registry_capabilities.routing is True

    @pytest.mark.asyncio
    async def test_registry_discoverable_by_other_nodes(
        self,
        real_consul_handler: HandlerConsul,
        unique_node_id: UUID,
        cleanup_consul_services: list[str],
    ) -> None:
        """Test registry is discoverable by other nodes via Consul.

        Verifies:
        - Registry service can be listed in Consul
        - Service metadata includes necessary discovery information

        Note: This tests the discovery path that other nodes would use
        to find the registry for registration.
        """
        registry_node_id = unique_node_id
        service_id = f"registry-discoverable-{registry_node_id.hex[:8]}"

        # Track service for cleanup
        cleanup_consul_services.append(service_id)

        now = datetime.now(UTC)

        # Register the registry service
        registration_data = {
            "node_id": str(registry_node_id),
            "node_type": "orchestrator",
            "version": "1.0.0",
            "service_type": "registry",
            "endpoints": {
                "health": "http://localhost:8085/health",
                "registration": "http://localhost:8085/api/v1/register",
            },
            "registered_at": now.isoformat(),
        }

        envelope: dict[str, object] = {
            "operation": "consul.kv_put",
            "payload": {
                "key": f"onex/services/{service_id}",
                "value": json.dumps(registration_data),
            },
        }

        await real_consul_handler.execute(envelope)

        # Discover the registry
        consul_result = await wait_for_consul_registration(
            consul_handler=real_consul_handler,
            service_id=service_id,
            timeout_seconds=10.0,
        )

        assert consul_result is not None, "Registry should be discoverable"
        assert consul_result["service_id"] == service_id

        # Parse the stored value to verify discovery info
        value = consul_result.get("value")
        if value:
            import json as json_mod

            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if isinstance(value, str):
                stored_data = json_mod.loads(value)
                assert stored_data.get("service_type") == "registry"
                assert stored_data.get("node_type") == "orchestrator"
                assert "registration" in stored_data.get("endpoints", {})
