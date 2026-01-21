# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for projector notification integration.

Tests the full integration flow between ProjectorShell and
TransitionNotificationPublisher using real (or realistic mock) dependencies.

This verifies:
    - Complete event -> projection -> notification flow
    - State transition tracking (from_state/to_state)
    - Correlation ID and causation ID propagation
    - Best-effort notification semantics (projection succeeds even if notification fails)
    - Configuration variations (enabled/disabled, custom columns)

Architecture tested:
    Event -> ProjectorShell.project() -> Database Commit
                   |                          |
          _fetch_current_state()    _publish_transition_notification()
                   |                          |
              from_state               Event Bus (mock)

Related Tickets:
    - OMN-1139: Integrate TransitionNotificationPublisher with ProjectorShell
    - OMN-1169: ProjectorShell contract-driven projections
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.notifications import ModelStateTransitionNotification
from omnibase_core.models.projectors import (
    ModelProjectorBehavior,
    ModelProjectorColumn,
    ModelProjectorContract,
    ModelProjectorSchema,
)
from omnibase_core.protocols.notifications import (
    ProtocolTransitionNotificationPublisher,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime import ProjectorShell, TransitionNotificationPublisher
from omnibase_infra.runtime.models import ModelProjectorNotificationConfig

if TYPE_CHECKING:
    import asyncpg


# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.asyncio,
]


# =============================================================================
# Mock Event Models
# =============================================================================


class MockRegistrationEvent(BaseModel):
    """Mock registration event payload for testing."""

    entity_id: UUID
    current_state: str
    node_type: str = "effect"
    node_version: str = "1.0.0"
    version: int = 1
    event_type: str = "registration.state.changed.v1"


class MockHeartbeatEvent(BaseModel):
    """Mock heartbeat event payload for testing."""

    entity_id: UUID
    heartbeat_count: int = 1
    event_type: str = "registration.heartbeat.v1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def integration_contract() -> ModelProjectorContract:
    """Create a projector contract for integration testing.

    This contract mimics the registration projection schema with state
    tracking columns needed for notification publishing.
    """
    columns = [
        ModelProjectorColumn(
            name="entity_id",
            type="UUID",
            source="payload.entity_id",
        ),
        ModelProjectorColumn(
            name="current_state",
            type="TEXT",
            source="payload.current_state",
        ),
        ModelProjectorColumn(
            name="node_type",
            type="TEXT",
            source="payload.node_type",
        ),
        ModelProjectorColumn(
            name="node_version",
            type="TEXT",
            source="payload.node_version",
        ),
        ModelProjectorColumn(
            name="version",
            type="INTEGER",
            source="payload.version",
            default="0",
        ),
    ]

    schema = ModelProjectorSchema(
        table="notification_test_entities",
        primary_key="entity_id",
        columns=columns,
    )

    behavior = ModelProjectorBehavior(
        mode="upsert",
        upsert_key="entity_id",
    )

    return ModelProjectorContract(
        projector_kind="materialized_view",
        projector_id="notification-test-projector",
        name="Notification Test Projector",
        version="1.0.0",
        aggregate_type="registration",
        consumed_events=["registration.state.changed.v1"],
        projection_schema=schema,
        behavior=behavior,
    )


@pytest.fixture
def notification_config() -> ModelProjectorNotificationConfig:
    """Create notification config for integration testing."""
    return ModelProjectorNotificationConfig(
        state_column="current_state",
        aggregate_id_column="entity_id",
        version_column="version",
        enabled=True,
    )


@pytest.fixture
async def mock_event_bus() -> AsyncMock:
    """Create mock event bus for integration testing.

    The mock captures all published envelopes for assertion.
    """
    bus = AsyncMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()
    bus.publish = AsyncMock()
    # Store published envelopes for verification (no type annotation on mock attribute)
    published: list[tuple[object, str]] = []
    bus.published_envelopes = published

    # Capture published envelopes for verification
    async def capture_envelope(envelope: object, topic: str) -> None:
        published.append((envelope, topic))

    bus.publish_envelope.side_effect = capture_envelope
    return bus


@pytest.fixture
async def notification_publisher(
    mock_event_bus: AsyncMock,
) -> TransitionNotificationPublisher:
    """Create real publisher with mock event bus."""
    return TransitionNotificationPublisher(
        event_bus=mock_event_bus,
        topic="test.fsm.state.transitions.v1",
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=30.0,
    )


@pytest.fixture
async def notification_test_pool(
    pg_pool: asyncpg.Pool,
) -> asyncpg.Pool:
    """Create test table for notification integration tests.

    Uses the existing pg_pool from conftest.py and adds a test table.
    """
    # Create test table for notification tests
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS notification_test_entities (
            entity_id UUID PRIMARY KEY,
            current_state TEXT NOT NULL,
            node_type TEXT NOT NULL DEFAULT 'effect',
            node_version TEXT NOT NULL DEFAULT '1.0.0',
            version INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """

    async with pg_pool.acquire() as conn:
        await conn.execute(create_table_sql)

    yield pg_pool

    # Cleanup
    async with pg_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS notification_test_entities CASCADE")


def make_registration_envelope(
    entity_id: UUID | None = None,
    current_state: str = "pending_registration",
    correlation_id: UUID | None = None,
    version: int = 1,
) -> ModelEventEnvelope[MockRegistrationEvent]:
    """Create a registration event envelope for testing."""
    if entity_id is None:
        entity_id = uuid4()
    if correlation_id is None:
        correlation_id = uuid4()

    payload = MockRegistrationEvent(
        entity_id=entity_id,
        current_state=current_state,
        version=version,
    )

    return ModelEventEnvelope(
        payload=payload,
        correlation_id=correlation_id,
        source_tool="notification-test",
        metadata=ModelEnvelopeMetadata(
            tags={"event_type": "registration.state.changed.v1"},
        ),
    )


# =============================================================================
# Integration Test Classes
# =============================================================================


class TestProjectorNotificationIntegration:
    """Integration tests for projector + notification publisher flow."""

    async def test_full_projection_publishes_notification(
        self,
        notification_test_pool: asyncpg.Pool,
        notification_publisher: TransitionNotificationPublisher,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Complete projection flow publishes state transition notification."""
        # Setup projector with publisher
        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=notification_publisher,
            notification_config=notification_config,
        )

        # Create event
        entity_id = uuid4()
        correlation_id = uuid4()
        envelope = make_registration_envelope(
            entity_id=entity_id,
            current_state="active",
            correlation_id=correlation_id,
            version=1,
        )

        # Execute projection
        result = await projector.project(envelope, correlation_id)

        # Verify projection succeeded
        assert result.success is True
        assert result.rows_affected == 1

        # Verify notification was published
        assert len(mock_event_bus.published_envelopes) == 1
        _published_envelope, topic = mock_event_bus.published_envelopes[0]
        assert topic == "test.fsm.state.transitions.v1"

    async def test_notification_contains_correct_aggregate_info(
        self,
        notification_test_pool: asyncpg.Pool,
        notification_publisher: TransitionNotificationPublisher,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Published notification has correct aggregate_type and aggregate_id."""
        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=notification_publisher,
            notification_config=notification_config,
        )

        entity_id = uuid4()
        correlation_id = uuid4()
        envelope = make_registration_envelope(
            entity_id=entity_id,
            current_state="active",
            correlation_id=correlation_id,
        )

        await projector.project(envelope, correlation_id)

        # Extract notification from published envelope
        assert len(mock_event_bus.published_envelopes) == 1
        published_envelope, _ = mock_event_bus.published_envelopes[0]

        # The envelope payload contains the notification data
        payload = published_envelope.payload
        assert payload["aggregate_type"] == "registration"
        # UUID fields may be UUID objects or strings depending on serialization
        assert str(payload["aggregate_id"]) == str(entity_id)

    async def test_notification_tracks_state_transition(
        self,
        notification_test_pool: asyncpg.Pool,
        notification_publisher: TransitionNotificationPublisher,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Notification correctly tracks from_state and to_state."""
        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=notification_publisher,
            notification_config=notification_config,
        )

        entity_id = uuid4()
        correlation_id = uuid4()

        # First transition: empty -> pending_registration
        envelope1 = make_registration_envelope(
            entity_id=entity_id,
            current_state="pending_registration",
            correlation_id=correlation_id,
            version=1,
        )
        await projector.project(envelope1, correlation_id)

        # Verify first notification (new entity, from_state is empty)
        assert len(mock_event_bus.published_envelopes) == 1
        payload1 = mock_event_bus.published_envelopes[0][0].payload
        assert payload1["from_state"] == ""  # New entity
        assert payload1["to_state"] == "pending_registration"

        # Second transition: pending_registration -> active
        mock_event_bus.published_envelopes.clear()
        envelope2 = make_registration_envelope(
            entity_id=entity_id,
            current_state="active",
            correlation_id=correlation_id,
            version=2,
        )
        await projector.project(envelope2, correlation_id)

        # Verify second notification (existing entity, has from_state)
        assert len(mock_event_bus.published_envelopes) == 1
        payload2 = mock_event_bus.published_envelopes[0][0].payload
        assert payload2["from_state"] == "pending_registration"
        assert payload2["to_state"] == "active"

    async def test_notification_has_correct_correlation_ids(
        self,
        notification_test_pool: asyncpg.Pool,
        notification_publisher: TransitionNotificationPublisher,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """correlation_id and causation_id are correctly propagated."""
        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=notification_publisher,
            notification_config=notification_config,
        )

        entity_id = uuid4()
        correlation_id = uuid4()
        envelope = make_registration_envelope(
            entity_id=entity_id,
            current_state="active",
            correlation_id=correlation_id,
        )

        await projector.project(envelope, correlation_id)

        # Verify IDs
        assert len(mock_event_bus.published_envelopes) == 1
        payload = mock_event_bus.published_envelopes[0][0].payload
        # UUID fields may be UUID objects or strings depending on serialization
        assert str(payload["correlation_id"]) == str(correlation_id)
        # causation_id should be the envelope_id of the triggering event
        assert str(payload["causation_id"]) == str(envelope.envelope_id)

    async def test_no_notification_on_zero_rows_affected(
        self,
        notification_test_pool: asyncpg.Pool,
        notification_publisher: TransitionNotificationPublisher,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """No notification published when projection affects zero rows.

        This tests the scenario where an event type is not consumed
        (resulting in a skipped projection with zero rows affected).
        """
        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=notification_publisher,
            notification_config=notification_config,
        )

        # Create an event type that's not in consumed_events
        payload = MockHeartbeatEvent(
            entity_id=uuid4(),
            heartbeat_count=1,
        )
        non_consumed_envelope = ModelEventEnvelope(
            payload=payload,
            correlation_id=uuid4(),
            source_tool="test",
            metadata=ModelEnvelopeMetadata(
                tags={"event_type": "registration.heartbeat.v1"},
            ),
        )

        result = await projector.project(non_consumed_envelope, uuid4())

        # Projection should be skipped (not consumed)
        assert result.success is True
        assert result.skipped is True
        assert result.rows_affected == 0

        # No notification should be published
        assert len(mock_event_bus.published_envelopes) == 0

    async def test_notification_failure_does_not_fail_projection(
        self,
        notification_test_pool: asyncpg.Pool,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
    ) -> None:
        """Projection succeeds even if notification publishing fails."""
        # Create a failing mock event bus
        failing_bus = AsyncMock(spec=ProtocolEventBusLike)
        failing_bus.publish_envelope.side_effect = Exception("Kafka unavailable")

        failing_publisher = TransitionNotificationPublisher(
            event_bus=failing_bus,
            topic="test.fsm.state.transitions.v1",
        )

        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=failing_publisher,
            notification_config=notification_config,
        )

        entity_id = uuid4()
        correlation_id = uuid4()
        envelope = make_registration_envelope(
            entity_id=entity_id,
            current_state="active",
            correlation_id=correlation_id,
        )

        # Projection should still succeed even though notification fails
        result = await projector.project(envelope, correlation_id)

        assert result.success is True
        assert result.rows_affected == 1

        # Verify data was actually persisted
        async with notification_test_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT current_state FROM notification_test_entities WHERE entity_id = $1",
                entity_id,
            )
        assert row is not None
        assert row["current_state"] == "active"

    async def test_multiple_projections_publish_multiple_notifications(
        self,
        notification_test_pool: asyncpg.Pool,
        notification_publisher: TransitionNotificationPublisher,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Each projection triggers its own notification."""
        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=notification_publisher,
            notification_config=notification_config,
        )

        # Project multiple entities
        entity_ids = [uuid4() for _ in range(3)]
        correlation_id = uuid4()

        for i, entity_id in enumerate(entity_ids):
            envelope = make_registration_envelope(
                entity_id=entity_id,
                current_state=f"state_{i}",
                correlation_id=correlation_id,
                version=i + 1,
            )
            await projector.project(envelope, correlation_id)

        # Verify all notifications were published
        assert len(mock_event_bus.published_envelopes) == 3

        # Verify each notification corresponds to correct entity
        # UUID fields may be UUID objects or strings depending on serialization
        published_aggregate_ids = [
            str(envelope.payload["aggregate_id"])
            for envelope, _ in mock_event_bus.published_envelopes
        ]
        for entity_id in entity_ids:
            assert str(entity_id) in published_aggregate_ids


class TestProjectorNotificationConfigIntegration:
    """Integration tests for notification configuration."""

    async def test_disabled_config_skips_notification(
        self,
        notification_test_pool: asyncpg.Pool,
        notification_publisher: TransitionNotificationPublisher,
        integration_contract: ModelProjectorContract,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Notifications are not published when config.enabled=False."""
        disabled_config = ModelProjectorNotificationConfig(
            state_column="current_state",
            aggregate_id_column="entity_id",
            version_column="version",
            enabled=False,  # Disabled
        )

        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=notification_publisher,
            notification_config=disabled_config,
        )

        entity_id = uuid4()
        correlation_id = uuid4()
        envelope = make_registration_envelope(
            entity_id=entity_id,
            current_state="active",
            correlation_id=correlation_id,
        )

        result = await projector.project(envelope, correlation_id)

        # Projection should succeed
        assert result.success is True
        assert result.rows_affected == 1

        # But no notification should be published
        assert len(mock_event_bus.published_envelopes) == 0

    async def test_no_publisher_skips_notification(
        self,
        notification_test_pool: asyncpg.Pool,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
    ) -> None:
        """Notifications are not published when no publisher is configured."""
        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=None,  # No publisher
            notification_config=notification_config,
        )

        entity_id = uuid4()
        correlation_id = uuid4()
        envelope = make_registration_envelope(
            entity_id=entity_id,
            current_state="active",
            correlation_id=correlation_id,
        )

        result = await projector.project(envelope, correlation_id)

        # Projection should succeed silently
        assert result.success is True
        assert result.rows_affected == 1

    async def test_notification_includes_version_from_values(
        self,
        notification_test_pool: asyncpg.Pool,
        notification_publisher: TransitionNotificationPublisher,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Version column value is correctly included in notification."""
        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=notification_publisher,
            notification_config=notification_config,
        )

        entity_id = uuid4()
        correlation_id = uuid4()
        envelope = make_registration_envelope(
            entity_id=entity_id,
            current_state="active",
            correlation_id=correlation_id,
            version=42,  # Specific version
        )

        await projector.project(envelope, correlation_id)

        assert len(mock_event_bus.published_envelopes) == 1
        payload = mock_event_bus.published_envelopes[0][0].payload
        assert payload["projection_version"] == 42


class TestProjectorNotificationConcurrency:
    """Integration tests for concurrent notification scenarios."""

    async def test_concurrent_projections_all_publish_notifications(
        self,
        notification_test_pool: asyncpg.Pool,
        notification_publisher: TransitionNotificationPublisher,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Concurrent projections all publish their notifications."""
        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=notification_publisher,
            notification_config=notification_config,
        )

        # Create multiple projections to run concurrently
        num_entities = 10
        entities = [(uuid4(), uuid4(), f"state_{i}") for i in range(num_entities)]

        async def project_entity(
            entity_id: UUID,
            correlation_id: UUID,
            state: str,
        ) -> bool:
            envelope = make_registration_envelope(
                entity_id=entity_id,
                current_state=state,
                correlation_id=correlation_id,
            )
            result = await projector.project(envelope, correlation_id)
            return bool(result.success)

        # Run all projections concurrently
        results = await asyncio.gather(
            *[
                project_entity(entity_id, correlation_id, state)
                for entity_id, correlation_id, state in entities
            ]
        )

        # All projections should succeed
        assert all(results)

        # All notifications should be published
        assert len(mock_event_bus.published_envelopes) == num_entities

    async def test_state_transition_chain_notifications(
        self,
        notification_test_pool: asyncpg.Pool,
        notification_publisher: TransitionNotificationPublisher,
        integration_contract: ModelProjectorContract,
        notification_config: ModelProjectorNotificationConfig,
        mock_event_bus: AsyncMock,
    ) -> None:
        """State transition chain produces correct from/to state sequence."""
        projector = ProjectorShell(
            contract=integration_contract,
            pool=notification_test_pool,
            notification_publisher=notification_publisher,
            notification_config=notification_config,
        )

        entity_id = uuid4()
        correlation_id = uuid4()

        # Define state transition chain
        states = [
            "pending_registration",
            "accepted",
            "awaiting_ack",
            "ack_received",
            "active",
        ]

        # Execute transitions sequentially
        for i, state in enumerate(states):
            envelope = make_registration_envelope(
                entity_id=entity_id,
                current_state=state,
                correlation_id=correlation_id,
                version=i + 1,
            )
            result = await projector.project(envelope, correlation_id)
            assert result.success is True

        # Verify all notifications were published
        assert len(mock_event_bus.published_envelopes) == len(states)

        # Verify from_state -> to_state chain is correct
        for i, (envelope, _) in enumerate(mock_event_bus.published_envelopes):
            payload = envelope.payload
            expected_from = "" if i == 0 else states[i - 1]
            expected_to = states[i]
            assert payload["from_state"] == expected_from, (
                f"Expected from_state '{expected_from}' at step {i}, "
                f"got '{payload['from_state']}'"
            )
            assert payload["to_state"] == expected_to, (
                f"Expected to_state '{expected_to}' at step {i}, "
                f"got '{payload['to_state']}'"
            )


class TestTransitionNotificationPublisherIntegration:
    """Integration tests for TransitionNotificationPublisher directly."""

    async def test_publisher_metrics_track_successful_publishes(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Publisher metrics correctly track successful publishes."""
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.metrics.v1",
        )

        # Publish several notifications
        for i in range(5):
            notification = ModelStateTransitionNotification(
                aggregate_type="test",
                aggregate_id=uuid4(),
                from_state="pending",
                to_state="active",
                projection_version=i,
                correlation_id=uuid4(),
                causation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )
            await publisher.publish(notification)

        # Check metrics
        metrics = publisher.get_metrics()
        assert metrics.notifications_published == 5
        assert metrics.notifications_failed == 0
        assert metrics.last_publish_at is not None
        assert metrics.is_healthy() is True

    async def test_publisher_metrics_track_failed_publishes(
        self,
    ) -> None:
        """Publisher metrics correctly track failed publishes."""
        # Create failing bus
        failing_bus = AsyncMock(spec=ProtocolEventBusLike)
        failing_bus.publish_envelope.side_effect = Exception("Connection refused")

        publisher = TransitionNotificationPublisher(
            event_bus=failing_bus,
            topic="test.metrics.v1",
            circuit_breaker_threshold=10,  # High threshold for test
        )

        # Attempt publishes (they will fail)
        for i in range(3):
            notification = ModelStateTransitionNotification(
                aggregate_type="test",
                aggregate_id=uuid4(),
                from_state="pending",
                to_state="active",
                projection_version=i,
                correlation_id=uuid4(),
                causation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )
            try:
                await publisher.publish(notification)
            except Exception:
                pass  # Expected to fail

        # Check metrics
        metrics = publisher.get_metrics()
        assert metrics.notifications_published == 0
        assert metrics.notifications_failed == 3
        assert metrics.consecutive_failures == 3

    async def test_publisher_batch_operation(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Publisher batch publish works correctly."""
        publisher = TransitionNotificationPublisher(
            event_bus=mock_event_bus,
            topic="test.batch.v1",
        )

        correlation_id = uuid4()
        notifications = [
            ModelStateTransitionNotification(
                aggregate_type="test",
                aggregate_id=uuid4(),
                from_state="pending",
                to_state=f"state_{i}",
                projection_version=i,
                correlation_id=correlation_id,
                causation_id=uuid4(),
                timestamp=datetime.now(UTC),
            )
            for i in range(5)
        ]

        await publisher.publish_batch(notifications)

        # Check all were published
        metrics = publisher.get_metrics()
        assert metrics.notifications_published == 5
        assert metrics.batch_operations == 1
        assert metrics.batch_notifications_total == 5


__all__: list[str] = [
    "TestProjectorNotificationIntegration",
    "TestProjectorNotificationConfigIntegration",
    "TestProjectorNotificationConcurrency",
    "TestTransitionNotificationPublisherIntegration",
]
