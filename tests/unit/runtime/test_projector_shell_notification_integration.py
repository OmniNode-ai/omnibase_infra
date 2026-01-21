# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for ProjectorShell notification integration.

Tests the integration between ProjectorShell and TransitionNotificationPublisher:
- Configuration validation
- Pre-projection state fetching
- Post-commit notification publishing
- Error handling and best-effort semantics

Related:
    - OMN-1139: Integrate TransitionNotificationPublisher with ProjectorShell
    - src/omnibase_infra/runtime/projector_shell.py
    - src/omnibase_infra/runtime/mixins/mixin_projector_notification_publishing.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import asyncpg
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
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.constants_notification import FROM_STATE_INITIAL
from omnibase_infra.runtime.models import ModelProjectorNotificationConfig
from omnibase_infra.runtime.projector_shell import ProjectorShell

# --- Test Fixtures ---


class MockStateChangeEvent(BaseModel):
    """Mock event payload for testing state transitions."""

    entity_id: UUID
    current_state: str
    version: int = 1
    event_type: str = "state.changed.v1"


@pytest.fixture
def basic_contract() -> ModelProjectorContract:
    """Create a basic projector contract for testing."""
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
            name="version",
            type="INTEGER",
            source="payload.version",
            default="0",
        ),
    ]

    schema = ModelProjectorSchema(
        table="test_entities",
        primary_key="entity_id",
        columns=columns,
    )

    behavior = ModelProjectorBehavior(
        mode="upsert",
        upsert_key="entity_id",
    )

    return ModelProjectorContract(
        projector_kind="materialized_view",
        projector_id="test-projector",
        name="Test Projector",
        version="1.0.0",
        aggregate_type="test_entity",
        consumed_events=["state.changed.v1"],
        projection_schema=schema,
        behavior=behavior,
    )


@pytest.fixture
def notification_config() -> ModelProjectorNotificationConfig:
    """Create a notification config for testing."""
    return ModelProjectorNotificationConfig(
        topic="test.fsm.state.transitions.v1",
        state_column="current_state",
        aggregate_id_column="entity_id",
        version_column="version",
        enabled=True,
    )


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg pool."""
    pool = MagicMock(spec=asyncpg.Pool)
    pool.acquire.return_value.__aenter__ = AsyncMock()
    pool.acquire.return_value.__aexit__ = AsyncMock()
    return pool


@pytest.fixture
def mock_notification_publisher() -> MagicMock:
    """Create a mock notification publisher."""
    publisher = MagicMock(spec=ProtocolTransitionNotificationPublisher)
    publisher.publish = AsyncMock()
    publisher.publish_batch = AsyncMock()
    return publisher


@pytest.fixture
def state_change_envelope() -> ModelEventEnvelope[MockStateChangeEvent]:
    """Create an event envelope with state change payload."""
    entity_id = uuid4()
    payload = MockStateChangeEvent(
        entity_id=entity_id,
        current_state="active",
        version=2,
    )
    return ModelEventEnvelope(
        payload=payload,
        correlation_id=uuid4(),
        source_tool="test",
        metadata=ModelEnvelopeMetadata(
            tags={"event_type": "state.changed.v1"},
        ),
    )


# --- ModelProjectorNotificationConfig Tests ---


class TestModelProjectorNotificationConfig:
    """Tests for ModelProjectorNotificationConfig model."""

    def test_basic_config_creation(self) -> None:
        """Test creating a basic notification config using expected_topic."""
        config = ModelProjectorNotificationConfig(
            expected_topic="test.fsm.state.transitions.v1",
            state_column="current_state",
            aggregate_id_column="entity_id",
        )
        assert config.expected_topic == "test.fsm.state.transitions.v1"
        assert config.state_column == "current_state"
        assert config.aggregate_id_column == "entity_id"
        assert config.version_column is None
        assert config.enabled is True

    def test_backwards_compatible_topic_alias(self) -> None:
        """Test creating a config using topic alias for backwards compatibility."""
        config = ModelProjectorNotificationConfig(
            topic="test.fsm.state.transitions.v1",  # Using alias
            state_column="current_state",
            aggregate_id_column="entity_id",
        )
        # Attribute is always expected_topic, but accepts topic as input alias
        assert config.expected_topic == "test.fsm.state.transitions.v1"
        assert config.state_column == "current_state"
        assert config.aggregate_id_column == "entity_id"

    def test_full_config_creation(self) -> None:
        """Test creating a config with all fields."""
        config = ModelProjectorNotificationConfig(
            expected_topic="custom.notifications.v1",
            state_column="fsm_state",
            aggregate_id_column="node_id",
            version_column="projection_version",
            enabled=False,
        )
        assert config.expected_topic == "custom.notifications.v1"
        assert config.state_column == "fsm_state"
        assert config.aggregate_id_column == "node_id"
        assert config.version_column == "projection_version"
        assert config.enabled is False

    def test_config_is_immutable(self) -> None:
        """Test that config is frozen after creation."""
        config = ModelProjectorNotificationConfig(
            expected_topic="test.fsm.state.transitions.v1",
            state_column="current_state",
            aggregate_id_column="entity_id",
        )
        with pytest.raises(Exception):  # Pydantic validation error
            config.state_column = "new_value"  # type: ignore[misc]

    def test_state_column_validation(self) -> None:
        """Test state_column validation."""
        # Empty string should fail
        with pytest.raises(Exception):
            ModelProjectorNotificationConfig(
                topic="test.fsm.state.transitions.v1",
                state_column="",
                aggregate_id_column="entity_id",
            )

    def test_aggregate_id_column_validation(self) -> None:
        """Test aggregate_id_column validation."""
        # Empty string should fail
        with pytest.raises(Exception):
            ModelProjectorNotificationConfig(
                topic="test.fsm.state.transitions.v1",
                state_column="current_state",
                aggregate_id_column="",
            )


# --- ProjectorShell Notification Config Validation Tests ---


class TestProjectorShellNotificationConfigValidation:
    """Tests for ProjectorShell notification config validation."""

    def test_valid_config_accepted(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
    ) -> None:
        """Test that valid notification config is accepted."""
        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )
        assert projector._notification_publisher is not None
        assert projector._notification_config is not None

    def test_invalid_state_column_rejected(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
    ) -> None:
        """Test that invalid state_column raises ProtocolConfigurationError."""
        invalid_config = ModelProjectorNotificationConfig(
            topic="test.fsm.state.transitions.v1",
            state_column="nonexistent_column",
            aggregate_id_column="entity_id",
        )
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            ProjectorShell(
                contract=basic_contract,
                pool=mock_pool,
                notification_publisher=mock_notification_publisher,
                notification_config=invalid_config,
            )
        assert "state_column" in str(exc_info.value)
        assert "nonexistent_column" in str(exc_info.value)

    def test_invalid_aggregate_id_column_rejected(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
    ) -> None:
        """Test that invalid aggregate_id_column raises ProtocolConfigurationError."""
        invalid_config = ModelProjectorNotificationConfig(
            topic="test.fsm.state.transitions.v1",
            state_column="current_state",
            aggregate_id_column="nonexistent_id",
        )
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            ProjectorShell(
                contract=basic_contract,
                pool=mock_pool,
                notification_publisher=mock_notification_publisher,
                notification_config=invalid_config,
            )
        assert "aggregate_id_column" in str(exc_info.value)
        assert "nonexistent_id" in str(exc_info.value)

    def test_invalid_version_column_rejected(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
    ) -> None:
        """Test that invalid version_column raises ProtocolConfigurationError."""
        invalid_config = ModelProjectorNotificationConfig(
            topic="test.fsm.state.transitions.v1",
            state_column="current_state",
            aggregate_id_column="entity_id",
            version_column="nonexistent_version",
        )
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            ProjectorShell(
                contract=basic_contract,
                pool=mock_pool,
                notification_publisher=mock_notification_publisher,
                notification_config=invalid_config,
            )
        assert "version_column" in str(exc_info.value)
        assert "nonexistent_version" in str(exc_info.value)

    def test_publisher_without_config_no_notification(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
    ) -> None:
        """Test that publisher without config doesn't enable notifications."""
        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=None,  # No config
        )
        assert not projector._is_notification_enabled()

    def test_config_without_publisher_no_notification(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
    ) -> None:
        """Test that config without publisher doesn't enable notifications."""
        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=None,  # No publisher
            notification_config=notification_config,
        )
        assert not projector._is_notification_enabled()

    def test_disabled_config_no_notification(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
    ) -> None:
        """Test that disabled config doesn't enable notifications."""
        disabled_config = ModelProjectorNotificationConfig(
            topic="test.fsm.state.transitions.v1",
            state_column="current_state",
            aggregate_id_column="entity_id",
            enabled=False,  # Explicitly disabled
        )
        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=disabled_config,
        )
        assert not projector._is_notification_enabled()


# --- ProjectorShell Notification Publishing Tests ---


class TestProjectorShellNotificationPublishing:
    """Tests for ProjectorShell notification publishing integration."""

    @pytest.mark.asyncio
    async def test_notification_published_on_successful_projection(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
        state_change_envelope: ModelEventEnvelope[MockStateChangeEvent],
    ) -> None:
        """Test that notification is published after successful projection."""
        # Setup mock for fetchrow (state lookup) to return previous state
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"current_state": "pending"}
        mock_conn.execute.return_value = "INSERT 0 1"
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        result = await projector.project(
            state_change_envelope,
            uuid4(),
        )

        assert result.success is True
        assert result.rows_affected == 1

        # Verify notification was published
        mock_notification_publisher.publish.assert_called_once()
        call_args = mock_notification_publisher.publish.call_args
        notification = call_args[0][0]
        assert isinstance(notification, ModelStateTransitionNotification)
        assert notification.aggregate_type == "test_entity"
        assert notification.from_state == "pending"
        assert notification.to_state == "active"

    @pytest.mark.asyncio
    async def test_notification_published_for_new_entity(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
        state_change_envelope: ModelEventEnvelope[MockStateChangeEvent],
    ) -> None:
        """Test that notification is published for new entity (no previous state)."""
        # Setup mock for fetchrow (state lookup) to return None (new entity)
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None  # New entity
        mock_conn.execute.return_value = "INSERT 0 1"
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        result = await projector.project(
            state_change_envelope,
            uuid4(),
        )

        assert result.success is True

        # Verify notification was published with FROM_STATE_INITIAL sentinel
        mock_notification_publisher.publish.assert_called_once()
        call_args = mock_notification_publisher.publish.call_args
        notification = call_args[0][0]
        assert notification.from_state == FROM_STATE_INITIAL  # Sentinel for new entity
        assert notification.to_state == "active"

    @pytest.mark.asyncio
    async def test_notification_not_published_when_rows_affected_zero(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
        state_change_envelope: ModelEventEnvelope[MockStateChangeEvent],
    ) -> None:
        """Test that notification is NOT published when no rows affected."""
        # Setup mock for fetchrow (state lookup)
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"current_state": "pending"}
        mock_conn.execute.return_value = "INSERT 0 0"  # No rows affected
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        result = await projector.project(
            state_change_envelope,
            uuid4(),
        )

        assert result.success is True
        assert result.rows_affected == 0

        # Verify notification was NOT published
        mock_notification_publisher.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_notification_not_published_when_disabled(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        state_change_envelope: ModelEventEnvelope[MockStateChangeEvent],
    ) -> None:
        """Test that notification is NOT published when config is disabled."""
        disabled_config = ModelProjectorNotificationConfig(
            topic="test.fsm.state.transitions.v1",
            state_column="current_state",
            aggregate_id_column="entity_id",
            enabled=False,
        )

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "INSERT 0 1"
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=disabled_config,
        )

        result = await projector.project(
            state_change_envelope,
            uuid4(),
        )

        assert result.success is True
        assert result.rows_affected == 1

        # Verify notification was NOT published
        mock_notification_publisher.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_projection_succeeds_even_if_notification_fails(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
        state_change_envelope: ModelEventEnvelope[MockStateChangeEvent],
    ) -> None:
        """Test that projection succeeds even if notification publishing fails."""
        # Setup mock
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"current_state": "pending"}
        mock_conn.execute.return_value = "INSERT 0 1"
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Make notification publishing fail
        mock_notification_publisher.publish.side_effect = Exception("Kafka unavailable")

        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        # Projection should still succeed
        result = await projector.project(
            state_change_envelope,
            uuid4(),
        )

        assert result.success is True
        assert result.rows_affected == 1

    @pytest.mark.asyncio
    async def test_notification_includes_correct_causation_id(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
        state_change_envelope: ModelEventEnvelope[MockStateChangeEvent],
    ) -> None:
        """Test that notification includes the event envelope_id as causation_id."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"current_state": "pending"}
        mock_conn.execute.return_value = "INSERT 0 1"
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        await projector.project(
            state_change_envelope,
            uuid4(),
        )

        # Verify causation_id is the envelope_id
        call_args = mock_notification_publisher.publish.call_args
        notification = call_args[0][0]
        assert notification.causation_id == state_change_envelope.envelope_id

    @pytest.mark.asyncio
    async def test_notification_includes_projection_version(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
        state_change_envelope: ModelEventEnvelope[MockStateChangeEvent],
    ) -> None:
        """Test that notification includes the projection version from values."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"current_state": "pending"}
        mock_conn.execute.return_value = "INSERT 0 1"
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        await projector.project(
            state_change_envelope,
            uuid4(),
        )

        # The payload has version=2
        call_args = mock_notification_publisher.publish.call_args
        notification = call_args[0][0]
        assert notification.projection_version == 2


# --- Mixin Helper Method Tests ---


class TestMixinHelperMethods:
    """Tests for mixin helper methods."""

    def test_extract_state_from_values(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
    ) -> None:
        """Test _extract_state_from_values extracts state correctly."""
        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        values = {"current_state": "active", "entity_id": uuid4()}
        state = projector._extract_state_from_values(values)
        assert state == "active"

    def test_extract_state_from_values_missing_column(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
    ) -> None:
        """Test _extract_state_from_values returns None when column missing."""
        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        values = {"entity_id": uuid4()}  # No current_state
        state = projector._extract_state_from_values(values)
        assert state is None

    def test_extract_aggregate_id_from_values_uuid(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
    ) -> None:
        """Test _extract_aggregate_id_from_values handles UUID correctly."""
        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        expected_id = uuid4()
        values = {"entity_id": expected_id, "current_state": "active"}
        aggregate_id = projector._extract_aggregate_id_from_values(values)
        assert aggregate_id == expected_id

    def test_extract_aggregate_id_from_values_string(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
    ) -> None:
        """Test _extract_aggregate_id_from_values converts string to UUID."""
        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        values = {"entity_id": uuid_str, "current_state": "active"}
        aggregate_id = projector._extract_aggregate_id_from_values(values)
        assert aggregate_id == UUID(uuid_str)

    def test_extract_version_from_values_int(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
    ) -> None:
        """Test _extract_version_from_values handles int correctly."""
        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        values = {"entity_id": uuid4(), "current_state": "active", "version": 5}
        version = projector._extract_version_from_values(values)
        assert version == 5

    def test_extract_version_from_values_no_version_column(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
    ) -> None:
        """Test _extract_version_from_values returns 0 when no version_column configured."""
        config_no_version = ModelProjectorNotificationConfig(
            topic="test.fsm.state.transitions.v1",
            state_column="current_state",
            aggregate_id_column="entity_id",
            version_column=None,  # No version column
        )

        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=config_no_version,
        )

        values = {"entity_id": uuid4(), "current_state": "active", "version": 5}
        version = projector._extract_version_from_values(values)
        assert version == 0


# --- Event Type Filtering Tests ---


class TestEventTypeFiltering:
    """Tests for event type filtering with notifications."""

    @pytest.mark.asyncio
    async def test_notification_not_published_for_skipped_events(
        self,
        basic_contract: ModelProjectorContract,
        mock_pool: MagicMock,
        mock_notification_publisher: MagicMock,
        notification_config: ModelProjectorNotificationConfig,
    ) -> None:
        """Test that notification is NOT published when event is skipped."""
        # Create an event that the projector doesn't consume
        non_consumed_payload = MagicMock()
        non_consumed_payload.event_type = "other.event.v1"
        non_consumed_envelope = ModelEventEnvelope(
            payload=non_consumed_payload,
            correlation_id=uuid4(),
            source_tool="test",
            metadata=ModelEnvelopeMetadata(
                tags={"event_type": "other.event.v1"},  # Not in consumed_events
            ),
        )

        projector = ProjectorShell(
            contract=basic_contract,
            pool=mock_pool,
            notification_publisher=mock_notification_publisher,
            notification_config=notification_config,
        )

        result = await projector.project(
            non_consumed_envelope,
            uuid4(),
        )

        assert result.success is True
        assert result.skipped is True

        # Verify notification was NOT published
        mock_notification_publisher.publish.assert_not_called()
