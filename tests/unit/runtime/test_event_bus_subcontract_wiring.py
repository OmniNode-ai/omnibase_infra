# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for EventBusSubcontractWiring.

Tests the contract-driven Kafka subscription wiring functionality including:
- Topic resolution with environment prefixes
- Subscription creation from subcontract
- Callback creation and dispatch bridging
- Cleanup and lifecycle management

Related:
    - OMN-1621: Runtime consumes event_bus subcontract for contract-driven wiring
    - src/omnibase_infra/runtime/event_bus_subcontract_wiring.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.models.contracts.subcontracts import ModelEventBusSubcontract
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
    load_event_bus_subcontract,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Create mock event bus with subscribe/publish methods."""
    bus = AsyncMock()
    # Subscribe returns an unsubscribe callable
    unsubscribe_callable = AsyncMock()
    bus.subscribe = AsyncMock(return_value=unsubscribe_callable)
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def mock_dispatch_engine() -> AsyncMock:
    """Create mock dispatch engine."""
    engine = AsyncMock()
    engine.dispatch = AsyncMock()
    return engine


@pytest.fixture
def wiring(
    mock_event_bus: AsyncMock,
    mock_dispatch_engine: AsyncMock,
) -> EventBusSubcontractWiring:
    """Create wiring instance with dev environment."""
    return EventBusSubcontractWiring(
        event_bus=mock_event_bus,
        dispatch_engine=mock_dispatch_engine,
        environment="dev",
    )


@pytest.fixture
def sample_subcontract() -> ModelEventBusSubcontract:
    """Create sample event bus subcontract with topics."""
    return ModelEventBusSubcontract(
        version=ModelSemVer(major=1, minor=0, patch=0),
        subscribe_topics=[
            "onex.evt.node.introspected.v1",
            "onex.evt.node.registered.v1",
        ],
        publish_topics=["onex.cmd.node.register.v1"],
    )


@pytest.fixture
def sample_event_message() -> ModelEventMessage:
    """Create sample event message for callback testing."""
    payload = {
        "event_type": "test.event",
        "correlation_id": str(uuid4()),
        "payload": {"key": "value"},
    }
    return ModelEventMessage(
        topic="dev.onex.evt.test.v1",
        key=b"test-key",
        value=json.dumps(payload).encode("utf-8"),
        headers=ModelEventHeaders(
            source="test-service",
            event_type="test.event",
            timestamp=datetime.now(UTC),
        ),
    )


# =============================================================================
# Topic Resolution Tests
# =============================================================================


class TestTopicResolution:
    """Tests for topic suffix to full topic name resolution."""

    def test_resolve_topic_adds_environment_prefix(
        self,
        wiring: EventBusSubcontractWiring,
    ) -> None:
        """Test topic resolution adds environment prefix."""
        topic_suffix = "onex.evt.test-service.test-event.v1"
        result = wiring.resolve_topic(topic_suffix)
        assert result == "dev.onex.evt.test-service.test-event.v1"

    def test_resolve_topic_with_prod_environment(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test topic resolution with production environment."""
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="prod",
        )
        result = wiring.resolve_topic("onex.evt.node.registered.v1")
        assert result == "prod.onex.evt.node.registered.v1"

    def test_resolve_topic_with_staging_environment(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test topic resolution with staging environment."""
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="staging",
        )
        result = wiring.resolve_topic("onex.cmd.process.v1")
        assert result == "staging.onex.cmd.process.v1"


# =============================================================================
# Wire Subscriptions Tests
# =============================================================================


class TestWireSubscriptions:
    """Tests for wiring subscriptions from subcontract."""

    @pytest.mark.asyncio
    async def test_wire_subscriptions_creates_subscriptions(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test wiring creates subscriptions for each topic."""
        await wiring.wire_subscriptions(sample_subcontract, node_name="test-handler")

        # Should subscribe to both topics
        assert mock_event_bus.subscribe.call_count == 2

    @pytest.mark.asyncio
    async def test_wire_subscriptions_uses_correct_topics(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test wiring uses resolved topic names."""
        await wiring.wire_subscriptions(sample_subcontract, node_name="test-handler")

        calls = mock_event_bus.subscribe.call_args_list
        topics = [call.kwargs["topic"] for call in calls]
        assert "dev.onex.evt.node.introspected.v1" in topics
        assert "dev.onex.evt.node.registered.v1" in topics

    @pytest.mark.asyncio
    async def test_wire_subscriptions_uses_correct_group_ids(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test wiring uses environment-prefixed group IDs."""
        await wiring.wire_subscriptions(
            sample_subcontract, node_name="registration-handler"
        )

        calls = mock_event_bus.subscribe.call_args_list
        group_ids = [call.kwargs["group_id"] for call in calls]
        assert all(gid == "dev.registration-handler" for gid in group_ids)

    @pytest.mark.asyncio
    async def test_wire_subscriptions_stores_unsubscribe_callables(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test wiring stores unsubscribe callables for cleanup."""
        await wiring.wire_subscriptions(sample_subcontract, node_name="test-handler")

        # Wiring should have stored 2 unsubscribe callables
        assert len(wiring._unsubscribe_callables) == 2

    @pytest.mark.asyncio
    async def test_wire_subscriptions_with_empty_topics(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test wiring with empty subscribe_topics is a no-op."""
        subcontract = ModelEventBusSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            subscribe_topics=[],
            publish_topics=[],
        )
        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        mock_event_bus.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_wire_subscriptions_with_default_topics(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Test wiring with default (unset) subscribe_topics is a no-op."""
        subcontract = ModelEventBusSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            # subscribe_topics defaults to None/empty
        )
        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        mock_event_bus.subscribe.assert_not_called()


# =============================================================================
# Dispatch Callback Tests
# =============================================================================


class TestDispatchCallback:
    """Tests for callback creation and dispatch bridging."""

    @pytest.mark.asyncio
    async def test_dispatch_callback_calls_dispatch_engine(
        self,
        wiring: EventBusSubcontractWiring,
        mock_dispatch_engine: AsyncMock,
        sample_event_message: ModelEventMessage,
    ) -> None:
        """Test callback dispatches to engine."""
        callback = wiring._create_dispatch_callback("dev.onex.evt.test.v1")

        await callback(sample_event_message)

        mock_dispatch_engine.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_callback_passes_topic(
        self,
        wiring: EventBusSubcontractWiring,
        mock_dispatch_engine: AsyncMock,
        sample_event_message: ModelEventMessage,
    ) -> None:
        """Test callback passes correct topic to dispatch engine."""
        topic = "dev.onex.evt.test.v1"
        callback = wiring._create_dispatch_callback(topic)

        await callback(sample_event_message)

        call_args = mock_dispatch_engine.dispatch.call_args
        assert call_args[0][0] == topic

    @pytest.mark.asyncio
    async def test_dispatch_callback_passes_envelope(
        self,
        wiring: EventBusSubcontractWiring,
        mock_dispatch_engine: AsyncMock,
        sample_event_message: ModelEventMessage,
    ) -> None:
        """Test callback passes deserialized envelope to dispatch engine."""
        callback = wiring._create_dispatch_callback("dev.onex.evt.test.v1")

        await callback(sample_event_message)

        call_args = mock_dispatch_engine.dispatch.call_args
        envelope = call_args[0][1]
        # Envelope should be deserialized from message
        assert envelope is not None

    @pytest.mark.asyncio
    async def test_dispatch_callback_raises_on_invalid_json(
        self,
        wiring: EventBusSubcontractWiring,
    ) -> None:
        """Test callback raises RuntimeHostError on invalid JSON in message."""
        callback = wiring._create_dispatch_callback("dev.onex.evt.test.v1")

        invalid_message = ModelEventMessage(
            topic="dev.onex.evt.test.v1",
            key=b"key",
            value=b"not valid json",
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        with pytest.raises(RuntimeHostError, match="Failed to deserialize"):
            await callback(invalid_message)

    @pytest.mark.asyncio
    async def test_dispatch_callback_propagates_dispatch_errors(
        self,
        wiring: EventBusSubcontractWiring,
        mock_dispatch_engine: AsyncMock,
        sample_event_message: ModelEventMessage,
    ) -> None:
        """Test callback wraps dispatch engine errors in RuntimeHostError."""
        mock_dispatch_engine.dispatch.side_effect = RuntimeError("Dispatch failed")
        callback = wiring._create_dispatch_callback("dev.onex.evt.test.v1")

        with pytest.raises(RuntimeHostError, match="Failed to dispatch"):
            await callback(sample_event_message)


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanup:
    """Tests for cleanup and lifecycle management."""

    @pytest.mark.asyncio
    async def test_cleanup_calls_all_unsubscribe_callables(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test cleanup unsubscribes from all topics."""
        # Create separate mock unsubscribe callables
        unsubscribe_mock_1 = AsyncMock()
        unsubscribe_mock_2 = AsyncMock()
        mock_event_bus.subscribe.side_effect = [unsubscribe_mock_1, unsubscribe_mock_2]

        await wiring.wire_subscriptions(sample_subcontract, node_name="test-handler")
        await wiring.cleanup()

        unsubscribe_mock_1.assert_called_once()
        unsubscribe_mock_2.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_clears_callables_list(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test cleanup clears the unsubscribe callables list."""
        await wiring.wire_subscriptions(sample_subcontract, node_name="test-handler")
        assert len(wiring._unsubscribe_callables) == 2

        await wiring.cleanup()
        assert len(wiring._unsubscribe_callables) == 0

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test cleanup can be called multiple times safely."""
        unsubscribe_mock = AsyncMock()
        mock_event_bus.subscribe.return_value = unsubscribe_mock

        await wiring.wire_subscriptions(sample_subcontract, node_name="test-handler")

        await wiring.cleanup()
        await wiring.cleanup()  # Second call should be no-op

        # Unsubscribe should only be called once per subscription
        assert unsubscribe_mock.call_count == 2  # 2 topics, 1 call each

    @pytest.mark.asyncio
    async def test_cleanup_handles_unsubscribe_errors(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test cleanup continues even if unsubscribe raises."""
        unsubscribe_error = AsyncMock(side_effect=RuntimeError("Unsubscribe failed"))
        unsubscribe_success = AsyncMock()
        mock_event_bus.subscribe.side_effect = [unsubscribe_error, unsubscribe_success]

        await wiring.wire_subscriptions(sample_subcontract, node_name="test-handler")
        # Should not raise, just log warning
        await wiring.cleanup()

        # Both unsubscribe callables should have been called
        unsubscribe_error.assert_called_once()
        unsubscribe_success.assert_called_once()


# =============================================================================
# Load Subcontract Tests
# =============================================================================


class TestLoadEventBusSubcontract:
    """Tests for load_event_bus_subcontract function."""

    def test_loads_valid_contract(self, tmp_path: Path) -> None:
        """Test loading valid event_bus subcontract."""
        # Topic format: onex.kind.producer.event-name.version (5 segments)
        contract_content = """
event_bus:
  version:
    major: 1
    minor: 0
    patch: 0
  subscribe_topics:
    - "onex.evt.test-service.test-event.v1"
  publish_topics:
    - "onex.evt.result-service.result-event.v1"
"""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(contract_content)

        result = load_event_bus_subcontract(contract_file)

        assert result is not None
        assert result.subscribe_topics == ["onex.evt.test-service.test-event.v1"]
        assert result.publish_topics == ["onex.evt.result-service.result-event.v1"]

    def test_loads_contract_with_multiple_topics(self, tmp_path: Path) -> None:
        """Test loading contract with multiple subscribe/publish topics."""
        contract_content = """
event_bus:
  version:
    major: 1
    minor: 0
    patch: 0
  subscribe_topics:
    - "onex.evt.node.introspected.v1"
    - "onex.evt.node.registered.v1"
    - "onex.cmd.registration.request.v1"
  publish_topics:
    - "onex.evt.node.processed.v1"
    - "onex.cmd.node.register.v1"
"""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(contract_content)

        result = load_event_bus_subcontract(contract_file)

        assert result is not None
        assert len(result.subscribe_topics) == 3
        assert len(result.publish_topics) == 2

    def test_returns_none_for_missing_event_bus_section(self, tmp_path: Path) -> None:
        """Test returns None when no event_bus section."""
        contract_content = """
name: "test-handler"
version: "1.0.0"
"""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(contract_content)

        result = load_event_bus_subcontract(contract_file)

        assert result is None

    def test_returns_none_for_nonexistent_file(self) -> None:
        """Test returns None for non-existent file."""
        result = load_event_bus_subcontract(Path("/nonexistent/contract.yaml"))
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        """Test returns None for empty contract file."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text("")

        result = load_event_bus_subcontract(contract_file)

        assert result is None

    def test_returns_none_for_invalid_yaml(self, tmp_path: Path) -> None:
        """Test returns None for invalid YAML."""
        contract_content = """
event_bus:
  subscribe_topics:
    - this is not valid yaml: because: of: colons
"""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(contract_content)

        result = load_event_bus_subcontract(contract_file)

        # Should return None due to YAML parse error
        assert result is None

    def test_returns_none_for_empty_event_bus_section(self, tmp_path: Path) -> None:
        """Test returns None for empty event_bus section."""
        contract_content = """
event_bus:
name: "test-handler"
"""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(contract_content)

        result = load_event_bus_subcontract(contract_file)

        assert result is None

    def test_uses_provided_logger(self, tmp_path: Path) -> None:
        """Test function uses provided logger."""
        contract_file = tmp_path / "contract.yaml"
        # Non-existent file to trigger warning

        mock_logger = MagicMock()
        load_event_bus_subcontract(
            Path("/nonexistent/contract.yaml"),
            logger=mock_logger,
        )

        mock_logger.warning.assert_called()


# =============================================================================
# Deserialization Tests
# =============================================================================


class TestDeserialization:
    """Tests for message deserialization."""

    def test_deserialize_valid_envelope(
        self,
        wiring: EventBusSubcontractWiring,
    ) -> None:
        """Test deserializing valid event envelope."""
        payload = {
            "event_type": "node.introspected",
            "correlation_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {"node_id": "test-123"},
        }
        message = ModelEventMessage(
            topic="dev.onex.evt.test.v1",
            key=b"key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        envelope = wiring._deserialize_to_envelope(message)

        assert envelope is not None

    def test_deserialize_raises_on_invalid_json(
        self,
        wiring: EventBusSubcontractWiring,
    ) -> None:
        """Test deserialization raises on invalid JSON."""
        message = ModelEventMessage(
            topic="dev.onex.evt.test.v1",
            key=b"key",
            value=b"not json",
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        with pytest.raises(json.JSONDecodeError):
            wiring._deserialize_to_envelope(message)
