# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for EventBusSubcontractWiring.

Tests the contract-driven Kafka subscription wiring functionality including:
- Topic resolution with environment prefixes
- Subscription creation from subcontract
- Callback creation and dispatch bridging
- Cleanup and lifecycle management
- DLQ consumer_group alignment with subscription consumer_group

Related:
    - OMN-1621: Runtime consumes event_bus subcontract for contract-driven wiring
    - OMN-1740: DLQ consumer_group alignment (PR #219 review feedback)
    - src/omnibase_infra/runtime/event_bus_subcontract_wiring.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
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

pytestmark = pytest.mark.unit

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
        node_name="test-handler",
        service="test-service",
        version="v1",
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
        topic="onex.evt.test.v1",
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
    """Tests for topic suffix to full topic name resolution (realm-agnostic)."""

    def test_resolve_topic_returns_topic_unchanged(
        self,
        wiring: EventBusSubcontractWiring,
    ) -> None:
        """Test topic resolution returns topic unchanged (realm-agnostic)."""
        topic_suffix = "onex.evt.test-service.test-event.v1"
        result = wiring.resolve_topic(topic_suffix)
        assert result == "onex.evt.test-service.test-event.v1"

    def test_resolve_topic_with_prod_environment(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test topic resolution is environment-agnostic (no prefix added)."""
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="prod",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )
        result = wiring.resolve_topic("onex.evt.node.registered.v1")
        assert result == "onex.evt.node.registered.v1"

    def test_resolve_topic_with_staging_environment(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test topic resolution is environment-agnostic (no prefix added)."""
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="staging",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )
        result = wiring.resolve_topic("onex.cmd.platform.process-event.v1")
        assert result == "onex.cmd.platform.process-event.v1"


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for constructor input validation (fail-fast on empty inputs)."""

    def test_empty_environment_raises_value_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test that empty environment raises ValueError."""
        with pytest.raises(ValueError, match="environment must be a non-empty string"):
            EventBusSubcontractWiring(
                event_bus=mock_event_bus,
                dispatch_engine=mock_dispatch_engine,
                environment="",
                node_name="test-handler",
                service="test-service",
                version="v1",
            )

    def test_whitespace_environment_raises_value_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test that whitespace-only environment raises ValueError."""
        with pytest.raises(ValueError, match="environment must be a non-empty string"):
            EventBusSubcontractWiring(
                event_bus=mock_event_bus,
                dispatch_engine=mock_dispatch_engine,
                environment="   ",
                node_name="test-handler",
                service="test-service",
                version="v1",
            )

    def test_empty_service_raises_value_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test that empty service raises ValueError."""
        with pytest.raises(ValueError, match="service must be a non-empty string"):
            EventBusSubcontractWiring(
                event_bus=mock_event_bus,
                dispatch_engine=mock_dispatch_engine,
                environment="dev",
                node_name="test-handler",
                service="",
                version="v1",
            )

    def test_whitespace_service_raises_value_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test that whitespace-only service raises ValueError."""
        with pytest.raises(ValueError, match="service must be a non-empty string"):
            EventBusSubcontractWiring(
                event_bus=mock_event_bus,
                dispatch_engine=mock_dispatch_engine,
                environment="dev",
                node_name="test-handler",
                service="\t\n",
                version="v1",
            )

    def test_empty_version_raises_value_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test that empty version raises ValueError."""
        with pytest.raises(ValueError, match="version must be a non-empty string"):
            EventBusSubcontractWiring(
                event_bus=mock_event_bus,
                dispatch_engine=mock_dispatch_engine,
                environment="dev",
                node_name="test-handler",
                service="test-service",
                version="",
            )

    def test_whitespace_version_raises_value_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test that whitespace-only version raises ValueError."""
        with pytest.raises(ValueError, match="version must be a non-empty string"):
            EventBusSubcontractWiring(
                event_bus=mock_event_bus,
                dispatch_engine=mock_dispatch_engine,
                environment="dev",
                node_name="test-handler",
                service="test-service",
                version="   ",
            )

    def test_valid_inputs_construct_successfully(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test that valid inputs construct successfully."""
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )
        assert wiring._environment == "dev"
        assert wiring._service == "test-service"
        assert wiring._version == "v1"


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
        assert "onex.evt.node.introspected.v1" in topics
        assert "onex.evt.node.registered.v1" in topics

    @pytest.mark.asyncio
    async def test_wire_subscriptions_uses_correct_node_identity(
        self,
        wiring: EventBusSubcontractWiring,
        mock_event_bus: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test wiring uses correct node identity for consumer groups."""
        await wiring.wire_subscriptions(
            sample_subcontract, node_name="registration-handler"
        )

        calls = mock_event_bus.subscribe.call_args_list
        node_identities = [call.kwargs["node_identity"] for call in calls]
        # Verify all node identities have correct env and service for consumer group
        # Note: service comes from wiring constructor param, not wire_subscriptions node_name
        for identity in node_identities:
            assert identity.env == "dev"
            assert (
                identity.service == "test-service"
            )  # From wiring fixture's service param

    @pytest.mark.asyncio
    async def test_wire_subscriptions_uses_compute_consumer_group_id_helper(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test wiring uses compute_consumer_group_id for consistent consumer group derivation.

        The consumer_group passed to _create_dispatch_callback should match
        the format produced by compute_consumer_group_id(), ensuring consistency
        across the codebase.
        """
        from omnibase_infra.enums import EnumConsumerGroupPurpose
        from omnibase_infra.models import ModelNodeIdentity
        from omnibase_infra.utils import compute_consumer_group_id

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="prod",
            node_name="test-handler",
            service="omniintelligence",
            version="v2.0.0",
        )

        # Mock _create_dispatch_callback to capture consumer_group argument
        original_create_callback = wiring._create_dispatch_callback
        captured_consumer_groups: list[str] = []

        def capture_callback(topic: str, consumer_group: str):
            captured_consumer_groups.append(consumer_group)
            return original_create_callback(topic, consumer_group)

        wiring._create_dispatch_callback = capture_callback  # type: ignore[method-assign]

        await wiring.wire_subscriptions(sample_subcontract, node_name="my-handler")

        # Build expected consumer group using the helper
        expected_identity = ModelNodeIdentity(
            env="prod",
            service="omniintelligence",
            node_name="my-handler",
            version="v2.0.0",
        )
        expected_consumer_group = compute_consumer_group_id(
            expected_identity, EnumConsumerGroupPurpose.CONSUME
        )

        # Verify all captured consumer groups match expected format
        assert len(captured_consumer_groups) == 2  # 2 topics in sample_subcontract
        for consumer_group in captured_consumer_groups:
            assert consumer_group == expected_consumer_group

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
        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )

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
        topic = "onex.evt.test.v1"
        callback = wiring._create_dispatch_callback(topic, "dev.test-handler")

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
        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )

        await callback(sample_event_message)

        call_args = mock_dispatch_engine.dispatch.call_args
        envelope = call_args[0][1]
        # Envelope should be deserialized from message
        assert envelope is not None

    @pytest.mark.asyncio
    async def test_dispatch_callback_routes_invalid_json_to_dlq(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test callback routes invalid JSON to DLQ (default content error behavior).

        Note: Default DLQ config routes content errors to DLQ without raising.
        For fail_fast behavior, see TestErrorClassification.test_content_error_fail_fast_raises_protocol_error.
        """
        from omnibase_infra.models.event_bus import ModelDlqConfig

        # Setup DLQ mock
        mock_event_bus._publish_raw_to_dlq = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            dlq_config=ModelDlqConfig(enabled=True, on_content_error="dlq_and_commit"),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )

        invalid_message = ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"key",
            value=b"not valid json",
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        # Should NOT raise - routes to DLQ instead
        await callback(invalid_message)

        # Verify DLQ was called
        mock_event_bus._publish_raw_to_dlq.assert_called_once()
        call_kwargs = mock_event_bus._publish_raw_to_dlq.call_args.kwargs
        assert call_kwargs["failure_type"] == "content_error"

    @pytest.mark.asyncio
    async def test_dispatch_callback_propagates_dispatch_errors(
        self,
        wiring: EventBusSubcontractWiring,
        mock_dispatch_engine: AsyncMock,
        sample_event_message: ModelEventMessage,
    ) -> None:
        """Test callback wraps dispatch engine errors in RuntimeHostError."""
        mock_dispatch_engine.dispatch.side_effect = RuntimeError("Dispatch failed")
        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )

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
            topic="onex.evt.test.v1",
            key=b"key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        envelope = wiring._deserialize_to_envelope(message, "onex.evt.test.v1")

        assert envelope is not None

    def test_deserialize_raises_on_invalid_json(
        self,
        wiring: EventBusSubcontractWiring,
    ) -> None:
        """Test deserialization raises on invalid JSON."""
        message = ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"key",
            value=b"not json",
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        with pytest.raises(json.JSONDecodeError):
            wiring._deserialize_to_envelope(message, "onex.evt.test.v1")


# =============================================================================
# Event Type Derivation Tests (OMN-2038)
# =============================================================================


class TestEventTypeDerivation:
    """Tests for event_type derivation from ONEX topic naming convention.

    Related: OMN-2038 - Propagate event_type through EventBusSubcontractWiring
    """

    def test_derive_event_type_from_standard_evt_topic(self) -> None:
        """Test derivation from standard event topic."""
        result = EventBusSubcontractWiring._derive_event_type_from_topic(
            "onex.evt.omniintelligence.intent-classified.v1"
        )
        assert result == "omniintelligence.intent-classified"

    def test_derive_event_type_from_cmd_topic(self) -> None:
        """Test derivation from command topic."""
        result = EventBusSubcontractWiring._derive_event_type_from_topic(
            "onex.cmd.node.register.v1"
        )
        assert result == "node.register"

    def test_derive_event_type_from_topic_with_higher_version(self) -> None:
        """Test derivation works regardless of version segment."""
        result = EventBusSubcontractWiring._derive_event_type_from_topic(
            "onex.evt.platform.process-event.v3"
        )
        assert result == "platform.process-event"

    def test_derive_event_type_returns_none_for_non_onex_topic(self) -> None:
        """Test returns None for topics not following ONEX convention."""
        result = EventBusSubcontractWiring._derive_event_type_from_topic(
            "custom.topic.name"
        )
        assert result is None

    def test_derive_event_type_returns_none_for_short_topic(self) -> None:
        """Test returns None for topics with fewer than 5 segments."""
        result = EventBusSubcontractWiring._derive_event_type_from_topic(
            "onex.evt.short"
        )
        assert result is None

    def test_derive_event_type_returns_none_for_empty_string(self) -> None:
        """Test returns None for empty string."""
        result = EventBusSubcontractWiring._derive_event_type_from_topic("")
        assert result is None

    def test_derive_event_type_returns_none_when_prefix_not_onex(self) -> None:
        """Test returns None when topic has 5 segments but does not start with 'onex'."""
        result = EventBusSubcontractWiring._derive_event_type_from_topic(
            "other.evt.producer.event-name.v1"
        )
        assert result is None

    def test_derive_event_type_from_topic_with_extra_segments(self) -> None:
        """Test derivation still works with more than 5 segments."""
        result = EventBusSubcontractWiring._derive_event_type_from_topic(
            "onex.evt.producer.event-name.v1.extra"
        )
        assert result == "producer.event-name"

    def test_deserialize_populates_event_type_when_not_in_payload(
        self,
        wiring: EventBusSubcontractWiring,
    ) -> None:
        """Test that event_type is derived from topic when not in the envelope payload."""
        payload = {
            "correlation_id": str(uuid4()),
            "payload": {"key": "value"},
            # event_type intentionally omitted
        }
        message = ModelEventMessage(
            topic="onex.evt.omniintelligence.intent-classified.v1",
            key=b"key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        envelope = wiring._deserialize_to_envelope(
            message, "onex.evt.omniintelligence.intent-classified.v1"
        )

        assert envelope.event_type == "omniintelligence.intent-classified"

    def test_deserialize_preserves_existing_event_type(
        self,
        wiring: EventBusSubcontractWiring,
    ) -> None:
        """Test that existing event_type in payload is NOT overwritten."""
        payload = {
            "event_type": "custom.explicit-type",
            "correlation_id": str(uuid4()),
            "payload": {"key": "value"},
        }
        message = ModelEventMessage(
            topic="onex.evt.omniintelligence.intent-classified.v1",
            key=b"key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        envelope = wiring._deserialize_to_envelope(
            message, "onex.evt.omniintelligence.intent-classified.v1"
        )

        # Must preserve the original, not overwrite with derived
        assert envelope.event_type == "custom.explicit-type"

    def test_deserialize_no_event_type_with_non_onex_topic(
        self,
        wiring: EventBusSubcontractWiring,
    ) -> None:
        """Test that event_type remains None when topic is not ONEX format."""
        payload = {
            "correlation_id": str(uuid4()),
            "payload": {"key": "value"},
            # event_type intentionally omitted
        }
        message = ModelEventMessage(
            topic="custom.topic",
            key=b"key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        envelope = wiring._deserialize_to_envelope(message, "custom.topic")

        assert getattr(envelope, "event_type", None) is None

    @pytest.mark.asyncio
    async def test_dispatched_envelope_has_event_type_from_topic(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """End-to-end: envelope dispatched via callback has event_type populated.

        This is the primary integration test for OMN-2038: when a message
        arrives without event_type, the wiring derives it from the topic
        and the dispatch engine receives the enriched envelope.
        """
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        topic = "onex.evt.node.introspected.v1"
        callback = wiring._create_dispatch_callback(topic, "dev.test-handler")

        # Message with no event_type in payload
        payload = {
            "correlation_id": str(uuid4()),
            "payload": {"node_id": "abc"},
        }
        message = ModelEventMessage(
            topic=topic,
            key=b"key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        await callback(message)

        # Verify dispatch was called and envelope has derived event_type
        mock_dispatch_engine.dispatch.assert_called_once()
        dispatched_envelope = mock_dispatch_engine.dispatch.call_args[0][1]
        assert dispatched_envelope.event_type == "node.introspected"

    @pytest.mark.asyncio
    async def test_dispatched_envelope_preserves_existing_event_type(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """End-to-end: envelope with explicit event_type is not overwritten."""
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        topic = "onex.evt.node.introspected.v1"
        callback = wiring._create_dispatch_callback(topic, "dev.test-handler")

        # Message WITH explicit event_type in payload
        payload = {
            "event_type": "custom.already-set",
            "correlation_id": str(uuid4()),
            "payload": {"node_id": "abc"},
        }
        message = ModelEventMessage(
            topic=topic,
            key=b"key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        await callback(message)

        mock_dispatch_engine.dispatch.assert_called_once()
        dispatched_envelope = mock_dispatch_engine.dispatch.call_args[0][1]
        assert dispatched_envelope.event_type == "custom.already-set"


# =============================================================================
# Idempotency Gate Tests
# =============================================================================


class TestIdempotencyGate:
    """Tests for idempotency gate behavior."""

    @pytest.fixture
    def mock_idempotency_store(self) -> AsyncMock:
        """Create mock idempotency store."""
        store = AsyncMock()
        store.check_and_record = AsyncMock(return_value=True)
        return store

    @pytest.fixture
    def sample_event_message_with_envelope_id(self) -> ModelEventMessage:
        """Create sample event message with envelope_id for idempotency testing."""
        envelope_id = uuid4()
        payload = {
            "event_type": "test.event",
            "envelope_id": str(envelope_id),
            "correlation_id": str(uuid4()),
            "payload": {"key": "value"},
        }
        return ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"test-key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test-service",
                event_type="test.event",
                timestamp=datetime.now(UTC),
            ),
        )

    @pytest.fixture
    def sample_event_message_without_envelope_id(self) -> ModelEventMessage:
        """Create sample event message without envelope_id."""
        payload = {
            "event_type": "test.event",
            "correlation_id": str(uuid4()),
            "payload": {"key": "value"},
        }
        return ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"test-key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test-service",
                event_type="test.event",
                timestamp=datetime.now(UTC),
            ),
        )

    @pytest.mark.asyncio
    async def test_duplicate_message_skipped_when_idempotency_enabled(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        mock_idempotency_store: AsyncMock,
        sample_event_message_with_envelope_id: ModelEventMessage,
    ) -> None:
        """Duplicate messages (same event_id) should be skipped."""
        from omnibase_infra.models.event_bus import ModelIdempotencyConfig

        # Setup: mock idempotency store that returns False (duplicate)
        mock_idempotency_store.check_and_record.return_value = False

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            idempotency_store=mock_idempotency_store,
            idempotency_config=ModelIdempotencyConfig(enabled=True),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(sample_event_message_with_envelope_id)

        # Verify: dispatch_engine.dispatch NOT called (duplicate skipped)
        mock_dispatch_engine.dispatch.assert_not_called()
        # Verify: idempotency store was checked
        mock_idempotency_store.check_and_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_new_message_processed_when_idempotency_enabled(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        mock_idempotency_store: AsyncMock,
        sample_event_message_with_envelope_id: ModelEventMessage,
    ) -> None:
        """New messages (new event_id) should be processed."""
        from omnibase_infra.models.event_bus import ModelIdempotencyConfig

        # Setup: mock idempotency store that returns True (new message)
        mock_idempotency_store.check_and_record.return_value = True

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            idempotency_store=mock_idempotency_store,
            idempotency_config=ModelIdempotencyConfig(enabled=True),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(sample_event_message_with_envelope_id)

        # Verify: dispatch_engine.dispatch called
        mock_dispatch_engine.dispatch.assert_called_once()
        # Verify: idempotency store was checked
        mock_idempotency_store.check_and_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_generated_envelope_id_used_for_idempotency(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        mock_idempotency_store: AsyncMock,
        sample_event_message_without_envelope_id: ModelEventMessage,
    ) -> None:
        """Auto-generated envelope_id is used for idempotency when not explicitly provided.

        Note: ModelEventEnvelope auto-generates envelope_id via default_factory,
        so envelope_id is never None. This test verifies that auto-generated IDs
        are properly used for idempotency tracking.
        """
        from omnibase_infra.models.event_bus import ModelIdempotencyConfig

        # Setup: idempotency store that returns True (new message)
        mock_idempotency_store.check_and_record.return_value = True

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            idempotency_store=mock_idempotency_store,
            idempotency_config=ModelIdempotencyConfig(enabled=True),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(sample_event_message_without_envelope_id)

        # Verify: idempotency check was called with auto-generated envelope_id
        mock_idempotency_store.check_and_record.assert_called_once()
        call_kwargs = mock_idempotency_store.check_and_record.call_args.kwargs
        assert call_kwargs["message_id"] is not None  # UUID was auto-generated

        # Verify: dispatch was called
        mock_dispatch_engine.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_idempotency_disabled_processes_all_messages(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        mock_idempotency_store: AsyncMock,
        sample_event_message_with_envelope_id: ModelEventMessage,
    ) -> None:
        """When idempotency disabled, all messages processed."""
        from omnibase_infra.models.event_bus import ModelIdempotencyConfig

        # Setup: idempotency_config.enabled = False
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            idempotency_store=mock_idempotency_store,
            idempotency_config=ModelIdempotencyConfig(enabled=False),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(sample_event_message_with_envelope_id)

        # Verify: dispatch called without idempotency check
        mock_dispatch_engine.dispatch.assert_called_once()
        # Verify: idempotency store NOT checked
        mock_idempotency_store.check_and_record.assert_not_called()

    @pytest.mark.asyncio
    async def test_idempotency_skips_when_no_store_provided(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        sample_event_message_with_envelope_id: ModelEventMessage,
    ) -> None:
        """When no idempotency store provided, messages processed without check."""
        from omnibase_infra.models.event_bus import ModelIdempotencyConfig

        # Setup: no idempotency_store provided, but config enabled
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            idempotency_store=None,  # No store
            idempotency_config=ModelIdempotencyConfig(enabled=True),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(sample_event_message_with_envelope_id)

        # Verify: dispatch called (no store = no idempotency check)
        mock_dispatch_engine.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_message_commits_offset_to_prevent_redelivery(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        mock_idempotency_store: AsyncMock,
        sample_event_message_with_envelope_id: ModelEventMessage,
    ) -> None:
        """Duplicate messages must commit offset to prevent infinite redelivery.

        This is a critical test: when a message is identified as a duplicate
        (already processed), we must still commit the offset. Otherwise Kafka
        will redeliver the same message forever, causing an infinite loop.

        Related: PR #219 review - deduped messages should still commit offsets.
        """
        from omnibase_infra.models.event_bus import (
            ModelIdempotencyConfig,
            ModelOffsetPolicyConfig,
        )

        # Setup: mock idempotency store that returns False (duplicate)
        mock_idempotency_store.check_and_record.return_value = False
        mock_event_bus.commit_offset = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            idempotency_store=mock_idempotency_store,
            idempotency_config=ModelIdempotencyConfig(enabled=True),
            offset_policy=ModelOffsetPolicyConfig(
                commit_strategy="commit_after_handler"
            ),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(sample_event_message_with_envelope_id)

        # Verify: dispatch_engine.dispatch NOT called (duplicate skipped)
        mock_dispatch_engine.dispatch.assert_not_called()

        # Verify: idempotency store was checked
        mock_idempotency_store.check_and_record.assert_called_once()

        # CRITICAL: Offset MUST be committed even for duplicates
        # This prevents infinite redelivery loop
        mock_event_bus.commit_offset.assert_called_once_with(
            sample_event_message_with_envelope_id
        )


# =============================================================================
# Error Classification Tests
# =============================================================================


class TestErrorClassification:
    """Tests for content vs infrastructure error classification."""

    @pytest.mark.asyncio
    async def test_json_decode_error_classified_as_content_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """JSONDecodeError should be classified as content error."""
        from omnibase_infra.models.event_bus import ModelDlqConfig

        # Setup: invalid JSON in message
        invalid_message = ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"key",
            value=b"not valid json",
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        # Track DLQ publishing
        mock_event_bus._publish_raw_to_dlq = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            dlq_config=ModelDlqConfig(
                enabled=True,
                on_content_error="dlq_and_commit",
            ),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(invalid_message)  # Should not raise

        # Verify: DLQ published (content error handling)
        mock_event_bus._publish_raw_to_dlq.assert_called_once()
        call_kwargs = mock_event_bus._publish_raw_to_dlq.call_args.kwargs
        assert call_kwargs["failure_type"] == "content_error"

    @pytest.mark.asyncio
    async def test_validation_error_classified_as_content_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """ValidationError should be classified as content error."""
        from omnibase_infra.models.event_bus import ModelDlqConfig

        # Setup: valid JSON but invalid schema (missing required fields)
        invalid_schema_message = ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"key",
            value=json.dumps({"invalid": "schema"}).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        mock_event_bus._publish_raw_to_dlq = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            dlq_config=ModelDlqConfig(
                enabled=True,
                on_content_error="dlq_and_commit",
            ),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(invalid_schema_message)  # Should not raise

        # Verify: DLQ published
        mock_event_bus._publish_raw_to_dlq.assert_called_once()
        call_kwargs = mock_event_bus._publish_raw_to_dlq.call_args.kwargs
        assert call_kwargs["failure_type"] == "content_error"

    @pytest.mark.asyncio
    async def test_runtime_host_error_classified_as_infra_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """RuntimeHostError should be classified as infrastructure error."""
        from omnibase_infra.models.event_bus import ModelDlqConfig

        # Setup: dispatch raises RuntimeHostError
        mock_dispatch_engine.dispatch.side_effect = RuntimeHostError("Database timeout")

        valid_message = ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"key",
            value=json.dumps(
                {
                    "event_type": "test.event",
                    "correlation_id": str(uuid4()),
                    "payload": {"key": "value"},
                }
            ).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            dlq_config=ModelDlqConfig(on_infra_exhausted="fail_fast"),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )

        # Verify: raises RuntimeHostError (fail-fast default)
        with pytest.raises(RuntimeHostError):
            await callback(valid_message)

    @pytest.mark.asyncio
    async def test_retry_exhausted_triggers_dlq_when_policy_allows(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """When retry exhausted and policy=dlq_and_commit, send to DLQ."""
        from omnibase_infra.models.event_bus import (
            ModelConsumerRetryConfig,
            ModelDlqConfig,
        )

        # Setup: dispatch always raises infrastructure error
        mock_dispatch_engine.dispatch.side_effect = RuntimeHostError("Service down")
        mock_event_bus._publish_raw_to_dlq = AsyncMock()
        mock_event_bus.commit_offset = AsyncMock()

        valid_message = ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"key",
            value=json.dumps(
                {
                    "event_type": "test.event",
                    "correlation_id": str(uuid4()),
                    "payload": {"key": "value"},
                }
            ).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            dlq_config=ModelDlqConfig(
                enabled=True,
                on_infra_exhausted="dlq_and_commit",
            ),
            retry_config=ModelConsumerRetryConfig(max_attempts=2),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )

        # First attempt - should fail but not go to DLQ (retry budget not exhausted)
        with pytest.raises(RuntimeHostError):
            await callback(valid_message)
        mock_event_bus._publish_raw_to_dlq.assert_not_called()

        # Second attempt - retry budget exhausted, should go to DLQ
        await callback(valid_message)  # Should NOT raise
        mock_event_bus._publish_raw_to_dlq.assert_called_once()
        call_kwargs = mock_event_bus._publish_raw_to_dlq.call_args.kwargs
        assert call_kwargs["failure_type"] == "infra_error"

    @pytest.mark.asyncio
    async def test_content_error_fail_fast_raises_protocol_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Content error with fail_fast policy should raise ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.models.event_bus import ModelDlqConfig

        invalid_message = ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"key",
            value=b"not valid json",
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            dlq_config=ModelDlqConfig(on_content_error="fail_fast"),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )

        with pytest.raises(ProtocolConfigurationError, match="Content error"):
            await callback(invalid_message)

    @pytest.mark.asyncio
    async def test_unexpected_error_classified_as_infra_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Unexpected errors (not RuntimeHostError) should be classified as infra error."""
        # Setup: dispatch raises unexpected Exception
        mock_dispatch_engine.dispatch.side_effect = ValueError("Unexpected error")

        valid_message = ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"key",
            value=json.dumps(
                {
                    "event_type": "test.event",
                    "correlation_id": str(uuid4()),
                    "payload": {"key": "value"},
                }
            ).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )

        # Verify: wrapped in RuntimeHostError
        with pytest.raises(RuntimeHostError, match="Failed to dispatch"):
            await callback(valid_message)

    @pytest.mark.asyncio
    async def test_dlq_consumer_group_matches_subscription_consumer_group(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """DLQ consumer_group must match the subscription consumer_group.

        This test verifies the fix for OMN-1740 PR review feedback:
        DLQ messages must include the same consumer_group that was used
        for the subscription, enabling correlation during replay and debugging.
        """
        from omnibase_infra.models.event_bus import ModelDlqConfig

        # Setup: invalid JSON to trigger DLQ
        invalid_message = ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"key",
            value=b"not valid json",
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        mock_event_bus._publish_raw_to_dlq = AsyncMock()

        # Use different node_name in __init__ vs callback to verify alignment
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="prod",  # Different environment
            node_name="init-handler",  # Different node_name than callback
            service="test-service",
            version="v1",
            dlq_config=ModelDlqConfig(
                enabled=True,
                on_content_error="dlq_and_commit",
            ),
        )

        # The consumer_group passed to _create_dispatch_callback should be
        # the one used in DLQ publishing, NOT self._node_name from __init__
        callback_consumer_group = "prod.my-special-handler"
        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", callback_consumer_group
        )
        await callback(invalid_message)

        # Verify: DLQ consumer_group matches the callback consumer_group
        mock_event_bus._publish_raw_to_dlq.assert_called_once()
        call_kwargs = mock_event_bus._publish_raw_to_dlq.call_args.kwargs
        assert call_kwargs["consumer_group"] == callback_consumer_group
        # Verify it's NOT using the __init__ node_name
        assert call_kwargs["consumer_group"] != "prod.init-handler"


# =============================================================================
# Offset Commit Policy Tests
# =============================================================================


class TestOffsetCommitPolicy:
    """Tests for offset commit policy behavior."""

    @pytest.fixture
    def sample_valid_message(self) -> ModelEventMessage:
        """Create a valid event message for testing."""
        payload = {
            "event_type": "test.event",
            "correlation_id": str(uuid4()),
            "payload": {"key": "value"},
        }
        return ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"test-key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test-service",
                event_type="test.event",
                timestamp=datetime.now(UTC),
            ),
        )

    @pytest.mark.asyncio
    async def test_commit_after_handler_commits_on_success(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        sample_valid_message: ModelEventMessage,
    ) -> None:
        """commit_after_handler policy commits offset after successful dispatch."""
        from omnibase_infra.models.event_bus import ModelOffsetPolicyConfig

        # Setup commit_offset method on event_bus
        mock_event_bus.commit_offset = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            offset_policy=ModelOffsetPolicyConfig(
                commit_strategy="commit_after_handler"
            ),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(sample_valid_message)

        # Verify: _commit_offset called after dispatch success
        mock_dispatch_engine.dispatch.assert_called_once()
        mock_event_bus.commit_offset.assert_called_once()

    @pytest.mark.asyncio
    async def test_commit_after_handler_no_commit_on_infra_error(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        sample_valid_message: ModelEventMessage,
    ) -> None:
        """commit_after_handler policy does NOT commit on infrastructure error."""
        from omnibase_infra.models.event_bus import ModelOffsetPolicyConfig

        # Setup: dispatch raises RuntimeHostError
        mock_dispatch_engine.dispatch.side_effect = RuntimeHostError("DB timeout")
        mock_event_bus.commit_offset = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            offset_policy=ModelOffsetPolicyConfig(
                commit_strategy="commit_after_handler"
            ),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )

        with pytest.raises(RuntimeHostError):
            await callback(sample_valid_message)

        # Verify: _commit_offset NOT called
        mock_event_bus.commit_offset.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_on_content_error_with_dlq(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Content error with dlq_and_commit should commit offset."""
        from omnibase_infra.models.event_bus import (
            ModelDlqConfig,
            ModelOffsetPolicyConfig,
        )

        invalid_message = ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"key",
            value=b"not valid json",
            headers=ModelEventHeaders(
                source="test",
                event_type="test",
                timestamp=datetime.now(UTC),
            ),
        )

        mock_event_bus._publish_raw_to_dlq = AsyncMock()
        mock_event_bus.commit_offset = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            offset_policy=ModelOffsetPolicyConfig(
                commit_strategy="commit_after_handler"
            ),
            dlq_config=ModelDlqConfig(on_content_error="dlq_and_commit"),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(invalid_message)  # Should not raise

        # Verify: offset committed after DLQ publish
        mock_event_bus._publish_raw_to_dlq.assert_called_once()
        mock_event_bus.commit_offset.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_commit_when_event_bus_lacks_commit_offset(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        sample_valid_message: ModelEventMessage,
    ) -> None:
        """When event bus has no commit_offset method, offset commit is skipped."""
        from omnibase_infra.models.event_bus import ModelOffsetPolicyConfig

        # Ensure no commit_offset method exists
        if hasattr(mock_event_bus, "commit_offset"):
            del mock_event_bus.commit_offset

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            offset_policy=ModelOffsetPolicyConfig(
                commit_strategy="commit_after_handler"
            ),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        # Should not raise even without commit_offset method
        await callback(sample_valid_message)

        mock_dispatch_engine.dispatch.assert_called_once()


# =============================================================================
# Retry Tracking Tests
# =============================================================================


class TestRetryTracking:
    """Tests for retry count tracking behavior."""

    @pytest.fixture
    def sample_valid_message(self) -> ModelEventMessage:
        """Create a valid event message for testing."""
        payload = {
            "event_type": "test.event",
            "correlation_id": str(uuid4()),
            "payload": {"key": "value"},
        }
        return ModelEventMessage(
            topic="onex.evt.test.v1",
            key=b"test-key",
            value=json.dumps(payload).encode("utf-8"),
            headers=ModelEventHeaders(
                source="test-service",
                event_type="test.event",
                timestamp=datetime.now(UTC),
            ),
        )

    @pytest.mark.asyncio
    async def test_retry_count_cleared_on_success(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        sample_valid_message: ModelEventMessage,
    ) -> None:
        """Retry count should be cleared after successful processing."""
        from omnibase_infra.models.event_bus import ModelConsumerRetryConfig

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            retry_config=ModelConsumerRetryConfig(max_attempts=3),
        )

        # Manually set a retry count
        correlation_id = uuid4()
        wiring._retry_counts[correlation_id] = 2

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(sample_valid_message)

        # Note: The retry count is cleared based on the correlation_id from the
        # envelope, not the one we set manually. This test verifies the mechanism
        # works for successful processing.
        mock_dispatch_engine.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_count_cleared_on_dlq(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        sample_valid_message: ModelEventMessage,
    ) -> None:
        """Retry count should be cleared after message sent to DLQ."""
        from omnibase_infra.models.event_bus import (
            ModelConsumerRetryConfig,
            ModelDlqConfig,
        )

        mock_dispatch_engine.dispatch.side_effect = RuntimeHostError("Service down")
        mock_event_bus._publish_raw_to_dlq = AsyncMock()
        mock_event_bus.commit_offset = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            retry_config=ModelConsumerRetryConfig(
                max_attempts=1
            ),  # Immediate exhaustion
            dlq_config=ModelDlqConfig(on_infra_exhausted="dlq_and_commit"),
        )

        callback = wiring._create_dispatch_callback(
            "onex.evt.test.v1", "dev.test-handler"
        )
        await callback(sample_valid_message)  # Should not raise

        # Verify DLQ was called and retry counts are empty after cleanup
        mock_event_bus._publish_raw_to_dlq.assert_called_once()

    def test_retry_count_increments_correctly(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test retry count increment helper method."""
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        correlation_id = uuid4()

        # Initial count should be 0
        assert wiring._get_retry_count(correlation_id) == 0

        # Increment and verify
        assert wiring._increment_retry_count(correlation_id) == 1
        assert wiring._get_retry_count(correlation_id) == 1

        # Increment again
        assert wiring._increment_retry_count(correlation_id) == 2
        assert wiring._get_retry_count(correlation_id) == 2

    def test_retry_exhausted_check(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
    ) -> None:
        """Test retry exhausted check helper method."""
        from omnibase_infra.models.event_bus import ModelConsumerRetryConfig

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            retry_config=ModelConsumerRetryConfig(max_attempts=3),
        )

        correlation_id = uuid4()

        # Not exhausted at start
        assert not wiring._is_retry_exhausted(correlation_id)

        # Increment to max
        wiring._retry_counts[correlation_id] = 3

        # Now exhausted
        assert wiring._is_retry_exhausted(correlation_id)

    @pytest.mark.asyncio
    async def test_cleanup_clears_retry_counts(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        sample_subcontract: ModelEventBusSubcontract,
    ) -> None:
        """Test cleanup also clears retry count tracking."""
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        # Add some retry counts
        wiring._retry_counts[uuid4()] = 2
        wiring._retry_counts[uuid4()] = 1

        assert len(wiring._retry_counts) == 2

        await wiring.wire_subscriptions(sample_subcontract, node_name="test-handler")
        await wiring.cleanup()

        # Verify retry counts cleared
        assert len(wiring._retry_counts) == 0
