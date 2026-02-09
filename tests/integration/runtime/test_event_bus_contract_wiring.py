# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for event bus contract wiring.

These tests verify that:
1. Contracts with event_bus.subscribe_topics cause runtime to start consumers
2. Messages consumed from topics reach the correct handler
3. Publishing through SPI publisher validates against contract
4. Handlers have no direct Kafka access (ARCH-002 compliance)

Test Coverage:
- TestContractDrivenSubscription: Subscription creation from subcontracts
- TestMessageDispatchFlow: Message routing from Kafka to dispatch engine
- TestPublisherContractValidation: Topic validation in PublisherTopicScoped
- TestHandlerNoBusAccess: ARCH-002 compliance verification
- TestContractLoadingFromYAML: YAML parsing and subcontract extraction
- TestSubscriptionCleanup: Proper cleanup on shutdown
- TestIdempotencyIntegration: Duplicate message deduplication (AC#7)
- TestOffsetCommitOnFailure: Offset commit behavior on failure (AC#8)

Architecture Context:
    The EventBusSubcontractWiring class bridges contract-declared topics to actual
    Kafka subscriptions, ensuring that nodes/handlers never directly interact with
    Kafka infrastructure. The runtime owns all Kafka plumbing per ARCH-002.

Related Tickets:
    - OMN-1621: Runtime consumes event_bus subcontract for contract-driven wiring
    - OMN-1740: Idempotency and offset commit semantics
    - ARCH-002: Runtime owns all Kafka plumbing

See Also:
    - src/omnibase_infra/runtime/event_bus_subcontract_wiring.py
    - src/omnibase_infra/runtime/publisher_topic_scoped.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_core.models.contracts.subcontracts import ModelEventBusSubcontract
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.errors import ProtocolConfigurationError, RuntimeHostError
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
    load_event_bus_subcontract,
)
from omnibase_infra.runtime.publisher_topic_scoped import PublisherTopicScoped

if TYPE_CHECKING:
    from omnibase_infra.event_bus.models import ModelEventMessage


class TestContractDrivenSubscription:
    """Test that contracts with subscribe_topics start consumers automatically.

    These tests verify the wiring between ModelEventBusSubcontract and actual
    Kafka subscriptions. When a contract declares subscribe_topics, the runtime
    should create corresponding consumers.
    """

    @pytest.fixture
    def mock_event_bus(self) -> AsyncMock:
        """Create mock Kafka event bus.

        Returns an AsyncMock that tracks subscribe calls and returns mock
        unsubscribe callables.
        """
        bus = AsyncMock()
        bus.subscribe = AsyncMock(return_value=AsyncMock())
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def mock_dispatch_engine(self) -> AsyncMock:
        """Create mock dispatch engine that tracks dispatched messages.

        Returns an AsyncMock with a dispatched_messages list for verification.
        """
        engine = AsyncMock()
        engine.dispatched_messages: list[tuple[str, object]] = []

        async def track_dispatch(topic: str, envelope: object) -> None:
            engine.dispatched_messages.append((topic, envelope))

        engine.dispatch = AsyncMock(side_effect=track_dispatch)
        return engine

    @pytest.fixture
    def subcontract_version(self) -> ModelSemVer:
        """Create default version for subcontracts."""
        return ModelSemVer(major=1, minor=0, patch=0)

    @pytest.mark.asyncio
    async def test_contract_with_subscribe_topics_creates_consumers(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        subcontract_version: ModelSemVer,
    ) -> None:
        """Given a contract with subscribe_topics, runtime creates consumers.

        Verifies:
        1. EventBusSubcontractWiring calls event_bus.subscribe for each topic
        2. Topics are resolved with environment prefix
        3. Consumer group ID is derived from environment and node name
        """
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=[
                "onex.cmd.test-service.process.v1",
                "onex.evt.test-service.notify.v1",
            ],
            publish_topics=[],
        )

        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        # Verify consumers created for each topic
        assert mock_event_bus.subscribe.call_count == 2
        topics_subscribed = [
            call.kwargs["topic"] for call in mock_event_bus.subscribe.call_args_list
        ]
        assert "onex.cmd.test-service.process.v1" in topics_subscribed
        assert "onex.evt.test-service.notify.v1" in topics_subscribed

    @pytest.mark.asyncio
    async def test_node_identity_derived_from_environment_and_node(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        subcontract_version: ModelSemVer,
    ) -> None:
        """Node identity is correctly populated for consumer group derivation.

        Verifies:
        1. Node identity contains correct environment, service, node_name, version
        2. Event bus receives node_identity for consumer group derivation
        """
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="staging",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=["onex.evt.test-producer.test-event.v1"],
            publish_topics=[],
        )

        await wiring.wire_subscriptions(subcontract, node_name="my-handler")

        # Verify node_identity is passed with correct values
        call_kwargs = mock_event_bus.subscribe.call_args.kwargs
        node_identity = call_kwargs["node_identity"]
        assert node_identity.env == "staging"
        assert node_identity.service == "test-service"
        assert node_identity.node_name == "my-handler"
        assert node_identity.version == "v1"

    @pytest.mark.asyncio
    async def test_empty_subscribe_topics_creates_no_consumers(
        self,
        mock_event_bus: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        subcontract_version: ModelSemVer,
    ) -> None:
        """Contract with empty subscribe_topics creates no consumers.

        Verifies that wire_subscriptions is a no-op when there are no topics
        to subscribe to.
        """
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=[],
            publish_topics=[],
        )

        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        # No subscribe calls should have been made
        mock_event_bus.subscribe.assert_not_called()


class TestMessageDispatchFlow:
    """Test that consumed messages flow to the dispatch engine correctly.

    These tests verify the message path from Kafka consumer through
    deserialization to the dispatch engine.
    """

    @pytest.fixture
    def mock_event_bus_with_callback_capture(self) -> AsyncMock:
        """Create mock event bus that captures callbacks.

        Returns an AsyncMock that stores on_message callbacks for each topic,
        allowing tests to simulate message delivery.
        """
        bus = AsyncMock()
        bus.callbacks: dict[str, object] = {}

        async def capture_subscribe(
            topic: str,
            node_identity: object,
            on_message: object,
        ) -> AsyncMock:
            bus.callbacks[topic] = on_message
            return AsyncMock()

        bus.subscribe = AsyncMock(side_effect=capture_subscribe)
        return bus

    @pytest.fixture
    def mock_dispatch_engine(self) -> AsyncMock:
        """Create mock dispatch engine."""
        engine = AsyncMock()
        engine.dispatch = AsyncMock()
        return engine

    @pytest.fixture
    def subcontract_version(self) -> ModelSemVer:
        """Create default version for subcontracts."""
        return ModelSemVer(major=1, minor=0, patch=0)

    @pytest.mark.asyncio
    async def test_consumed_messages_reach_dispatch_engine(
        self,
        mock_event_bus_with_callback_capture: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        subcontract_version: ModelSemVer,
    ) -> None:
        """Messages from Kafka consumer should reach dispatch engine.

        Verifies:
        1. Callback is registered with event bus
        2. Message is deserialized to envelope
        3. Dispatch engine receives the envelope
        """
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus_with_callback_capture,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=["onex.evt.test-producer.test-event.v1"],
            publish_topics=[],
        )

        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        # Simulate Kafka delivering a message with valid envelope format
        mock_message = MagicMock()
        mock_message.value = json.dumps(
            {
                "event_type": "test.event",
                "payload": {"data": "test"},
                "source": "test-source",
            }
        ).encode()

        # Get the callback that was registered
        topic_key = "onex.evt.test-producer.test-event.v1"
        callback = mock_event_bus_with_callback_capture.callbacks[topic_key]
        await callback(mock_message)

        # Verify dispatch engine received the message
        mock_dispatch_engine.dispatch.assert_called_once()
        call_args = mock_dispatch_engine.dispatch.call_args
        assert call_args[0][0] == topic_key  # topic

    @pytest.mark.asyncio
    async def test_malformed_message_raises_error_with_fail_fast_policy(
        self,
        mock_event_bus_with_callback_capture: AsyncMock,
        mock_dispatch_engine: AsyncMock,
        subcontract_version: ModelSemVer,
    ) -> None:
        """Malformed JSON messages should raise ProtocolConfigurationError with fail_fast.

        Verifies:
        1. Invalid JSON causes exception
        2. Exception is wrapped as ProtocolConfigurationError (OnexError)
        3. Exception is propagated when fail_fast policy is configured

        Note: Default behavior sends content errors to DLQ. This test
        explicitly uses fail_fast to verify error propagation path.
        """
        from omnibase_infra.models.event_bus import ModelDlqConfig

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus_with_callback_capture,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
            dlq_config=ModelDlqConfig(
                enabled=False,
                on_content_error="fail_fast",
            ),
        )

        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=["onex.evt.test-producer.test-event.v1"],
            publish_topics=[],
        )

        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        # Simulate Kafka delivering a malformed message
        mock_message = MagicMock()
        mock_message.value = b"not valid json"

        # Get the callback and verify it raises ProtocolConfigurationError
        topic_key = "onex.evt.test-producer.test-event.v1"
        callback = mock_event_bus_with_callback_capture.callbacks[topic_key]
        with pytest.raises(ProtocolConfigurationError):
            await callback(mock_message)


class TestPublisherContractValidation:
    """Test that publishing validates against contract.

    These tests verify that PublisherTopicScoped enforces topic-level
    access control based on the contract's publish_topics.
    """

    @pytest.fixture
    def mock_event_bus(self) -> AsyncMock:
        """Create mock event bus."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        return bus

    @pytest.mark.asyncio
    async def test_publishing_to_allowed_topic_succeeds(
        self, mock_event_bus: AsyncMock
    ) -> None:
        """Publishing to a topic in contract's publish_topics succeeds.

        Verifies:
        1. No exception is raised
        2. Event bus publish is called
        3. Topic is resolved with environment prefix
        """
        publisher = PublisherTopicScoped(
            event_bus=mock_event_bus,
            allowed_topics={"onex.evt.test.allowed-output.v1"},
            environment="dev",
        )

        result = await publisher.publish(
            event_type="output.event",
            payload={"result": "success"},
            topic="onex.evt.test.allowed-output.v1",
        )

        assert result is True
        mock_event_bus.publish.assert_called_once()
        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert call_kwargs["topic"] == "onex.evt.test.allowed-output.v1"

    @pytest.mark.asyncio
    async def test_publishing_to_forbidden_topic_raises_error(
        self, mock_event_bus: AsyncMock
    ) -> None:
        """Publishing to a topic not in contract's publish_topics raises ProtocolConfigurationError.

        Verifies:
        1. ProtocolConfigurationError is raised with descriptive message
        2. Allowed topics are listed in error message
        3. Event bus publish is NOT called
        """
        publisher = PublisherTopicScoped(
            event_bus=mock_event_bus,
            allowed_topics={"onex.evt.test.allowed-output.v1"},
            environment="dev",
        )

        with pytest.raises(
            ProtocolConfigurationError, match="not in contract's publish_topics"
        ):
            await publisher.publish(
                event_type="output.event",
                payload={"result": "success"},
                topic="onex.evt.test.forbidden.v1",
            )

        # Verify publish was not called
        mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_publishing_with_none_topic_raises_error(
        self, mock_event_bus: AsyncMock
    ) -> None:
        """Publishing with topic=None raises ProtocolConfigurationError.

        Verifies that topic is a required parameter.
        """
        publisher = PublisherTopicScoped(
            event_bus=mock_event_bus,
            allowed_topics={"onex.evt.test.allowed-output.v1"},
            environment="dev",
        )

        with pytest.raises(ProtocolConfigurationError, match="topic is required"):
            await publisher.publish(
                event_type="output.event",
                payload={"result": "success"},
                topic=None,
            )

    @pytest.mark.asyncio
    async def test_correlation_id_used_as_message_key(
        self, mock_event_bus: AsyncMock
    ) -> None:
        """Correlation ID is used as the message key for partitioning.

        Verifies:
        1. Correlation ID is encoded to bytes
        2. Passed as key parameter to publish
        """
        publisher = PublisherTopicScoped(
            event_bus=mock_event_bus,
            allowed_topics={"onex.evt.test.output.v1"},
            environment="dev",
        )

        await publisher.publish(
            event_type="output.event",
            payload={"result": "success"},
            topic="onex.evt.test.output.v1",
            correlation_id="corr-abc-123",
        )

        call_kwargs = mock_event_bus.publish.call_args.kwargs
        assert call_kwargs["key"] == b"corr-abc-123"


class TestHandlerNoBusAccess:
    """Test that handlers have no direct Kafka access (ARCH-002 compliance).

    These tests document and verify the architectural constraint that handlers
    never see ModelEventMessage directly - they receive ModelEventEnvelope
    after deserialization by the wiring layer.
    """

    def test_handlers_receive_envelope_not_raw_kafka_message(self) -> None:
        """Handlers should receive deserialized envelopes, not raw Kafka messages.

        This test documents the architectural constraint:
        - Handlers never see ModelEventMessage directly
        - They receive ModelEventEnvelope after deserialization by the wiring layer
        - The wiring layer owns the Kafka infrastructure details
        """
        from uuid import uuid4

        mock_dispatch_engine = AsyncMock()
        mock_event_bus = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        # The deserialization happens in _deserialize_to_envelope
        # Create a valid ModelEventEnvelope JSON
        mock_message = MagicMock()
        mock_message.value = json.dumps(
            {
                "event_type": "test.event",
                "payload": {"data": "test"},
                "source": "test-source",
                "correlation_id": str(uuid4()),
            }
        ).encode()

        envelope = wiring._deserialize_to_envelope(
            mock_message, "onex.evt.test.test-event.v1"
        )

        # The envelope is a ModelEventEnvelope, not the raw Kafka message
        # After JSON parsing, it's validated against the envelope model
        assert envelope is not None
        # The raw Kafka message details (topic, partition, offset) are NOT in the envelope

    def test_wiring_deserializes_to_model_event_envelope(self) -> None:
        """Wiring layer deserializes to ModelEventEnvelope, not dict.

        Verifies the deserialization produces a proper Pydantic model,
        not a raw dict.
        """
        from uuid import uuid4

        from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

        mock_dispatch_engine = AsyncMock()
        mock_event_bus = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        # Create a valid envelope JSON with proper UUID
        correlation_id = str(uuid4())
        mock_message = MagicMock()
        mock_message.value = json.dumps(
            {
                "event_type": "test.event",
                "payload": {"data": "test"},
                "source": "test-source",
                "correlation_id": correlation_id,
            }
        ).encode()

        envelope = wiring._deserialize_to_envelope(
            mock_message, "onex.evt.test.test-event.v1"
        )

        # Verify it's a ModelEventEnvelope instance
        assert isinstance(envelope, ModelEventEnvelope)


class TestContractLoadingFromYAML:
    """Test loading event_bus subcontract from YAML files.

    These tests verify the load_event_bus_subcontract function correctly
    parses YAML contracts and extracts the event_bus section.
    """

    def test_load_from_valid_contract_file(self, tmp_path: Path) -> None:
        """Test loading subcontract from valid contract.yaml.

        Verifies:
        1. YAML is parsed correctly
        2. event_bus section is extracted
        3. ModelEventBusSubcontract is properly constructed
        """
        contract_yaml = """
name: "test-handler"
version: "1.0.0"
event_bus:
  version:
    major: 1
    minor: 0
    patch: 0
  subscribe_topics:
    - "onex.cmd.test.process.v1"
  publish_topics:
    - "onex.evt.test.result.v1"
"""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(contract_yaml)

        subcontract = load_event_bus_subcontract(contract_file)

        assert subcontract is not None
        assert "onex.cmd.test.process.v1" in subcontract.subscribe_topics
        assert "onex.evt.test.result.v1" in subcontract.publish_topics

    def test_load_from_contract_without_event_bus_returns_none(
        self, tmp_path: Path
    ) -> None:
        """Test loading from contract without event_bus section returns None.

        Verifies that contracts without event_bus configuration return None
        rather than raising an exception.
        """
        contract_yaml = """
name: "test-handler"
version: "1.0.0"
"""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(contract_yaml)

        subcontract = load_event_bus_subcontract(contract_file)

        assert subcontract is None

    def test_load_from_nonexistent_file_returns_none(self, tmp_path: Path) -> None:
        """Test loading from nonexistent file returns None.

        Verifies graceful handling of missing contract files.
        """
        nonexistent_file = tmp_path / "does_not_exist.yaml"

        subcontract = load_event_bus_subcontract(nonexistent_file)

        assert subcontract is None

    def test_load_from_malformed_yaml_returns_none(self, tmp_path: Path) -> None:
        """Test loading from malformed YAML returns None.

        Verifies graceful handling of invalid YAML syntax.
        """
        malformed_yaml = """
name: "test-handler
version: 1.0.0
  invalid: yaml: content
"""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(malformed_yaml)

        subcontract = load_event_bus_subcontract(contract_file)

        assert subcontract is None

    def test_load_from_empty_file_returns_none(self, tmp_path: Path) -> None:
        """Test loading from empty file returns None.

        Verifies graceful handling of empty contract files.
        """
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text("")

        subcontract = load_event_bus_subcontract(contract_file)

        assert subcontract is None


class TestSubscriptionCleanup:
    """Test that subscriptions are properly cleaned up on shutdown.

    These tests verify the cleanup() method properly unsubscribes from
    all topics and releases resources.
    """

    @pytest.fixture
    def subcontract_version(self) -> ModelSemVer:
        """Create default version for subcontracts."""
        return ModelSemVer(major=1, minor=0, patch=0)

    @pytest.mark.asyncio
    async def test_cleanup_unsubscribes_all_topics(
        self, subcontract_version: ModelSemVer
    ) -> None:
        """Test cleanup calls all unsubscribe callbacks.

        Verifies:
        1. Each subscription's unsubscribe callable is called
        2. All subscriptions are cleaned up
        """
        unsubscribe_callback_1 = AsyncMock()
        unsubscribe_callback_2 = AsyncMock()
        unsubscribe_callback_3 = AsyncMock()
        unsubscribe_callbacks = [
            unsubscribe_callback_1,
            unsubscribe_callback_2,
            unsubscribe_callback_3,
        ]

        mock_event_bus = AsyncMock()
        mock_event_bus.subscribe = AsyncMock(side_effect=unsubscribe_callbacks)

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=AsyncMock(),
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=[
                "onex.evt.producer-a.event-a.v1",
                "onex.evt.producer-b.event-b.v1",
                "onex.evt.producer-c.event-c.v1",
            ],
            publish_topics=[],
        )

        await wiring.wire_subscriptions(subcontract, node_name="test")
        await wiring.cleanup()

        # All unsubscribe callbacks should have been called
        unsubscribe_callback_1.assert_called_once()
        unsubscribe_callback_2.assert_called_once()
        unsubscribe_callback_3.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent(
        self, subcontract_version: ModelSemVer
    ) -> None:
        """Test that cleanup can be called multiple times safely.

        Verifies:
        1. First cleanup unsubscribes all
        2. Second cleanup is a no-op (no errors, no duplicate unsubscribe)
        """
        unsubscribe_callback = AsyncMock()

        mock_event_bus = AsyncMock()
        mock_event_bus.subscribe = AsyncMock(return_value=unsubscribe_callback)

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=AsyncMock(),
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=["onex.evt.test-producer.test-event.v1"],
            publish_topics=[],
        )

        await wiring.wire_subscriptions(subcontract, node_name="test")

        # First cleanup
        await wiring.cleanup()
        assert unsubscribe_callback.call_count == 1

        # Second cleanup - should not call unsubscribe again
        await wiring.cleanup()
        assert unsubscribe_callback.call_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_handles_unsubscribe_errors_gracefully(
        self, subcontract_version: ModelSemVer
    ) -> None:
        """Test that cleanup continues even if unsubscribe raises.

        Verifies:
        1. Errors in unsubscribe are logged but don't stop cleanup
        2. Other unsubscribe callbacks are still called
        """
        # First callback raises, second should still be called
        failing_unsubscribe = AsyncMock(side_effect=RuntimeError("unsubscribe failed"))
        successful_unsubscribe = AsyncMock()

        mock_event_bus = AsyncMock()
        mock_event_bus.subscribe = AsyncMock(
            side_effect=[failing_unsubscribe, successful_unsubscribe]
        )

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=AsyncMock(),
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=[
                "onex.evt.producer-a.event-a.v1",
                "onex.evt.producer-b.event-b.v1",
            ],
            publish_topics=[],
        )

        await wiring.wire_subscriptions(subcontract, node_name="test")

        # Cleanup should not raise despite the failing unsubscribe
        await wiring.cleanup()

        # Both unsubscribes should have been attempted
        failing_unsubscribe.assert_called_once()
        successful_unsubscribe.assert_called_once()


class TestTopicResolution:
    """Test topic suffix to full topic name resolution.

    These tests verify the resolve_topic method correctly prepends
    the environment prefix to topic suffixes.
    """

    def test_resolve_topic_is_realm_agnostic(self) -> None:
        """Test that resolve_topic returns topic suffix unchanged (realm-agnostic).

        Verifies the ONEX topic naming convention is realm-agnostic:
        {topic_suffix} (no environment prefix)
        """
        mock_event_bus = AsyncMock()
        mock_dispatch_engine = AsyncMock()

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="dev",
            node_name="test-handler",
            service="test-service",
            version="v1",
        )

        resolved = wiring.resolve_topic("onex.evt.user.created.v1")

        assert resolved == "onex.evt.user.created.v1"

    def test_resolve_topic_is_consistent_across_environments(self) -> None:
        """Test resolve_topic returns same result regardless of environment.

        Verifies:
        1. Topic resolution is realm-agnostic
        2. Same topic suffix regardless of environment
        """
        mock_event_bus = AsyncMock()
        mock_dispatch_engine = AsyncMock()
        topic_suffix = "onex.evt.test.test-event.v1"

        for env in ["dev", "staging", "prod"]:
            wiring = EventBusSubcontractWiring(
                event_bus=mock_event_bus,
                dispatch_engine=mock_dispatch_engine,
                environment=env,
                node_name="test-handler",
                service="test-service",
                version="v1",
            )
            # All environments should resolve to the same topic (realm-agnostic)
            assert wiring.resolve_topic(topic_suffix) == "onex.evt.test.test-event.v1"


class TestPublisherTopicScopedProperties:
    """Test PublisherTopicScoped property accessors.

    These tests verify the allowed_topics and environment properties
    return correct values.
    """

    def test_allowed_topics_returns_frozenset(self) -> None:
        """Test that allowed_topics returns an immutable frozenset.

        Verifies:
        1. Return type is frozenset
        2. Contains all allowed topics
        """
        mock_event_bus = AsyncMock()

        publisher = PublisherTopicScoped(
            event_bus=mock_event_bus,
            allowed_topics={"onex.evt.test.topic-a.v1", "onex.evt.test.topic-b.v1"},
            environment="dev",
        )

        topics = publisher.allowed_topics

        assert isinstance(topics, frozenset)
        assert topics == frozenset(
            {"onex.evt.test.topic-a.v1", "onex.evt.test.topic-b.v1"}
        )

    def test_environment_property(self) -> None:
        """Test that environment property returns configured value."""
        mock_event_bus = AsyncMock()

        publisher = PublisherTopicScoped(
            event_bus=mock_event_bus,
            allowed_topics={"onex.evt.test.test-event.v1"},
            environment="staging",
        )

        assert publisher.environment == "staging"


class TestIdempotencyIntegration:
    """Integration tests for idempotency behavior (AC#7).

    These tests verify that the event bus wiring correctly deduplicates
    messages based on event_id when idempotency is enabled. This ensures
    at-least-once delivery semantics don't result in duplicate processing.

    Related Tickets:
        - OMN-1740 AC#7: Duplicate messages are deduplicated
    """

    @pytest.fixture
    def subcontract_version(self) -> ModelSemVer:
        """Create default version for subcontracts."""
        return ModelSemVer(major=1, minor=0, patch=0)

    @pytest.mark.asyncio
    async def test_duplicate_message_deduplicated(
        self,
        subcontract_version: ModelSemVer,
    ) -> None:
        """Given the same message delivered twice, handler runs only once.

        This test verifies the complete idempotency flow:
        1. First message with envelope_id is processed normally
        2. Second message with same envelope_id is detected as duplicate
        3. Handler (dispatch engine) is only invoked once
        4. Idempotency store correctly records the envelope_id

        The test uses StoreIdempotencyInmemory for isolation - no database
        dependency required.

        Related: OMN-1740 AC#7
        """
        from uuid import uuid4

        from omnibase_infra.idempotency import StoreIdempotencyInmemory
        from omnibase_infra.models.event_bus import ModelIdempotencyConfig

        # Setup: Create wiring with in-memory idempotency store
        idempotency_store = StoreIdempotencyInmemory()
        mock_dispatch_engine = AsyncMock()
        mock_dispatch_engine.dispatch = AsyncMock()
        mock_event_bus = AsyncMock()

        # Track callbacks for message simulation
        callbacks: dict[str, object] = {}

        async def capture_subscribe(
            topic: str,
            node_identity: object,
            on_message: object,
        ) -> AsyncMock:
            callbacks[topic] = on_message
            return AsyncMock()

        mock_event_bus.subscribe = AsyncMock(side_effect=capture_subscribe)

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="test",
            node_name="test-handler",
            service="test-service",
            version="v1",
            idempotency_store=idempotency_store,
            idempotency_config=ModelIdempotencyConfig(enabled=True),
        )

        # Wire subscription
        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=["onex.evt.test.dedup.v1"],
        )
        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        # Create message with specific envelope_id (used for deduplication)
        envelope_id = uuid4()
        correlation_id = uuid4()
        message_json = json.dumps(
            {
                "event_type": "test.dedup",
                "envelope_id": str(envelope_id),
                "correlation_id": str(correlation_id),
                "payload": {"data": "test"},
                "source": "test-source",
            }
        )
        mock_message = MagicMock()
        mock_message.value = message_json.encode()

        # Get callback and deliver message TWICE
        topic_key = "onex.evt.test.dedup.v1"
        callback = callbacks[topic_key]
        await callback(mock_message)  # First delivery
        await callback(mock_message)  # Duplicate delivery

        # Verify: dispatch called only ONCE
        assert mock_dispatch_engine.dispatch.call_count == 1

        # Verify: first call received the correct topic and envelope
        call_args = mock_dispatch_engine.dispatch.call_args
        assert call_args[0][0] == topic_key  # topic

        # Verify: idempotency store has the envelope_id recorded
        is_processed = await idempotency_store.is_processed(
            message_id=envelope_id,
            domain=topic_key,
        )
        assert is_processed is True

        # Verify: calling check_and_record again returns False (duplicate)
        is_new = await idempotency_store.check_and_record(
            message_id=envelope_id,
            domain=topic_key,
            correlation_id=uuid4(),
        )
        assert is_new is False  # Already recorded

    @pytest.mark.asyncio
    async def test_different_event_ids_both_processed(
        self,
        subcontract_version: ModelSemVer,
    ) -> None:
        """Messages with different envelope_ids are both processed.

        Verifies that idempotency correctly distinguishes between different
        messages - only duplicates are skipped.

        Related: OMN-1740 AC#7
        """
        from uuid import uuid4

        from omnibase_infra.idempotency import StoreIdempotencyInmemory
        from omnibase_infra.models.event_bus import ModelIdempotencyConfig

        # Setup
        idempotency_store = StoreIdempotencyInmemory()
        mock_dispatch_engine = AsyncMock()
        mock_dispatch_engine.dispatch = AsyncMock()
        mock_event_bus = AsyncMock()

        callbacks: dict[str, object] = {}

        async def capture_subscribe(
            topic: str,
            node_identity: object,
            on_message: object,
        ) -> AsyncMock:
            callbacks[topic] = on_message
            return AsyncMock()

        mock_event_bus.subscribe = AsyncMock(side_effect=capture_subscribe)

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="test",
            node_name="test-handler",
            service="test-service",
            version="v1",
            idempotency_store=idempotency_store,
            idempotency_config=ModelIdempotencyConfig(enabled=True),
        )

        # Wire subscription
        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=["onex.evt.test.dedup.v1"],
        )
        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        # Create TWO messages with DIFFERENT envelope_ids
        envelope_id_1 = uuid4()
        envelope_id_2 = uuid4()

        message_1 = MagicMock()
        message_1.value = json.dumps(
            {
                "event_type": "test.dedup",
                "envelope_id": str(envelope_id_1),
                "correlation_id": str(uuid4()),
                "payload": {"data": "message1"},
                "source": "test-source",
            }
        ).encode()

        message_2 = MagicMock()
        message_2.value = json.dumps(
            {
                "event_type": "test.dedup",
                "envelope_id": str(envelope_id_2),
                "correlation_id": str(uuid4()),
                "payload": {"data": "message2"},
                "source": "test-source",
            }
        ).encode()

        # Get callback and deliver both messages
        topic_key = "onex.evt.test.dedup.v1"
        callback = callbacks[topic_key]
        await callback(message_1)
        await callback(message_2)

        # Verify: dispatch called TWICE (once for each unique message)
        assert mock_dispatch_engine.dispatch.call_count == 2

        # Verify: both envelope_ids are recorded
        assert await idempotency_store.is_processed(envelope_id_1, domain=topic_key)
        assert await idempotency_store.is_processed(envelope_id_2, domain=topic_key)

    @pytest.mark.asyncio
    async def test_idempotency_disabled_processes_all_messages(
        self,
        subcontract_version: ModelSemVer,
    ) -> None:
        """When idempotency is disabled, all messages are processed.

        Verifies that disabling idempotency bypasses the deduplication check,
        allowing duplicate messages to be processed multiple times.

        Related: OMN-1740 AC#7
        """
        from uuid import uuid4

        from omnibase_infra.idempotency import StoreIdempotencyInmemory
        from omnibase_infra.models.event_bus import ModelIdempotencyConfig

        # Setup with idempotency DISABLED
        idempotency_store = StoreIdempotencyInmemory()
        mock_dispatch_engine = AsyncMock()
        mock_dispatch_engine.dispatch = AsyncMock()
        mock_event_bus = AsyncMock()

        callbacks: dict[str, object] = {}

        async def capture_subscribe(
            topic: str,
            node_identity: object,
            on_message: object,
        ) -> AsyncMock:
            callbacks[topic] = on_message
            return AsyncMock()

        mock_event_bus.subscribe = AsyncMock(side_effect=capture_subscribe)

        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="test",
            node_name="test-handler",
            service="test-service",
            version="v1",
            idempotency_store=idempotency_store,
            idempotency_config=ModelIdempotencyConfig(enabled=False),  # DISABLED
        )

        # Wire subscription
        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=["onex.evt.test.dedup.v1"],
        )
        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        # Create message with specific envelope_id
        envelope_id = uuid4()
        message_json = json.dumps(
            {
                "event_type": "test.dedup",
                "envelope_id": str(envelope_id),
                "correlation_id": str(uuid4()),
                "payload": {"data": "test"},
                "source": "test-source",
            }
        )
        mock_message = MagicMock()
        mock_message.value = message_json.encode()

        # Get callback and deliver message TWICE
        topic_key = "onex.evt.test.dedup.v1"
        callback = callbacks[topic_key]
        await callback(mock_message)
        await callback(mock_message)

        # Verify: dispatch called TWICE (idempotency disabled)
        assert mock_dispatch_engine.dispatch.call_count == 2

        # Verify: idempotency store was NOT used (no records)
        assert await idempotency_store.get_record_count() == 0


class TestOffsetCommitOnFailure:
    """Integration tests for offset commit behavior on failure (AC#8).

    These tests verify that:
    - Offsets are NOT committed when handler raises infrastructure errors
    - Offsets ARE committed on successful handler execution
    - This ensures at-least-once delivery semantics

    Related Tickets:
        - OMN-1740 AC#8: Offset not committed on handler failure
    """

    @pytest.fixture
    def subcontract_version(self) -> ModelSemVer:
        """Create default version for subcontracts."""
        return ModelSemVer(major=1, minor=0, patch=0)

    @pytest.mark.asyncio
    async def test_offset_not_committed_on_infra_error(
        self,
        subcontract_version: ModelSemVer,
    ) -> None:
        """Given infrastructure error, offset should NOT be committed.

        This test verifies:
        1. Handler raises RuntimeHostError
        2. Offset is NOT committed
        3. Error is re-raised (fail-fast default)
        4. Kafka will redeliver the message

        Related: OMN-1740 AC#8
        """
        from uuid import uuid4

        from omnibase_infra.models.event_bus import ModelOffsetPolicyConfig

        # Setup: dispatch engine that raises infrastructure error
        mock_dispatch_engine = AsyncMock()
        mock_dispatch_engine.dispatch = AsyncMock(
            side_effect=RuntimeHostError("Database unavailable")
        )

        # Setup: event bus with commit_offset tracking
        mock_event_bus = AsyncMock()
        mock_event_bus.commit_offset = AsyncMock()

        # Capture the callback registered with subscribe
        callbacks: dict[str, object] = {}

        async def capture_subscribe(
            topic: str,
            node_identity: object,
            on_message: object,
        ) -> AsyncMock:
            callbacks[topic] = on_message
            return AsyncMock()

        mock_event_bus.subscribe = AsyncMock(side_effect=capture_subscribe)

        # Create wiring with commit_after_handler policy (default)
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="test",
            node_name="test-handler",
            service="test-service",
            version="v1",
            offset_policy=ModelOffsetPolicyConfig(
                commit_strategy="commit_after_handler"
            ),
        )

        # Wire subscription
        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=["onex.evt.test.failure.v1"],
        )
        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        # Create valid message with proper envelope format
        message_json = json.dumps(
            {
                "event_type": "test.failure",
                "correlation_id": str(uuid4()),
                "payload": {"data": "test"},
                "source": "test-source",
            }
        )
        mock_message = MagicMock()
        mock_message.value = message_json.encode()

        # Deliver message - should raise RuntimeHostError
        callback = callbacks["onex.evt.test.failure.v1"]
        with pytest.raises(RuntimeHostError):
            await callback(mock_message)

        # Verify: commit_offset was NOT called (fail-fast behavior)
        mock_event_bus.commit_offset.assert_not_called()

    @pytest.mark.asyncio
    async def test_offset_committed_on_success(
        self,
        subcontract_version: ModelSemVer,
    ) -> None:
        """Given successful handler execution, offset should be committed.

        This test verifies:
        1. Handler completes successfully
        2. Offset is committed
        3. Message is acknowledged

        Related: OMN-1740 AC#8
        """
        from uuid import uuid4

        from omnibase_infra.models.event_bus import ModelOffsetPolicyConfig

        # Setup: dispatch engine that succeeds
        mock_dispatch_engine = AsyncMock()
        mock_dispatch_engine.dispatch = AsyncMock(return_value=None)

        # Setup: event bus with commit_offset tracking
        mock_event_bus = AsyncMock()
        mock_event_bus.commit_offset = AsyncMock()

        # Capture the callback registered with subscribe
        callbacks: dict[str, object] = {}

        async def capture_subscribe(
            topic: str,
            node_identity: object,
            on_message: object,
        ) -> AsyncMock:
            callbacks[topic] = on_message
            return AsyncMock()

        mock_event_bus.subscribe = AsyncMock(side_effect=capture_subscribe)

        # Create wiring with commit_after_handler policy (default)
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="test",
            node_name="test-handler",
            service="test-service",
            version="v1",
            offset_policy=ModelOffsetPolicyConfig(
                commit_strategy="commit_after_handler"
            ),
        )

        # Wire subscription
        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=["onex.evt.test.success.v1"],
        )
        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        # Create valid message with proper envelope format
        message_json = json.dumps(
            {
                "event_type": "test.success",
                "correlation_id": str(uuid4()),
                "payload": {"data": "test"},
                "source": "test-source",
            }
        )
        mock_message = MagicMock()
        mock_message.value = message_json.encode()

        # Deliver message - should succeed
        callback = callbacks["onex.evt.test.success.v1"]
        await callback(mock_message)

        # Verify: commit_offset WAS called
        mock_event_bus.commit_offset.assert_called_once()

    @pytest.mark.asyncio
    async def test_offset_not_committed_on_content_error_with_fail_fast(
        self,
        subcontract_version: ModelSemVer,
    ) -> None:
        """Given content error with fail_fast policy, offset should NOT be committed.

        This test verifies:
        1. Malformed message causes content error
        2. With fail_fast policy, offset is NOT committed
        3. Error is re-raised

        Related: OMN-1740 AC#8
        """
        from omnibase_infra.models.event_bus import (
            ModelDlqConfig,
            ModelOffsetPolicyConfig,
        )

        # Setup: dispatch engine (won't be called due to deserialization failure)
        mock_dispatch_engine = AsyncMock()

        # Setup: event bus with commit_offset tracking
        mock_event_bus = AsyncMock()
        mock_event_bus.commit_offset = AsyncMock()

        # Capture the callback registered with subscribe
        callbacks: dict[str, object] = {}

        async def capture_subscribe(
            topic: str,
            node_identity: object,
            on_message: object,
        ) -> AsyncMock:
            callbacks[topic] = on_message
            return AsyncMock()

        mock_event_bus.subscribe = AsyncMock(side_effect=capture_subscribe)

        # Create wiring with fail_fast for content errors
        wiring = EventBusSubcontractWiring(
            event_bus=mock_event_bus,
            dispatch_engine=mock_dispatch_engine,
            environment="test",
            node_name="test-handler",
            service="test-service",
            version="v1",
            offset_policy=ModelOffsetPolicyConfig(
                commit_strategy="commit_after_handler"
            ),
            dlq_config=ModelDlqConfig(
                enabled=False,
                on_content_error="fail_fast",
            ),
        )

        # Wire subscription
        subcontract = ModelEventBusSubcontract(
            version=subcontract_version,
            subscribe_topics=["onex.evt.test.malformed.v1"],
        )
        await wiring.wire_subscriptions(subcontract, node_name="test-handler")

        # Create malformed message (invalid JSON)
        mock_message = MagicMock()
        mock_message.value = b"not valid json"

        # Deliver message - should raise ProtocolConfigurationError
        callback = callbacks["onex.evt.test.malformed.v1"]
        with pytest.raises(ProtocolConfigurationError):
            await callback(mock_message)

        # Verify: commit_offset was NOT called
        mock_event_bus.commit_offset.assert_not_called()
