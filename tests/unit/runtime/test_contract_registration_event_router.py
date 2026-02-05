# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ContractRegistrationEventRouter.

This module tests the ContractRegistrationEventRouter class which routes
contract registration events from the event bus to the ContractRegistryReducer.

Test Coverage:
- Initialization with valid and invalid parameters
- Correlation ID extraction from headers and payload
- Event type determination from topic suffix
- Message handling for valid events
- Message handling for invalid messages
- Property accessors

Related:
- OMN-1869: Contract Registration Event Router
- ContractRegistrationEventRouter source implementation
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums import EnumReductionType, EnumStreamingMode
from omnibase_core.models.events import (
    ModelContractDeregisteredEvent,
    ModelContractRegisteredEvent,
    ModelNodeHeartbeatEvent,
)
from omnibase_core.nodes import ModelReducerOutput
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.contract_registry_reducer.models.model_contract_registry_state import (
    ModelContractRegistryState,
)
from omnibase_infra.runtime.contract_registration_event_router import (
    TOPIC_SUFFIX_CONTRACT_DEREGISTERED,
    TOPIC_SUFFIX_CONTRACT_REGISTERED,
    TOPIC_SUFFIX_NODE_HEARTBEAT,
    ContractRegistrationEventRouter,
)


class TestContractRegistrationEventRouterInit:
    """Tests for ContractRegistrationEventRouter initialization."""

    def test_init_with_valid_parameters(self) -> None:
        """Should initialize successfully with valid parameters."""
        container = MagicMock(spec=ModelONEXContainer)
        reducer = MagicMock()
        event_bus = MagicMock()
        output_topic = "test.output.topic"

        router = ContractRegistrationEventRouter(
            container=container,
            reducer=reducer,
            event_bus=event_bus,
            output_topic=output_topic,
        )

        assert router.container is container
        assert router.reducer is reducer
        assert router.event_bus is event_bus
        assert router.output_topic == output_topic
        assert isinstance(router.state, ModelContractRegistryState)

    def test_init_raises_value_error_for_empty_output_topic(self) -> None:
        """Should raise ValueError when output_topic is empty."""
        container = MagicMock(spec=ModelONEXContainer)
        reducer = MagicMock()
        event_bus = MagicMock()

        with pytest.raises(ValueError, match="output_topic cannot be empty"):
            ContractRegistrationEventRouter(
                container=container,
                reducer=reducer,
                event_bus=event_bus,
                output_topic="",
            )

    def test_init_raises_value_error_for_whitespace_output_topic(self) -> None:
        """Should raise ValueError when output_topic is only whitespace."""
        container = MagicMock(spec=ModelONEXContainer)
        reducer = MagicMock()
        event_bus = MagicMock()

        # Empty string after stripping is still falsy
        with pytest.raises(ValueError, match="output_topic cannot be empty"):
            ContractRegistrationEventRouter(
                container=container,
                reducer=reducer,
                event_bus=event_bus,
                output_topic="",
            )


class TestExtractCorrelationIdFromMessage:
    """Tests for _extract_correlation_id_from_message method."""

    @pytest.fixture
    def router(self) -> ContractRegistrationEventRouter:
        """Create a router instance for testing."""
        return ContractRegistrationEventRouter(
            container=MagicMock(spec=ModelONEXContainer),
            reducer=MagicMock(),
            event_bus=MagicMock(),
            output_topic="test.output.topic",
        )

    def test_extract_correlation_id_from_headers(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should extract correlation ID from message headers."""
        expected_correlation_id = uuid4()
        headers = MagicMock()
        headers.correlation_id = str(expected_correlation_id)

        msg = MagicMock()
        msg.headers = headers
        msg.value = None

        result = router.extract_correlation_id_from_message(msg)

        assert result == expected_correlation_id

    def test_extract_correlation_id_from_headers_bytes(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should extract correlation ID from headers when stored as bytes."""
        expected_correlation_id = uuid4()
        headers = MagicMock()
        # Simulate bytes-like correlation_id
        correlation_bytes = MagicMock()
        correlation_bytes.decode.return_value = str(expected_correlation_id)
        headers.correlation_id = correlation_bytes

        msg = MagicMock()
        msg.headers = headers
        msg.value = None

        result = router.extract_correlation_id_from_message(msg)

        assert result == expected_correlation_id

    def test_extract_correlation_id_from_payload(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should extract correlation ID from JSON payload when not in headers."""
        expected_correlation_id = uuid4()

        msg = MagicMock()
        msg.headers = None
        msg.value = json.dumps({"correlation_id": str(expected_correlation_id)})

        result = router.extract_correlation_id_from_message(msg)

        assert result == expected_correlation_id

    def test_extract_correlation_id_from_payload_bytes(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should extract correlation ID from bytes payload."""
        expected_correlation_id = uuid4()

        msg = MagicMock()
        msg.headers = None
        # Create bytes-like value
        value_bytes = MagicMock()
        value_bytes.decode.return_value = json.dumps(
            {"correlation_id": str(expected_correlation_id)}
        )
        msg.value = value_bytes

        result = router.extract_correlation_id_from_message(msg)

        assert result == expected_correlation_id

    def test_generates_new_uuid_when_not_found(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should generate new UUID when correlation ID not in headers or payload."""
        msg = MagicMock()
        msg.headers = None
        msg.value = json.dumps({"some_field": "value"})

        result = router.extract_correlation_id_from_message(msg)

        assert isinstance(result, UUID)

    def test_generates_new_uuid_for_invalid_payload(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should generate new UUID when payload is not valid JSON."""
        msg = MagicMock()
        msg.headers = None
        msg.value = "not valid json"

        result = router.extract_correlation_id_from_message(msg)

        assert isinstance(result, UUID)

    def test_generates_new_uuid_for_none_value(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should generate new UUID when message value is None."""
        msg = MagicMock()
        msg.headers = None
        msg.value = None

        result = router.extract_correlation_id_from_message(msg)

        assert isinstance(result, UUID)


class TestDetermineEventType:
    """Tests for _determine_event_type method."""

    @pytest.fixture
    def router(self) -> ContractRegistrationEventRouter:
        """Create a router instance for testing."""
        return ContractRegistrationEventRouter(
            container=MagicMock(spec=ModelONEXContainer),
            reducer=MagicMock(),
            event_bus=MagicMock(),
            output_topic="test.output.topic",
        )

    def test_returns_contract_registered_event_for_registered_topic(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should return ModelContractRegisteredEvent for contract-registered topic."""
        topic = f"dev.{TOPIC_SUFFIX_CONTRACT_REGISTERED}"

        result = router._determine_event_type(topic)

        assert result is ModelContractRegisteredEvent

    def test_returns_contract_deregistered_event_for_deregistered_topic(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should return ModelContractDeregisteredEvent for contract-deregistered topic."""
        topic = f"dev.{TOPIC_SUFFIX_CONTRACT_DEREGISTERED}"

        result = router._determine_event_type(topic)

        assert result is ModelContractDeregisteredEvent

    def test_returns_node_heartbeat_event_for_heartbeat_topic(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should return ModelNodeHeartbeatEvent for node-heartbeat topic."""
        topic = f"dev.{TOPIC_SUFFIX_NODE_HEARTBEAT}"

        result = router._determine_event_type(topic)

        assert result is ModelNodeHeartbeatEvent

    def test_returns_none_for_unrecognized_topic(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should return None for unrecognized topic suffix."""
        topic = "onex.evt.platform.unknown-event.v1"

        result = router._determine_event_type(topic)

        assert result is None

    def test_returns_none_for_empty_topic(
        self, router: ContractRegistrationEventRouter
    ) -> None:
        """Should return None for empty topic string."""
        result = router._determine_event_type("")

        assert result is None


class TestHandleMessage:
    """Tests for handle_message async method."""

    @pytest.fixture
    def mock_reducer(self) -> MagicMock:
        """Create a mock reducer that returns valid output."""
        reducer = MagicMock()
        reducer.reduce.return_value = ModelReducerOutput(
            result=ModelContractRegistryState(),
            intents=(),
            items_processed=1,
            reduction_type=EnumReductionType.TRANSFORM,
            streaming_mode=EnumStreamingMode.BATCH,
            operation_id=uuid4(),
            processing_time_ms=10.0,
        )
        return reducer

    @pytest.fixture
    def router(self, mock_reducer: MagicMock) -> ContractRegistrationEventRouter:
        """Create a router instance with mock reducer."""
        return ContractRegistrationEventRouter(
            container=MagicMock(spec=ModelONEXContainer),
            reducer=mock_reducer,
            event_bus=MagicMock(),
            output_topic="test.output.topic",
        )

    def _create_contract_registered_event(self) -> dict:
        """Create a valid contract-registered event payload."""
        return {
            "event_id": str(uuid4()),
            "event_type": "contract-registered",
            "timestamp": datetime.now(UTC).isoformat(),
            "node_name": "test-node",
            "node_version": {"major": 1, "minor": 0, "patch": 0},
            "contract_hash": "sha256:abc123",
            "contract_yaml": "name: test\nversion: 1.0.0",
            "correlation_id": str(uuid4()),
        }

    def _create_message(
        self, topic: str, payload: dict, correlation_id: UUID | None = None
    ) -> ModelEventMessage:
        """Create a mock event message."""
        headers = ModelEventHeaders(
            source="test-source",
            event_type="test-event",
            timestamp=datetime.now(UTC),
            correlation_id=correlation_id or uuid4(),
        )
        return ModelEventMessage(
            topic=topic,
            value=json.dumps(payload).encode("utf-8"),
            headers=headers,
            partition=0,
            offset="0",
        )

    @pytest.mark.asyncio
    async def test_handle_message_routes_valid_contract_registered_event(
        self, router: ContractRegistrationEventRouter, mock_reducer: MagicMock
    ) -> None:
        """Should route valid contract-registered event to reducer."""
        payload = self._create_contract_registered_event()
        topic = f"dev.{TOPIC_SUFFIX_CONTRACT_REGISTERED}"
        msg = self._create_message(topic, payload)

        result = await router.handle_message(msg)

        assert result is not None
        assert mock_reducer.reduce.called

    @pytest.mark.asyncio
    async def test_handle_message_returns_none_for_unrecognized_topic(
        self, router: ContractRegistrationEventRouter, mock_reducer: MagicMock
    ) -> None:
        """Should return None for unrecognized topic."""
        payload = {"some": "data"}
        msg = self._create_message("unknown.topic", payload)

        result = await router.handle_message(msg)

        assert result is None
        assert not mock_reducer.reduce.called

    @pytest.mark.asyncio
    async def test_handle_message_returns_none_for_none_value(
        self, router: ContractRegistrationEventRouter, mock_reducer: MagicMock
    ) -> None:
        """Should return None when message value is None."""
        topic = f"dev.{TOPIC_SUFFIX_CONTRACT_REGISTERED}"
        headers = ModelEventHeaders(
            source="test-source",
            event_type="test-event",
            timestamp=datetime.now(UTC),
        )

        # Create mock message with None value
        msg = MagicMock()
        msg.topic = topic
        msg.value = None
        msg.headers = headers
        msg.partition = 0
        msg.offset = 0

        result = await router.handle_message(msg)

        assert result is None
        assert not mock_reducer.reduce.called

    @pytest.mark.asyncio
    async def test_handle_message_returns_none_for_invalid_json(
        self, router: ContractRegistrationEventRouter, mock_reducer: MagicMock
    ) -> None:
        """Should return None for invalid JSON payload."""
        topic = f"dev.{TOPIC_SUFFIX_CONTRACT_REGISTERED}"
        headers = ModelEventHeaders(
            source="test-source",
            event_type="test-event",
            timestamp=datetime.now(UTC),
        )

        msg = MagicMock()
        msg.topic = topic
        msg.value = b"not valid json"
        msg.headers = headers
        msg.partition = 0
        msg.offset = 0

        result = await router.handle_message(msg)

        assert result is None
        assert not mock_reducer.reduce.called

    @pytest.mark.asyncio
    async def test_handle_message_returns_none_for_validation_error(
        self, router: ContractRegistrationEventRouter, mock_reducer: MagicMock
    ) -> None:
        """Should return None when event payload fails validation."""
        # Missing required fields for contract-registered event
        payload = {"event_id": str(uuid4())}
        topic = f"dev.{TOPIC_SUFFIX_CONTRACT_REGISTERED}"
        msg = self._create_message(topic, payload)

        result = await router.handle_message(msg)

        assert result is None
        assert not mock_reducer.reduce.called

    @pytest.mark.asyncio
    async def test_handle_message_updates_internal_state(
        self, router: ContractRegistrationEventRouter, mock_reducer: MagicMock
    ) -> None:
        """Should update internal state after successful processing."""
        initial_state = router.state
        payload = self._create_contract_registered_event()
        topic = f"dev.{TOPIC_SUFFIX_CONTRACT_REGISTERED}"
        msg = self._create_message(topic, payload)

        # Configure reducer to return new state
        new_state = ModelContractRegistryState()
        mock_reducer.reduce.return_value = ModelReducerOutput(
            result=new_state,
            intents=(),
            items_processed=1,
            reduction_type=EnumReductionType.TRANSFORM,
            streaming_mode=EnumStreamingMode.BATCH,
            operation_id=uuid4(),
            processing_time_ms=10.0,
        )

        await router.handle_message(msg)

        # State should be updated to the one returned by reducer
        assert router.state is new_state


class TestContractRegistrationEventRouterProperties:
    """Tests for ContractRegistrationEventRouter properties."""

    def test_container_property_returns_container(self) -> None:
        """Should return the container passed during initialization."""
        container = MagicMock(spec=ModelONEXContainer)
        router = ContractRegistrationEventRouter(
            container=container,
            reducer=MagicMock(),
            event_bus=MagicMock(),
            output_topic="test.topic",
        )

        assert router.container is container

    def test_output_topic_property_returns_topic(self) -> None:
        """Should return the output topic passed during initialization."""
        router = ContractRegistrationEventRouter(
            container=MagicMock(spec=ModelONEXContainer),
            reducer=MagicMock(),
            event_bus=MagicMock(),
            output_topic="expected.output.topic",
        )

        assert router.output_topic == "expected.output.topic"

    def test_reducer_property_returns_reducer(self) -> None:
        """Should return the reducer passed during initialization."""
        reducer = MagicMock()
        router = ContractRegistrationEventRouter(
            container=MagicMock(spec=ModelONEXContainer),
            reducer=reducer,
            event_bus=MagicMock(),
            output_topic="test.topic",
        )

        assert router.reducer is reducer

    def test_event_bus_property_returns_event_bus(self) -> None:
        """Should return the event bus passed during initialization."""
        event_bus = MagicMock()
        router = ContractRegistrationEventRouter(
            container=MagicMock(spec=ModelONEXContainer),
            reducer=MagicMock(),
            event_bus=event_bus,
            output_topic="test.topic",
        )

        assert router.event_bus is event_bus

    def test_state_property_returns_initial_state(self) -> None:
        """Should return initial empty state after construction."""
        router = ContractRegistrationEventRouter(
            container=MagicMock(spec=ModelONEXContainer),
            reducer=MagicMock(),
            event_bus=MagicMock(),
            output_topic="test.topic",
        )

        state = router.state
        assert isinstance(state, ModelContractRegistryState)
