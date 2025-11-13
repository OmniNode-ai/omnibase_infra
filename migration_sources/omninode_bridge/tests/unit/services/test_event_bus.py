#!/usr/bin/env python3
"""
Unit Tests for EventBusService.

Tests event publishing, subscription, correlation filtering, and timeout handling.

ONEX v2.0 Compliance:
- Pydantic v2 models
- OnexError error handling
- Correlation-based event routing

Wave 5 Refactor - Event-Driven Orchestration
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from omnibase_core import EnumCoreErrorCode, ModelOnexError

from omninode_bridge.services.event_bus import EventBusService
from omninode_bridge.services.kafka_client import KafkaClient


@pytest.fixture
def mock_kafka_client():
    """Create mock KafkaClient for testing."""
    client = MagicMock(spec=KafkaClient)
    # Start as not connected so EventBus will call connect()
    client.is_connected = False
    client.connect = AsyncMock()

    # After connect is called, set is_connected to True
    async def _connect():
        client.is_connected = True

    client.connect.side_effect = _connect
    client.publish_with_envelope = AsyncMock(return_value=True)
    client.consume_messages = AsyncMock(return_value=[])
    client.health_check = AsyncMock(
        return_value={"status": "healthy", "connected": True}
    )
    return client


@pytest.fixture
async def event_bus(mock_kafka_client):
    """Create EventBusService instance for testing."""
    bus = EventBusService(
        kafka_client=mock_kafka_client,
        node_id="test-orchestrator",
        namespace="test",
    )
    yield bus
    # Cleanup
    if bus.is_initialized:
        await bus.shutdown()


class TestEventBusInitialization:
    """Test EventBus initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, event_bus, mock_kafka_client):
        """Test successful EventBus initialization."""
        # Initialize EventBus
        await event_bus.initialize()

        # Verify initialization
        assert event_bus.is_initialized
        assert event_bus._consumer_task is not None
        assert not event_bus._consumer_task.done()

        # Verify Kafka client connected
        mock_kafka_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, event_bus, mock_kafka_client):
        """Test initializing already initialized EventBus."""
        # Initialize twice
        await event_bus.initialize()
        await event_bus.initialize()

        # Should only connect once
        assert mock_kafka_client.connect.call_count == 1

    @pytest.mark.asyncio
    async def test_initialize_kafka_connection_failure(
        self, event_bus, mock_kafka_client
    ):
        """Test initialization with Kafka connection failure."""
        # Simulate connection failure
        mock_kafka_client.is_connected = False
        mock_kafka_client.connect.side_effect = Exception("Connection failed")

        # Should raise OnexError
        with pytest.raises(ModelOnexError) as exc_info:
            await event_bus.initialize()

        assert exc_info.value.error_code == EnumCoreErrorCode.INITIALIZATION_FAILED
        assert "Connection failed" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_shutdown(self, event_bus):
        """Test EventBus shutdown."""
        # Initialize and shutdown
        await event_bus.initialize()
        await event_bus.shutdown()

        # Verify shutdown
        assert not event_bus.is_initialized
        assert len(event_bus._event_listeners) == 0


class TestEventPublishing:
    """Test event publishing capabilities."""

    @pytest.mark.asyncio
    async def test_publish_action_event_success(self, event_bus, mock_kafka_client):
        """Test successful Action event publishing."""
        await event_bus.initialize()

        correlation_id = uuid4()
        action_type = "AGGREGATE_STAMPS"
        payload = {"workflow_key": "test-workflow", "items": [1, 2, 3]}

        # Publish event
        success = await event_bus.publish_action_event(
            correlation_id=correlation_id,
            action_type=action_type,
            payload=payload,
        )

        # Verify success
        assert success is True
        assert event_bus._events_published == 1

        # Verify Kafka publish was called
        mock_kafka_client.publish_with_envelope.assert_called_once()

        # Verify call arguments
        call_args = mock_kafka_client.publish_with_envelope.call_args
        assert call_args.kwargs["event_type"] == "ACTION"
        assert call_args.kwargs["source_node_id"] == "test-orchestrator"
        assert call_args.kwargs["correlation_id"] == correlation_id
        assert call_args.kwargs["topic"] == "test.omninode_bridge.onex.evt.action.v1"

        # Verify payload structure
        payload_data = call_args.kwargs["payload"]
        assert payload_data["action_type"] == action_type
        assert payload_data["correlation_id"] == str(correlation_id)
        assert payload_data["payload"] == payload

    @pytest.mark.asyncio
    async def test_publish_action_event_not_initialized(
        self, event_bus, mock_kafka_client
    ):
        """Test publishing event when EventBus not initialized."""
        correlation_id = uuid4()

        # Try to publish without initializing
        with pytest.raises(ModelOnexError) as exc_info:
            await event_bus.publish_action_event(
                correlation_id=correlation_id,
                action_type="TEST",
                payload={},
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE
        assert "not initialized" in str(exc_info.value.message).lower()

    @pytest.mark.asyncio
    async def test_publish_action_event_kafka_failure(
        self, event_bus, mock_kafka_client
    ):
        """Test handling Kafka publish failure."""
        await event_bus.initialize()

        # Simulate Kafka publish failure
        mock_kafka_client.publish_with_envelope.side_effect = Exception("Kafka error")

        correlation_id = uuid4()

        # Should raise OnexError
        with pytest.raises(ModelOnexError) as exc_info:
            await event_bus.publish_action_event(
                correlation_id=correlation_id,
                action_type="TEST",
                payload={},
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.OPERATION_FAILED
        assert "Kafka error" in str(exc_info.value.__cause__)


class TestEventWaiting:
    """Test event waiting and correlation filtering."""

    @pytest.mark.asyncio
    async def test_wait_for_completion_success(self, event_bus, mock_kafka_client):
        """Test successful event waiting with StateCommitted."""
        await event_bus.initialize()

        correlation_id = uuid4()

        # Simulate receiving StateCommitted event
        async def simulate_event():
            await asyncio.sleep(0.1)
            correlation_id_str = str(correlation_id)

            # Create event queue and add event
            if correlation_id_str in event_bus._event_listeners:
                event_data = {
                    "event_type": "STATE_COMMITTED",
                    "correlation_id": correlation_id_str,
                    "payload": {"version": 2, "state": {"items": [1, 2, 3]}},
                    "metadata": {},
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                await event_bus._event_listeners[correlation_id_str].put(event_data)

        # Start simulation task
        simulation_task = asyncio.create_task(simulate_event())

        # Wait for completion
        result = await event_bus.wait_for_completion(
            correlation_id=correlation_id,
            timeout_seconds=5.0,
        )

        # Verify result
        assert result["event_type"] == "STATE_COMMITTED"
        assert result["correlation_id"] == str(correlation_id)
        assert result["payload"]["version"] == 2

        # Verify listener was cleaned up
        assert str(correlation_id) not in event_bus._event_listeners

        await simulation_task

    @pytest.mark.asyncio
    async def test_wait_for_completion_reducer_gave_up(
        self, event_bus, mock_kafka_client
    ):
        """Test event waiting with ReducerGaveUp (failure)."""
        await event_bus.initialize()

        correlation_id = uuid4()

        # Simulate receiving ReducerGaveUp event
        async def simulate_event():
            await asyncio.sleep(0.1)
            correlation_id_str = str(correlation_id)

            if correlation_id_str in event_bus._event_listeners:
                event_data = {
                    "event_type": "REDUCER_GAVE_UP",
                    "correlation_id": correlation_id_str,
                    "payload": {"error": "Max retries exceeded"},
                    "metadata": {},
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                await event_bus._event_listeners[correlation_id_str].put(event_data)

        # Start simulation task
        simulation_task = asyncio.create_task(simulate_event())

        # Wait for completion - should raise error
        with pytest.raises(ModelOnexError) as exc_info:
            await event_bus.wait_for_completion(
                correlation_id=correlation_id,
                timeout_seconds=5.0,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.OPERATION_FAILED
        assert "gave up" in str(exc_info.value.message).lower()

        # Verify listener was cleaned up
        assert str(correlation_id) not in event_bus._event_listeners

        await simulation_task

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self, event_bus, mock_kafka_client):
        """Test event waiting with timeout."""
        await event_bus.initialize()

        correlation_id = uuid4()

        # Wait for completion with short timeout (no event will arrive)
        result = await event_bus.wait_for_completion(
            correlation_id=correlation_id,
            timeout_seconds=0.5,
        )

        # Verify timeout result
        assert result == {}
        assert event_bus._events_timeout == 1

        # Verify listener was cleaned up
        assert str(correlation_id) not in event_bus._event_listeners

    @pytest.mark.asyncio
    async def test_wait_for_completion_not_initialized(
        self, event_bus, mock_kafka_client
    ):
        """Test waiting for event when EventBus not initialized."""
        correlation_id = uuid4()

        # Try to wait without initializing
        with pytest.raises(ModelOnexError) as exc_info:
            await event_bus.wait_for_completion(
                correlation_id=correlation_id,
                timeout_seconds=5.0,
            )

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE
        assert "not initialized" in str(exc_info.value.message).lower()


class TestHealthCheck:
    """Test health check capabilities."""

    @pytest.mark.asyncio
    async def test_health_check_initialized(self, event_bus, mock_kafka_client):
        """Test health check when initialized."""
        # Set client as connected for health check
        mock_kafka_client.is_connected = True
        await event_bus.initialize()

        health = await event_bus.health_check()

        # Verify health status
        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert health["kafka_connected"] is True
        assert health["consumer_task_running"] is True
        assert health["active_listeners"] == 0
        assert health["metrics"]["events_published"] == 0
        assert health["metrics"]["events_consumed"] == 0
        assert health["metrics"]["events_timeout"] == 0

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, event_bus, mock_kafka_client):
        """Test health check when not initialized."""
        health = await event_bus.health_check()

        # Verify health status
        assert health["status"] == "not_initialized"
        assert health["initialized"] is False
        assert health["consumer_task_running"] is False


class TestEventConsumption:
    """Test background event consumption."""

    @pytest.mark.asyncio
    async def test_handle_consumed_event_with_listener(
        self, event_bus, mock_kafka_client
    ):
        """Test handling consumed event with active listener."""
        await event_bus.initialize()

        correlation_id = uuid4()
        correlation_id_str = str(correlation_id)

        # Create listener queue
        event_queue: asyncio.Queue = asyncio.Queue()
        event_bus._event_listeners[correlation_id_str] = event_queue

        # Simulate consumed message
        message = {
            "topic": "test.omninode_bridge.onex.evt.state-committed.v1",
            "partition": 0,
            "offset": 123,
            "key": correlation_id_str,
            "value": {
                "event_type": "STATE_COMMITTED",
                "correlation_id": correlation_id_str,
                "payload": {"version": 1},
                "metadata": {},
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }

        # Handle event
        await event_bus._handle_consumed_event(message)

        # Verify event was routed to listener
        assert not event_queue.empty()
        event_data = await event_queue.get()
        assert event_data["event_type"] == "STATE_COMMITTED"
        assert event_data["correlation_id"] == correlation_id_str
        assert event_bus._events_consumed == 1

    @pytest.mark.asyncio
    async def test_handle_consumed_event_without_listener(
        self, event_bus, mock_kafka_client
    ):
        """Test handling consumed event without active listener."""
        await event_bus.initialize()

        correlation_id = uuid4()

        # Simulate consumed message (no listener registered)
        message = {
            "topic": "test.omninode_bridge.onex.evt.state-committed.v1",
            "partition": 0,
            "offset": 123,
            "key": str(correlation_id),
            "value": {
                "event_type": "STATE_COMMITTED",
                "correlation_id": str(correlation_id),
                "payload": {"version": 1},
                "metadata": {},
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }

        # Handle event - should not raise error
        await event_bus._handle_consumed_event(message)

        # Verify event was not consumed (no listener)
        assert event_bus._events_consumed == 0

    @pytest.mark.asyncio
    async def test_handle_consumed_event_missing_correlation_id(
        self, event_bus, mock_kafka_client
    ):
        """Test handling consumed event without correlation_id."""
        await event_bus.initialize()

        # Simulate consumed message without correlation_id
        message = {
            "topic": "test.omninode_bridge.onex.evt.state-committed.v1",
            "partition": 0,
            "offset": 123,
            "key": None,
            "value": {
                "event_type": "STATE_COMMITTED",
                "payload": {"version": 1},
                "metadata": {},
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }

        # Handle event - should not raise error, just log warning
        await event_bus._handle_consumed_event(message)

        # Verify event was not consumed
        assert event_bus._events_consumed == 0
