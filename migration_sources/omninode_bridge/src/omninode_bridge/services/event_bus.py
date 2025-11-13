#!/usr/bin/env python3
"""
EventBus Service - Event-Driven Coordination.

Provides event publishing and subscription capabilities for bridge nodes.
Enables event-driven coordination between orchestrator and reducer nodes.

ONEX v2.0 Compliance:
- Suffix-based naming: EventBusService
- OnexEnvelopeV1 for all events
- Correlation-based event filtering
- Comprehensive error handling with OnexError

Wave 5 Refactor - Event-Driven Orchestration
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

# Import with fallback when omnibase_core is not available
try:
    from omnibase_core import EnumCoreErrorCode, ModelOnexError
except ImportError:
    # Fallback to minimal stubs for testing/demo mode
    from enum import Enum

    class EnumCoreErrorCode(str, Enum):
        """Stub error code enum."""

        OPERATION_FAILED = "OPERATION_FAILED"
        VALIDATION_ERROR = "VALIDATION_ERROR"
        TIMEOUT = "TIMEOUT"

    class ModelOnexError(Exception):
        """Stub ONEX error class."""

        def __init__(self, code: EnumCoreErrorCode, message: str, context: dict = None):
            super().__init__(message)
            self.code = code
            self.message = message
            self.context = context or {}


from prometheus_client import Counter, Histogram

from .kafka_client import KafkaClient

# Alias for consistency
OnexError = ModelOnexError

logger = logging.getLogger(__name__)

# Prometheus Metrics
event_bus_published_total = Counter(
    "event_bus_published_total",
    "Total number of events published",
    ["event_type"],
)

event_bus_consumed_total = Counter(
    "event_bus_consumed_total",
    "Total number of events consumed",
    ["event_type"],
)

event_bus_timeout_total = Counter(
    "event_bus_timeout_total",
    "Total number of event wait timeouts",
)

event_bus_wait_time_ms = Histogram(
    "event_bus_wait_time_ms",
    "Event wait time in milliseconds",
    buckets=[10, 50, 100, 500, 1000, 5000, 10000, 30000],
)


class EventBusService:
    """
    Event bus for orchestrator-reducer coordination via Kafka.

    Features:
    - Event publishing with OnexEnvelopeV1 wrapping
    - Event subscription with correlation filtering
    - Timeout-based event waiting
    - Health monitoring and connection management

    Usage:
        >>> bus = EventBusService(kafka_client)
        >>> await bus.initialize()
        >>> await bus.publish_action_event(workflow_id, action_data)
        >>> result = await bus.wait_for_completion(workflow_id, timeout_seconds=30)
    """

    def __init__(
        self,
        kafka_client: KafkaClient,
        node_id: str = "orchestrator",
        namespace: str = "dev",
    ):
        """
        Initialize EventBus service.

        Args:
            kafka_client: KafkaClient instance for publishing/consuming
            node_id: Node identifier for event sourcing
            namespace: Multi-tenant namespace
        """
        self.kafka_client = kafka_client
        self.node_id = node_id
        self.namespace = namespace
        self._initialized = False

        # Event listeners for correlation-based routing
        self._event_listeners: dict[str, asyncio.Queue[dict[str, Any]]] = {}

        # Background task for consumer
        self._consumer_task: asyncio.Task | None = None

        # Metrics tracking
        self._events_published = 0
        self._events_consumed = 0
        self._events_timeout = 0

    async def initialize(self) -> None:
        """
        Initialize EventBus service.

        Connects to Kafka and starts background consumer task.

        Raises:
            OnexError: If initialization fails
        """
        if self._initialized:
            logger.warning("EventBus already initialized")
            return

        try:
            # Ensure Kafka client is connected
            if not self.kafka_client.is_connected:
                await self.kafka_client.connect()

            # Start background consumer task
            self._consumer_task = asyncio.create_task(self._consume_events_background())

            self._initialized = True
            logger.info(
                f"EventBus initialized successfully (node_id={self.node_id}, namespace={self.namespace})"
            )

        except Exception as e:
            raise OnexError(
                error_code=EnumCoreErrorCode.INITIALIZATION_FAILED,
                message=f"EventBus initialization failed: {e!s}",
                details={
                    "node_id": self.node_id,
                    "namespace": self.namespace,
                    "error_type": type(e).__name__,
                },
            ) from e

    async def shutdown(self) -> None:
        """
        Shutdown EventBus service.

        Stops background consumer task and disconnects from Kafka.
        """
        if not self._initialized:
            return

        try:
            # Cancel background consumer task
            if self._consumer_task and not self._consumer_task.done():
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass

            # Clear event listeners
            self._event_listeners.clear()

            self._initialized = False
            logger.info("EventBus shutdown complete")

        except Exception as e:
            logger.warning(f"EventBus shutdown encountered error: {e}")

    async def publish_action_event(
        self,
        correlation_id: UUID,
        action_type: str,
        payload: dict[str, Any],
    ) -> bool:
        """
        Publish Action event to trigger reducer processing.

        Args:
            correlation_id: Workflow correlation ID
            action_type: Type of action (e.g., "AGGREGATE_STAMPS")
            payload: Action payload data

        Returns:
            True if published successfully, False otherwise

        Raises:
            OnexError: If publishing fails critically
        """
        if not self._initialized:
            raise OnexError(
                error_code=EnumCoreErrorCode.INVALID_STATE,
                message="EventBus not initialized - call initialize() first",
                details={"node_id": self.node_id},
            )

        try:
            # Create action event payload
            action_payload = {
                "correlation_id": str(correlation_id),
                "action_type": action_type,
                "payload": payload,
                "timestamp": datetime.now(UTC).isoformat(),
                "source_node": self.node_id,
            }

            # Publish with OnexEnvelopeV1 wrapping
            success = await self.kafka_client.publish_with_envelope(
                event_type="ACTION",
                source_node_id=str(self.node_id),
                payload=action_payload,
                topic=f"{self.namespace}.omninode_bridge.onex.evt.action.v1",
                correlation_id=correlation_id,
                metadata={
                    "action_type": action_type,
                    "namespace": self.namespace,
                },
            )

            if success:
                self._events_published += 1
                event_bus_published_total.labels(event_type="ACTION").inc()
                logger.info(
                    f"Published Action event (correlation_id={correlation_id}, action_type={action_type})"
                )
            else:
                logger.warning(
                    f"Failed to publish Action event (correlation_id={correlation_id})"
                )

            return success

        except Exception as e:
            raise OnexError(
                error_code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"Failed to publish Action event: {e!s}",
                details={
                    "node_id": self.node_id,
                    "correlation_id": str(correlation_id),
                    "action_type": action_type,
                    "error_type": type(e).__name__,
                },
            ) from e

    async def wait_for_completion(
        self,
        correlation_id: UUID,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """
        Wait for StateCommitted or ReducerGaveUp event with timeout.

        Args:
            correlation_id: Workflow correlation ID to filter events
            timeout_seconds: Maximum wait time in seconds

        Returns:
            Event data if received, empty dict on timeout

        Raises:
            OnexError: If event indicates failure or critical error
        """
        if not self._initialized:
            raise OnexError(
                error_code=EnumCoreErrorCode.INVALID_STATE,
                message="EventBus not initialized - call initialize() first",
                details={"node_id": self.node_id},
            )

        correlation_id_str = str(correlation_id)

        # Create event queue for this correlation ID
        event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._event_listeners[correlation_id_str] = event_queue

        # Track wait time
        start_time = time.time()

        try:
            # Wait for event with timeout
            event = await asyncio.wait_for(
                event_queue.get(),
                timeout=timeout_seconds,
            )

            # Record wait time
            wait_time_ms = (time.time() - start_time) * 1000
            event_bus_wait_time_ms.observe(wait_time_ms)

            logger.info(
                f"Received event for correlation_id={correlation_id_str}: {event.get('event_type')}"
            )

            # Check event type
            event_type = event.get("event_type")

            if event_type == "STATE_COMMITTED":
                # Success - return event data
                return event

            elif event_type == "REDUCER_GAVE_UP":
                # Reducer gave up - raise error
                raise OnexError(
                    error_code=EnumCoreErrorCode.OPERATION_FAILED,
                    message="Reducer gave up processing workflow",
                    details={
                        "node_id": self.node_id,
                        "correlation_id": correlation_id_str,
                        "event_data": event,
                    },
                )

            else:
                # Unknown event type
                logger.warning(
                    f"Received unexpected event type: {event_type} for correlation_id={correlation_id_str}"
                )
                return event

        except TimeoutError:
            # Timeout - increment metric and return empty dict
            self._events_timeout += 1
            event_bus_timeout_total.inc()
            logger.warning(
                f"Timeout waiting for event (correlation_id={correlation_id_str}, timeout={timeout_seconds}s)"
            )
            return {}

        finally:
            # Clean up event listener
            self._event_listeners.pop(correlation_id_str, None)

    async def _consume_events_background(self) -> None:
        """
        Background task to consume events from Kafka topics.

        Subscribes to:
        - StateCommitted events (success path)
        - ReducerGaveUp events (failure path)

        Routes events to correlation-based listeners.
        """
        try:
            # Topics to subscribe to
            topics = [
                f"{self.namespace}.omninode_bridge.onex.evt.state-committed.v1",
                f"{self.namespace}.omninode_bridge.onex.evt.reducer-gave-up.v1",
            ]

            logger.info(
                f"Starting background event consumer (topics={topics}, node_id={self.node_id})"
            )

            # Consume events in a loop
            while True:
                try:
                    # Check if Kafka is connected before consuming
                    if not self.kafka_client.is_connected:
                        logger.debug(
                            "Kafka client not connected - event bus consumer in degraded mode (node_id=%s)",
                            self.node_id,
                        )
                        # Sleep to avoid tight loop when Kafka is unavailable
                        await asyncio.sleep(5.0)
                        continue

                    # Consume messages from all topics
                    for topic in topics:
                        messages = await self.kafka_client.consume_messages(
                            topic=topic,
                            group_id=f"{self.node_id}-event-bus",
                            max_messages=10,
                            timeout_ms=1000,  # 1 second poll timeout
                        )

                        for msg in messages:
                            await self._handle_consumed_event(msg)

                    # Small delay between polling cycles
                    await asyncio.sleep(0.1)

                except asyncio.CancelledError:
                    logger.info("Background consumer cancelled")
                    raise

                except Exception as e:
                    logger.error(f"Error consuming events: {e}")
                    await asyncio.sleep(1.0)  # Backoff on error

        except asyncio.CancelledError:
            logger.info("Background event consumer stopped")

    async def _handle_consumed_event(self, message: dict[str, Any]) -> None:
        """
        Handle a consumed event from Kafka.

        Routes event to appropriate correlation-based listener.

        Args:
            message: Kafka message with event data
        """
        try:
            # Extract envelope from message value
            envelope = message.get("value", {})

            # Get correlation ID from envelope
            correlation_id = envelope.get("correlation_id")
            if not correlation_id:
                logger.warning("Received event without correlation_id, skipping")
                return

            correlation_id_str = str(correlation_id)

            # Get event type from envelope
            event_type = envelope.get("event_type", "UNKNOWN")

            # Check if we have a listener for this correlation ID
            if correlation_id_str in self._event_listeners:
                # Route event to listener
                event_data = {
                    "event_type": event_type,
                    "correlation_id": correlation_id_str,
                    "payload": envelope.get("payload", {}),
                    "metadata": envelope.get("metadata", {}),
                    "timestamp": envelope.get("timestamp"),
                }

                await self._event_listeners[correlation_id_str].put(event_data)
                self._events_consumed += 1
                event_bus_consumed_total.labels(event_type=event_type).inc()

                logger.debug(
                    f"Routed event to listener (correlation_id={correlation_id_str}, event_type={event_type})"
                )
            else:
                logger.debug(
                    f"No listener for correlation_id={correlation_id_str}, event_type={event_type}"
                )

        except Exception as e:
            logger.error(f"Error handling consumed event: {e}")

    async def health_check(self) -> dict[str, Any]:
        """
        Get EventBus health status.

        Returns:
            Health status dictionary
        """
        kafka_health = await self.kafka_client.health_check()

        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "kafka_connected": self.kafka_client.is_connected,
            "kafka_health": kafka_health,
            "consumer_task_running": (
                self._consumer_task is not None and not self._consumer_task.done()
            ),
            "active_listeners": len(self._event_listeners),
            "metrics": {
                "events_published": self._events_published,
                "events_consumed": self._events_consumed,
                "events_timeout": self._events_timeout,
            },
        }

    @property
    def is_initialized(self) -> bool:
        """Check if EventBus is initialized."""
        return self._initialized
