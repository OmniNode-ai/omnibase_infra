# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""In-Memory Event Bus implementation for local development and testing.

Implements ProtocolEventBus interface using deque-based event history with
direct subscriber callback invocation. This implementation is designed for
local development and testing scenarios where a full message broker (Kafka)
is not needed.

Features:
    - Topic-based message routing with FIFO ordering
    - Async publish/subscribe with callback handlers
    - Event history tracking for debugging and testing
    - Async-safe operations using asyncio.Lock
    - No external dependencies required
    - Support for environment/group-based routing

Usage:
    ```python
    from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus

    bus = InMemoryEventBus(environment="dev", group="test")
    await bus.start()

    # Subscribe to a topic
    async def handler(msg):
        print(f"Received: {msg.value}")
    unsubscribe = await bus.subscribe("events", "group1", handler)

    # Publish a message
    await bus.publish("events", b"key", b"value")

    # Cleanup
    await unsubscribe()
    await bus.close()
    ```

Protocol Compatibility:
    This class implements ProtocolEventBus from omnibase_core using duck typing
    (no explicit inheritance required per ONEX patterns).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Optional
from uuid import uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage

logger = logging.getLogger(__name__)


class InMemoryEventBus:
    """In-memory event bus for local development and testing.

    Implements ProtocolEventBus interface using deque-based event history
    with direct subscriber callback invocation. Async-safe operations are
    ensured via asyncio.Lock. This implementation provides a lightweight
    event bus for testing and local development without external dependencies.

    Features:
        - Topic-based message routing with FIFO ordering
        - Multiple subscribers per topic with group-based filtering
        - Event history tracking with configurable retention
        - Async-safe operations using asyncio.Lock
        - Environment and group-based message routing
        - Debugging utilities for inspecting event flow

    Attributes:
        environment: Environment identifier (e.g., "local", "dev", "test")
        group: Consumer group identifier
        adapter: Returns self (no separate adapter for in-memory)

    Example:
        ```python
        bus = InMemoryEventBus(environment="dev", group="test")
        await bus.start()

        # Subscribe
        async def handler(msg):
            print(f"Received: {msg.value}")
        unsubscribe = await bus.subscribe("events", "group1", handler)

        # Publish
        await bus.publish("events", b"key", b"value")

        # Cleanup
        await unsubscribe()
        await bus.close()
        ```
    """

    def __init__(
        self,
        environment: str = "local",
        group: str = "default",
        max_history: int = 1000,
        circuit_breaker_threshold: int = 5,
    ) -> None:
        """Initialize the in-memory event bus.

        Args:
            environment: Environment identifier for message routing
            group: Consumer group identifier for message routing
            max_history: Maximum number of events to retain in history
            circuit_breaker_threshold: Number of consecutive failures before circuit opens

        Raises:
            ValueError: If circuit_breaker_threshold is not a positive integer
        """
        if circuit_breaker_threshold < 1:
            raise ValueError(
                f"circuit_breaker_threshold must be a positive integer, got {circuit_breaker_threshold}"
            )
        self._environment = environment
        self._group = group
        self._max_history = max_history

        # Topic -> list of (group_id, callback) tuples
        self._subscribers: dict[
            str, list[tuple[str, Callable[[ModelEventMessage], Awaitable[None]]]]
        ] = defaultdict(list)

        # Event history for debugging (circular buffer with O(1) operations)
        self._event_history: deque[ModelEventMessage] = deque(maxlen=max_history)

        # Topic -> offset counter for message ordering
        self._topic_offsets: dict[str, int] = defaultdict(int)

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Started flag
        self._started = False

        # Shutdown flag for consuming loop
        self._shutdown = False

        # Subscriber failure tracking for circuit breaker pattern
        # Maps (topic, group_id) to consecutive failure count
        self._subscriber_failures: dict[tuple[str, str], int] = {}
        self._max_consecutive_failures: int = circuit_breaker_threshold

    @property
    def adapter(self) -> InMemoryEventBus:
        """No adapter for in-memory - returns self.

        Returns:
            Self reference (in-memory bus is its own adapter)
        """
        return self

    @property
    def environment(self) -> str:
        """Get the environment identifier.

        Returns:
            Environment string (e.g., "local", "dev", "test")
        """
        return self._environment

    @property
    def group(self) -> str:
        """Get the consumer group identifier.

        Returns:
            Consumer group string
        """
        return self._group

    async def start(self) -> None:
        """Start the event bus.

        Initializes internal state and marks the bus as ready for operations.
        This is a no-op for in-memory implementation but required for protocol.
        """
        async with self._lock:
            self._started = True
            self._shutdown = False
        logger.info(
            "InMemoryEventBus started",
            extra={"environment": self._environment, "group": self._group},
        )

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the event bus with configuration.

        Protocol method for compatibility with ProtocolEventBus.
        Extracts configuration and delegates to start().

        Args:
            config: Configuration dictionary with optional keys:
                - environment: Override environment setting
                - group: Override group setting
                - max_history: Override max_history setting
        """
        if "environment" in config:
            self._environment = str(config["environment"])
        if "group" in config:
            self._group = str(config["group"])
        if "max_history" in config:
            self._max_history = int(str(config["max_history"]))
            # Recreate deque with new maxlen, preserving existing history
            self._event_history = deque(self._event_history, maxlen=self._max_history)
        await self.start()

    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus.

        Protocol method that stops consuming and clears resources.
        """
        await self.close()

    async def publish(
        self,
        topic: str,
        key: Optional[bytes],
        value: bytes,
        headers: Optional[ModelEventHeaders] = None,
    ) -> None:
        """Publish message to topic.

        Delivers the message to all subscribers registered for the topic.
        Messages are delivered asynchronously but in FIFO order per subscriber.

        Args:
            topic: Target topic name
            key: Optional message key (for future partitioning support)
            value: Message payload as bytes
            headers: Optional event headers with metadata

        Raises:
            InfraUnavailableError: If the bus has not been started
        """
        if not self._started:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="publish",
                target_name=f"event_bus.{self._environment}",
                correlation_id=(
                    headers.correlation_id if headers is not None else uuid4()
                ),
            )
            raise InfraUnavailableError(
                "Event bus not started. Call start() first.",
                context=context,
                topic=topic,
            )

        # Create headers if not provided
        if headers is None:
            headers = ModelEventHeaders(
                source=f"{self._environment}.{self._group}",
                event_type=topic,
            )

        async with self._lock:
            # Get next offset for topic
            offset = self._topic_offsets[topic]
            self._topic_offsets[topic] = offset + 1

            message = ModelEventMessage(
                topic=topic,
                key=key,
                value=value,
                headers=headers,
                offset=str(offset),  # Convert int to string for Pydantic model
                partition=0,
            )

            # Add to history (deque handles maxlen automatically with O(1) performance)
            self._event_history.append(message)

            # Get subscribers snapshot
            subscribers = list(self._subscribers.get(topic, []))

        # Call subscribers outside lock to avoid deadlocks
        for group_id, callback in subscribers:
            failure_key = (topic, group_id)

            # Check if circuit is open (too many consecutive failures)
            if (
                self._subscriber_failures.get(failure_key, 0)
                >= self._max_consecutive_failures
            ):
                logger.warning(
                    "Subscriber circuit breaker open - skipping callback",
                    extra={
                        "topic": topic,
                        "group_id": group_id,
                        "consecutive_failures": self._subscriber_failures[failure_key],
                        "correlation_id": str(headers.correlation_id),
                    },
                )
                continue

            try:
                await callback(message)
                # Reset failure count on success
                if failure_key in self._subscriber_failures:
                    del self._subscriber_failures[failure_key]
            except Exception as e:
                # Increment failure count
                self._subscriber_failures[failure_key] = (
                    self._subscriber_failures.get(failure_key, 0) + 1
                )
                # Log but don't fail other subscribers
                logger.exception(
                    "Subscriber callback failed",
                    extra={
                        "topic": topic,
                        "group_id": group_id,
                        "error": str(e),
                        "consecutive_failures": self._subscriber_failures[failure_key],
                        "correlation_id": str(headers.correlation_id),
                    },
                )

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
    ) -> None:
        """Publish an OnexEnvelope to a topic.

        Protocol method for ProtocolEventBus compatibility.
        Serializes the envelope to JSON bytes and publishes.

        Args:
            envelope: Envelope object to publish (ModelOnexEnvelope)
            topic: Target topic name
        """
        # Serialize envelope to JSON bytes
        # Note: envelope is expected to have a model_dump() method (Pydantic)
        envelope_dict: object
        if hasattr(envelope, "model_dump"):
            envelope_dict = envelope.model_dump(mode="json")  # type: ignore[union-attr]
        elif hasattr(envelope, "dict"):
            envelope_dict = envelope.dict()  # type: ignore[union-attr]
        elif isinstance(envelope, dict):
            envelope_dict = envelope
        else:
            # Fallback for non-Pydantic, non-dict types (e.g., primitive JSON-serializable
            # types like str, int, list). json.dumps() will raise TypeError if the object
            # is not JSON-serializable, providing a clear error at serialization time.
            envelope_dict = envelope

        value = json.dumps(envelope_dict).encode("utf-8")

        headers = ModelEventHeaders(
            source=f"{self._environment}.{self._group}",
            event_type=topic,
            content_type="application/json",
        )

        await self.publish(topic, None, value, headers)

    async def subscribe(
        self,
        topic: str,
        group_id: str,
        on_message: Callable[[ModelEventMessage], Awaitable[None]],
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe to topic with callback handler.

        Registers a callback to be invoked for each message published to the topic.
        Returns an unsubscribe function to remove the subscription.

        Args:
            topic: Topic to subscribe to
            group_id: Consumer group identifier for this subscription
            on_message: Async callback invoked for each message

        Returns:
            Async unsubscribe function to remove this subscription

        Example:
            ```python
            async def handler(msg):
                print(f"Received: {msg.value}")

            unsubscribe = await bus.subscribe("events", "group1", handler)
            # ... later ...
            await unsubscribe()
            ```
        """
        async with self._lock:
            self._subscribers[topic].append((group_id, on_message))
            logger.debug(
                "Subscriber added",
                extra={"topic": topic, "group_id": group_id},
            )

        async def unsubscribe() -> None:
            """Remove this subscription from the topic."""
            async with self._lock:
                try:
                    self._subscribers[topic].remove((group_id, on_message))
                    logger.debug(
                        "Subscriber removed",
                        extra={"topic": topic, "group_id": group_id},
                    )
                except ValueError:
                    # Already unsubscribed
                    pass

        return unsubscribe

    async def start_consuming(self) -> None:
        """Start the consumer loop.

        Protocol method for ProtocolEventBus compatibility.
        For in-memory implementation, this is a no-op as messages are
        delivered synchronously in publish().

        This method blocks until shutdown() is called (for protocol compatibility).
        """
        if not self._started:
            await self.start()

        # For in-memory, we don't need a consuming loop since publish
        # delivers messages synchronously. But we provide an async wait
        # for protocol compatibility.
        while not self._shutdown:
            await asyncio.sleep(0.1)

    async def broadcast_to_environment(
        self,
        command: str,
        payload: dict[str, object],
        target_environment: Optional[str] = None,
    ) -> None:
        """Broadcast command to environment.

        Sends a command message to all subscribers in the target environment.

        Args:
            command: Command identifier
            payload: Command payload data
            target_environment: Target environment (defaults to current)
        """
        env = target_environment or self._environment
        topic = f"{env}.broadcast"
        value_dict = {"command": command, "payload": payload}
        value = json.dumps(value_dict).encode("utf-8")

        headers = ModelEventHeaders(
            source=f"{self._environment}.{self._group}",
            event_type="broadcast",
            content_type="application/json",
        )

        await self.publish(topic, None, value, headers)

    async def send_to_group(
        self,
        command: str,
        payload: dict[str, object],
        target_group: str,
    ) -> None:
        """Send command to specific group.

        Sends a command message to all subscribers in a specific group.

        Args:
            command: Command identifier
            payload: Command payload data
            target_group: Target group identifier
        """
        topic = f"{self._environment}.{target_group}"
        value_dict = {"command": command, "payload": payload}
        value = json.dumps(value_dict).encode("utf-8")

        headers = ModelEventHeaders(
            source=f"{self._environment}.{self._group}",
            event_type="group_command",
            content_type="application/json",
        )

        await self.publish(topic, None, value, headers)

    async def close(self) -> None:
        """Close the event bus and release resources.

        Clears all subscribers, failure tracking, and marks the bus as stopped.
        """
        async with self._lock:
            self._subscribers.clear()
            self._subscriber_failures.clear()
            self._started = False
            self._shutdown = True
        logger.info(
            "InMemoryEventBus closed",
            extra={"environment": self._environment, "group": self._group},
        )

    async def health_check(self) -> dict[str, object]:
        """Check event bus health.

        Protocol method for ProtocolEventBus compatibility.

        Returns:
            Dictionary with health status information:
                - healthy: Whether the bus is operational
                - started: Whether start() has been called
                - environment: Current environment
                - group: Current consumer group
                - subscriber_count: Total number of active subscriptions
                - topic_count: Number of topics with subscribers
                - history_size: Current event history size
        """
        async with self._lock:
            subscriber_count = sum(len(subs) for subs in self._subscribers.values())
            topic_count = len(self._subscribers)
            history_size = len(self._event_history)

        return {
            "healthy": self._started,
            "started": self._started,
            "environment": self._environment,
            "group": self._group,
            "subscriber_count": subscriber_count,
            "topic_count": topic_count,
            "history_size": history_size,
        }

    # =========================================================================
    # Debugging/Observability Methods
    # =========================================================================

    async def get_event_history(
        self,
        limit: int = 100,
        topic: Optional[str] = None,
    ) -> list[ModelEventMessage]:
        """Get recent events for debugging.

        Args:
            limit: Maximum number of events to return
            topic: Optional topic filter

        Returns:
            List of recent events (most recent last)
        """
        async with self._lock:
            # Convert deque to list for slicing operations
            history_list = list(self._event_history)
            history = (
                history_list[-limit:] if limit < len(history_list) else history_list
            )
            if topic:
                history = [msg for msg in history if msg.topic == topic]
            return list(history)

    async def clear_event_history(self) -> None:
        """Clear event history.

        Useful for test isolation between test cases.
        """
        async with self._lock:
            self._event_history.clear()
        logger.debug("Event history cleared")

    async def get_subscriber_count(self, topic: Optional[str] = None) -> int:
        """Get subscriber count, optionally filtered by topic.

        Args:
            topic: Optional topic to filter by

        Returns:
            Number of active subscriptions
        """
        async with self._lock:
            if topic:
                return len(self._subscribers.get(topic, []))
            return sum(len(subs) for subs in self._subscribers.values())

    async def get_topics(self) -> list[str]:
        """Get list of topics with active subscribers.

        Returns:
            List of topic names with at least one subscriber
        """
        async with self._lock:
            return [topic for topic, subs in self._subscribers.items() if subs]

    async def get_topic_offset(self, topic: str) -> int:
        """Get current offset for a topic.

        Args:
            topic: Topic name

        Returns:
            Current offset (number of messages published to topic)
        """
        async with self._lock:
            return self._topic_offsets.get(topic, 0)

    # =========================================================================
    # Circuit Breaker Methods
    # =========================================================================

    async def reset_subscriber_circuit(self, topic: str, group_id: str) -> bool:
        """Reset the circuit breaker for a specific subscriber.

        Clears the failure count for the specified topic/group_id combination,
        allowing the subscriber to receive messages again.

        Args:
            topic: Topic name
            group_id: Consumer group identifier

        Returns:
            True if the circuit was reset, False if there was no circuit to reset
        """
        failure_key = (topic, group_id)
        async with self._lock:
            if failure_key in self._subscriber_failures:
                del self._subscriber_failures[failure_key]
                logger.info(
                    "Subscriber circuit breaker reset",
                    extra={"topic": topic, "group_id": group_id},
                )
                return True
            return False

    async def get_circuit_breaker_status(self) -> dict[str, object]:
        """Get circuit breaker status for all subscribers.

        Returns:
            Dictionary with circuit breaker status information:
                - open_circuits: List of dicts with topic/group_id for open circuits
                - failure_counts: Dict mapping "topic:group_id" to failure count
        """
        async with self._lock:
            open_circuits = [
                {"topic": topic, "group_id": group_id}
                for (topic, group_id), count in self._subscriber_failures.items()
                if count >= self._max_consecutive_failures
            ]
            return {
                "open_circuits": open_circuits,
                "failure_counts": {
                    f"{topic}:{group_id}": count
                    for (topic, group_id), count in self._subscriber_failures.items()
                },
            }


__all__: list[str] = ["InMemoryEventBus"]
