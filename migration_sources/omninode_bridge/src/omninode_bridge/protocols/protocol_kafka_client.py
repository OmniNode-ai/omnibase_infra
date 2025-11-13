"""
Kafka Client Protocol for type-safe Kafka operations.

This module defines the KafkaClientProtocol for structural typing
of Kafka clients, enabling type-safe dependency injection without
tight coupling to specific Kafka client implementations.

ONEX Compliance:
- Suffix-based naming: KafkaClientProtocol (Protocol suffix)
- Protocol-based typing for flexibility
- Async-first design

Usage:
    ```python
    from omninode_bridge.protocols import KafkaClientProtocol

    async def publish_event(
        kafka: KafkaClientProtocol,
        topic: str,
        event: dict[str, Any]
    ) -> None:
        if kafka.is_connected:
            await kafka.publish(topic=topic, value=event)
    ```
"""

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class KafkaClientProtocol(Protocol):
    """
    Protocol for Kafka clients supporting async event publishing.

    This protocol defines the minimal interface required for Kafka
    operations in the omninode_bridge infrastructure. Any Kafka client
    implementing these methods can be used with components requiring
    event publishing capabilities.

    Properties:
        is_connected: Check if client is connected to Kafka cluster

    Methods:
        publish: Publish event to Kafka topic with optional key

    Example:
        ```python
        from omninode_bridge.protocols import KafkaClientProtocol
        from omninode_bridge.services.kafka_client import KafkaClient

        # KafkaClient implements KafkaClientProtocol protocol
        kafka: KafkaClientProtocol = KafkaClient()
        await kafka.connect()

        # Type-safe event publishing
        if kafka.is_connected:
            await kafka.publish(
                topic="events.workflow.started",
                value={"workflow_id": "123", "status": "started"},
                key="123"
            )
        ```

    Protocol Benefits:
        - Type safety without tight coupling to KafkaClient implementation
        - Easy testing with mock implementations
        - Flexibility for different Kafka client implementations
        - Static type checking with mypy/pyright
        - Duck typing support for workflow orchestration
    """

    @property
    def is_connected(self) -> bool:
        """
        Check if client is connected to Kafka cluster.

        Returns:
            True if connected and ready to publish, False otherwise

        Example:
            ```python
            if kafka.is_connected:
                await kafka.publish(topic="events", value=event_data)
            else:
                logger.warning("Kafka client not connected")
            ```
        """
        ...

    async def publish(
        self, topic: str, value: dict[str, Any], key: Optional[str] = None
    ) -> None:
        """
        Publish event to Kafka topic with optional partition key.

        Args:
            topic: Kafka topic name (e.g., "events.workflow.started")
            value: Event payload as dictionary
            key: Optional partition key for ordering guarantees

        Raises:
            Exception: Kafka-specific exceptions for publish failures

        Example:
            ```python
            await kafka.publish(
                topic="events.workflow.completed",
                value={
                    "workflow_id": "abc-123",
                    "status": "completed",
                    "duration_seconds": 42.5
                },
                key="abc-123"  # Same key ensures ordering
            )
            ```

        Note:
            When a key is provided, messages with the same key are guaranteed
            to be delivered to the same partition, preserving message ordering
            for that key. Without a key, messages are distributed across
            partitions using round-robin or other partitioning strategies.
        """
        ...
