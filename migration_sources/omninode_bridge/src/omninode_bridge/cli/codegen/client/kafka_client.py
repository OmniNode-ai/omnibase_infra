"""
Kafka Client for CLI operations.

Provides simplified Kafka producer/consumer interface for CLI tools.
Handles event publishing and consumption with correlation ID tracking.
"""

import json
from collections.abc import Callable
from typing import Any
from uuid import UUID

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from omninode_bridge.cli.codegen.config import _get_kafka_bootstrap_servers
from omninode_bridge.events.codegen import (
    KAFKA_TOPICS,
    ModelEventNodeGenerationRequested,
)


class CLIKafkaClient:
    """
    Simplified Kafka client for CLI tools.

    Features:
    - Async producer for publishing requests
    - Async consumer for tracking progress
    - Correlation ID filtering
    - Automatic reconnection

    This implementation is designed for testability with dependency injection.
    """

    def __init__(self, bootstrap_servers: str | None = None):
        """
        Initialize CLI Kafka client.

        Args:
            bootstrap_servers: Kafka broker addresses (defaults to environment config)
        """
        self.bootstrap_servers = bootstrap_servers or _get_kafka_bootstrap_servers()
        self.producer: AIOKafkaProducer | None = None
        self.consumer: AIOKafkaConsumer | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect producer to Kafka."""
        if self._connected:
            return

        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda v: str(v).encode("utf-8") if v else None,
        )

        await self.producer.start()
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()
        self._connected = False

    async def publish_request(self, event: ModelEventNodeGenerationRequested) -> None:
        """
        Publish node generation request event.

        Args:
            event: Generation request event

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected or not self.producer:
            raise RuntimeError("Kafka client not connected. Call connect() first.")

        topic = KAFKA_TOPICS["NODE_GENERATION_REQUESTED"]

        await self.producer.send(
            topic,
            value=event.model_dump(),
            key=str(event.correlation_id),
        )

    async def consume_progress_events(
        self,
        correlation_id: UUID,
        callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """
        Consume progress events for a specific correlation ID.

        Args:
            correlation_id: Correlation ID to filter events
            callback: Callback function(event_type, event_data)

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected:
            raise RuntimeError("Kafka client not connected. Call connect() first.")

        # Subscribe to relevant topics
        topics = [
            KAFKA_TOPICS["NODE_GENERATION_STARTED"],
            KAFKA_TOPICS["NODE_GENERATION_STAGE_COMPLETED"],
            KAFKA_TOPICS["NODE_GENERATION_COMPLETED"],
            KAFKA_TOPICS["NODE_GENERATION_FAILED"],
            KAFKA_TOPICS["ORCHESTRATOR_CHECKPOINT_REACHED"],
        ]

        self.consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",  # Only consume new messages
            group_id=f"cli_{correlation_id}",
        )

        await self.consumer.start()

        try:
            async for message in self.consumer:
                event_data = message.value

                # Filter by correlation ID
                if str(event_data.get("correlation_id")) != str(correlation_id):
                    continue

                event_type = event_data.get("event_type")
                callback(event_type, event_data)

                # Stop consuming after completion or failure
                if event_type in [
                    "NODE_GENERATION_COMPLETED",
                    "NODE_GENERATION_FAILED",
                ]:
                    break

        finally:
            await self.consumer.stop()

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected
