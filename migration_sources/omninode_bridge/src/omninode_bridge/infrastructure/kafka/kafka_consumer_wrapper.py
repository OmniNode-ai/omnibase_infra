"""
Kafka Consumer Wrapper implementing ProtocolKafkaConsumer.

Provides protocol-compliant Kafka consumer for event-driven database operations
following ONEX v2.0 architecture patterns.

Features:
- Wraps AIOKafkaConsumer for async Kafka operations
- Implements ProtocolKafkaConsumer from omnibase_core
- Supports environment-based topic naming
- Structured logging with correlation IDs
- Proper error handling with OnexError
- Async context manager pattern

Implementation: Phase 2, Kafka Integration
"""

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, Optional

try:
    from aiokafka import AIOKafkaConsumer, ConsumerRebalanceListener
    from aiokafka.errors import KafkaError

    AIOKAFKA_AVAILABLE = True
except ImportError:
    AIOKAFKA_AVAILABLE = False
    # Fallback for type hints when aiokafka is not installed
    AIOKafkaConsumer = Any  # type: ignore
    ConsumerRebalanceListener = Any  # type: ignore
    KafkaError = Exception  # type: ignore

from omnibase_core import EnumCoreErrorCode, ModelOnexError

OnexError = ModelOnexError

logger = logging.getLogger(__name__)


class KafkaConsumerWrapper:
    """
    Production-ready Kafka consumer wrapper implementing ProtocolKafkaConsumer.

    Wraps AIOKafkaConsumer to provide protocol-compliant event consumption
    for Database Adapter Effect node with proper error handling, logging,
    and lifecycle management.

    Protocol Implementation:
        - subscribe_to_topics(): Subscribe to Kafka topics with consumer group
        - consume_messages_stream(): Async generator for message streaming
        - commit_offsets(): Manual offset commit for at-least-once delivery
        - close_consumer(): Graceful shutdown with offset commit

    Environment Variables:
        - KAFKA_BOOTSTRAP_SERVERS: Comma-separated Kafka brokers (default: localhost:29092)
        - KAFKA_SECURITY_PROTOCOL: Security protocol (PLAINTEXT, SSL, SASL_SSL)
        - KAFKA_SASL_MECHANISM: SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512)
        - KAFKA_SASL_USERNAME: SASL username
        - KAFKA_SASL_PASSWORD: SASL password
        - OMNINODE_ENV: Environment prefix for topic naming (dev, staging, prod)
        - OMNINODE_TENANT: Tenant namespace for topics (omnibase, omninode_bridge)
        - OMNINODE_CONTEXT: Context for topics (onex, bridge)

    Topic Naming Convention:
        {env}.{tenant}.{context}.{class}.{topic_name}.{version}
        Example: dev.omninode_bridge.onex.evt.workflow-started.v1

    Usage:
        ```python
        # Create consumer
        consumer = KafkaConsumerWrapper()

        # Subscribe to topics
        await consumer.subscribe_to_topics(
            topics=["workflow-started", "metadata-stamp-created"],
            group_id="database_adapter_consumers"
        )

        # Consume events in batches
        async for messages in consumer.consume_messages_stream(
            batch_timeout_ms=1000, max_records=500
        ):
            for message in messages:
                # Process message
                await process_event(message)
            # Commit offsets after successful batch processing
            await consumer.commit_offsets()

        # Cleanup
        await consumer.close_consumer()
        ```
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        security_protocol: str = "PLAINTEXT",
        sasl_mechanism: Optional[str] = None,
        sasl_username: Optional[str] = None,
        sasl_password: Optional[str] = None,
    ):
        """
        Initialize Kafka consumer wrapper.

        Args:
            bootstrap_servers: Kafka bootstrap servers (comma-separated)
            security_protocol: Security protocol (PLAINTEXT, SSL, SASL_SSL)
            sasl_mechanism: SASL mechanism (PLAIN, SCRAM-SHA-256, etc.)
            sasl_username: SASL username
            sasl_password: SASL password

        Raises:
            OnexError: If aiokafka is not available
        """
        # Check aiokafka availability
        if not AIOKAFKA_AVAILABLE:
            raise OnexError(
                error_code=EnumCoreErrorCode.DEPENDENCY_ERROR,
                message="aiokafka not installed - required for Kafka consumer",
                context={"install_command": "pip install aiokafka"},
            )

        # Load configuration from environment or parameters
        # Default to remote infrastructure (resolves to 192.168.86.200:9092 via /etc/hosts)
        self._bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
        )
        self._security_protocol = security_protocol or os.getenv(
            "KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"
        )
        self._sasl_mechanism = sasl_mechanism or os.getenv("KAFKA_SASL_MECHANISM")
        self._sasl_username = sasl_username or os.getenv("KAFKA_SASL_USERNAME")
        self._sasl_password = sasl_password or os.getenv("KAFKA_SASL_PASSWORD")

        # Environment-based topic naming
        self._env = os.getenv("OMNINODE_ENV", "dev")
        self._tenant = os.getenv("OMNINODE_TENANT", "omninode_bridge")
        self._context = os.getenv("OMNINODE_CONTEXT", "onex")

        # Consumer state
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._subscribed_topics: list[str] = []
        self._consumer_group: Optional[str] = None
        self._is_running = False

        logger.info(
            "KafkaConsumerWrapper initialized",
            extra={
                "bootstrap_servers": self._bootstrap_servers,
                "security_protocol": self._security_protocol,
                "env": self._env,
                "tenant": self._tenant,
                "context": self._context,
            },
        )

    def _build_topic_name(self, short_name: str, topic_class: str = "evt") -> str:
        """
        Build full Kafka topic name from short name.

        Args:
            short_name: Short topic name (e.g., "workflow-started")
            topic_class: Topic class (evt, cmd, qrs, doc)

        Returns:
            Full topic name following OmniNode convention

        Example:
            _build_topic_name("workflow-started", "evt")
            → "dev.omninode_bridge.onex.evt.workflow-started.v1"
        """
        return (
            f"{self._env}.{self._tenant}.{self._context}.{topic_class}.{short_name}.v1"
        )

    async def subscribe_to_topics(
        self, topics: list[str], group_id: str, topic_class: str = "evt"
    ) -> None:
        """
        Subscribe to Kafka topics with consumer group.

        Creates AIOKafkaConsumer and subscribes to specified topics.
        Topic names are automatically expanded to full OmniNode format.

        Args:
            topics: List of short topic names (e.g., ["workflow-started", "stamp-created"])
            group_id: Consumer group ID for offset coordination
            topic_class: Topic class for naming (evt, cmd, qrs, doc)

        Raises:
            OnexError: If subscription fails or consumer already subscribed
        """
        if self._consumer is not None:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_FAILED,
                message="Consumer already subscribed - call close_consumer() first",
                context={
                    "current_group": self._consumer_group,
                    "current_topics": self._subscribed_topics,
                },
            )

        try:
            # Build full topic names
            full_topic_names = [
                self._build_topic_name(topic, topic_class) for topic in topics
            ]

            logger.info(
                "Subscribing to Kafka topics",
                extra={
                    "group_id": group_id,
                    "topics": full_topic_names,
                    "topic_class": topic_class,
                },
            )

            # Build security configuration
            security_config = self._build_security_config()

            # Create AIOKafkaConsumer
            self._consumer = AIOKafkaConsumer(
                *full_topic_names,
                bootstrap_servers=self._bootstrap_servers.split(","),
                group_id=group_id,
                auto_offset_reset="latest",  # Start from latest on first connect
                enable_auto_commit=False,  # Manual commit for control
                value_deserializer=lambda v: (
                    json.loads(v.decode("utf-8")) if v else None
                ),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                **security_config,
            )

            # Start consumer
            await self._consumer.start()

            # Store subscription info
            self._subscribed_topics = full_topic_names
            self._consumer_group = group_id
            self._is_running = True

            logger.info(
                "Successfully subscribed to Kafka topics",
                extra={
                    "group_id": group_id,
                    "topics": full_topic_names,
                    "bootstrap_servers": self._bootstrap_servers,
                },
            )

        except KafkaError as e:
            logger.error(f"Kafka subscription failed: {e}", exc_info=True)
            raise OnexError(
                error_code=EnumCoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Failed to subscribe to Kafka topics: {e!s}",
                context={
                    "group_id": group_id,
                    "topics": topics,
                    "bootstrap_servers": self._bootstrap_servers,
                },
            ) from e
        except Exception as e:
            logger.error(f"Consumer subscription failed: {e}", exc_info=True)
            raise OnexError(
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                message=f"Unexpected error subscribing to topics: {e!s}",
                context={"group_id": group_id, "topics": topics},
            ) from e

    def _build_security_config(self) -> dict[str, Any]:
        """
        Build Kafka security configuration from environment.

        Returns:
            Dictionary with security configuration for AIOKafkaConsumer
        """
        config: dict[str, Any] = {}

        # Security protocol
        if self._security_protocol and self._security_protocol != "PLAINTEXT":
            config["security_protocol"] = self._security_protocol

        # SASL configuration
        if self._sasl_mechanism:
            config["sasl_mechanism"] = self._sasl_mechanism
            if self._sasl_username:
                config["sasl_plain_username"] = self._sasl_username
            if self._sasl_password:
                config["sasl_plain_password"] = self._sasl_password

        return config

    async def consume_messages_stream(
        self, batch_timeout_ms: int = 1000, max_records: int = 500
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """
        Async generator for consuming message batches from Kafka.

        Yields batches of messages for processing. Use with manual offset commit
        to ensure at-least-once delivery semantics.

        Args:
            batch_timeout_ms: Maximum time to wait for batch (milliseconds)
            max_records: Maximum number of records to fetch per batch

        Yields:
            List of messages as dictionaries with keys:
            - key: Message key (str or None)
            - value: Deserialized message value (dict)
            - topic: Topic name (str)
            - partition: Partition number (int)
            - offset: Message offset (int)
            - timestamp: Message timestamp (int, milliseconds since epoch)

        Raises:
            OnexError: If consumer not subscribed or consumption fails

        Example:
            async for messages in consumer.consume_messages_stream():
                for msg in messages:
                    print(f"Topic: {msg['topic']}, Value: {msg['value']}")
                await consumer.commit_offsets()
        """
        if not self._consumer or not self._is_running:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_FAILED,
                message="Consumer not subscribed - call subscribe_to_topics() first",
            )

        try:
            logger.info(
                "Starting message consumption stream",
                extra={
                    "topics": self._subscribed_topics,
                    "group_id": self._consumer_group,
                    "batch_timeout_ms": batch_timeout_ms,
                    "max_records": max_records,
                },
            )

            # Consume messages in batches using getmany()
            while self._is_running:
                # Fetch batch of messages with timeout
                # Returns: Dict[TopicPartition, List[ConsumerRecord]]
                data = await self._consumer.getmany(
                    timeout_ms=batch_timeout_ms, max_records=max_records
                )

                # Skip empty batches (timeout with no messages)
                if not data:
                    continue

                # Convert ConsumerRecords to dictionaries
                batch_messages: list[dict[str, Any]] = []
                for topic_partition, messages in data.items():
                    for msg in messages:
                        message_dict = {
                            "key": msg.key,
                            "value": msg.value,
                            "topic": msg.topic,
                            "partition": msg.partition,
                            "offset": msg.offset,
                            "timestamp": msg.timestamp,
                            "headers": dict(msg.headers) if msg.headers else {},
                        }
                        batch_messages.append(message_dict)

                # Yield batch of messages
                if batch_messages:
                    logger.debug(
                        "Yielding message batch",
                        extra={
                            "batch_size": len(batch_messages),
                            "partitions": len(data),
                        },
                    )
                    yield batch_messages

        except KafkaError as e:
            logger.error(f"Kafka consumption error: {e}", exc_info=True)
            raise OnexError(
                error_code=EnumCoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Kafka message consumption failed: {e!s}",
                context={
                    "topics": self._subscribed_topics,
                    "group_id": self._consumer_group,
                },
            ) from e
        except asyncio.CancelledError:
            logger.info("Message consumption cancelled - shutting down gracefully")
            raise
        except Exception as e:
            logger.error(f"Unexpected consumption error: {e}", exc_info=True)
            raise OnexError(
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                message=f"Unexpected error during message consumption: {e!s}",
            ) from e

    async def commit_offsets(self) -> None:
        """
        Manually commit consumer offsets to Kafka.

        Call after successfully processing a batch of messages to ensure
        at-least-once delivery semantics.

        Raises:
            OnexError: If consumer not subscribed or commit fails
        """
        if not self._consumer or not self._is_running:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_FAILED,
                message="Consumer not subscribed - cannot commit offsets",
            )

        try:
            await self._consumer.commit()
            logger.debug("Offsets committed successfully")

        except KafkaError as e:
            logger.error(f"Offset commit failed: {e}", exc_info=True)
            raise OnexError(
                error_code=EnumCoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Failed to commit Kafka offsets: {e!s}",
                context={"group_id": self._consumer_group},
            ) from e
        except Exception as e:
            logger.error(f"Unexpected commit error: {e}", exc_info=True)
            raise OnexError(
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                message=f"Unexpected error committing offsets: {e!s}",
            ) from e

    async def close_consumer(self) -> None:
        """
        Gracefully close Kafka consumer with offset commit.

        Commits pending offsets and stops consumer. Safe to call multiple times.
        """
        if not self._consumer:
            logger.debug("Consumer already closed or never started")
            return

        try:
            logger.info(
                "Closing Kafka consumer",
                extra={
                    "group_id": self._consumer_group,
                    "topics": self._subscribed_topics,
                },
            )

            # Commit final offsets
            if self._is_running:
                try:
                    await self._consumer.commit()
                    logger.debug("Final offsets committed")
                except Exception as e:
                    logger.warning(f"Final offset commit failed: {e}")

            # Stop consumer
            await self._consumer.stop()
            self._is_running = False

            logger.info(
                "Kafka consumer closed successfully",
                extra={"group_id": self._consumer_group},
            )

        except Exception as e:
            logger.error(f"Error closing consumer: {e}", exc_info=True)
            # Don't raise - cleanup should be best-effort
        finally:
            self._consumer = None
            self._subscribed_topics = []
            self._consumer_group = None
            self._is_running = False

    @property
    def is_subscribed(self) -> bool:
        """Check if consumer is currently subscribed and running."""
        return self._is_running and self._consumer is not None

    @property
    def subscribed_topics(self) -> list[str]:
        """Get list of currently subscribed topics."""
        return self._subscribed_topics.copy()

    @property
    def consumer_group(self) -> Optional[str]:
        """Get current consumer group ID."""
        return self._consumer_group
