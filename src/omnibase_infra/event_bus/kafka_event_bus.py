# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka Event Bus implementation for production message streaming.

Implements ProtocolEventBus interface using Apache Kafka (via aiokafka) for
production-grade message delivery with resilience patterns including circuit
breaker, retry with exponential backoff, and dead letter queue support.

Features:
    - Topic-based message routing with Kafka partitioning
    - Async publish/subscribe with callback handlers
    - Circuit breaker for connection failure protection
    - Retry with exponential backoff on publish failures
    - Graceful degradation when Kafka is unavailable
    - Support for environment/group-based routing
    - Proper producer/consumer lifecycle management

Usage:
    ```python
    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus

    bus = KafkaEventBus(bootstrap_servers="localhost:9092", environment="dev", group="test")
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
import random
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker state machine."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class KafkaEventBus:
    """Kafka-backed event bus for production message streaming.

    Implements ProtocolEventBus interface using Apache Kafka (via aiokafka)
    with resilience patterns including circuit breaker, retry with exponential
    backoff, and graceful degradation when Kafka is unavailable.

    Features:
        - Topic-based message routing with Kafka partitioning
        - Multiple subscribers per topic with callback-based delivery
        - Circuit breaker for connection failure protection
        - Retry with exponential backoff on publish failures
        - Environment and group-based message routing
        - Proper async producer/consumer lifecycle management

    Attributes:
        environment: Environment identifier (e.g., "local", "dev", "prod")
        group: Consumer group identifier
        adapter: Returns self (for protocol compatibility)

    Example:
        ```python
        bus = KafkaEventBus(bootstrap_servers="kafka:9092", environment="dev")
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
        config: Optional[ModelKafkaEventBusConfig] = None,
        # Backwards compatibility parameters (override config if provided)
        bootstrap_servers: Optional[str] = None,
        environment: Optional[str] = None,
        group: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_retry_attempts: Optional[int] = None,
        retry_backoff_base: Optional[float] = None,
        circuit_breaker_threshold: Optional[int] = None,
        circuit_breaker_reset_timeout: Optional[float] = None,
    ) -> None:
        """Initialize the Kafka event bus.

        Args:
            config: Configuration model containing all settings. If not provided,
                defaults are used with environment variable overrides.
            bootstrap_servers: Override bootstrap servers from config
            environment: Override environment identifier from config
            group: Override consumer group identifier from config
            timeout_seconds: Override timeout from config
            max_retry_attempts: Override max retry attempts from config
            retry_backoff_base: Override retry backoff base from config
            circuit_breaker_threshold: Override circuit breaker threshold from config
            circuit_breaker_reset_timeout: Override circuit breaker reset timeout from config

        Raises:
            ProtocolConfigurationError: If circuit_breaker_threshold is not a positive integer

        Example:
            ```python
            # Using config model (recommended)
            config = ModelKafkaEventBusConfig(
                bootstrap_servers="kafka:9092",
                environment="prod",
            )
            bus = KafkaEventBus(config=config)

            # Using factory methods
            bus = KafkaEventBus.default()
            bus = KafkaEventBus.from_yaml(Path("kafka.yaml"))

            # Backwards compatible direct parameters
            bus = KafkaEventBus(bootstrap_servers="kafka:9092", environment="dev")
            ```
        """
        # Use provided config or create default with environment overrides
        if config is None:
            config = ModelKafkaEventBusConfig.default()

        # Store config reference
        self._config = config

        # Apply parameter overrides for backwards compatibility
        self._bootstrap_servers = (
            bootstrap_servers
            if bootstrap_servers is not None
            else config.bootstrap_servers
        )
        self._environment = (
            environment if environment is not None else config.environment
        )
        self._group = group if group is not None else config.group
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else config.timeout_seconds
        )
        self._max_retry_attempts = (
            max_retry_attempts
            if max_retry_attempts is not None
            else config.max_retry_attempts
        )
        self._retry_backoff_base = (
            retry_backoff_base
            if retry_backoff_base is not None
            else config.retry_backoff_base
        )

        # Circuit breaker configuration with override support
        effective_threshold = (
            circuit_breaker_threshold
            if circuit_breaker_threshold is not None
            else config.circuit_breaker_threshold
        )
        if effective_threshold < 1:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="init",
                target_name="kafka_event_bus",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                f"circuit_breaker_threshold must be a positive integer, got {effective_threshold}",
                context=context,
                parameter="circuit_breaker_threshold",
                value=effective_threshold,
            )
        self._circuit_breaker_threshold = effective_threshold
        self._circuit_breaker_reset_timeout = (
            circuit_breaker_reset_timeout
            if circuit_breaker_reset_timeout is not None
            else config.circuit_breaker_reset_timeout
        )
        self._circuit_state = CircuitState.CLOSED
        self._circuit_failure_count = 0
        self._circuit_last_failure_time: float = 0.0

        # Kafka producer and consumer
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumers: dict[str, AIOKafkaConsumer] = {}

        # Subscriber registry: topic -> list of (group_id, subscription_id, callback) tuples
        self._subscribers: dict[
            str, list[tuple[str, str, Callable[[ModelEventMessage], Awaitable[None]]]]
        ] = defaultdict(list)

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # State flags
        self._started = False
        self._shutdown = False

        # Background consumer tasks
        self._consumer_tasks: dict[str, asyncio.Task[None]] = {}

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_config(cls, config: ModelKafkaEventBusConfig) -> KafkaEventBus:
        """Create KafkaEventBus from a configuration model.

        Args:
            config: Configuration model containing all settings

        Returns:
            KafkaEventBus instance configured with the provided settings

        Example:
            ```python
            config = ModelKafkaEventBusConfig(
                bootstrap_servers="kafka:9092",
                environment="prod",
                timeout_seconds=60,
            )
            bus = KafkaEventBus.from_config(config)
            ```
        """
        return cls(config=config)

    @classmethod
    def from_yaml(cls, path: Path) -> KafkaEventBus:
        """Create KafkaEventBus from a YAML configuration file.

        Loads configuration from a YAML file with environment variable
        overrides applied automatically.

        Args:
            path: Path to YAML configuration file

        Returns:
            KafkaEventBus instance configured from the YAML file

        Raises:
            FileNotFoundError: If the YAML file does not exist
            ValueError: If the YAML content is invalid

        Example:
            ```python
            bus = KafkaEventBus.from_yaml(Path("/etc/kafka/config.yaml"))
            ```
        """
        config = ModelKafkaEventBusConfig.from_yaml(path)
        return cls(config=config)

    @classmethod
    def default(cls) -> KafkaEventBus:
        """Create KafkaEventBus with default configuration.

        Creates an instance with default settings and environment variable
        overrides applied automatically. This is the recommended way to
        create a KafkaEventBus for most use cases.

        Returns:
            KafkaEventBus instance with default configuration

        Example:
            ```python
            bus = KafkaEventBus.default()
            await bus.start()
            ```
        """
        return cls(config=ModelKafkaEventBusConfig.default())

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> ModelKafkaEventBusConfig:
        """Get the configuration model.

        Returns:
            Configuration model instance used by this event bus
        """
        return self._config

    @property
    def adapter(self) -> KafkaEventBus:
        """Return self for protocol compatibility.

        Returns:
            Self reference (Kafka bus is its own adapter)
        """
        return self

    @property
    def environment(self) -> str:
        """Get the environment identifier.

        Returns:
            Environment string (e.g., "local", "dev", "prod")
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
        """Start the event bus and connect to Kafka.

        Initializes the Kafka producer with connection retry and circuit
        breaker protection. If connection fails, the bus operates in
        degraded mode where publishes will fail gracefully.

        Raises:
            InfraConnectionError: If connection fails after all retries and
                circuit breaker is open
        """
        if self._started:
            logger.debug("KafkaEventBus already started")
            return

        async with self._lock:
            if self._started:
                return

            # Check circuit breaker before attempting connection
            # No correlation_id available during startup - generate new if needed
            self._check_circuit_breaker()

            try:
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self._bootstrap_servers,
                    acks="all",
                    enable_idempotence=True,
                )

                await asyncio.wait_for(
                    self._producer.start(),
                    timeout=self._timeout_seconds,
                )

                self._started = True
                self._shutdown = False
                self._reset_circuit_breaker()

                logger.info(
                    "KafkaEventBus started",
                    extra={
                        "environment": self._environment,
                        "group": self._group,
                        "bootstrap_servers": self._bootstrap_servers,
                    },
                )

            except TimeoutError as e:
                # Clean up producer on failure to prevent resource leak
                self._producer = None
                self._record_circuit_failure()
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="start",
                    target_name=f"kafka.{self._bootstrap_servers}",
                    correlation_id=uuid4(),
                )
                logger.warning(
                    f"Timeout connecting to Kafka after {self._timeout_seconds}s",
                    extra={"bootstrap_servers": self._bootstrap_servers},
                )
                raise InfraTimeoutError(
                    f"Timeout connecting to Kafka after {self._timeout_seconds}s",
                    context=context,
                    bootstrap_servers=self._bootstrap_servers,
                    timeout_seconds=self._timeout_seconds,
                ) from e

            except Exception as e:
                # Clean up producer on failure to prevent resource leak
                self._producer = None
                self._record_circuit_failure()
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="start",
                    target_name=f"kafka.{self._bootstrap_servers}",
                    correlation_id=uuid4(),
                )
                logger.warning(
                    f"Failed to connect to Kafka: {e}",
                    extra={
                        "bootstrap_servers": self._bootstrap_servers,
                        "error": str(e),
                    },
                )
                raise InfraConnectionError(
                    f"Failed to connect to Kafka: {e}",
                    context=context,
                    bootstrap_servers=self._bootstrap_servers,
                ) from e

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the event bus with configuration.

        Protocol method for compatibility with ProtocolEventBus.
        Extracts configuration and delegates to start(). Config updates
        are applied before start() which acquires its own lock.

        Args:
            config: Configuration dictionary with optional keys:
                - environment: Override environment setting
                - group: Override group setting
                - bootstrap_servers: Override bootstrap servers
                - timeout_seconds: Override timeout setting
        """
        # Apply config updates, then call start() which handles locking
        if "environment" in config:
            self._environment = str(config["environment"])
        if "group" in config:
            self._group = str(config["group"])
        if "bootstrap_servers" in config:
            self._bootstrap_servers = str(config["bootstrap_servers"])
        if "timeout_seconds" in config:
            self._timeout_seconds = int(str(config["timeout_seconds"]))

        await self.start()

    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus.

        Protocol method that stops consuming and closes connections.
        """
        await self.close()

    async def close(self) -> None:
        """Close the event bus and release all resources.

        Stops all background consumer tasks, closes all consumers, and
        stops the producer. Safe to call multiple times.
        """
        async with self._lock:
            self._shutdown = True
            self._started = False

            # Cancel all consumer tasks
            for task in self._consumer_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self._consumer_tasks.clear()

            # Close all consumers
            for consumer in self._consumers.values():
                try:
                    await consumer.stop()
                except Exception as e:
                    logger.warning(f"Error stopping consumer: {e}")

            self._consumers.clear()

            # Close producer
            if self._producer is not None:
                try:
                    await self._producer.stop()
                except Exception as e:
                    logger.warning(f"Error stopping producer: {e}")
                self._producer = None

            # Clear subscribers
            self._subscribers.clear()

        logger.info(
            "KafkaEventBus closed",
            extra={"environment": self._environment, "group": self._group},
        )

    async def publish(
        self,
        topic: str,
        key: Optional[bytes],
        value: bytes,
        headers: Optional[ModelEventHeaders] = None,
    ) -> None:
        """Publish message to topic.

        Publishes a message to the specified Kafka topic with retry and
        circuit breaker protection.

        Args:
            topic: Target topic name
            key: Optional message key (for partitioning)
            value: Message payload as bytes
            headers: Optional event headers with metadata

        Raises:
            InfraUnavailableError: If the bus has not been started
            InfraConnectionError: If publish fails after all retries
        """
        if not self._started:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="publish",
                target_name=f"kafka.{self._environment}",
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

        # Check circuit breaker - propagate correlation_id from headers
        self._check_circuit_breaker(correlation_id=headers.correlation_id)

        # Convert headers to Kafka format
        kafka_headers = self._model_headers_to_kafka(headers)

        # Publish with retry
        await self._publish_with_retry(topic, key, value, kafka_headers, headers)

    async def _publish_with_retry(
        self,
        topic: str,
        key: Optional[bytes],
        value: bytes,
        kafka_headers: list[tuple[str, bytes]],
        headers: ModelEventHeaders,
    ) -> None:
        """Publish message with exponential backoff retry.

        Args:
            topic: Target topic name
            key: Optional message key
            value: Message payload
            kafka_headers: Kafka-formatted headers
            headers: Original headers model

        Raises:
            InfraConnectionError: If publish fails after all retries
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self._max_retry_attempts + 1):
            try:
                if self._producer is None:
                    raise InfraConnectionError(
                        "Kafka producer not initialized",
                        context=ModelInfraErrorContext(
                            transport_type=EnumInfraTransportType.KAFKA,
                            operation="publish",
                            target_name=f"kafka.{topic}",
                            correlation_id=headers.correlation_id,
                        ),
                    )

                future = await self._producer.send(
                    topic,
                    value=value,
                    key=key,
                    headers=kafka_headers,
                )
                record_metadata = await asyncio.wait_for(
                    future,
                    timeout=self._timeout_seconds,
                )

                # Success - reset circuit breaker
                self._reset_circuit_breaker()

                logger.debug(
                    f"Published to topic {topic}",
                    extra={
                        "partition": record_metadata.partition,
                        "offset": record_metadata.offset,
                        "correlation_id": str(headers.correlation_id),
                    },
                )
                return

            except TimeoutError as e:
                # Clean up producer on failure to prevent resource leak
                self._producer = None
                last_exception = e
                self._record_circuit_failure()
                logger.warning(
                    f"Publish timeout (attempt {attempt + 1}/{self._max_retry_attempts + 1})",
                    extra={
                        "topic": topic,
                        "correlation_id": str(headers.correlation_id),
                    },
                )

            except KafkaError as e:
                last_exception = e
                self._record_circuit_failure()
                logger.warning(
                    f"Kafka error on publish (attempt {attempt + 1}/{self._max_retry_attempts + 1}): {e}",
                    extra={
                        "topic": topic,
                        "correlation_id": str(headers.correlation_id),
                    },
                )

            except Exception as e:
                last_exception = e
                self._record_circuit_failure()
                logger.warning(
                    f"Publish error (attempt {attempt + 1}/{self._max_retry_attempts + 1}): {e}",
                    extra={
                        "topic": topic,
                        "correlation_id": str(headers.correlation_id),
                    },
                )

            # Calculate backoff with jitter
            if attempt < self._max_retry_attempts:
                delay = self._retry_backoff_base * (2**attempt)
                jitter = random.uniform(0.5, 1.5)
                delay *= jitter
                await asyncio.sleep(delay)

        # All retries exhausted - differentiate timeout vs connection errors
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="publish",
            target_name=f"kafka.{topic}",
            correlation_id=headers.correlation_id,
        )
        if isinstance(last_exception, TimeoutError):
            raise InfraTimeoutError(
                f"Timeout publishing to topic {topic} after {self._max_retry_attempts + 1} attempts",
                context=context,
                topic=topic,
                retry_count=self._max_retry_attempts + 1,
                timeout_seconds=self._timeout_seconds,
            ) from last_exception
        raise InfraConnectionError(
            f"Failed to publish to topic {topic} after {self._max_retry_attempts + 1} attempts",
            context=context,
            topic=topic,
            retry_count=self._max_retry_attempts + 1,
        ) from last_exception

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
        envelope_dict: object
        if hasattr(envelope, "model_dump"):
            envelope_dict = envelope.model_dump(mode="json")  # type: ignore[union-attr]
        elif hasattr(envelope, "dict"):
            envelope_dict = envelope.dict()  # type: ignore[union-attr]
        elif isinstance(envelope, dict):
            envelope_dict = envelope
        else:
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

        Registers a callback to be invoked for each message received on the topic.
        Returns an unsubscribe function to remove the subscription.

        Note: Unlike typical Kafka consumer groups, this implementation maintains
        a subscriber registry and fans out messages to all registered callbacks,
        matching the InMemoryEventBus interface.

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
        subscription_id = str(uuid4())

        async with self._lock:
            # Add to subscriber registry
            self._subscribers[topic].append((group_id, subscription_id, on_message))

            # Start consumer for this topic if not already running
            if topic not in self._consumers and self._started:
                await self._start_consumer_for_topic(topic, group_id)

            logger.debug(
                "Subscriber added",
                extra={
                    "topic": topic,
                    "group_id": group_id,
                    "subscription_id": subscription_id,
                },
            )

        async def unsubscribe() -> None:
            """Remove this subscription from the topic."""
            async with self._lock:
                try:
                    # Find and remove the subscription
                    subs = self._subscribers.get(topic, [])
                    for i, (gid, sid, _) in enumerate(subs):
                        if sid == subscription_id:
                            subs.pop(i)
                            break

                    logger.debug(
                        "Subscriber removed",
                        extra={
                            "topic": topic,
                            "group_id": group_id,
                            "subscription_id": subscription_id,
                        },
                    )

                    # Stop consumer if no more subscribers for this topic
                    if not self._subscribers.get(topic):
                        await self._stop_consumer_for_topic(topic)

                except Exception as e:
                    logger.warning(f"Error during unsubscribe: {e}")

        return unsubscribe

    async def _start_consumer_for_topic(self, topic: str, group_id: str) -> None:
        """Start a Kafka consumer for a specific topic.

        Args:
            topic: Topic to consume from
            group_id: Consumer group ID
        """
        if topic in self._consumers:
            return

        try:
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self._bootstrap_servers,
                group_id=f"{self._environment}.{group_id}",
                auto_offset_reset="latest",
                enable_auto_commit=True,
            )
            await asyncio.wait_for(
                consumer.start(),
                timeout=self._timeout_seconds,
            )

            self._consumers[topic] = consumer

            # Start background task to consume messages
            task = asyncio.create_task(self._consume_loop(topic))
            self._consumer_tasks[topic] = task

            logger.debug(f"Started consumer for topic {topic}")

        except Exception as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="start_consumer",
                target_name=f"kafka.{topic}",
                correlation_id=uuid4(),
            )
            logger.exception(f"Failed to start consumer for topic {topic}")
            raise InfraConnectionError(
                f"Failed to start consumer for topic {topic}",
                context=context,
                topic=topic,
                bootstrap_servers=self._bootstrap_servers,
            ) from e

    async def _stop_consumer_for_topic(self, topic: str) -> None:
        """Stop the consumer for a specific topic.

        Args:
            topic: Topic to stop consuming from
        """
        # Cancel consumer task
        if topic in self._consumer_tasks:
            task = self._consumer_tasks.pop(topic)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop consumer
        if topic in self._consumers:
            consumer = self._consumers.pop(topic)
            try:
                await consumer.stop()
            except Exception as e:
                logger.warning(f"Error stopping consumer for topic {topic}: {e}")

    async def _consume_loop(self, topic: str) -> None:
        """Background loop to consume messages and dispatch to subscribers.

        Args:
            topic: Topic being consumed
        """
        consumer = self._consumers.get(topic)
        if consumer is None:
            return

        try:
            async for msg in consumer:
                if self._shutdown:
                    break

                # Convert Kafka message to ModelEventMessage
                event_message = self._kafka_msg_to_model(msg, topic)

                # Get subscribers snapshot
                async with self._lock:
                    subscribers = list(self._subscribers.get(topic, []))

                # Dispatch to all subscribers
                for group_id, subscription_id, callback in subscribers:
                    try:
                        await callback(event_message)
                    except Exception as e:
                        logger.exception(
                            "Subscriber callback failed",
                            extra={
                                "topic": topic,
                                "group_id": group_id,
                                "subscription_id": subscription_id,
                                "error": str(e),
                            },
                        )

        except asyncio.CancelledError:
            logger.debug(f"Consumer loop cancelled for topic {topic}")
        except Exception:
            logger.exception(f"Consumer loop error for topic {topic}")

    async def start_consuming(self) -> None:
        """Start the consumer loop.

        Protocol method for ProtocolEventBus compatibility.
        Blocks until shutdown() is called.
        """
        if not self._started:
            await self.start()

        # Collect topics that need consumers while holding lock briefly
        topics_to_start: list[tuple[str, str]] = []
        async with self._lock:
            for topic in self._subscribers:
                if topic not in self._consumers:
                    subs = self._subscribers[topic]
                    if subs:
                        group_id = subs[0][0]
                        topics_to_start.append((topic, group_id))

        # Start consumers outside the lock to avoid blocking
        for topic, group_id in topics_to_start:
            await self._start_consumer_for_topic(topic, group_id)

        # Block until shutdown
        while not self._shutdown:
            await asyncio.sleep(self._config.consumer_sleep_interval)

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

    async def health_check(self) -> dict[str, object]:
        """Check event bus health.

        Protocol method for ProtocolEventBus compatibility.

        Returns:
            Dictionary with health status information:
                - healthy: Whether the bus is operational
                - started: Whether start() has been called
                - environment: Current environment
                - group: Current consumer group
                - bootstrap_servers: Kafka bootstrap servers
                - circuit_state: Current circuit breaker state
                - subscriber_count: Total number of active subscriptions
                - topic_count: Number of topics with subscribers
                - consumer_count: Number of active consumers
        """
        async with self._lock:
            subscriber_count = sum(len(subs) for subs in self._subscribers.values())
            topic_count = len(self._subscribers)
            consumer_count = len(self._consumers)

        # Check if producer is healthy
        producer_healthy = False
        if self._producer is not None:
            try:
                # Check if producer client is not closed
                producer_healthy = not getattr(self._producer, "_closed", True)
            except Exception:
                producer_healthy = False

        return {
            "healthy": self._started and producer_healthy,
            "started": self._started,
            "environment": self._environment,
            "group": self._group,
            "bootstrap_servers": self._bootstrap_servers,
            "circuit_state": self._circuit_state.value,
            "subscriber_count": subscriber_count,
            "topic_count": topic_count,
            "consumer_count": consumer_count,
        }

    # =========================================================================
    # Circuit Breaker Methods
    # =========================================================================

    def _check_circuit_breaker(self, correlation_id: Optional[UUID] = None) -> None:
        """Check circuit breaker state and raise if open.

        Args:
            correlation_id: Optional correlation ID to propagate from caller.
                If not provided, a new UUID will be generated.

        Raises:
            InfraUnavailableError: If circuit breaker is open
        """
        current_time = time.time()

        if self._circuit_state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if (
                current_time - self._circuit_last_failure_time
                > self._circuit_breaker_reset_timeout
            ):
                self._circuit_state = CircuitState.HALF_OPEN
                self._circuit_failure_count = 0
                logger.info("Circuit breaker transitioning to half-open")
            else:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="circuit_check",
                    target_name=f"kafka.{self._bootstrap_servers}",
                    correlation_id=correlation_id if correlation_id else uuid4(),
                )
                raise InfraUnavailableError(
                    "Circuit breaker is open - Kafka temporarily unavailable",
                    context=context,
                    circuit_state=self._circuit_state.value,
                    retry_after_seconds=int(
                        self._circuit_breaker_reset_timeout
                        - (current_time - self._circuit_last_failure_time)
                    ),
                )

    def _record_circuit_failure(self) -> None:
        """Record a failure for circuit breaker tracking."""
        self._circuit_failure_count += 1
        self._circuit_last_failure_time = time.time()

        if self._circuit_failure_count >= self._circuit_breaker_threshold:
            self._circuit_state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self._circuit_failure_count} failures",
                extra={"bootstrap_servers": self._bootstrap_servers},
            )

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker on successful operation."""
        if self._circuit_state != CircuitState.CLOSED:
            logger.info(
                f"Circuit breaker reset from {self._circuit_state.value} to closed"
            )
        self._circuit_state = CircuitState.CLOSED
        self._circuit_failure_count = 0

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _model_headers_to_kafka(
        self, headers: ModelEventHeaders
    ) -> list[tuple[str, bytes]]:
        """Convert ModelEventHeaders to Kafka header format.

        Args:
            headers: Model headers

        Returns:
            List of (key, value) tuples with bytes values
        """
        kafka_headers: list[tuple[str, bytes]] = [
            ("content_type", headers.content_type.encode("utf-8")),
            ("correlation_id", str(headers.correlation_id).encode("utf-8")),
            ("message_id", str(headers.message_id).encode("utf-8")),
            ("timestamp", headers.timestamp.isoformat().encode("utf-8")),
            ("source", headers.source.encode("utf-8")),
            ("event_type", headers.event_type.encode("utf-8")),
            ("schema_version", headers.schema_version.encode("utf-8")),
            ("priority", headers.priority.encode("utf-8")),
            ("retry_count", str(headers.retry_count).encode("utf-8")),
            ("max_retries", str(headers.max_retries).encode("utf-8")),
        ]

        # Add optional headers if present
        if headers.destination:
            kafka_headers.append(("destination", headers.destination.encode("utf-8")))
        if headers.trace_id:
            kafka_headers.append(("trace_id", headers.trace_id.encode("utf-8")))
        if headers.span_id:
            kafka_headers.append(("span_id", headers.span_id.encode("utf-8")))
        if headers.parent_span_id:
            kafka_headers.append(
                ("parent_span_id", headers.parent_span_id.encode("utf-8"))
            )
        if headers.operation_name:
            kafka_headers.append(
                ("operation_name", headers.operation_name.encode("utf-8"))
            )
        if headers.routing_key:
            kafka_headers.append(("routing_key", headers.routing_key.encode("utf-8")))
        if headers.partition_key:
            kafka_headers.append(
                ("partition_key", headers.partition_key.encode("utf-8"))
            )
        if headers.ttl_seconds is not None:
            kafka_headers.append(
                ("ttl_seconds", str(headers.ttl_seconds).encode("utf-8"))
            )

        return kafka_headers

    def _kafka_headers_to_model(
        self, kafka_headers: Optional[list[tuple[str, bytes]]]
    ) -> ModelEventHeaders:
        """Convert Kafka headers to ModelEventHeaders.

        Args:
            kafka_headers: Kafka header list

        Returns:
            ModelEventHeaders instance
        """
        if not kafka_headers:
            return ModelEventHeaders(source="unknown", event_type="unknown")

        headers_dict: dict[str, str] = {}
        for key, value in kafka_headers:
            if value is not None:
                headers_dict[key] = value.decode("utf-8")

        # Parse correlation_id from string to UUID (with fallback to new UUID)
        correlation_id_str = headers_dict.get("correlation_id")
        correlation_id = UUID(correlation_id_str) if correlation_id_str else uuid4()

        # Parse message_id from string to UUID (with fallback to new UUID)
        message_id_str = headers_dict.get("message_id")
        message_id = UUID(message_id_str) if message_id_str else uuid4()

        # Parse timestamp from ISO format string to datetime (with fallback to now)
        timestamp_str = headers_dict.get("timestamp")
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
        else:
            timestamp = datetime.now(UTC)

        # Parse priority with validation (default to "normal" if invalid)
        priority_str = headers_dict.get("priority", "normal")
        valid_priorities = ("low", "normal", "high", "critical")
        priority = priority_str if priority_str in valid_priorities else "normal"

        # Parse integer fields with fallback defaults
        retry_count_str = headers_dict.get("retry_count")
        retry_count = int(retry_count_str) if retry_count_str else 0

        max_retries_str = headers_dict.get("max_retries")
        max_retries = int(max_retries_str) if max_retries_str else 3

        ttl_seconds_str = headers_dict.get("ttl_seconds")
        ttl_seconds = int(ttl_seconds_str) if ttl_seconds_str else None

        return ModelEventHeaders(
            content_type=headers_dict.get("content_type", "application/json"),
            correlation_id=correlation_id,
            message_id=message_id,
            timestamp=timestamp,
            source=headers_dict.get("source", "unknown"),
            event_type=headers_dict.get("event_type", "unknown"),
            schema_version=headers_dict.get("schema_version", "1.0.0"),
            destination=headers_dict.get("destination"),
            trace_id=headers_dict.get("trace_id"),
            span_id=headers_dict.get("span_id"),
            parent_span_id=headers_dict.get("parent_span_id"),
            operation_name=headers_dict.get("operation_name"),
            priority=priority,  # type: ignore[arg-type]
            routing_key=headers_dict.get("routing_key"),
            partition_key=headers_dict.get("partition_key"),
            retry_count=retry_count,
            max_retries=max_retries,
            ttl_seconds=ttl_seconds,
        )

    def _kafka_msg_to_model(self, msg: object, topic: str) -> ModelEventMessage:
        """Convert Kafka ConsumerRecord to ModelEventMessage.

        Args:
            msg: Kafka ConsumerRecord
            topic: Topic name

        Returns:
            ModelEventMessage instance
        """
        # Extract fields from Kafka message
        key = getattr(msg, "key", None)
        value = getattr(msg, "value", b"")
        offset = getattr(msg, "offset", None)
        partition = getattr(msg, "partition", None)
        kafka_headers = getattr(msg, "headers", None)

        # Convert key to bytes if it's a string
        if isinstance(key, str):
            key = key.encode("utf-8")

        # Ensure value is bytes
        if isinstance(value, str):
            value = value.encode("utf-8")

        headers = self._kafka_headers_to_model(kafka_headers)

        return ModelEventMessage(
            topic=topic,
            key=key,
            value=value,
            headers=headers,
            offset=str(offset) if offset is not None else None,
            partition=partition,
        )


__all__: list[str] = ["KafkaEventBus"]
