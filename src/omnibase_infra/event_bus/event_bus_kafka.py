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
    - Dead letter queue (DLQ) for failed message processing
    - Graceful degradation when Kafka is unavailable
    - Support for environment/group-based routing
    - Proper producer/consumer lifecycle management

Environment Variables:
    Configuration can be overridden using environment variables. All variables
    are optional and fall back to defaults if not set.

    Connection Settings:
        KAFKA_BOOTSTRAP_SERVERS: Kafka broker addresses (comma-separated)
            Default: "localhost:9092"
            Example: "kafka1:9092,kafka2:9092,kafka3:9092"

        KAFKA_ENVIRONMENT: Environment identifier for message routing
            Default: "local"
            Example: "dev", "staging", "prod"

    Timeout and Retry Settings:
        KAFKA_TIMEOUT_SECONDS: Timeout for Kafka operations (integer seconds)
            Default: 30
            Range: 1-300
            Example: "60"

        KAFKA_MAX_RETRY_ATTEMPTS: Maximum publish retry attempts
            Default: 3
            Range: 0-10
            Example: "5"

            NOTE: This is the BUS-LEVEL retry for Kafka connection/publish failures.
            This is distinct from MESSAGE-LEVEL retry tracked in ModelEventHeaders
            (retry_count/max_retries), which is for application-level message
            delivery tracking across services. See "Dual Retry Configuration" below.

        KAFKA_RETRY_BACKOFF_BASE: Base delay for exponential backoff (float seconds)
            Default: 1.0
            Range: 0.1-60.0
            Example: "2.0"

    Circuit Breaker Settings:
        KAFKA_CIRCUIT_BREAKER_THRESHOLD: Failures before circuit opens
            Default: 5
            Range: 1-100
            Example: "10"

        KAFKA_CIRCUIT_BREAKER_RESET_TIMEOUT: Seconds before circuit resets
            Default: 30.0
            Range: 1.0-3600.0
            Example: "60.0"

    Consumer Settings:
        KAFKA_CONSUMER_SLEEP_INTERVAL: Sleep between poll iterations (float seconds)
            Default: 0.1
            Range: 0.01-10.0
            Example: "0.2"

        KAFKA_AUTO_OFFSET_RESET: Offset reset policy
            Default: "latest"
            Options: "earliest", "latest"

        KAFKA_ENABLE_AUTO_COMMIT: Auto-commit consumer offsets
            Default: true
            Options: "true", "1", "yes", "on" (case-insensitive) = True
                     All other values = False
            Example: "false"

    Producer Settings:
        KAFKA_ACKS: Producer acknowledgment policy
            Default: "all"
            Options: "all" (all replicas), "1" (leader only), "0" (no ack)

        KAFKA_ENABLE_IDEMPOTENCE: Enable idempotent producer
            Default: true
            Options: "true", "1", "yes", "on" (case-insensitive) = True
                     All other values = False
            Example: "true"

    Dead Letter Queue Settings:
        KAFKA_DEAD_LETTER_TOPIC: Topic name for failed messages
            Default: None (DLQ disabled)
            Example: "dlq-events"

            When configured, messages that fail processing will be published
            to this topic with comprehensive failure metadata including:
            - Original topic and message
            - Failure reason and timestamp
            - Correlation ID for tracking
            - Retry count and error type

Dual Retry Configuration:
    ONEX uses TWO distinct retry mechanisms that serve different purposes:

    1. **Bus-Level Retry** (EventBusKafka internal):
       - Configured via: max_retry_attempts, retry_backoff_base
       - Purpose: Handle transient Kafka connection/publish failures
       - Scope: Single publish operation within the event bus
       - Applies to: Producer.send() failures, timeouts, connection errors
       - Example: If Kafka broker is temporarily unreachable, retry 3 times
         with exponential backoff before failing

    2. **Message-Level Retry** (ModelEventHeaders):
       - Configured via: retry_count, max_retries in message headers
       - Purpose: Track application-level message delivery attempts
       - Scope: End-to-end message delivery across services
       - Applies to: Business logic failures, handler exceptions
       - Example: If order processing fails, increment retry_count and
         republish; stop after max_retries reached

    These mechanisms are INDEPENDENT and work together:
    - Bus-level retry handles infrastructure failures (network, broker)
    - Message-level retry handles application failures (handler errors)

    A single message publish may trigger multiple bus-level retries,
    while still counting as a single message-level delivery attempt.

Usage:
    ```python
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
    from omnibase_infra.models import ModelNodeIdentity

    # Option 1: Use defaults with environment variable overrides
    bus = EventBusKafka.default()
    await bus.start()

    # Option 2: Explicit configuration via config model
    config = ModelKafkaEventBusConfig(
        bootstrap_servers="kafka:9092",
        environment="dev",
    )
    bus = EventBusKafka(config=config)
    await bus.start()

    # Subscribe to a topic with node identity
    identity = ModelNodeIdentity(
        env="dev",
        service="my-service",
        node_name="event-processor",
        version="v1",
    )

    async def handler(msg):
        print(f"Received: {msg.value}")
    unsubscribe = await bus.subscribe("events", identity, handler)

    # Publish a message
    await bus.publish("events", b"key", b"value")

    # Cleanup
    await unsubscribe()
    await bus.close()
    ```

Protocol Compatibility:
    This class implements ProtocolEventBus from omnibase_core using duck typing
    (no explicit inheritance required per ONEX patterns).

    TODO: Consider formalizing the EventBusKafka interface as a Protocol
    (ProtocolEventBusKafka) in the future to enable better static type checking
    and IDE support for consumers that depend on Kafka-specific features.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError

from omnibase_infra.enums import EnumConsumerGroupPurpose, EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.event_bus.mixin_kafka_broadcast import MixinKafkaBroadcast
from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq
from omnibase_infra.event_bus.models import (
    ModelEventHeaders,
    ModelEventMessage,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.utils import compute_consumer_group_id

logger = logging.getLogger(__name__)


class EventBusKafka(MixinKafkaBroadcast, MixinKafkaDlq, MixinAsyncCircuitBreaker):
    """Kafka-backed event bus for production message streaming.

    Implements ProtocolEventBus interface using Apache Kafka (via aiokafka)
    with resilience patterns including circuit breaker, retry with exponential
    backoff, dead letter queue support, and graceful degradation when Kafka
    is unavailable.

    Features:
        - Topic-based message routing with Kafka partitioning
        - Multiple subscribers per topic with callback-based delivery
        - Circuit breaker for connection failure protection
        - Retry with exponential backoff on publish failures
        - Dead letter queue (DLQ) for failed message processing
        - Environment-based message routing
        - Proper async producer/consumer lifecycle management

    Attributes:
        environment: Environment identifier (e.g., "local", "dev", "prod")
        adapter: Returns self (for protocol compatibility)

    Architecture:
        This class uses mixin composition to organize functionality:
        - MixinKafkaBroadcast: Environment broadcast messaging, envelope publishing
        - MixinKafkaDlq: Dead letter queue handling and metrics
        - MixinAsyncCircuitBreaker: Circuit breaker resilience pattern

        The core class provides:
        - Factory methods (3): from_config, from_yaml, default
        - Properties (3): config, adapter, environment
        - Lifecycle methods (4): start, initialize, shutdown, close
        - Pub/Sub methods (3): publish, subscribe, start_consuming
        - Health check (1): health_check

    Example:
        ```python
        from omnibase_infra.models import ModelNodeIdentity

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="kafka:9092",
            environment="dev",
        )
        bus = EventBusKafka(config=config)
        await bus.start()

        # Subscribe with node identity
        identity = ModelNodeIdentity(
            env="dev",
            service="my-service",
            node_name="event-processor",
            version="v1",
        )

        async def handler(msg):
            print(f"Received: {msg.value}")
        unsubscribe = await bus.subscribe("events", identity, handler)

        # Publish
        await bus.publish("events", b"key", b"value")

        # Cleanup
        await unsubscribe()
        await bus.close()
        ```
    """

    def __init__(
        self,
        config: ModelKafkaEventBusConfig | None = None,
    ) -> None:
        """Initialize the Kafka event bus.

        Args:
            config: Configuration model containing all settings. If not provided,
                defaults are used with environment variable overrides.

        Raises:
            ProtocolConfigurationError: If circuit_breaker_threshold is not a positive integer

        Example:
            ```python
            # Using config model (recommended)
            config = ModelKafkaEventBusConfig(
                bootstrap_servers="kafka:9092",
                environment="prod",
            )
            bus = EventBusKafka(config=config)

            # Using factory methods
            bus = EventBusKafka.default()
            bus = EventBusKafka.from_yaml(Path("kafka.yaml"))
            ```
        """
        # Use provided config or create default with environment overrides
        if config is None:
            config = ModelKafkaEventBusConfig.default()

        # Store config reference
        self._config = config

        # Apply config values
        self._bootstrap_servers = config.bootstrap_servers
        self._environment = config.environment
        self._timeout_seconds = config.timeout_seconds
        self._max_retry_attempts = config.max_retry_attempts
        self._retry_backoff_base = config.retry_backoff_base

        # Circuit breaker configuration
        if config.circuit_breaker_threshold < 1:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="init",
                target_name="kafka_event_bus",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                f"circuit_breaker_threshold must be a positive integer, got {config.circuit_breaker_threshold}",
                context=context,
                parameter="circuit_breaker_threshold",
                value=config.circuit_breaker_threshold,
            )

        # Initialize circuit breaker mixin
        self._init_circuit_breaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout,
            service_name=f"kafka.{self._environment}",
            transport_type=EnumInfraTransportType.KAFKA,
        )

        # Kafka producer and consumer
        self._producer: AIOKafkaProducer | None = None
        self._consumers: dict[str, AIOKafkaConsumer] = {}

        # Subscriber registry: topic -> list of (group_id, subscription_id, callback) tuples
        self._subscribers: dict[
            str, list[tuple[str, str, Callable[[ModelEventMessage], Awaitable[None]]]]
        ] = defaultdict(list)

        # Lock for coroutine safety (protects all shared state)
        self._lock = asyncio.Lock()

        # State flags
        self._started = False
        self._shutdown = False

        # Background consumer tasks
        self._consumer_tasks: dict[str, asyncio.Task[None]] = {}

        # Producer lock for independent producer access (avoids deadlock with main lock)
        self._producer_lock = asyncio.Lock()

        # Initialize DLQ mixin (metrics tracking, callback hooks)
        self._init_dlq()

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_config(cls, config: ModelKafkaEventBusConfig) -> EventBusKafka:
        """Create EventBusKafka from a configuration model.

        Args:
            config: Configuration model containing all settings

        Returns:
            EventBusKafka instance configured with the provided settings

        Example:
            ```python
            config = ModelKafkaEventBusConfig(
                bootstrap_servers="kafka:9092",
                environment="prod",
                timeout_seconds=60,
            )
            bus = EventBusKafka.from_config(config)
            ```
        """
        return cls(config=config)

    @classmethod
    def from_yaml(cls, path: Path) -> EventBusKafka:
        """Create EventBusKafka from a YAML configuration file.

        Loads configuration from a YAML file with environment variable
        overrides applied automatically.

        Args:
            path: Path to YAML configuration file

        Returns:
            EventBusKafka instance configured from the YAML file

        Raises:
            FileNotFoundError: If the YAML file does not exist
            ValueError: If the YAML content is invalid

        Example:
            ```python
            bus = EventBusKafka.from_yaml(Path("/etc/kafka/config.yaml"))
            ```
        """
        config = ModelKafkaEventBusConfig.from_yaml(path)
        return cls(config=config)

    @classmethod
    def default(cls) -> EventBusKafka:
        """Create EventBusKafka with default configuration.

        Creates an instance with default settings and environment variable
        overrides applied automatically. This is the recommended way to
        create a EventBusKafka for most use cases.

        Returns:
            EventBusKafka instance with default configuration

        Example:
            ```python
            bus = EventBusKafka.default()
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
    def adapter(self) -> EventBusKafka:
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
            logger.debug("EventBusKafka already started")
            return

        correlation_id = uuid4()

        async with self._lock:
            if self._started:
                return

            # Check circuit breaker before attempting connection
            # Note: Circuit breaker requires its own lock to be held
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker(
                    operation="start", correlation_id=correlation_id
                )

            try:
                # Apply producer configuration from config model
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self._bootstrap_servers,
                    acks=self._config.acks_aiokafka,
                    enable_idempotence=self._config.enable_idempotence,
                )

                await asyncio.wait_for(
                    self._producer.start(),
                    timeout=self._timeout_seconds,
                )

                self._started = True
                self._shutdown = False

                # Reset circuit breaker on success
                async with self._circuit_breaker_lock:
                    await self._reset_circuit_breaker()

                logger.info(
                    "EventBusKafka started",
                    extra={
                        "environment": self._environment,
                        "bootstrap_servers": self._sanitize_bootstrap_servers(
                            self._bootstrap_servers
                        ),
                    },
                )

            except TimeoutError as e:
                # Clean up producer on failure to prevent resource leak (thread-safe)
                async with self._producer_lock:
                    if self._producer is not None:
                        try:
                            await self._producer.stop()
                        except Exception as cleanup_err:
                            logger.warning(
                                "Cleanup failed for Kafka producer stop: %s",
                                cleanup_err,
                                exc_info=True,
                            )
                    self._producer = None
                # Record failure (circuit breaker lock required)
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation="start", correlation_id=correlation_id
                    )
                # Sanitize servers for safe logging (remove credentials)
                sanitized_servers = self._sanitize_bootstrap_servers(
                    self._bootstrap_servers
                )
                timeout_ctx = ModelTimeoutErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="start",
                    target_name=f"kafka.{self._environment}",
                    correlation_id=correlation_id,
                    timeout_seconds=self._timeout_seconds,
                )
                logger.warning(
                    f"Timeout connecting to Kafka after {self._timeout_seconds}s",
                    extra={
                        "environment": self._environment,
                        "correlation_id": str(correlation_id),
                    },
                )
                raise InfraTimeoutError(
                    f"Timeout connecting to Kafka after {self._timeout_seconds}s",
                    context=timeout_ctx,
                    servers=sanitized_servers,
                ) from e

            except Exception as e:
                # Clean up producer on failure to prevent resource leak (thread-safe)
                async with self._producer_lock:
                    if self._producer is not None:
                        try:
                            await self._producer.stop()
                        except Exception as cleanup_err:
                            logger.warning(
                                "Cleanup failed for Kafka producer stop: %s",
                                cleanup_err,
                                exc_info=True,
                            )
                    self._producer = None
                # Record failure (circuit breaker lock required)
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation="start", correlation_id=correlation_id
                    )
                # Sanitize servers for safe logging (remove credentials)
                sanitized_servers = self._sanitize_bootstrap_servers(
                    self._bootstrap_servers
                )
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="start",
                    target_name=f"kafka.{self._environment}",
                    correlation_id=correlation_id,
                )
                logger.warning(
                    f"Failed to connect to Kafka: {e}",
                    extra={
                        "environment": self._environment,
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )
                raise InfraConnectionError(
                    f"Failed to connect to Kafka: {e}",
                    context=context,
                    servers=sanitized_servers,
                ) from e

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the event bus with configuration.

        Protocol method for compatibility with ProtocolEventBus.
        Extracts configuration and delegates to start(). Config updates
        are applied atomically with lock protection to prevent races.

        Args:
            config: Configuration dictionary with optional keys:
                - environment: Override environment setting
                - bootstrap_servers: Override bootstrap servers
                - timeout_seconds: Override timeout setting
        """
        # Apply config updates atomically under lock to prevent races
        async with self._lock:
            if "environment" in config:
                self._environment = str(config["environment"])
            if "bootstrap_servers" in config:
                self._bootstrap_servers = str(config["bootstrap_servers"])
            if "timeout_seconds" in config:
                self._timeout_seconds = int(str(config["timeout_seconds"]))

        # Start after config updates are complete
        await self.start()

    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus.

        Protocol method that stops consuming and closes connections.
        """
        await self.close()

    async def close(self) -> None:
        """Close the event bus and release all resources.

        Stops all background consumer tasks, closes all consumers, and
        stops the producer. Safe to call multiple times. Uses proper
        synchronization to prevent races during shutdown.
        """
        # First, signal shutdown to all background tasks
        async with self._lock:
            if self._shutdown:
                # Already shutting down or shutdown
                return
            self._shutdown = True
            self._started = False

        # Cancel all consumer tasks (outside main lock to avoid deadlock)
        tasks_to_cancel = []
        async with self._lock:
            tasks_to_cancel = list(self._consumer_tasks.values())

        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Clear task registry
        async with self._lock:
            self._consumer_tasks.clear()

        # Close all consumers
        consumers_to_close = []
        async with self._lock:
            consumers_to_close = list(self._consumers.values())
            self._consumers.clear()

        for consumer in consumers_to_close:
            try:
                await consumer.stop()
            except Exception as e:
                logger.warning(f"Error stopping consumer: {e}")

        # Close producer with proper locking
        async with self._producer_lock:
            if self._producer is not None:
                try:
                    await self._producer.stop()
                except Exception as e:
                    logger.warning(f"Error stopping producer: {e}")
                self._producer = None

        # Clear subscribers
        async with self._lock:
            self._subscribers.clear()

        logger.info(
            "EventBusKafka closed",
            extra={"environment": self._environment},
        )

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None = None,
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
                source=self._environment,
                event_type=topic,
                timestamp=datetime.now(UTC),
            )

        # Validate topic name
        self._validate_topic_name(topic, headers.correlation_id)

        # Check circuit breaker - propagate correlation_id from headers (thread-safe)
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="publish", correlation_id=headers.correlation_id
            )

        # Convert headers to Kafka format
        kafka_headers = self._model_headers_to_kafka(headers)

        # Publish with retry
        await self._publish_with_retry(topic, key, value, kafka_headers, headers)

    async def _publish_with_retry(
        self,
        topic: str,
        key: bytes | None,
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
        last_exception: Exception | None = None

        for attempt in range(self._max_retry_attempts + 1):
            try:
                # Thread-safe producer access - acquire lock to check and use producer
                async with self._producer_lock:
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

                # Wait for completion outside lock to allow other operations
                record_metadata = await asyncio.wait_for(
                    future,
                    timeout=self._timeout_seconds,
                )

                # Success - reset circuit breaker (thread-safe)
                async with self._circuit_breaker_lock:
                    await self._reset_circuit_breaker()

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
                # Clean up producer on timeout to prevent resource leak (thread-safe)
                async with self._producer_lock:
                    if self._producer is not None:
                        try:
                            await self._producer.stop()
                        except Exception as cleanup_err:
                            logger.warning(
                                "Cleanup failed for Kafka producer stop during publish: %s",
                                cleanup_err,
                                exc_info=True,
                            )
                    self._producer = None
                last_exception = e
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation="publish", correlation_id=headers.correlation_id
                    )
                logger.warning(
                    f"Publish timeout (attempt {attempt + 1}/{self._max_retry_attempts + 1})",
                    extra={
                        "topic": topic,
                        "correlation_id": str(headers.correlation_id),
                    },
                )

            except KafkaError as e:
                last_exception = e
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation="publish", correlation_id=headers.correlation_id
                    )
                logger.warning(
                    f"Kafka error on publish (attempt {attempt + 1}/{self._max_retry_attempts + 1}): {e}",
                    extra={
                        "topic": topic,
                        "correlation_id": str(headers.correlation_id),
                    },
                )

            except Exception as e:
                last_exception = e
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation="publish", correlation_id=headers.correlation_id
                    )
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
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="publish",
                target_name=f"kafka.{topic}",
                correlation_id=headers.correlation_id,
                timeout_seconds=self._timeout_seconds,
            )
            raise InfraTimeoutError(
                f"Timeout publishing to topic {topic} after {self._max_retry_attempts + 1} attempts",
                context=timeout_ctx,
                topic=topic,
                retry_count=self._max_retry_attempts + 1,
            ) from last_exception
        raise InfraConnectionError(
            f"Failed to publish to topic {topic} after {self._max_retry_attempts + 1} attempts",
            context=context,
            topic=topic,
            retry_count=self._max_retry_attempts + 1,
        ) from last_exception

    async def subscribe(
        self,
        topic: str,
        node_identity: ModelNodeIdentity,
        on_message: Callable[[ModelEventMessage], Awaitable[None]],
        *,
        purpose: EnumConsumerGroupPurpose = EnumConsumerGroupPurpose.CONSUME,
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe to topic with callback handler.

        Registers a callback to be invoked for each message received on the topic.
        Returns an unsubscribe function to remove the subscription.

        The consumer group ID is derived from the node identity using the canonical
        format: ``{env}.{service}.{node_name}.{purpose}.{version}``.

        Note: Unlike typical Kafka consumer groups, this implementation maintains
        a subscriber registry and fans out messages to all registered callbacks,
        matching the EventBusInmemory interface.

        Args:
            topic: Topic to subscribe to
            node_identity: Node identity used to derive the consumer group ID.
                Contains env, service, node_name, and version components.
            on_message: Async callback invoked for each message
            purpose: Consumer group purpose classification. Defaults to CONSUME.
                Used in the consumer group ID derivation for disambiguation.

        Returns:
            Async unsubscribe function to remove this subscription

        Example:
            ```python
            from omnibase_infra.models import ModelNodeIdentity
            from omnibase_infra.enums import EnumConsumerGroupPurpose

            identity = ModelNodeIdentity(
                env="dev",
                service="my-service",
                node_name="event-processor",
                version="v1",
            )

            async def handler(msg):
                print(f"Received: {msg.value}")

            # Standard subscription (group_id: dev.my-service.event-processor.consume.v1)
            unsubscribe = await bus.subscribe("events", identity, handler)

            # With explicit purpose
            unsubscribe = await bus.subscribe(
                "events", identity, handler,
                purpose=EnumConsumerGroupPurpose.INTROSPECTION,
            )

            # ... later ...
            await unsubscribe()
            ```
        """
        subscription_id = str(uuid4())
        correlation_id = uuid4()

        # Derive consumer group ID from node identity (no overrides allowed)
        effective_group_id = compute_consumer_group_id(node_identity, purpose)

        # Validate topic name
        self._validate_topic_name(topic, correlation_id)

        async with self._lock:
            # Add to subscriber registry
            self._subscribers[topic].append(
                (effective_group_id, subscription_id, on_message)
            )

            # Start consumer for this topic if not already running
            if topic not in self._consumers and self._started:
                await self._start_consumer_for_topic(topic, effective_group_id)

            logger.debug(
                "Subscriber added",
                extra={
                    "topic": topic,
                    "group_id": effective_group_id,
                    "subscription_id": subscription_id,
                },
            )

        async def unsubscribe() -> None:
            """Remove this subscription from the topic."""
            async with self._lock:
                try:
                    # Find and remove the subscription
                    subs = self._subscribers.get(topic, [])
                    for i, (_gid, sid, _) in enumerate(subs):
                        if sid == subscription_id:
                            subs.pop(i)
                            break

                    logger.debug(
                        "Subscriber removed",
                        extra={
                            "topic": topic,
                            "group_id": effective_group_id,
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

        This method creates and starts a Kafka consumer for the specified topic,
        then launches a background task to consume messages. All startup failures
        are logged and propagated to the caller.

        Args:
            topic: Topic to consume from
            group_id: Fully qualified consumer group ID. This should be derived
                from ``compute_consumer_group_id()`` or an explicit override.
                The ID is used directly without any prefix modification.

        Raises:
            ProtocolConfigurationError: If group_id is empty (must be derived from
                compute_consumer_group_id or provided as explicit override)
            InfraTimeoutError: If consumer startup times out after timeout_seconds
            InfraConnectionError: If consumer fails to connect to Kafka brokers
        """
        if topic in self._consumers:
            return

        correlation_id = uuid4()
        sanitized_servers = self._sanitize_bootstrap_servers(self._bootstrap_servers)

        # Use group_id directly - it's already fully qualified from compute_consumer_group_id()
        # or an explicit override. Empty group_id indicates a bug in the caller.
        effective_group_id = group_id.strip()
        if not effective_group_id:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="start_consumer",
                target_name=f"kafka.{topic}",
            )
            raise ProtocolConfigurationError(
                f"Consumer group ID is required for topic '{topic}'. "
                "Internal error: compute_consumer_group_id() should have been called.",
                context=context,
                parameter="group_id",
                value=group_id,
            )

        # Apply consumer configuration from config model
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=effective_group_id,
            auto_offset_reset=self._config.auto_offset_reset,
            enable_auto_commit=self._config.enable_auto_commit,
        )

        try:
            await asyncio.wait_for(
                consumer.start(),
                timeout=self._timeout_seconds,
            )

            self._consumers[topic] = consumer

            # Start background task to consume messages with correlation tracking
            task = asyncio.create_task(self._consume_loop(topic, correlation_id))
            self._consumer_tasks[topic] = task

            logger.info(
                f"Started consumer for topic {topic}",
                extra={
                    "topic": topic,
                    "group_id": effective_group_id,
                    "correlation_id": str(correlation_id),
                    "servers": sanitized_servers,
                },
            )

        except TimeoutError as e:
            # Clean up consumer on failure to prevent resource leak
            try:
                await consumer.stop()
            except Exception as cleanup_err:
                logger.warning(
                    "Cleanup failed for Kafka consumer stop (topic=%s): %s",
                    topic,
                    cleanup_err,
                    exc_info=True,
                )

            # Propagate timeout error to surface startup failures (differentiate from connection errors)
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="start_consumer",
                target_name=f"kafka.{topic}",
                correlation_id=correlation_id,
                timeout_seconds=self._timeout_seconds,
            )
            logger.exception(
                f"Timeout starting consumer for topic {topic} after {self._timeout_seconds}s",
                extra={
                    "topic": topic,
                    "group_id": group_id,
                    "correlation_id": str(correlation_id),
                    "timeout_seconds": self._timeout_seconds,
                    "servers": sanitized_servers,
                    "error_type": "timeout",
                },
            )
            raise InfraTimeoutError(
                f"Timeout starting consumer for topic {topic} after {self._timeout_seconds}s",
                context=timeout_ctx,
                topic=topic,
                servers=sanitized_servers,
            ) from e

        except Exception as e:
            # Clean up consumer on failure to prevent resource leak
            try:
                await consumer.stop()
            except Exception as cleanup_err:
                logger.warning(
                    "Cleanup failed for Kafka consumer stop (topic=%s): %s",
                    topic,
                    cleanup_err,
                    exc_info=True,
                )

            # Propagate connection error to surface startup failures (differentiate from timeout)
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="start_consumer",
                target_name=f"kafka.{topic}",
                correlation_id=correlation_id,
            )
            logger.exception(
                f"Failed to start consumer for topic {topic}: {e}",
                extra={
                    "topic": topic,
                    "group_id": group_id,
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "servers": sanitized_servers,
                },
            )
            raise InfraConnectionError(
                f"Failed to start consumer for topic {topic}: {e}",
                context=context,
                topic=topic,
                servers=sanitized_servers,
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

    async def _consume_loop(self, topic: str, correlation_id: UUID) -> None:
        """Background loop to consume messages and dispatch to subscribers.

        This method runs in a background task and continuously polls the Kafka consumer
        for new messages. It handles graceful cancellation, dispatches messages to all
        registered subscribers, and logs all errors without terminating the loop.

        Args:
            topic: Topic being consumed
            correlation_id: Correlation ID for tracking this consumer task
        """
        consumer = self._consumers.get(topic)
        if consumer is None:
            logger.warning(
                f"Consumer not found for topic {topic} in consume loop",
                extra={
                    "topic": topic,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        logger.debug(
            f"Consumer loop started for topic {topic}",
            extra={
                "topic": topic,
                "correlation_id": str(correlation_id),
            },
        )

        try:
            async for msg in consumer:
                if self._shutdown:
                    logger.debug(
                        f"Consumer loop shutdown signal received for topic {topic}",
                        extra={
                            "topic": topic,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    break

                # Get subscribers snapshot early - needed for consumer group in DLQ
                async with self._lock:
                    subscribers = list(self._subscribers.get(topic, []))

                # Extract consumer group for DLQ traceability (all subscribers share the same consumer)
                effective_consumer_group = (
                    subscribers[0][0] if subscribers else "unknown"
                )

                # Convert Kafka message to ModelEventMessage - handle conversion errors
                try:
                    event_message = self._kafka_msg_to_model(msg, topic)
                except Exception as e:
                    logger.exception(
                        f"Failed to convert Kafka message to event model for topic {topic}",
                        extra={
                            "topic": topic,
                            "correlation_id": str(correlation_id),
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    # Deserialization errors are permanent failures - route to DLQ
                    # Create minimal message from raw Kafka data for DLQ context
                    await self._publish_raw_to_dlq(
                        original_topic=topic,
                        raw_msg=msg,
                        error=e,
                        correlation_id=correlation_id,
                        failure_type="deserialization_error",
                        consumer_group=effective_consumer_group,
                    )
                    continue  # Skip this message but continue consuming

                # Dispatch to all subscribers
                for group_id, subscription_id, callback in subscribers:
                    try:
                        await callback(event_message)
                    except Exception as e:
                        # Check if message-level retries are exhausted
                        retry_count = event_message.headers.retry_count
                        max_retries = event_message.headers.max_retries
                        retries_exhausted = retry_count >= max_retries

                        logger.exception(
                            "Subscriber callback failed",
                            extra={
                                "topic": topic,
                                "group_id": group_id,
                                "subscription_id": subscription_id,
                                "correlation_id": str(correlation_id),
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "retry_count": retry_count,
                                "max_retries": max_retries,
                                "retries_exhausted": retries_exhausted,
                            },
                        )

                        # Route to DLQ when retries exhausted (permanent failure)
                        # Per ModelEventHeaders: "When retry_count >= max_retries, message should go to DLQ"
                        if retries_exhausted:
                            await self._publish_to_dlq(
                                original_topic=topic,
                                failed_message=event_message,
                                error=e,
                                correlation_id=correlation_id,
                                consumer_group=group_id,
                            )
                        else:
                            # Message still has retries available - log for potential republish
                            # Note: Republishing logic is the responsibility of the caller/handler
                            logger.warning(
                                f"Handler failed but retries available ({retry_count}/{max_retries})",
                                extra={
                                    "topic": topic,
                                    "correlation_id": str(correlation_id),
                                    "retry_count": retry_count,
                                    "max_retries": max_retries,
                                },
                            )
                        # Continue dispatching to other subscribers even if one fails

        except asyncio.CancelledError:
            # Graceful cancellation - this is expected during shutdown
            logger.info(
                f"Consumer loop cancelled for topic {topic}",
                extra={
                    "topic": topic,
                    "correlation_id": str(correlation_id),
                },
            )
            raise  # Re-raise to properly handle task cancellation

        except Exception as e:
            # Unexpected error in consumer loop - log with full context
            logger.exception(
                f"Consumer loop error for topic {topic}: {e}",
                extra={
                    "topic": topic,
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Don't raise - allow task to complete and cleanup to proceed

        finally:
            logger.info(
                f"Consumer loop exiting for topic {topic}",
                extra={
                    "topic": topic,
                    "correlation_id": str(correlation_id),
                },
            )

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

    async def health_check(self) -> dict[str, object]:
        """Check event bus health.

        Protocol method for ProtocolEventBus compatibility.

        Returns:
            Dictionary with health status information:
                - healthy: Whether the bus is operational
                - started: Whether start() has been called
                - environment: Current environment
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
            started = self._started

        # Get circuit breaker state (thread-safe access)
        async with self._circuit_breaker_lock:
            circuit_state = "open" if self._circuit_breaker_open else "closed"

        # Check if producer is healthy (thread-safe access)
        producer_healthy = False
        async with self._producer_lock:
            if self._producer is not None:
                try:
                    # Check if producer client is not closed
                    producer_healthy = not getattr(self._producer, "_closed", True)
                except Exception:
                    producer_healthy = False

        return {
            "healthy": started and producer_healthy,
            "started": started,
            "environment": self._environment,
            "bootstrap_servers": self._sanitize_bootstrap_servers(
                self._bootstrap_servers
            ),
            "circuit_state": circuit_state,
            "subscriber_count": subscriber_count,
            "topic_count": topic_count,
            "consumer_count": consumer_count,
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _sanitize_bootstrap_servers(self, servers: str) -> str:
        """Sanitize bootstrap servers string to remove potential credentials.

        Removes any authentication tokens, passwords, or sensitive data from
        the bootstrap servers string before logging or including in errors.

        Args:
            servers: Raw bootstrap servers string (may contain credentials)

        Returns:
            Sanitized servers string safe for logging and error messages

        Example:
            "user:pass@kafka:9092" -> "kafka:9092"
            "kafka:9092,kafka2:9092" -> "kafka:9092,kafka2:9092"
        """
        if not servers:
            return "unknown"

        # Split by comma for multiple servers
        server_list = [s.strip() for s in servers.split(",")]
        sanitized = []

        for server in server_list:
            # Remove any user:pass@ prefix (credentials)
            if "@" in server:
                # Keep only the part after @
                server = server.split("@", 1)[1]
            sanitized.append(server)

        return ",".join(sanitized)

    def _validate_topic_name(self, topic: str, correlation_id: UUID) -> None:
        """Validate Kafka topic name according to Kafka naming rules.

        Kafka topic names must:
        - Not be empty
        - Be 255 characters or less
        - Contain only: a-z, A-Z, 0-9, period (.), underscore (_), hyphen (-)
        - Not be "." or ".." (reserved)

        Args:
            topic: Topic name to validate
            correlation_id: Correlation ID for error context

        Raises:
            ProtocolConfigurationError: If topic name is invalid

        Reference:
            https://kafka.apache.org/documentation/#topicconfigs
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_topic",
            target_name=f"kafka.{self._environment}",
            correlation_id=correlation_id,
        )

        if not topic:
            raise ProtocolConfigurationError(
                "Topic name cannot be empty",
                context=context,
                parameter="topic",
                value=topic,
            )

        if len(topic) > 255:
            raise ProtocolConfigurationError(
                f"Topic name '{topic}' exceeds maximum length of 255 characters",
                context=context,
                parameter="topic",
                value=topic,
            )

        if topic in (".", ".."):
            raise ProtocolConfigurationError(
                f"Topic name '{topic}' is reserved and cannot be used",
                context=context,
                parameter="topic",
                value=topic,
            )

        # Validate characters (a-z, A-Z, 0-9, '.', '_', '-')
        if not re.match(r"^[a-zA-Z0-9._-]+$", topic):
            raise ProtocolConfigurationError(
                f"Topic name '{topic}' contains invalid characters. "
                "Only alphanumeric characters, periods (.), underscores (_), "
                "and hyphens (-) are allowed",
                context=context,
                parameter="topic",
                value=topic,
            )

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
        self, kafka_headers: list[tuple[str, bytes]] | None
    ) -> ModelEventHeaders:
        """Convert Kafka headers to ModelEventHeaders.

        Args:
            kafka_headers: Kafka header list

        Returns:
            ModelEventHeaders instance
        """
        if not kafka_headers:
            return ModelEventHeaders(
                source="unknown",
                event_type="unknown",
                timestamp=datetime.now(UTC),
            )

        headers_dict: dict[str, str] = {}
        for key, value in kafka_headers:
            if value is not None:
                headers_dict[key] = value.decode("utf-8")

        # Parse correlation_id from string to UUID (with fallback to new UUID)
        correlation_id_str = headers_dict.get("correlation_id")
        if correlation_id_str:
            try:
                correlation_id = UUID(correlation_id_str)
            except (ValueError, AttributeError):
                # Invalid UUID format - generate new one
                correlation_id = uuid4()
        else:
            correlation_id = uuid4()

        # Parse message_id from string to UUID (with fallback to new UUID)
        message_id_str = headers_dict.get("message_id")
        if message_id_str:
            try:
                message_id = UUID(message_id_str)
            except (ValueError, AttributeError):
                # Invalid UUID format - generate new one
                message_id = uuid4()
        else:
            message_id = uuid4()

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
            priority=priority,
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


__all__: list[str] = ["EventBusKafka"]
