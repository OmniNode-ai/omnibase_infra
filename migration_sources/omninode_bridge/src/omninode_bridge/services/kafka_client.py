"""Kafka client for event streaming in OmniNode Bridge."""

import asyncio
import hashlib
import json
import logging
import os
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import UUID

from aiokafka import AIOKafkaClient, AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import KafkaError, KafkaTimeoutError

# Type variable for generic retry function
T = TypeVar("T")

from ..agents.type_defs import (
    KafkaEnvelopeMetricsDict,
    KafkaHealthCheckDict,
    KafkaResilienceMetricsDict,
    KafkaRetryResultDict,
)
from ..config.batch_sizes import get_batch_manager
from ..models.events import AnyEvent
from ..security.audit_logger import AuditEventType, AuditSeverity, get_audit_logger
from ..utils.circuit_breaker_config import KAFKA_CIRCUIT_BREAKER

# Apply hostname patch for RedPanda connectivity
from . import (  # noqa: F401 - side effect import for socket patching
    kafka_hostname_patch,
)

logger = logging.getLogger(__name__)


class KafkaClient:
    """Async Kafka client for publishing events to RedPanda/Kafka topics with resilience patterns."""

    def __init__(
        self,
        bootstrap_servers: str = None,
        enable_dead_letter_queue: bool = None,
        dead_letter_topic_suffix: str = None,
        max_retry_attempts: int = None,
        retry_backoff_base: float = None,
        timeout_seconds: int = None,
        # Performance optimization parameters
        compression_type: str | None = None,
        batch_size: int | None = None,
        linger_ms: int | None = None,
        buffer_memory: int | None = None,
        max_request_size: int | None = None,
    ):
        """Initialize Kafka client with resilience features.

        Args:
            bootstrap_servers: Kafka bootstrap servers (default uses omninode_bridge port)
            enable_dead_letter_queue: Enable dead letter queue for failed messages
            dead_letter_topic_suffix: Suffix for dead letter queue topics
            max_retry_attempts: Maximum retry attempts before sending to DLQ
            retry_backoff_base: Base delay for exponential backoff
            timeout_seconds: Timeout for operations
        """
        # Configure parameters from environment variables with sensible defaults
        # Default to remote infrastructure (resolves to 192.168.86.200:9092 via /etc/hosts)
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            "omninode-bridge-redpanda:9092",
        )
        self.producer: AIOKafkaProducer | None = None
        self._connected = False

        # Resilience configuration from environment
        self.enable_dead_letter_queue = (
            enable_dead_letter_queue
            if enable_dead_letter_queue is not None
            else os.getenv("KAFKA_ENABLE_DLQ", "true").lower() == "true"
        )
        self.dead_letter_topic_suffix = dead_letter_topic_suffix or os.getenv(
            "KAFKA_DLQ_SUFFIX",
            ".dlq",
        )
        self.max_retry_attempts = (
            max_retry_attempts
            if max_retry_attempts is not None
            else int(os.getenv("KAFKA_MAX_RETRY_ATTEMPTS", "3"))
        )
        self.retry_backoff_base = (
            retry_backoff_base
            if retry_backoff_base is not None
            else float(os.getenv("KAFKA_RETRY_BACKOFF_BASE", "1.0"))
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else int(os.getenv("KAFKA_TIMEOUT_SECONDS", "30"))
        )

        # Performance optimization configuration
        # Use batch size manager for environment-aware configuration
        batch_manager = get_batch_manager()
        default_batch_size = batch_manager.kafka_producer_batch_size

        # Environment-based defaults for other Kafka performance tuning
        environment = os.getenv("ENVIRONMENT", "development").lower()

        if environment == "production":
            # Production: Optimize for throughput and reliability
            default_compression = "lz4"  # Fast compression with good ratio
            default_linger_ms = 10  # 10ms batching window
            default_buffer_memory = 67108864  # 64MB buffer
            default_max_request_size = 1048576  # 1MB max request
        elif environment == "staging":
            # Staging: Balanced performance
            default_compression = "gzip"
            default_linger_ms = 5
            default_buffer_memory = 33554432  # 32MB buffer
            default_max_request_size = 524288  # 512KB max request
        else:  # development
            # Development: Minimize latency over throughput
            default_compression = None
            default_linger_ms = 0
            default_buffer_memory = 16777216  # 16MB buffer
            default_max_request_size = 262144  # 256KB max request

        self.compression_type = (
            compression_type
            if compression_type is not None
            else os.getenv("KAFKA_COMPRESSION_TYPE", default_compression)
        )
        self.batch_size = (
            batch_size
            if batch_size is not None
            else int(os.getenv("KAFKA_BATCH_SIZE", default_batch_size))
        )
        self.linger_ms = (
            linger_ms
            if linger_ms is not None
            else int(os.getenv("KAFKA_LINGER_MS", default_linger_ms))
        )
        self.buffer_memory = (
            buffer_memory
            if buffer_memory is not None
            else int(os.getenv("KAFKA_BUFFER_MEMORY", default_buffer_memory))
        )
        self.max_request_size = (
            max_request_size
            if max_request_size is not None
            else int(os.getenv("KAFKA_MAX_REQUEST_SIZE", default_max_request_size))
        )

        # Dead letter queue tracking
        self._failed_messages: list[dict[str, Any]] = []
        self._retry_counts: dict[str, int] = {}

        # Performance metrics tracking
        self._message_count = 0
        self._total_bytes_sent = 0
        self._batch_count = 0

        # Event publishing metrics (envelope-wrapped events)
        self._envelope_published_count = 0
        self._envelope_failed_count = 0
        self._envelope_publish_latencies: list[float] = []  # milliseconds

        # Intelligent partitioning configuration
        self.partitioning_strategy = os.getenv(
            "KAFKA_PARTITIONING_STRATEGY", "balanced"
        )
        self.partition_cache = {}  # Cache topic partition counts
        self.partition_load_tracker = {}  # Track partition load for balancing
        self.max_partition_skew = float(
            os.getenv("KAFKA_MAX_PARTITION_SKEW", "0.2")
        )  # 20% max skew

        # Initialize audit logger
        self.audit_logger = get_audit_logger("kafka_client", "0.1.0")

    @KAFKA_CIRCUIT_BREAKER()
    async def connect(self) -> None:
        """Connect to Kafka cluster with circuit breaker protection."""
        try:
            # Build producer configuration with performance optimizations
            producer_config = {
                "bootstrap_servers": self.bootstrap_servers,
                "value_serializer": lambda v: json.dumps(v, default=str).encode(
                    "utf-8",
                ),
                "key_serializer": lambda v: (
                    str(v).encode("utf-8") if v is not None else None
                ),
                # Reliability settings compatible with aiokafka 0.11.0
                "acks": "all",  # Wait for all replicas
                "enable_idempotence": True,  # Prevent duplicate messages
                # Note: retries and retry_backoff_ms not supported in aiokafka 0.11.0
                # "retries": self.max_retry_attempts,
                # "retry_backoff_ms": int(self.retry_backoff_base * 1000),
            }

            # Add compression if supported and configured
            if self.compression_type:
                try:
                    producer_config["compression_type"] = self.compression_type
                except Exception as e:
                    logger.warning(
                        f"Compression type {self.compression_type} not supported: {e}",
                    )

            # Note: For aiokafka 0.11.0, batch_size and linger_ms are not supported
            # When upgrading to newer aiokafka versions, uncomment these:
            # producer_config["batch_size"] = self.batch_size
            # producer_config["linger_ms"] = self.linger_ms
            # producer_config["buffer_memory"] = self.buffer_memory
            # producer_config["max_request_size"] = self.max_request_size

            # Create producer with optimized configuration
            self.producer = AIOKafkaProducer(**producer_config)

            # Connect with timeout - DO NOT set _connected=True until ALL operations complete
            await asyncio.wait_for(self.producer.start(), timeout=self.timeout_seconds)

            # Log successful Kafka connection for audit (connection established but not fully ready)
            self.audit_logger.log_event(
                event_type=AuditEventType.SERVICE_STARTUP,
                severity=AuditSeverity.LOW,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "bootstrap_servers": self.bootstrap_servers,
                    "connection_status": "successful",
                    "timeout_seconds": self.timeout_seconds,
                    "circuit_breaker_status": "healthy",
                },
                message=f"Kafka client connected successfully to {self.bootstrap_servers}",
            )

            # CRITICAL: Only set connected=True after ALL setup operations complete successfully
            # This prevents race conditions where connection state is set before full initialization
            self._connected = True
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")

        except TimeoutError:
            logger.warning(
                f"Timeout connecting to Kafka after {self.timeout_seconds}s - continuing in degraded mode",
            )
            self._connected = False

            # Log Kafka connection timeout for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.SERVICE_STARTUP,
                severity=AuditSeverity.HIGH,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "bootstrap_servers": self.bootstrap_servers,
                    "connection_status": "timeout",
                    "timeout_seconds": self.timeout_seconds,
                    "error_type": "timeout",
                    "degraded_mode": True,
                },
                message=f"Kafka client connection timeout after {self.timeout_seconds}s - operating in degraded mode",
            )
            # Don't raise - allow graceful degradation
            return
        except Exception as e:
            logger.warning(
                f"Failed to connect to Kafka: {e} - continuing in degraded mode",
            )
            self._connected = False

            # Log Kafka connection failure for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.SERVICE_STARTUP,
                severity=AuditSeverity.CRITICAL,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "bootstrap_servers": self.bootstrap_servers,
                    "connection_status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "degraded_mode": True,
                },
                message=f"Kafka client connection failed: {e!s} - operating in degraded mode",
            )
            # Don't raise - allow graceful degradation
            return

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        if self.producer:
            try:
                await asyncio.wait_for(
                    self.producer.stop(),
                    timeout=self.timeout_seconds,
                )

                # Log successful Kafka disconnection for audit
                self.audit_logger.log_event(
                    event_type=AuditEventType.SERVICE_SHUTDOWN,
                    severity=AuditSeverity.LOW,
                    request=None,
                    additional_data={
                        "service": "kafka_client",
                        "bootstrap_servers": self.bootstrap_servers,
                        "disconnection_status": "successful",
                    },
                    message=f"Kafka client disconnected successfully from {self.bootstrap_servers}",
                )

            except TimeoutError:
                logger.warning("Timeout disconnecting from Kafka")

                # Log Kafka disconnection timeout for audit
                self.audit_logger.log_event(
                    event_type=AuditEventType.SERVICE_SHUTDOWN,
                    severity=AuditSeverity.MEDIUM,
                    request=None,
                    additional_data={
                        "service": "kafka_client",
                        "bootstrap_servers": self.bootstrap_servers,
                        "disconnection_status": "timeout",
                        "timeout_seconds": self.timeout_seconds,
                    },
                    message=f"Kafka client disconnection timeout after {self.timeout_seconds}s",
                )

            finally:
                self._connected = False
                logger.info("Disconnected from Kafka")

    async def flush(self, timeout: float | None = None) -> None:
        """Flush all buffered messages in the producer.

        This ensures all messages are sent to Kafka immediately, which is
        useful in tests and critical workflows where immediate delivery is required.

        Args:
            timeout: Timeout in seconds (defaults to self.timeout_seconds)
        """
        if not self.producer or not self._connected:
            logger.warning("Cannot flush: Kafka producer not connected")
            return

        timeout_seconds = timeout or self.timeout_seconds

        try:
            await asyncio.wait_for(self.producer.flush(), timeout=timeout_seconds)
            logger.debug("Kafka producer flushed successfully")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout flushing Kafka producer after {timeout_seconds}s")
        except Exception as e:
            logger.error(f"Error flushing Kafka producer: {e}")

    async def _send_to_dead_letter_queue(
        self,
        original_topic: str,
        message_data: dict[str, Any],
        key: str | None = None,
        error_reason: str = "max_retries_exceeded",
    ) -> bool:
        """Send failed message to dead letter queue.

        Args:
            original_topic: Original topic name
            message_data: Message data
            key: Message key
            error_reason: Reason for failure

        Returns:
            True if sent to DLQ successfully, False otherwise
        """
        if not self.enable_dead_letter_queue:
            logger.warning(
                f"Dead letter queue disabled, dropping message for topic {original_topic}",
            )
            return False

        dlq_topic = f"{original_topic}{self.dead_letter_topic_suffix}"

        # Create DLQ message with metadata
        dlq_message = {
            "original_topic": original_topic,
            "original_key": key,
            "original_data": message_data,
            "failure_reason": error_reason,
            "failure_timestamp": datetime.now(UTC).isoformat(),
            "retry_attempts": self._retry_counts.get(f"{original_topic}:{key}", 0),
        }

        try:
            future = await self.producer.send(dlq_topic, value=dlq_message, key=key)
            record_metadata = await future

            logger.info(
                f"Sent message to dead letter queue {dlq_topic} "
                f"(partition: {record_metadata.partition}, offset: {record_metadata.offset})",
            )

            # Track failed message
            self._failed_messages.append(
                {
                    "original_topic": original_topic,
                    "dlq_topic": dlq_topic,
                    "key": key,
                    "failure_reason": error_reason,
                    "timestamp": datetime.now(UTC),
                },
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to send message to dead letter queue {dlq_topic}: {e}",
            )
            return False

    async def _retry_with_backoff(
        self,
        operation_func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """Retry operation with exponential backoff.

        Args:
            operation_func: Async function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Result of successful operation

        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None

        for attempt in range(self.max_retry_attempts + 1):
            try:
                return await operation_func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_retry_attempts:
                    delay = self.retry_backoff_base * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}",
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retry_attempts + 1} attempts failed")
                    break

        raise last_exception

    async def _get_optimal_partition(self, topic: str, key: str | None) -> int | None:
        """Get optimal partition for message to prevent hotspots.

        Args:
            topic: Topic name
            key: Message key

        Returns:
            Optimal partition number or None for default partitioning
        """
        try:
            # Get topic partition count (cached)
            if topic not in self.partition_cache:
                # Try to get partition count from producer metadata
                if self.producer and hasattr(self.producer, "_metadata"):
                    metadata = self.producer._metadata
                    # Use partitions_for_topic() method (returns Optional[set[int]])
                    partitions = (
                        metadata.partitions_for_topic(topic) if metadata else None
                    )
                    if partitions:
                        self.partition_cache[topic] = len(partitions)
                    else:
                        # Default assumption for new topics
                        self.partition_cache[topic] = int(
                            os.getenv("KAFKA_DEFAULT_PARTITIONS", "3")
                        )
                else:
                    self.partition_cache[topic] = int(
                        os.getenv("KAFKA_DEFAULT_PARTITIONS", "3")
                    )

            partition_count = self.partition_cache[topic]

            if self.partitioning_strategy == "balanced":
                return await self._get_balanced_partition(topic, key, partition_count)
            elif self.partitioning_strategy == "hash":
                return await self._get_hash_partition(topic, key, partition_count)
            elif self.partitioning_strategy == "round_robin":
                return await self._get_round_robin_partition(topic, partition_count)
            else:  # default - let Kafka decide
                return None

        except Exception as e:
            logger.warning(
                f"Failed to determine optimal partition for topic {topic}: {e}"
            )
            return None

    async def _get_balanced_partition(
        self, topic: str, key: str | None, partition_count: int
    ) -> int:
        """Get balanced partition to prevent hotspots with deterministic key-based partitioning.

        Uses SHA-256 for stable, process-independent partition assignment to ensure
        the same key always maps to the same partition across producer restarts,
        maintaining event ordering guarantees for correlation-based processing.
        """
        # Initialize load tracking for topic if needed
        if topic not in self.partition_load_tracker:
            self.partition_load_tracker[topic] = [0] * partition_count

        load_tracker = self.partition_load_tracker[topic]

        # Find partition with minimum load
        min_load = min(load_tracker)
        max_load = max(load_tracker)

        # Check if we have significant skew
        if max_load > 0:
            skew = (max_load - min_load) / max_load
            if skew > self.max_partition_skew:
                logger.info(f"Partition skew detected for topic {topic}: {skew:.2%}")

        # If key provided, use deterministic hash-based partitioning
        if key:
            # Use SHA-256 for stable, deterministic partition assignment across process restarts
            # Python's hash() is salted per-process, causing different partitions across restarts
            # SHA-256 ensures same key -> same partition, preserving event ordering guarantees
            #
            # NOTE: We do NOT apply load balancing for keyed messages because:
            # 1. Keys are used for event ordering guarantees (same key = same partition)
            # 2. Overriding partition assignment would break ordering for correlation-based processing
            # 3. For truly random load balancing without ordering, use round_robin strategy instead
            hash_value = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
            partition = hash_value % partition_count
        else:
            # No key - use least loaded partition
            min_partitions = [
                i for i, load in enumerate(load_tracker) if load == min_load
            ]
            partition = random.choice(min_partitions)

        # Update load tracker
        load_tracker[partition] += 1
        return partition

    async def _get_hash_partition(
        self, topic: str, key: str | None, partition_count: int
    ) -> int:
        """Get partition using consistent hashing."""
        if key:
            # Use SHA-256 for consistent hashing
            hash_input = f"{topic}:{key}".encode()
            hash_value = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
            return hash_value % partition_count
        else:
            # Random partition if no key
            return random.randint(0, partition_count - 1)

    async def _get_round_robin_partition(self, topic: str, partition_count: int) -> int:
        """Get partition using round-robin strategy."""
        # Initialize counter for topic if needed
        if not hasattr(self, "_round_robin_counters"):
            self._round_robin_counters = {}

        if topic not in self._round_robin_counters:
            self._round_robin_counters[topic] = 0

        partition = self._round_robin_counters[topic] % partition_count
        self._round_robin_counters[topic] += 1
        return partition

    async def _update_partition_metadata(self, topic: str) -> None:
        """Update partition metadata for a topic."""
        try:
            if self.producer and hasattr(self.producer, "_metadata"):
                metadata = self.producer._metadata
                if metadata and topic in metadata.partitions:
                    new_count = len(metadata.partitions[topic])
                    old_count = self.partition_cache.get(topic, 0)

                    if new_count != old_count:
                        logger.info(
                            f"Partition count changed for topic {topic}: {old_count} -> {new_count}"
                        )
                        self.partition_cache[topic] = new_count

                        # Reset load tracker if partition count changed
                        if topic in self.partition_load_tracker:
                            self.partition_load_tracker[topic] = [0] * new_count

        except Exception as e:
            logger.warning(
                f"Failed to update partition metadata for topic {topic}: {e}"
            )

    async def publish_event(
        self,
        event: AnyEvent,
        topic: str | None = None,
        key: str | None = None,
    ) -> bool:
        """Publish an event to Kafka topic with resilience features.

        Args:
            event: Event to publish
            topic: Topic name (if None, uses event's default topic)
            key: Message key for partitioning (if None, uses event ID)

        Returns:
            True if published successfully, False otherwise
        """
        if not self._connected or not self.producer:
            logger.error("Kafka client not connected")
            return False

        # Determine topic and key
        topic_name = topic or event.to_kafka_topic()
        message_key = key or str(event.id)
        event_data = event.model_dump()

        try:
            # Try to publish with retry and circuit breaker protection
            await self._publish_with_resilience(topic_name, event_data, message_key)

            logger.info(f"Published event {event.id} to topic {topic_name}")

            # Log successful message publication for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.WORKFLOW_EXECUTION_START,
                severity=AuditSeverity.LOW,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "operation": "publish_event",
                    "event_id": str(event.id),
                    "topic": topic_name,
                    "message_key": message_key,
                    "event_type": event.type,
                    "event_service": event.service,
                    "publication_status": "successful",
                },
                message=f"Event {event.id} published successfully to topic {topic_name}",
            )

            return True

        except Exception as e:
            logger.error(f"Failed to publish event {event.id} after all retries: {e}")

            # Log message publication failure for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.WORKFLOW_EXECUTION_FAILURE,
                severity=AuditSeverity.HIGH,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "operation": "publish_event",
                    "event_id": str(event.id),
                    "topic": topic_name,
                    "message_key": message_key,
                    "event_type": event.type,
                    "event_service": event.service,
                    "publication_status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                message=f"Event {event.id} publication failed: {e!s}",
            )

            # Send to dead letter queue if enabled
            if self.enable_dead_letter_queue:
                dlq_success = await self._send_to_dead_letter_queue(
                    topic_name,
                    event_data,
                    message_key,
                    str(e),
                )
                if dlq_success:
                    logger.info(f"Event {event.id} sent to dead letter queue")

                    # Log dead letter queue success for audit
                    self.audit_logger.log_event(
                        event_type=AuditEventType.WORKFLOW_EXECUTION_FAILURE,
                        severity=AuditSeverity.MEDIUM,
                        request=None,
                        additional_data={
                            "service": "kafka_client",
                            "operation": "dead_letter_queue",
                            "event_id": str(event.id),
                            "original_topic": topic_name,
                            "dlq_topic": f"{topic_name}{self.dead_letter_topic_suffix}",
                            "dlq_status": "successful",
                            "original_error": str(e),
                        },
                        message=f"Event {event.id} sent to dead letter queue after publication failure",
                    )

                    return True

            return False

    @KAFKA_CIRCUIT_BREAKER()
    async def _publish_with_resilience(
        self,
        topic: str,
        data: dict[str, Any],
        key: str | None = None,
    ) -> None:
        """Publish message with circuit breaker protection.

        Args:
            topic: Topic name
            data: Message data
            key: Message key

        Raises:
            Exception: If publish fails after retries
        """

        async def _publish_operation():
            # Get optimal partition to prevent hotspots
            optimal_partition = await self._get_optimal_partition(topic, key)

            try:
                if optimal_partition is not None:
                    # Use specific partition for load balancing
                    future = await self.producer.send(
                        topic, value=data, key=key, partition=optimal_partition
                    )
                    logger.debug(
                        f"Using optimal partition {optimal_partition} for topic {topic}"
                    )
                else:
                    # Use default Kafka partitioning
                    future = await self.producer.send(topic, value=data, key=key)

                return await asyncio.wait_for(future, timeout=self.timeout_seconds)

            except (AssertionError, ValueError) as e:
                # Partition assignment failed (topic doesn't exist or invalid partition)
                # Fall back to default Kafka partitioning
                if "partition" in str(e).lower():
                    logger.warning(
                        f"Partition assignment failed for topic {topic}, "
                        f"falling back to default partitioning: {e}"
                    )
                    # Clear cached partition info for this topic
                    self.partition_cache.pop(topic, None)

                    # Retry with default partitioning
                    future = await self.producer.send(topic, value=data, key=key)
                    return await asyncio.wait_for(future, timeout=self.timeout_seconds)
                else:
                    # Re-raise if it's not a partition-related error
                    raise

        # Track retry count
        retry_key = f"{topic}:{key}"
        self._retry_counts[retry_key] = self._retry_counts.get(retry_key, 0)

        try:
            record_metadata = await self._retry_with_backoff(_publish_operation)

            # Reset retry count on success
            self._retry_counts.pop(retry_key, None)

            logger.debug(
                f"Published to topic {topic} "
                f"(partition: {record_metadata.partition}, offset: {record_metadata.offset})",
            )

        except (
            TimeoutError,
            KafkaError,
            KafkaTimeoutError,
            asyncio.CancelledError,
        ) as e:
            # Update retry count for expected Kafka/async errors
            self._retry_counts[retry_key] = self.max_retry_attempts + 1
            logger.warning(
                f"Kafka publish failed for topic {topic} after retries: {type(e).__name__}: {e}"
            )
            raise
        except Exception as e:
            # Unexpected error - log with full context
            self._retry_counts[retry_key] = self.max_retry_attempts + 1
            logger.error(
                f"Unexpected error publishing to topic {topic}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    def _validate_topic_name(self, topic: str) -> bool:
        """Validate Kafka topic name against standard rules.

        Args:
            topic: Topic name to validate

        Returns:
            True if valid, False otherwise
        """
        if not topic:
            return False

        # Kafka topic name rules:
        # - Maximum 249 characters
        # - Only alphanumeric, dots, dashes, underscores
        # - Cannot be '.' or '..'
        if len(topic) > 249:
            logger.warning(f"Topic name too long ({len(topic)} chars): {topic[:50]}...")
            return False

        if topic in (".", ".."):
            logger.warning(f"Invalid topic name: {topic}")
            return False

        # Check for valid characters (allowing dots, dashes, underscores, alphanumeric)
        import re

        if not re.match(r"^[a-zA-Z0-9._-]+$", topic):
            logger.warning(f"Topic name contains invalid characters: {topic[:50]}...")
            return False

        return True

    async def publish_raw_event(
        self,
        topic: str,
        data: dict[str, Any],
        key: str | None = None,
    ) -> bool:
        """Publish raw event data to a specific topic with resilience features.

        Args:
            topic: Topic name
            data: Event data dictionary
            key: Message key for partitioning

        Returns:
            True if published successfully, False otherwise
        """
        if not self._connected or not self.producer:
            logger.error("Kafka client not connected")
            return False

        # Validate topic name to fail fast for invalid names
        if not self._validate_topic_name(topic):
            logger.error(f"Invalid topic name, rejecting publish: {topic[:50]}...")
            return False

        try:
            # Try to publish with retry and circuit breaker protection
            await self._publish_with_resilience(topic, data, key)

            logger.info(f"Published raw event to topic {topic}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to publish raw event to {topic} after all retries: {e}",
            )

            # Send to dead letter queue if enabled
            if self.enable_dead_letter_queue:
                dlq_success = await self._send_to_dead_letter_queue(
                    topic,
                    data,
                    key,
                    str(e),
                )
                if dlq_success:
                    logger.info(
                        f"Raw event sent to dead letter queue for topic {topic}",
                    )
                    return True

            return False

    async def publish_with_envelope(
        self,
        event_type: str,
        source_node_id: str,
        payload: dict[str, Any],
        topic: str | None = None,
        correlation_id: UUID | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish event wrapped in OnexEnvelopeV1 for standardized event format.

        Args:
            event_type: Type of event (e.g., "WORKFLOW_STARTED", "AGGREGATION_COMPLETED")
            source_node_id: Node identifier that generated this event
            payload: Event payload data
            topic: Optional override topic (if None, generates from event_type)
            correlation_id: Optional correlation ID for tracing (UUID or string)
            metadata: Optional additional envelope metadata

        Returns:
            True if published successfully, False otherwise
        """
        if not self._connected or not self.producer:
            logger.error("Kafka client not connected")
            return False

        # Track start time for latency metrics
        import time

        start_time = time.perf_counter()

        # Initialize variables before try block so they're available in exception handlers
        topic_name = topic or "unknown"
        kafka_key = None

        try:
            # Import ModelOnexEnvelopeV1 locally to avoid circular dependencies
            from ..nodes.registry.v1_0_0.models.model_onex_envelope_v1 import (
                ModelOnexEnvelopeV1,
            )

            # Convert correlation_id to UUID if it's a string
            corr_id = None
            if correlation_id:
                if isinstance(correlation_id, UUID):
                    corr_id = correlation_id
                elif isinstance(correlation_id, str):
                    try:
                        corr_id = UUID(correlation_id)
                    except (ValueError, AttributeError):
                        logger.warning(
                            f"Invalid correlation_id format: {correlation_id}, using None"
                        )
                        corr_id = None

            # Create envelope with OnexEnvelopeV1 standard format
            envelope = ModelOnexEnvelopeV1.create(
                event_type=event_type,
                source_node_id=source_node_id,
                payload=payload,
                correlation_id=corr_id,
                metadata=metadata or {},
            )

            # Determine topic (use provided or generate from envelope)
            topic_name = topic or envelope.to_kafka_topic(
                service_prefix="omninode_bridge"
            )

            # Get Kafka key for partitioning (correlation_id > source_node_id > event_id)
            kafka_key = envelope.get_kafka_key()

            # Convert envelope to dict for publishing
            envelope_data = envelope.to_dict()

            # Try to publish with retry and circuit breaker protection
            await self._publish_with_resilience(topic_name, envelope_data, kafka_key)

            # Calculate latency and update metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._envelope_published_count += 1
            self._envelope_publish_latencies.append(latency_ms)

            # Keep only last 1000 latency measurements for memory efficiency
            if len(self._envelope_publish_latencies) > 1000:
                self._envelope_publish_latencies = self._envelope_publish_latencies[
                    -1000:
                ]

            logger.info(
                f"Published envelope event {envelope.event_id} "
                f"(type: {event_type}) to topic {topic_name} in {latency_ms:.2f}ms"
            )

            # Log successful envelope publication for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.WORKFLOW_EXECUTION_START,
                severity=AuditSeverity.LOW,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "operation": "publish_with_envelope",
                    "event_id": str(envelope.event_id),
                    "event_type": event_type,
                    "topic": topic_name,
                    "kafka_key": kafka_key,
                    "source_node_id": source_node_id,
                    "correlation_id": str(corr_id) if corr_id else None,
                    "publication_status": "successful",
                    "latency_ms": latency_ms,
                },
                message=f"Envelope event {envelope.event_id} (type: {event_type}) published successfully to {topic_name}",
            )

            return True

        except Exception as e:
            # Update failure metrics
            self._envelope_failed_count += 1

            logger.error(
                f"Failed to publish envelope event (type: {event_type}) after all retries: {e}"
            )

            # Log envelope publication failure for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.WORKFLOW_EXECUTION_FAILURE,
                severity=AuditSeverity.HIGH,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "operation": "publish_with_envelope",
                    "event_type": event_type,
                    "topic": topic_name,
                    "source_node_id": source_node_id,
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "publication_status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                message=f"Envelope event publication failed (type: {event_type}): {e!s}",
            )

            # Send to dead letter queue if enabled
            if self.enable_dead_letter_queue and topic_name:
                # Preserve full envelope context in DLQ, not just payload
                dlq_data = envelope_data if "envelope_data" in locals() else payload
                dlq_success = await self._send_to_dead_letter_queue(
                    topic_name,
                    dlq_data,
                    kafka_key,
                    str(e),
                )
                if dlq_success:
                    logger.info("Envelope event sent to dead letter queue")
                    return True

            return False

    async def send_message(
        self,
        topic: str,
        data: dict[str, Any],
        key: str | None = None,
    ) -> bool:
        """Send a message to a Kafka topic.

        This is an alias for publish_raw_event to maintain compatibility
        with workflow coordinator interface.

        Args:
            topic: Topic name
            data: Message data dictionary
            key: Message key for partitioning

        Returns:
            True if sent successfully, False otherwise
        """
        return await self.publish_raw_event(topic, data, key)

    @KAFKA_CIRCUIT_BREAKER()
    async def create_topic(
        self,
        topic_name: str,
        num_partitions: int = 1,
        replication_factor: int = 1,
    ) -> bool:
        """Create a Kafka topic with specified configuration.

        Args:
            topic_name: Name of the topic to create
            num_partitions: Number of partitions for the topic
            replication_factor: Replication factor for the topic

        Returns:
            True if topic was created successfully, False otherwise
        """
        if not self._connected:
            logger.error("Kafka client not connected - cannot create topic")
            return False

        # Validate topic name to fail fast for invalid names
        if not self._validate_topic_name(topic_name):
            logger.error(
                f"Invalid topic name, rejecting creation: {topic_name[:50]}..."
            )
            return False

        admin_client = None
        try:
            # Create admin client for topic management
            admin_client = AIOKafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                request_timeout_ms=self.timeout_seconds * 1000,
            )
            await admin_client.start()

            # Create topic configuration
            topic = NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor,
            )

            # Create the topic
            await admin_client.create_topics([topic])

            logger.info(
                f"Created topic {topic_name} with {num_partitions} partitions "
                f"and replication factor {replication_factor}"
            )

            # Log successful topic creation for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.RESOURCE_CREATION,
                severity=AuditSeverity.MEDIUM,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "operation": "create_topic",
                    "topic_name": topic_name,
                    "num_partitions": num_partitions,
                    "replication_factor": replication_factor,
                    "creation_status": "successful",
                },
                message=f"Kafka topic {topic_name} created successfully",
            )

            return True

        except Exception as e:
            logger.error(f"Failed to create topic {topic_name}: {e}")

            # Log topic creation failure for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.RESOURCE_CREATION,
                severity=AuditSeverity.HIGH,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "operation": "create_topic",
                    "topic_name": topic_name,
                    "num_partitions": num_partitions,
                    "replication_factor": replication_factor,
                    "creation_status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                message=f"Kafka topic {topic_name} creation failed: {e!s}",
            )

            return False

        finally:
            if admin_client:
                try:
                    await admin_client.close()
                except Exception as e:
                    logger.warning(f"Failed to close admin client: {e}")

    @KAFKA_CIRCUIT_BREAKER()
    async def list_topics(self) -> list[str]:
        """List all available Kafka topics.

        Returns:
            List of topic names, empty list if operation fails
        """
        if not self._connected:
            logger.error("Kafka client not connected - cannot list topics")
            return []

        client = None
        try:
            # Create client for metadata operations
            client = AIOKafkaClient(
                bootstrap_servers=self.bootstrap_servers,
                request_timeout_ms=self.timeout_seconds * 1000,
            )
            await client.bootstrap()

            # Get cluster metadata and extract topic names
            # Force load metadata for all topics
            await client.check_version()
            topics = list(client.cluster.topics())

            logger.info(f"Found {len(topics)} topics")

            # Log successful topic listing for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.RESOURCE_ACCESS,
                severity=AuditSeverity.LOW,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "operation": "list_topics",
                    "topic_count": len(topics),
                    "listing_status": "successful",
                },
                message=f"Listed {len(topics)} Kafka topics successfully",
            )

            return topics

        except Exception as e:
            logger.error(f"Failed to list topics: {e}")

            # Log topic listing failure for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.RESOURCE_ACCESS,
                severity=AuditSeverity.MEDIUM,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "operation": "list_topics",
                    "listing_status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                message=f"Kafka topic listing failed: {e!s}",
            )

            return []

        finally:
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"Failed to close client: {e}")

    async def publish_message(
        self,
        topic: str,
        key: str | None,
        value: dict[str, Any] | str,
    ) -> bool:
        """Alternative publish method for compatibility with test frameworks.

        Args:
            topic: Topic name
            key: Message key for partitioning
            value: Message value (dict or string)

        Returns:
            True if published successfully, False otherwise
        """
        # Convert string values to dict format for consistency
        message_data = {"message": value} if isinstance(value, str) else value

        # Use existing publish_raw_event method
        return await self.publish_raw_event(topic, message_data, key)

    @KAFKA_CIRCUIT_BREAKER()
    async def consume_messages(
        self,
        topic: str,
        group_id: str,
        max_messages: int = 1,
        timeout_ms: int = 5000,
    ) -> list[dict[str, Any]]:
        """Consume messages from a Kafka topic.

        Args:
            topic: Topic name to consume from
            group_id: Consumer group ID
            max_messages: Maximum number of messages to consume
            timeout_ms: Timeout in milliseconds for consumption

        Returns:
            List of consumed messages with metadata
        """
        if not self._connected:
            logger.error("Kafka client not connected - cannot consume messages")
            return []

        consumer = None
        consumed_messages = []

        try:
            # Create consumer with configuration
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: (
                    json.loads(m.decode("utf-8")) if m else None
                ),
                key_deserializer=lambda m: m.decode("utf-8") if m else None,
                auto_offset_reset="earliest",  # Start from beginning if no offset
                enable_auto_commit=True,  # Auto-commit offsets
                consumer_timeout_ms=timeout_ms,
            )

            await consumer.start()

            # Consume messages up to max_messages or timeout
            start_time = asyncio.get_event_loop().time()
            timeout_seconds = timeout_ms / 1000

            while (
                len(consumed_messages) < max_messages
                and (asyncio.get_event_loop().time() - start_time) < timeout_seconds
            ):
                try:
                    # Get messages with remaining timeout
                    remaining_timeout = timeout_seconds - (
                        asyncio.get_event_loop().time() - start_time
                    )
                    if remaining_timeout <= 0:
                        break

                    msg = await asyncio.wait_for(
                        consumer.__anext__(),
                        timeout=remaining_timeout,
                    )

                    # Format message with metadata
                    message_data = {
                        "topic": msg.topic,
                        "partition": msg.partition,
                        "offset": msg.offset,
                        "key": msg.key,
                        "value": msg.value,
                        "timestamp": msg.timestamp,
                        "timestamp_type": msg.timestamp_type,
                    }

                    consumed_messages.append(message_data)

                    logger.debug(
                        f"Consumed message from {topic} "
                        f"(partition: {msg.partition}, offset: {msg.offset})"
                    )

                except TimeoutError:
                    # Timeout reached, return what we have
                    break
                except StopAsyncIteration:
                    # No more messages available
                    break

            logger.info(
                f"Consumed {len(consumed_messages)} messages from topic {topic}"
            )

            # Log successful message consumption for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.RESOURCE_ACCESS,
                severity=AuditSeverity.LOW,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "operation": "consume_messages",
                    "topic": topic,
                    "group_id": group_id,
                    "messages_consumed": len(consumed_messages),
                    "max_messages_requested": max_messages,
                    "timeout_ms": timeout_ms,
                    "consumption_status": "successful",
                },
                message=f"Consumed {len(consumed_messages)} messages from topic {topic}",
            )

            return consumed_messages

        except Exception as e:
            logger.error(f"Failed to consume messages from topic {topic}: {e}")

            # Log message consumption failure for audit
            self.audit_logger.log_event(
                event_type=AuditEventType.RESOURCE_ACCESS,
                severity=AuditSeverity.MEDIUM,
                request=None,
                additional_data={
                    "service": "kafka_client",
                    "operation": "consume_messages",
                    "topic": topic,
                    "group_id": group_id,
                    "max_messages_requested": max_messages,
                    "timeout_ms": timeout_ms,
                    "consumption_status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                message=f"Message consumption from topic {topic} failed: {e!s}",
            )

            return []

        finally:
            if consumer:
                try:
                    await consumer.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop consumer: {e}")

    async def health_check(self) -> KafkaHealthCheckDict:
        """Check Kafka connection health.

        Returns:
            Health status dictionary
        """
        if not self._connected or not self.producer:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Not connected to Kafka",
            }

        try:
            # For aiokafka 0.11.0, we can check if the producer is started
            # and the client connection is available
            if hasattr(self.producer, "_closed") and not self.producer._closed:
                return {
                    "status": "healthy",
                    "connected": True,
                    "bootstrap_servers": self.bootstrap_servers,
                    "producer_active": True,
                }
            else:
                return {
                    "status": "unhealthy",
                    "connected": False,
                    "error": "Producer is closed",
                }

        except Exception as e:
            return {"status": "unhealthy", "connected": False, "error": str(e)}

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to Kafka."""
        return self._connected and self.producer is not None

    async def get_failed_messages(self) -> list[dict[str, Any]]:
        """Get list of messages that were sent to dead letter queue.

        Returns:
            List of failed message metadata
        """
        return self._failed_messages.copy()

    async def get_resilience_metrics(self) -> KafkaResilienceMetricsDict:
        """Get resilience and performance metrics.

        Returns:
            Dictionary with resilience metrics
        """
        total_failures = len(self._failed_messages)
        active_retries = len([k for k, v in self._retry_counts.items() if v > 0])

        # Get recent failures (last hour)
        current_time = datetime.now(UTC)
        recent_failures = [
            msg
            for msg in self._failed_messages
            if (current_time - msg["timestamp"]).total_seconds() < 3600
        ]

        return {
            "connection_status": {
                "connected": self._connected,
                "bootstrap_servers": self.bootstrap_servers,
                "connection_timeout": self.timeout_seconds,
            },
            "resilience_config": {
                "dead_letter_queue_enabled": self.enable_dead_letter_queue,
                "max_retry_attempts": self.max_retry_attempts,
                "retry_backoff_base": self.retry_backoff_base,
                "circuit_breaker_enabled": True,
            },
            "failure_statistics": {
                "total_failures": total_failures,
                "recent_failures_1h": len(recent_failures),
                "active_retries": active_retries,
                "failed_messages_in_dlq": total_failures,
            },
            "performance_metrics": {
                "retry_counts": dict(self._retry_counts),
                "last_failure_time": (
                    self._failed_messages[-1]["timestamp"].isoformat()
                    if self._failed_messages
                    else None
                ),
            },
        }

    async def get_envelope_metrics(self) -> KafkaEnvelopeMetricsDict:
        """Get OnexEnvelopeV1 event publishing metrics.

        Returns:
            Dictionary with envelope publishing metrics
        """
        total_events = self._envelope_published_count + self._envelope_failed_count
        success_rate = (
            self._envelope_published_count / total_events if total_events > 0 else 0.0
        )

        # Calculate latency percentiles
        if self._envelope_publish_latencies:
            sorted_latencies = sorted(self._envelope_publish_latencies)
            count = len(sorted_latencies)
            p50_index = int(count * 0.50)
            p95_index = int(count * 0.95)
            p99_index = int(count * 0.99)

            avg_latency = sum(sorted_latencies) / count
            p50_latency = sorted_latencies[p50_index] if p50_index < count else 0.0
            p95_latency = sorted_latencies[p95_index] if p95_index < count else 0.0
            p99_latency = sorted_latencies[p99_index] if p99_index < count else 0.0
            min_latency = sorted_latencies[0]
            max_latency = sorted_latencies[-1]
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0.0
            min_latency = max_latency = 0.0

        return {
            "envelope_publishing": {
                "total_events_published": self._envelope_published_count,
                "total_events_failed": self._envelope_failed_count,
                "success_rate": success_rate,
            },
            "latency_metrics_ms": {
                "average": round(avg_latency, 2),
                "p50": round(p50_latency, 2),
                "p95": round(p95_latency, 2),
                "p99": round(p99_latency, 2),
                "min": round(min_latency, 2),
                "max": round(max_latency, 2),
                "sample_count": len(self._envelope_publish_latencies),
            },
            "performance_summary": {
                "meets_target_latency": avg_latency < 100.0,  # Target: <100ms avg
                "meets_success_rate": success_rate >= 0.95,  # Target: ≥95%
            },
        }

    async def clear_failed_messages_history(self) -> None:
        """Clear the history of failed messages (for maintenance purposes)."""
        self._failed_messages.clear()
        logger.info("Cleared failed messages history")

    async def retry_failed_messages(self) -> KafkaRetryResultDict:
        """Retry messages from dead letter queue.

        Returns:
            Dictionary with retry statistics
        """
        if not self._failed_messages:
            return {"total": 0, "successful": 0, "failed": 0}

        total = len(self._failed_messages)
        successful = 0
        failed = 0

        # Make a copy to avoid modification during iteration
        messages_to_retry = self._failed_messages.copy()
        self._failed_messages.clear()

        for msg_info in messages_to_retry:
            try:
                # Reset retry count for this message
                retry_key = f"{msg_info['original_topic']}:{msg_info['key']}"
                self._retry_counts.pop(retry_key, None)

                # Try to publish to original topic
                await self._publish_with_resilience(
                    msg_info["original_topic"],
                    msg_info.get("original_data", {}),
                    msg_info["key"],
                )
                successful += 1
                logger.info(
                    f"Successfully retried message for topic {msg_info['original_topic']}",
                )

            except Exception as e:
                failed += 1
                logger.error(
                    f"Failed to retry message for topic {msg_info['original_topic']}: {e}",
                )
                # Add back to failed messages
                self._failed_messages.append(msg_info)

        logger.info(
            f"Retry completed: {successful} successful, {failed} failed out of {total} total",
        )

        return {"total": total, "successful": successful, "failed": failed}
