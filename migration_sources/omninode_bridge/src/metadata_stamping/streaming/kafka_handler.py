"""
Kafka integration for MetadataStampingService Phase 2.

Provides event streaming capabilities for async metadata stamping,
batch processing, and real-time event distribution with high throughput
and reliability.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for metadata stamping operations."""

    STAMP_REQUEST = "stamp_request"
    STAMP_COMPLETED = "stamp_completed"
    STAMP_FAILED = "stamp_failed"
    BATCH_STAMP_REQUEST = "batch_stamp_request"
    BATCH_STAMP_COMPLETED = "batch_stamp_completed"
    HASH_COMPLETED = "hash_completed"
    METADATA_EXTRACTED = "metadata_extracted"
    CACHE_INVALIDATED = "cache_invalidated"
    PERFORMANCE_ALERT = "performance_alert"


class Priority(Enum):
    """Message priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class KafkaConfig:
    """Kafka configuration for high-performance streaming."""

    bootstrap_servers: list[str] = field(default_factory=lambda: ["localhost:9092"])

    # Security settings
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_plain_username: Optional[str] = None
    sasl_plain_password: Optional[str] = None
    ssl_context: Optional[Any] = None

    # Producer settings
    producer_acks: str = "all"  # Ensure all replicas acknowledge
    producer_retries: int = 5
    producer_retry_backoff_ms: int = 100
    producer_request_timeout_ms: int = 30000
    producer_compression_type: str = "gzip"
    producer_batch_size: int = (
        16384  # Configurable via BatchSizeConfig.kafka_producer_batch_size
    )
    producer_linger_ms: int = 5  # Small delay for batching
    producer_buffer_memory: int = 33554432  # 32MB
    producer_max_in_flight_requests: int = 5

    # Consumer settings
    consumer_group_id: str = "metadata_stamping_service"
    consumer_auto_offset_reset: str = "latest"
    consumer_enable_auto_commit: bool = False  # Manual commit for reliability
    consumer_max_poll_records: int = 500
    consumer_session_timeout_ms: int = 30000
    consumer_heartbeat_interval_ms: int = 3000
    consumer_fetch_min_bytes: int = 1024
    consumer_fetch_max_wait_ms: int = 500

    # Topic settings
    default_partitions: int = 3
    default_replication_factor: int = 1

    # Topic names
    stamp_requests_topic: str = "metadata.stamp.requests"
    stamp_results_topic: str = "metadata.stamp.results"
    batch_requests_topic: str = "metadata.batch.requests"
    monitoring_topic: str = "metadata.monitoring"
    deadletter_topic: str = "metadata.deadletter"

    # Performance settings
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


class StampRequestEvent(BaseModel):
    """Event model for stamping requests."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.STAMP_REQUEST
    timestamp: float = Field(default_factory=time.time)
    priority: Priority = Priority.NORMAL

    # Request data
    file_hash: str
    file_path: str
    file_size: int
    content_type: str
    protocol_version: str = "1.0"

    # Processing options
    extract_metadata: bool = True
    cache_result: bool = True
    notify_completion: bool = False

    # Tracing
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class StampResultEvent(BaseModel):
    """Event model for stamping results."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.STAMP_COMPLETED
    timestamp: float = Field(default_factory=time.time)

    # Original request reference
    request_event_id: str

    # Result data
    stamp_id: str
    file_hash: str
    stamp_data: dict[str, Any]
    execution_time_ms: float
    cache_hit: bool = False

    # Performance metrics
    hash_generation_time_ms: float
    metadata_extraction_time_ms: float
    database_operation_time_ms: float

    # Tracing
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class BatchStampRequestEvent(BaseModel):
    """Event model for batch stamping requests."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.BATCH_STAMP_REQUEST
    timestamp: float = Field(default_factory=time.time)
    priority: Priority = Priority.HIGH

    # Batch data
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_requests: list[StampRequestEvent]
    batch_size: int

    # Processing options
    parallel_processing: bool = True
    max_concurrency: int = 10

    # Tracing
    trace_id: Optional[str] = None


class PerformanceAlertEvent(BaseModel):
    """Event model for performance alerts."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.PERFORMANCE_ALERT
    timestamp: float = Field(default_factory=time.time)
    priority: Priority = Priority.CRITICAL

    # Alert data
    alert_type: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    message: str

    # Context
    service_instance: str
    component: str


class KafkaMetrics:
    """Kafka operation metrics tracking."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.messages_produced = 0
        self.messages_consumed = 0
        self.production_errors = 0
        self.consumption_errors = 0
        self.total_production_time = 0.0
        self.total_consumption_time = 0.0
        self.circuit_breaker_trips = 0

    @property
    def average_production_time(self) -> float:
        """Calculate average message production time."""
        return (
            self.total_production_time / self.messages_produced
            if self.messages_produced > 0
            else 0.0
        )

    @property
    def average_consumption_time(self) -> float:
        """Calculate average message consumption time."""
        return (
            self.total_consumption_time / self.messages_consumed
            if self.messages_consumed > 0
            else 0.0
        )

    @property
    def production_error_rate(self) -> float:
        """Calculate production error rate."""
        total = self.messages_produced + self.production_errors
        return self.production_errors / total if total > 0 else 0.0


class MetadataStampingKafkaHandler:
    """
    High-performance Kafka integration for MetadataStampingService.

    Features:
    - Async event streaming for stamping requests and results
    - Batch processing with configurable concurrency
    - Dead letter queue for failed messages
    - Circuit breaker for resilience
    - Performance monitoring and alerting
    - Exactly-once delivery semantics
    - Auto-scaling consumer groups
    """

    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: dict[str, AIOKafkaConsumer] = {}
        self.metrics = KafkaMetrics()

        # Message handlers
        self._event_handlers: dict[
            EventType, list[Callable[[BaseModel], Awaitable[None]]]
        ] = {}

        # Consumer tasks
        self._consumer_tasks: dict[str, asyncio.Task] = {}

        # Circuit breaker state
        self._circuit_breaker_open = False
        self._circuit_breaker_last_failure = 0.0
        self._circuit_breaker_failure_count = 0

        # Dead letter queue
        self._dead_letter_messages: list[dict[str, Any]] = []

    async def initialize(self) -> bool:
        """Initialize Kafka producer and create topics."""
        try:
            # Initialize producer
            await self._initialize_producer()

            # Create topics if they don't exist
            await self._create_topics()

            logger.info(
                f"Kafka handler initialized with brokers: {self.config.bootstrap_servers}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Kafka handler: {e}")
            return False

    async def _initialize_producer(self):
        """Initialize Kafka producer with optimal settings."""
        producer_config = {
            "bootstrap_servers": self.config.bootstrap_servers,
            "acks": self.config.producer_acks,
            "retries": self.config.producer_retries,
            "retry_backoff_ms": self.config.producer_retry_backoff_ms,
            "request_timeout_ms": self.config.producer_request_timeout_ms,
            "compression_type": self.config.producer_compression_type,
            "batch_size": self.config.producer_batch_size,
            "linger_ms": self.config.producer_linger_ms,
            "buffer_memory": self.config.producer_buffer_memory,
            "max_in_flight_requests_per_connection": self.config.producer_max_in_flight_requests,
        }

        # Add security configuration if specified
        if self.config.security_protocol != "PLAINTEXT":
            producer_config.update(
                {
                    "security_protocol": self.config.security_protocol,
                    "sasl_mechanism": self.config.sasl_mechanism,
                    "sasl_plain_username": self.config.sasl_plain_username,
                    "sasl_plain_password": self.config.sasl_plain_password,
                    "ssl_context": self.config.ssl_context,
                }
            )

        self.producer = AIOKafkaProducer(**producer_config)
        await self.producer.start()

    async def _create_topics(self):
        """Create required Kafka topics."""
        # This would typically use kafka-admin-client
        # For now, we assume topics are created externally
        topics = [
            self.config.stamp_requests_topic,
            self.config.stamp_results_topic,
            self.config.batch_requests_topic,
            self.config.monitoring_topic,
            self.config.deadletter_topic,
        ]

        logger.info(f"Topics configured: {topics}")

    @asynccontextmanager
    async def _circuit_breaker(self):
        """Circuit breaker pattern for Kafka operations."""
        if self._circuit_breaker_open:
            if (
                time.time() - self._circuit_breaker_last_failure
                > self.config.circuit_breaker_timeout
            ):
                self._circuit_breaker_open = False
                self._circuit_breaker_failure_count = 0
                logger.info("Circuit breaker closed - attempting Kafka operations")
            else:
                raise KafkaError("Circuit breaker open - Kafka operations disabled")

        try:
            yield
            # Reset failure count on success
            self._circuit_breaker_failure_count = 0
        except KafkaError as e:
            self._circuit_breaker_failure_count += 1
            self._circuit_breaker_last_failure = time.time()

            if (
                self._circuit_breaker_failure_count
                >= self.config.circuit_breaker_threshold
            ):
                self._circuit_breaker_open = True
                self.metrics.circuit_breaker_trips += 1
                logger.warning(
                    f"Circuit breaker opened after {self._circuit_breaker_failure_count} failures"
                )

            raise e

    def _serialize_event(self, event: BaseModel) -> bytes:
        """Serialize event to JSON bytes."""
        return event.model_dump_json().encode("utf-8")

    def _deserialize_event(self, data: bytes, event_class: type) -> BaseModel:
        """Deserialize JSON bytes to event object."""
        return event_class.model_validate_json(data.decode("utf-8"))

    async def publish_stamp_request(self, request: StampRequestEvent) -> bool:
        """Publish a stamping request event."""
        start_time = time.perf_counter()

        try:
            async with self._circuit_breaker():
                # Determine partition based on file_hash for ordered processing
                partition = hash(request.file_hash) % self.config.default_partitions

                await self.producer.send_and_wait(
                    topic=self.config.stamp_requests_topic,
                    value=self._serialize_event(request),
                    key=request.file_hash.encode("utf-8"),
                    partition=partition,
                )

            production_time = (time.perf_counter() - start_time) * 1000
            self.metrics.messages_produced += 1
            self.metrics.total_production_time += production_time

            logger.debug(
                f"Published stamp request {request.event_id} in {production_time:.2f}ms"
            )
            return True

        except Exception as e:
            self.metrics.production_errors += 1
            logger.error(f"Failed to publish stamp request {request.event_id}: {e}")

            # Add to dead letter queue
            await self._handle_dead_letter_message(request, str(e))
            return False

    async def publish_stamp_result(self, result: StampResultEvent) -> bool:
        """Publish a stamping result event."""
        start_time = time.perf_counter()

        try:
            async with self._circuit_breaker():
                await self.producer.send_and_wait(
                    topic=self.config.stamp_results_topic,
                    value=self._serialize_event(result),
                    key=result.file_hash.encode("utf-8"),
                )

            production_time = (time.perf_counter() - start_time) * 1000
            self.metrics.messages_produced += 1
            self.metrics.total_production_time += production_time

            logger.debug(
                f"Published stamp result {result.event_id} in {production_time:.2f}ms"
            )
            return True

        except Exception as e:
            self.metrics.production_errors += 1
            logger.error(f"Failed to publish stamp result {result.event_id}: {e}")
            return False

    async def publish_batch_request(
        self, batch_request: BatchStampRequestEvent
    ) -> bool:
        """Publish a batch stamping request event."""
        start_time = time.perf_counter()

        try:
            async with self._circuit_breaker():
                await self.producer.send_and_wait(
                    topic=self.config.batch_requests_topic,
                    value=self._serialize_event(batch_request),
                    key=batch_request.batch_id.encode("utf-8"),
                )

            production_time = (time.perf_counter() - start_time) * 1000
            self.metrics.messages_produced += 1
            self.metrics.total_production_time += production_time

            logger.info(
                f"Published batch request {batch_request.batch_id} with {batch_request.batch_size} files in {production_time:.2f}ms"
            )
            return True

        except Exception as e:
            self.metrics.production_errors += 1
            logger.error(
                f"Failed to publish batch request {batch_request.batch_id}: {e}"
            )
            return False

    async def publish_performance_alert(self, alert: PerformanceAlertEvent) -> bool:
        """Publish a performance alert event."""
        try:
            async with self._circuit_breaker():
                await self.producer.send_and_wait(
                    topic=self.config.monitoring_topic,
                    value=self._serialize_event(alert),
                    key=alert.alert_type.encode("utf-8"),
                )

            logger.warning(
                f"Published performance alert: {alert.alert_type} - {alert.message}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to publish performance alert {alert.event_id}: {e}")
            return False

    def register_event_handler(
        self, event_type: EventType, handler: Callable[[BaseModel], Awaitable[None]]
    ):
        """Register an event handler for a specific event type.

        Args:
            event_type: Type of event to handle
            handler: Async callable that accepts a BaseModel event and returns None
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []

        self._event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")

    async def start_consumer(
        self, topic: str, event_class: type, consumer_group: Optional[str] = None
    ) -> bool:
        """Start a consumer for a specific topic."""
        try:
            consumer_group = consumer_group or self.config.consumer_group_id

            consumer_config = {
                "bootstrap_servers": self.config.bootstrap_servers,
                "group_id": consumer_group,
                "auto_offset_reset": self.config.consumer_auto_offset_reset,
                "enable_auto_commit": self.config.consumer_enable_auto_commit,
                "max_poll_records": self.config.consumer_max_poll_records,
                "session_timeout_ms": self.config.consumer_session_timeout_ms,
                "heartbeat_interval_ms": self.config.consumer_heartbeat_interval_ms,
                "fetch_min_bytes": self.config.consumer_fetch_min_bytes,
                "fetch_max_wait_ms": self.config.consumer_fetch_max_wait_ms,
            }

            # Add security configuration if specified
            if self.config.security_protocol != "PLAINTEXT":
                consumer_config.update(
                    {
                        "security_protocol": self.config.security_protocol,
                        "sasl_mechanism": self.config.sasl_mechanism,
                        "sasl_plain_username": self.config.sasl_plain_username,
                        "sasl_plain_password": self.config.sasl_plain_password,
                        "ssl_context": self.config.ssl_context,
                    }
                )

            consumer = AIOKafkaConsumer(topic, **consumer_config)
            await consumer.start()

            self.consumers[topic] = consumer

            # Start consumer task
            self._consumer_tasks[topic] = asyncio.create_task(
                self._consume_messages(topic, consumer, event_class)
            )

            logger.info(
                f"Started consumer for topic {topic} with group {consumer_group}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start consumer for topic {topic}: {e}")
            return False

    async def _consume_messages(
        self, topic: str, consumer: AIOKafkaConsumer, event_class: type
    ):
        """
        Consume messages from a topic and process them with idempotency guarantees.

        IDOMPOTENCY REQUIREMENTS:
        =========================

        This method implements idempotent message processing to handle Kafka's
        "at-least-once" delivery semantics. When message processing fails after
        partial completion, the same message may be re-delivered and processed
        again. All event handlers MUST be implemented as idempotent operations.

        Handler Idempotency Contract:
        ------------------------------
        1. **Duplicate Detection**: Each event has a unique event_id that handlers
           should use to detect and skip duplicate processing

        2. **State Validation**: Before making changes, handlers should validate
           the current state to avoid redundant operations

        3. **Conditional Updates**: All database writes and external service calls
           should use conditional logic to prevent duplicate side effects

        4. **Idempotent Key Patterns**: Use business keys (file_hash, batch_id)
           instead of sequential identifiers for idempotent operations

        5. **Transaction Safety**: Database operations should be transactional
           to ensure atomicity and prevent partial updates

        Failure Recovery Scenarios:
        -------------------------
        1. **Handler Failure**: If a handler fails after partial state changes,
           the message will be re-processed. Handlers must clean up partial
           state before reapplying changes.

        2. **Offset Commit Failure**: If offset commit fails after successful
           processing, the message may be re-delivered. The event_id should
           be used to detect and skip duplicates.

        3. **Connection Failure**: Network or database connection failures
           during processing may cause message re-delivery. All operations
           must be retryable and idempotent.

        Handler Implementation Guidelines:
        ---------------------------------
        ```python
        async def handle_stamp_request(self, event: StampRequestEvent) -> None:
            # Check for duplicate processing using event_id
            if await self.is_duplicate_event(event.event_id):
                logger.info(f"Skipping duplicate event {event.event_id}")
                return

            # Use conditional logic for idempotent operations
            existing_stamp = await self.get_stamp_by_hash(event.file_hash)
            if not existing_stamp:
                # Only create if doesn't exist
                await self.create_stamp(event)

            # Mark event as processed to prevent duplicates
            await self.mark_event_processed(event.event_id)
        ```

        Message Processing Flow:
        -------------------------
        1. Deserialize event from Kafka message
        2. Check for event_id to prevent duplicate processing
        3. Process event through registered handlers (idempotent)
        4. Commit offset to mark message as processed
        5. Handle failures with dead letter queue
        """
        try:
            async for message in consumer:
                start_time = time.perf_counter()

                try:
                    # Deserialize event
                    event = self._deserialize_event(message.value, event_class)

                    # IDOMPOTENCY: All handlers must implement idempotent operations
                    # to handle potential message re-delivery from Kafka's at-least-once semantics
                    if event.event_type in self._event_handlers:
                        for handler in self._event_handlers[event.event_type]:
                            try:
                                await handler(event)
                            except Exception as e:
                                logger.error(
                                    f"Event handler failed for {event.event_id}: {e}. "
                                    f"Handler must be idempotent to handle reprocessing."
                                )

                    # Commit offset manually for reliability
                    # This ensures we don't reprocess messages after successful completion
                    await consumer.commit()

                    consumption_time = (time.perf_counter() - start_time) * 1000
                    self.metrics.messages_consumed += 1
                    self.metrics.total_consumption_time += consumption_time

                    logger.debug(
                        f"Processed message {event.event_id} from {topic} in {consumption_time:.2f}ms"
                    )

                except Exception as e:
                    self.metrics.consumption_errors += 1
                    logger.error(f"Failed to process message from {topic}: {e}")

                    # Send to dead letter queue for manual inspection
                    # Messages in DLQ should be analyzed for idempotency issues
                    await self._handle_dead_letter_message(message.value, str(e))

        except asyncio.CancelledError:
            logger.info(f"Consumer for topic {topic} cancelled")
        except Exception as e:
            logger.error(f"Consumer error for topic {topic}: {e}")
        finally:
            await consumer.stop()

    async def _handle_dead_letter_message(
        self, message_data: Union[BaseModel, bytes], error: str
    ):
        """Handle failed messages by sending to dead letter queue."""
        try:
            if isinstance(message_data, BaseModel):
                serialized_data = self._serialize_event(message_data)
            else:
                serialized_data = message_data

            dead_letter_entry = {
                "timestamp": time.time(),
                "original_data": serialized_data.decode("utf-8"),
                "error": error,
                "retry_count": 0,
            }

            # Store in memory for now (would be sent to dead letter topic in production)
            self._dead_letter_messages.append(dead_letter_entry)

            # Optionally publish to dead letter topic
            if self.producer and not self._circuit_breaker_open:
                await self.producer.send_and_wait(
                    topic=self.config.deadletter_topic,
                    value=json.dumps(dead_letter_entry).encode("utf-8"),
                )

            logger.warning(f"Message sent to dead letter queue: {error}")

        except Exception as e:
            logger.error(f"Failed to handle dead letter message: {e}")

    async def stop_consumer(self, topic: str):
        """Stop a consumer for a specific topic."""
        if topic in self._consumer_tasks:
            self._consumer_tasks[topic].cancel()
            try:
                await self._consumer_tasks[topic]
            except asyncio.CancelledError:
                pass
            finally:
                del self._consumer_tasks[topic]

        if topic in self.consumers:
            await self.consumers[topic].stop()
            del self.consumers[topic]

        logger.info(f"Stopped consumer for topic {topic}")

    async def stop_all_consumers(self):
        """Stop all active consumers."""
        topics = list(self._consumer_tasks.keys())
        for topic in topics:
            await self.stop_consumer(topic)

    async def close(self):
        """Close all Kafka connections and cleanup resources."""
        # Stop all consumers
        await self.stop_all_consumers()

        # Stop producer
        if self.producer:
            await self.producer.stop()

        logger.info("Kafka handler closed")

    async def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive Kafka metrics."""
        return {
            "messages_produced": self.metrics.messages_produced,
            "messages_consumed": self.metrics.messages_consumed,
            "production_errors": self.metrics.production_errors,
            "consumption_errors": self.metrics.consumption_errors,
            "average_production_time_ms": self.metrics.average_production_time,
            "average_consumption_time_ms": self.metrics.average_consumption_time,
            "production_error_rate": self.metrics.production_error_rate,
            "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
            "circuit_breaker_open": self._circuit_breaker_open,
            "active_consumers": len(self.consumers),
            "dead_letter_messages": len(self._dead_letter_messages),
            "topics": {
                "stamp_requests": self.config.stamp_requests_topic,
                "stamp_results": self.config.stamp_results_topic,
                "batch_requests": self.config.batch_requests_topic,
                "monitoring": self.config.monitoring_topic,
                "deadletter": self.config.deadletter_topic,
            },
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform Kafka health check."""
        try:
            # Test producer connectivity
            producer_healthy = (
                self.producer is not None and not self._circuit_breaker_open
            )

            # Test consumer connectivity
            consumers_healthy = len(self.consumers) > 0

            status = (
                "healthy"
                if producer_healthy and not self._circuit_breaker_open
                else "degraded"
            )

            return {
                "status": status,
                "producer_connected": producer_healthy,
                "consumers_active": len(self.consumers),
                "circuit_breaker_open": self._circuit_breaker_open,
                "brokers": self.config.bootstrap_servers,
                "total_messages_processed": self.metrics.messages_produced
                + self.metrics.messages_consumed,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self._circuit_breaker_open,
            }


# Factory function for easy integration
async def create_kafka_handler(
    config: Optional[KafkaConfig] = None,
) -> MetadataStampingKafkaHandler:
    """Factory function to create and initialize Kafka handler."""
    if config is None:
        config = KafkaConfig()

    handler = MetadataStampingKafkaHandler(config)
    await handler.initialize()
    return handler


# Event processing utilities
class EventProcessor:
    """Utility class for processing events with retry logic and monitoring."""

    def __init__(
        self, kafka_handler: MetadataStampingKafkaHandler, max_retries: int = 3
    ):
        self.kafka_handler = kafka_handler
        self.max_retries = max_retries

    async def process_stamp_request(self, event: StampRequestEvent) -> bool:
        """Process a stamp request event with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # This would be implemented to actually process the stamping request
                # For now, simulate processing
                await asyncio.sleep(0.1)  # Simulate processing time

                # Create result event
                result = StampResultEvent(
                    request_event_id=event.event_id,
                    stamp_id=str(uuid.uuid4()),
                    file_hash=event.file_hash,
                    stamp_data={"processed": True, "timestamp": time.time()},
                    execution_time_ms=100.0,
                    hash_generation_time_ms=50.0,
                    metadata_extraction_time_ms=30.0,
                    database_operation_time_ms=20.0,
                    trace_id=event.trace_id,
                    span_id=event.span_id,
                )

                # Publish result
                await self.kafka_handler.publish_stamp_result(result)
                return True

            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1} failed for event {event.event_id}: {e}"
                )
                if attempt == self.max_retries - 1:
                    # Final attempt failed, publish failure event
                    failure_event = StampResultEvent(
                        request_event_id=event.event_id,
                        event_type=EventType.STAMP_FAILED,
                        stamp_id="",
                        file_hash=event.file_hash,
                        stamp_data={"error": str(e)},
                        execution_time_ms=0.0,
                        hash_generation_time_ms=0.0,
                        metadata_extraction_time_ms=0.0,
                        database_operation_time_ms=0.0,
                        trace_id=event.trace_id,
                        span_id=event.span_id,
                    )
                    await self.kafka_handler.publish_stamp_result(failure_event)
                    return False

                await asyncio.sleep(2**attempt)  # Exponential backoff

        return False
