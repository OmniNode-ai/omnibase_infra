# Consumer Service Architecture - Metadata Stamping Service

**Status**: Design Phase
**Type**: Production-Ready Kafka Consumer
**Target**: >1000 events/sec with <100ms latency

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Service Structure](#service-structure)
3. [Processing Pipeline](#processing-pipeline)
4. [Batch Processing](#batch-processing)
5. [Dead Letter Queue](#dead-letter-queue)
6. [Idempotency](#idempotency)
7. [Health Checks](#health-checks)
8. [Metrics & Monitoring](#metrics--monitoring)
9. [Configuration](#configuration)
10. [Graceful Shutdown](#graceful-shutdown)
11. [Deployment](#deployment)
12. [Error Handling Matrix](#error-handling-matrix)
13. [Testing Strategy](#testing-strategy)
14. [Integration Steps](#integration-steps)

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Kafka Topics (Redpanda)                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ dev.omninode_bridge.onex.cmd.metadata-stamp-request.v1           │  │
│  └────────────────────────┬─────────────────────────────────────────┘  │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │ Consumer Group: stamping-consumers
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Stamping Consumer Service                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ KafkaConsumerWrapper (aiokafka)                                │    │
│  │  • Batch: 100 events (500ms timeout)                           │    │
│  │  • Manual commits                                              │    │
│  │  • OnexEnvelopeV1 deserialization                             │    │
│  └────────────────────┬───────────────────────────────────────────┘    │
│                       │                                                 │
│                       ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ Event Router & Validator                                       │    │
│  │  • Validate OnexEnvelopeV1 schema                             │    │
│  │  • Check idempotency (PostgreSQL)                             │    │
│  │  • Route to workflow orchestrator                             │    │
│  └────────────────────┬───────────────────────────────────────────┘    │
│                       │                                                 │
│                       ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ Stamping Workflow (Parallel Processing)                        │    │
│  │  1. Generate BLAKE3 hash                                       │    │
│  │  2. Trigger OnexTree intelligence (optional)                   │    │
│  │  3. Create metadata stamp                                      │    │
│  │  4. Store to PostgreSQL                                        │    │
│  │  5. Publish completion event                                   │    │
│  └────────────────────┬───────────────────────────────────────────┘    │
│                       │                                                 │
│                       ├─── Success ──────────────────────────┐         │
│                       │                                       ▼         │
│                       │          ┌────────────────────────────────┐    │
│                       │          │ Commit Offsets                 │    │
│                       │          │ Publish Success Event          │    │
│                       │          └────────────────────────────────┘    │
│                       │                                                 │
│                       └─── Failure ──────────────────────────┐         │
│                                                               ▼         │
│                                      ┌────────────────────────────┐    │
│                                      │ DLQ Publisher              │    │
│                                      │ Error Metrics              │    │
│                                      └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                            │                            │
                            ▼                            ▼
┌──────────────────────────────────┐    ┌──────────────────────────────┐
│ PostgreSQL                       │    │ DLQ Topics                   │
│  • processed_events (dedup)      │    │  • stamp-request.dlq         │
│  • metadata_stamps               │    │  • Retention: 7 days         │
│  • workflow_executions           │    │  • Manual review required    │
└──────────────────────────────────┘    └──────────────────────────────┘
```

### Performance Characteristics

| Metric | Target | Current |
|--------|--------|---------|
| **Throughput** | 1,000+ events/sec | TBD |
| **Batch Size** | 100 events | - |
| **Batch Timeout** | 500ms | - |
| **Processing Latency** | <100ms per event | - |
| **DLQ Rate** | <0.5% | - |
| **Consumer Lag** | <100 events | - |

---

## Service Structure

### File Organization

```
src/omninode_bridge/consumers/
├── __init__.py
├── stamping_consumer.py          # Main consumer service
├── event_processor.py             # Event processing logic
├── idempotency_checker.py         # Deduplication
├── dlq_publisher.py               # DLQ handling
└── models/
    ├── __init__.py
    ├── consumer_config.py         # Configuration models
    └── processing_result.py       # Result models
```

### Main Consumer Service

**Location**: `src/omninode_bridge/consumers/stamping_consumer.py`

```python
"""
Production-ready Kafka consumer for metadata stamping requests.

Features:
- Batch processing with configurable size and timeout
- Idempotency via PostgreSQL deduplication
- Dead Letter Queue for failed events
- Graceful shutdown with offset commit
- Health checks and metrics
- Circuit breaker integration
"""

import asyncio
import logging
import signal
from typing import Optional
from uuid import UUID

from omnibase_core import ModelOnexError, EnumCoreErrorCode
from ..infrastructure.kafka.kafka_consumer_wrapper import KafkaConsumerWrapper
from ..services.kafka_client import KafkaClient
from ..services.postgres_client import PostgresClient
from .event_processor import EventProcessor
from .idempotency_checker import IdempotencyChecker
from .dlq_publisher import DLQPublisher
from .models.consumer_config import ConsumerConfig

logger = logging.getLogger(__name__)


class StampingConsumerService:
    """
    Production-ready consumer service for metadata stamping requests.

    Consumes from: dev.omninode_bridge.onex.cmd.metadata-stamp-request.v1
    Produces to:
        - dev.omninode_bridge.onex.evt.metadata-stamp-created.v1 (success)
        - dev.omninode_bridge.onex.cmd.metadata-stamp-request.dlq (failure)

    Performance Targets:
        - Throughput: >1000 events/sec
        - Latency: <100ms per event
        - DLQ Rate: <0.5%
    """

    def __init__(
        self,
        config: Optional[ConsumerConfig] = None,
        kafka_consumer: Optional[KafkaConsumerWrapper] = None,
        kafka_producer: Optional[KafkaClient] = None,
        postgres_client: Optional[PostgresClient] = None,
    ):
        """
        Initialize stamping consumer service.

        Args:
            config: Consumer configuration (defaults to environment)
            kafka_consumer: Kafka consumer wrapper
            kafka_producer: Kafka producer client
            postgres_client: PostgreSQL client
        """
        self.config = config or ConsumerConfig.from_environment()

        # Initialize clients
        self.kafka_consumer = kafka_consumer or KafkaConsumerWrapper()
        self.kafka_producer = kafka_producer or KafkaClient()
        self.postgres_client = postgres_client or PostgresClient()

        # Initialize processing components
        self.event_processor = EventProcessor(
            kafka_producer=self.kafka_producer,
            postgres_client=self.postgres_client,
        )
        self.idempotency_checker = IdempotencyChecker(
            postgres_client=self.postgres_client,
        )
        self.dlq_publisher = DLQPublisher(
            kafka_producer=self.kafka_producer,
            dlq_topic_suffix=".dlq",
        )

        # Service state
        self._running = False
        self._shutdown_requested = False
        self._consumption_task: Optional[asyncio.Task] = None

        # Metrics
        self._events_consumed = 0
        self._events_processed = 0
        self._events_failed = 0
        self._events_deduplicated = 0

        logger.info("StampingConsumerService initialized", extra={
            "consumer_group": self.config.consumer_group,
            "batch_size": self.config.batch_size,
            "batch_timeout_ms": self.config.batch_timeout_ms,
        })

    async def start(self) -> None:
        """
        Start consumer service with graceful initialization.

        Raises:
            ModelOnexError: If service startup fails
        """
        try:
            logger.info("Starting StampingConsumerService...")

            # Connect to infrastructure
            await self._connect_infrastructure()

            # Subscribe to topics
            await self._subscribe_topics()

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            # Start consumption loop
            self._running = True
            self._consumption_task = asyncio.create_task(self._consumption_loop())

            logger.info("StampingConsumerService started successfully")

        except Exception as e:
            logger.error(f"Failed to start consumer service: {e}", exc_info=True)
            await self.stop()
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                message=f"Consumer service startup failed: {e!s}",
            ) from e

    async def _connect_infrastructure(self) -> None:
        """Connect to Kafka and PostgreSQL infrastructure."""
        logger.info("Connecting to infrastructure...")

        # Connect PostgreSQL (for idempotency checks and storage)
        await self.postgres_client.connect()
        logger.info("PostgreSQL connected")

        # Connect Kafka producer (for success/DLQ events)
        await self.kafka_producer.connect()
        logger.info("Kafka producer connected")

        logger.info("Infrastructure connection complete")

    async def _subscribe_topics(self) -> None:
        """Subscribe to stamping request topics."""
        topics = [
            "metadata-stamp-request",  # Short name, auto-expanded to full topic
        ]

        await self.kafka_consumer.subscribe_to_topics(
            topics=topics,
            group_id=self.config.consumer_group,
            topic_class="cmd",  # Command topic
        )

        logger.info(f"Subscribed to topics: {topics}")

    async def _consumption_loop(self) -> None:
        """
        Main consumption loop with batch processing.

        Processes events in batches for optimal throughput.
        Implements graceful shutdown on SIGTERM/SIGINT.
        """
        try:
            logger.info("Starting consumption loop")

            async for messages in self.kafka_consumer.consume_messages_stream(
                batch_timeout_ms=self.config.batch_timeout_ms,
                max_records=self.config.batch_size,
            ):
                # Check shutdown signal
                if self._shutdown_requested:
                    logger.info("Shutdown requested, breaking consumption loop")
                    break

                # Process batch
                await self._process_batch(messages)

                # Commit offsets after successful batch processing
                await self.kafka_consumer.commit_offsets()

        except asyncio.CancelledError:
            logger.info("Consumption loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Consumption loop error: {e}", exc_info=True)
            raise
        finally:
            logger.info("Consumption loop terminated")

    async def _process_batch(self, messages: list[dict]) -> None:
        """
        Process batch of stamping request events.

        Args:
            messages: List of Kafka messages to process
        """
        batch_size = len(messages)
        self._events_consumed += batch_size

        logger.debug(f"Processing batch of {batch_size} events")

        # Process events in parallel (asyncio.gather)
        tasks = [self._process_event(msg) for msg in messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        successes = sum(1 for r in results if r is True)
        failures = sum(1 for r in results if isinstance(r, Exception))

        self._events_processed += successes
        self._events_failed += failures

        logger.info(
            f"Batch processed: {successes} success, {failures} failed",
            extra={
                "batch_size": batch_size,
                "successes": successes,
                "failures": failures,
            }
        )

    async def _process_event(self, message: dict) -> bool:
        """
        Process single stamping request event.

        Args:
            message: Kafka message dictionary

        Returns:
            True if processed successfully, False otherwise
        """
        try:
            # Extract envelope from message value
            envelope = message["value"]  # Already deserialized by consumer
            correlation_id = envelope.get("correlation_id")
            event_id = envelope.get("event_id")

            logger.debug(f"Processing event {event_id}", extra={
                "event_id": event_id,
                "correlation_id": correlation_id,
            })

            # Check idempotency (already processed?)
            is_duplicate = await self.idempotency_checker.is_duplicate(event_id)
            if is_duplicate:
                logger.info(f"Event {event_id} already processed, skipping")
                self._events_deduplicated += 1
                return True  # Not an error, just skip

            # Process event through workflow
            result = await self.event_processor.process_stamping_request(envelope)

            if result.success:
                # Mark as processed in idempotency table
                await self.idempotency_checker.mark_processed(
                    event_id=event_id,
                    correlation_id=correlation_id,
                    result=result.data,
                )
                return True
            else:
                # Send to DLQ
                await self.dlq_publisher.publish_to_dlq(
                    original_topic=message["topic"],
                    event_data=envelope,
                    error_reason=result.error_message,
                    retry_count=0,
                )
                return False

        except Exception as e:
            logger.error(f"Event processing error: {e}", exc_info=True)

            # Send to DLQ for manual review
            try:
                await self.dlq_publisher.publish_to_dlq(
                    original_topic=message["topic"],
                    event_data=message["value"],
                    error_reason=str(e),
                    retry_count=0,
                )
            except Exception as dlq_error:
                logger.error(f"DLQ publish failed: {dlq_error}")

            return False

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(self._handle_shutdown_signal(sig))
            )

        logger.info("Signal handlers configured (SIGTERM, SIGINT)")

    async def _handle_shutdown_signal(self, sig: signal.Signals) -> None:
        """
        Handle shutdown signal with graceful cleanup.

        Args:
            sig: Signal received
        """
        logger.info(f"Received shutdown signal: {sig.name}")
        self._shutdown_requested = True

        # Give consumption loop time to finish current batch
        await asyncio.sleep(1)

        # Stop service
        await self.stop()

    async def stop(self) -> None:
        """
        Gracefully stop consumer service with cleanup.

        Ensures:
        - Current batch finishes processing
        - Offsets are committed
        - Connections are closed
        """
        if not self._running:
            logger.debug("Service already stopped")
            return

        try:
            logger.info("Stopping StampingConsumerService...")

            self._running = False
            self._shutdown_requested = True

            # Cancel consumption task
            if self._consumption_task:
                self._consumption_task.cancel()
                try:
                    await asyncio.wait_for(self._consumption_task, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("Consumption task cancellation timed out")
                except asyncio.CancelledError:
                    pass

            # Close consumer (commits offsets)
            await self.kafka_consumer.close_consumer()

            # Disconnect infrastructure
            await self.kafka_producer.disconnect()
            await self.postgres_client.disconnect()

            logger.info("StampingConsumerService stopped successfully")

        except Exception as e:
            logger.error(f"Error during service shutdown: {e}", exc_info=True)

    async def health_check(self) -> dict:
        """
        Comprehensive health check for consumer service.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "running": self._running,
            "shutdown_requested": self._shutdown_requested,
            "metrics": {
                "events_consumed": self._events_consumed,
                "events_processed": self._events_processed,
                "events_failed": self._events_failed,
                "events_deduplicated": self._events_deduplicated,
            },
            "dependencies": {},
        }

        # Check Kafka consumer
        health["dependencies"]["kafka_consumer"] = {
            "status": "healthy" if self.kafka_consumer.is_subscribed else "unhealthy",
            "subscribed": self.kafka_consumer.is_subscribed,
            "topics": self.kafka_consumer.subscribed_topics,
        }

        # Check Kafka producer
        producer_health = await self.kafka_producer.health_check()
        health["dependencies"]["kafka_producer"] = producer_health

        # Check PostgreSQL
        postgres_health = await self.postgres_client.health_check()
        health["dependencies"]["postgres"] = postgres_health

        # Overall status
        if any(
            dep.get("status") == "unhealthy"
            for dep in health["dependencies"].values()
        ):
            health["status"] = "degraded"

        return health

    def get_metrics(self) -> dict:
        """
        Get consumer service metrics.

        Returns:
            Metrics dictionary (Prometheus format)
        """
        return {
            "events_consumed_total": self._events_consumed,
            "events_processed_total": self._events_processed,
            "events_failed_total": self._events_failed,
            "events_deduplicated_total": self._events_deduplicated,
            "processing_success_rate": (
                self._events_processed / max(self._events_consumed, 1)
            ),
            "dlq_rate": (
                self._events_failed / max(self._events_consumed, 1)
            ),
        }


# Entry point for running as standalone service
async def main():
    """Run stamping consumer service."""
    service = StampingConsumerService()

    try:
        await service.start()

        # Keep service running
        while service._running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await service.stop()


if __name__ == "__main__":
    import logging.config

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run service
    asyncio.run(main())
```

---

## Processing Pipeline

### Event Processor

**Location**: `src/omninode_bridge/consumers/event_processor.py`

```python
"""Event processing logic for stamping requests."""

import logging
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from omnibase_core import ModelOnexError, EnumCoreErrorCode
from ..services.kafka_client import KafkaClient
from ..services.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of event processing."""
    success: bool
    data: Optional[dict] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0


class EventProcessor:
    """
    Process stamping request events through workflow.

    Workflow Steps:
    1. Validate event payload
    2. Generate BLAKE3 hash
    3. Trigger OnexTree intelligence (if configured)
    4. Create metadata stamp
    5. Store to PostgreSQL
    6. Publish success event
    """

    def __init__(
        self,
        kafka_producer: KafkaClient,
        postgres_client: PostgresClient,
    ):
        self.kafka_producer = kafka_producer
        self.postgres_client = postgres_client

    async def process_stamping_request(self, envelope: dict) -> ProcessingResult:
        """
        Process stamping request through complete workflow.

        Args:
            envelope: OnexEnvelopeV1 event data

        Returns:
            ProcessingResult with success status and data
        """
        import time
        start_time = time.perf_counter()

        try:
            # Extract payload
            payload = envelope.get("payload", {})
            correlation_id = envelope.get("correlation_id")

            # Validate payload
            if not self._validate_payload(payload):
                return ProcessingResult(
                    success=False,
                    error_message="Invalid payload schema",
                )

            # Step 1: Generate BLAKE3 hash
            file_hash = await self._generate_hash(payload)

            # Step 2: Trigger intelligence (optional, non-blocking)
            intelligence_data = await self._request_intelligence(
                file_hash=file_hash,
                correlation_id=correlation_id,
            )

            # Step 3: Create metadata stamp
            stamp_data = await self._create_stamp(
                file_hash=file_hash,
                payload=payload,
                intelligence=intelligence_data,
            )

            # Step 4: Store to PostgreSQL
            await self._persist_stamp(stamp_data)

            # Step 5: Publish success event
            await self._publish_success_event(
                stamp_data=stamp_data,
                correlation_id=correlation_id,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            return ProcessingResult(
                success=True,
                data=stamp_data,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Stamping workflow failed: {e}", exc_info=True)

            return ProcessingResult(
                success=False,
                error_message=str(e),
                duration_ms=duration_ms,
            )

    def _validate_payload(self, payload: dict) -> bool:
        """Validate stamping request payload."""
        required_fields = ["content", "file_path"]
        return all(field in payload for field in required_fields)

    async def _generate_hash(self, payload: dict) -> str:
        """Generate BLAKE3 hash of content."""
        # TODO: Integrate with BLAKE3HashGenerator
        content = payload.get("content", "")
        import hashlib
        return hashlib.blake3(content.encode()).hexdigest()

    async def _request_intelligence(
        self,
        file_hash: str,
        correlation_id: Optional[UUID],
    ) -> Optional[dict]:
        """Request OnexTree intelligence (non-blocking)."""
        try:
            # TODO: Integrate with OnexTree HTTP client
            # This is optional and should not block the workflow
            return None
        except Exception as e:
            logger.warning(f"Intelligence request failed (non-fatal): {e}")
            return None

    async def _create_stamp(
        self,
        file_hash: str,
        payload: dict,
        intelligence: Optional[dict],
    ) -> dict:
        """Create metadata stamp."""
        from datetime import UTC, datetime

        return {
            "file_hash": file_hash,
            "file_path": payload.get("file_path"),
            "content": payload.get("content"),
            "intelligence_data": intelligence,
            "created_at": datetime.now(UTC).isoformat(),
        }

    async def _persist_stamp(self, stamp_data: dict) -> None:
        """Persist stamp to PostgreSQL."""
        query = """
            INSERT INTO metadata_stamps (
                file_hash, file_path, content, intelligence_data, created_at
            )
            VALUES ($1, $2, $3, $4, $5)
        """

        await self.postgres_client.execute_query(
            query,
            stamp_data["file_hash"],
            stamp_data["file_path"],
            stamp_data["content"],
            stamp_data.get("intelligence_data"),
            stamp_data["created_at"],
        )

    async def _publish_success_event(
        self,
        stamp_data: dict,
        correlation_id: Optional[UUID],
    ) -> None:
        """Publish stamp-created success event."""
        await self.kafka_producer.publish_with_envelope(
            event_type="STAMP_CREATED",
            source_node_id="stamping-consumer",
            payload=stamp_data,
            correlation_id=correlation_id,
        )
```

---

## Batch Processing

### Batch Configuration

```python
# src/omninode_bridge/consumers/models/consumer_config.py

from dataclasses import dataclass
import os


@dataclass
class ConsumerConfig:
    """Consumer service configuration."""

    # Kafka configuration
    consumer_group: str = "stamping-consumers"
    batch_size: int = 100
    batch_timeout_ms: int = 500

    # Processing configuration
    max_workers: int = 10
    processing_timeout_seconds: int = 30

    # Retry configuration
    retry_max_attempts: int = 3
    retry_backoff_base: float = 1.0

    # DLQ configuration
    dlq_enabled: bool = True
    dlq_alert_threshold: int = 10

    # Idempotency configuration
    idempotency_ttl_hours: int = 168  # 7 days

    @classmethod
    def from_environment(cls) -> "ConsumerConfig":
        """Load configuration from environment variables."""
        return cls(
            consumer_group=os.getenv(
                "STAMPING_CONSUMER_GROUP",
                "stamping-consumers"
            ),
            batch_size=int(os.getenv("STAMPING_BATCH_SIZE", "100")),
            batch_timeout_ms=int(os.getenv("STAMPING_BATCH_TIMEOUT_MS", "500")),
            max_workers=int(os.getenv("STAMPING_MAX_WORKERS", "10")),
            processing_timeout_seconds=int(
                os.getenv("STAMPING_PROCESSING_TIMEOUT", "30")
            ),
            retry_max_attempts=int(os.getenv("STAMPING_RETRY_ATTEMPTS", "3")),
            retry_backoff_base=float(os.getenv("STAMPING_RETRY_BACKOFF", "1.0")),
            dlq_enabled=os.getenv("STAMPING_DLQ_ENABLED", "true").lower() == "true",
            dlq_alert_threshold=int(os.getenv("STAMPING_DLQ_THRESHOLD", "10")),
            idempotency_ttl_hours=int(os.getenv("STAMPING_IDEMPOTENCY_TTL", "168")),
        )
```

---

## Dead Letter Queue

### DLQ Publisher

**Location**: `src/omninode_bridge/consumers/dlq_publisher.py`

```python
"""Dead Letter Queue publisher for failed events."""

import logging
from datetime import UTC, datetime
from typing import Any, Optional

from ..services.kafka_client import KafkaClient

logger = logging.getLogger(__name__)


class DLQPublisher:
    """
    Publish failed events to Dead Letter Queue topics.

    DLQ Topics:
    - dev.omninode_bridge.onex.cmd.metadata-stamp-request.dlq

    Retention: 7 days
    Alert Threshold: >10 failures in 1 hour
    """

    def __init__(
        self,
        kafka_producer: KafkaClient,
        dlq_topic_suffix: str = ".dlq",
    ):
        self.kafka_producer = kafka_producer
        self.dlq_topic_suffix = dlq_topic_suffix

        # DLQ metrics
        self._dlq_count = 0
        self._last_alert_time: Optional[datetime] = None

    async def publish_to_dlq(
        self,
        original_topic: str,
        event_data: dict,
        error_reason: str,
        retry_count: int = 0,
    ) -> bool:
        """
        Publish failed event to DLQ topic.

        Args:
            original_topic: Original topic where event failed
            event_data: Event data (OnexEnvelopeV1)
            error_reason: Error message describing failure
            retry_count: Number of retry attempts made

        Returns:
            True if published to DLQ successfully
        """
        try:
            dlq_topic = f"{original_topic}{self.dlq_topic_suffix}"

            # Create DLQ envelope with error metadata
            dlq_envelope = {
                "original_topic": original_topic,
                "original_event": event_data,
                "failure_reason": error_reason,
                "failure_timestamp": datetime.now(UTC).isoformat(),
                "retry_count": retry_count,
                "dlq_metadata": {
                    "dlq_version": "1.0",
                    "requires_manual_review": retry_count >= 3,
                }
            }

            # Publish to DLQ topic
            success = await self.kafka_producer.publish_raw_event(
                topic=dlq_topic,
                data=dlq_envelope,
                key=event_data.get("event_id"),
            )

            if success:
                self._dlq_count += 1
                logger.warning(
                    f"Event sent to DLQ: {dlq_topic}",
                    extra={
                        "dlq_topic": dlq_topic,
                        "error_reason": error_reason,
                        "retry_count": retry_count,
                    }
                )

                # Check alert threshold
                await self._check_alert_threshold()

            return success

        except Exception as e:
            logger.error(f"Failed to publish to DLQ: {e}", exc_info=True)
            return False

    async def _check_alert_threshold(self) -> None:
        """Check if DLQ count exceeds alert threshold."""
        # TODO: Implement alert logic (PagerDuty, email, etc.)
        if self._dlq_count >= 10:
            logger.critical(
                f"DLQ threshold exceeded: {self._dlq_count} failures",
                extra={"dlq_count": self._dlq_count}
            )

    def get_dlq_metrics(self) -> dict:
        """Get DLQ metrics."""
        return {
            "dlq_messages_total": self._dlq_count,
        }
```

---

## Idempotency

### Idempotency Checker

**Location**: `src/omninode_bridge/consumers/idempotency_checker.py`

```python
"""Idempotency checker using PostgreSQL."""

import logging
from typing import Any, Optional
from uuid import UUID
from datetime import UTC, datetime

from ..services.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class IdempotencyChecker:
    """
    Check and track processed events for idempotency.

    PostgreSQL Schema:
        CREATE TABLE processed_events (
            event_id UUID PRIMARY KEY,
            correlation_id UUID,
            processed_at TIMESTAMP WITH TIME ZONE NOT NULL,
            result JSONB,
            ttl_hours INTEGER DEFAULT 168,
            expires_at TIMESTAMP WITH TIME ZONE
        );

        CREATE INDEX idx_processed_events_correlation
            ON processed_events(correlation_id);
        CREATE INDEX idx_processed_events_expires
            ON processed_events(expires_at);
    """

    def __init__(self, postgres_client: PostgresClient):
        self.postgres_client = postgres_client

    async def is_duplicate(self, event_id: str | UUID) -> bool:
        """
        Check if event has already been processed.

        Args:
            event_id: Event ID to check

        Returns:
            True if event already processed, False otherwise
        """
        try:
            query = """
                SELECT event_id
                FROM processed_events
                WHERE event_id = $1
                AND expires_at > NOW()
            """

            result = await self.postgres_client.fetch_one(query, str(event_id))
            return result is not None

        except Exception as e:
            logger.error(f"Idempotency check failed: {e}", exc_info=True)
            # On error, assume not duplicate to avoid blocking
            return False

    async def mark_processed(
        self,
        event_id: str | UUID,
        correlation_id: Optional[str | UUID],
        result: Optional[dict] = None,
        ttl_hours: int = 168,
    ) -> bool:
        """
        Mark event as processed in idempotency table.

        Args:
            event_id: Event ID
            correlation_id: Correlation ID for tracking
            result: Processing result data
            ttl_hours: Time-to-live in hours (default: 7 days)

        Returns:
            True if marked successfully
        """
        try:
            query = """
                INSERT INTO processed_events (
                    event_id,
                    correlation_id,
                    processed_at,
                    result,
                    ttl_hours,
                    expires_at
                )
                VALUES ($1, $2, $3, $4, $5, NOW() + ($5 || ' hours')::INTERVAL)
                ON CONFLICT (event_id) DO NOTHING
            """

            await self.postgres_client.execute_query(
                query,
                str(event_id),
                str(correlation_id) if correlation_id else None,
                datetime.now(UTC),
                result,
                ttl_hours,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to mark event as processed: {e}", exc_info=True)
            return False

    async def cleanup_expired(self) -> int:
        """
        Clean up expired idempotency records.

        Returns:
            Number of records deleted
        """
        try:
            query = """
                DELETE FROM processed_events
                WHERE expires_at < NOW()
            """

            result = await self.postgres_client.execute_query(query)

            # Extract row count from result
            # Format: "DELETE N" where N is count
            count = int(result.split()[-1]) if result else 0

            if count > 0:
                logger.info(f"Cleaned up {count} expired idempotency records")

            return count

        except Exception as e:
            logger.error(f"Idempotency cleanup failed: {e}", exc_info=True)
            return 0
```

### PostgreSQL Schema

```sql
-- Idempotency tracking table
CREATE TABLE IF NOT EXISTS processed_events (
    event_id UUID PRIMARY KEY,
    correlation_id UUID,
    processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    result JSONB DEFAULT '{}',
    ttl_hours INTEGER DEFAULT 168,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Constraints
    CONSTRAINT expires_at_valid CHECK (expires_at > processed_at)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_processed_events_correlation
    ON processed_events(correlation_id);

CREATE INDEX IF NOT EXISTS idx_processed_events_expires
    ON processed_events(expires_at)
    WHERE expires_at > NOW();

-- Auto-cleanup function
CREATE OR REPLACE FUNCTION cleanup_expired_events() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    DELETE FROM processed_events
    WHERE expires_at < NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
```

---

## Health Checks

### FastAPI Health Check Endpoints

```python
# src/omninode_bridge/consumers/api.py
"""
Optional FastAPI endpoints for consumer health checks.

Run alongside consumer service for monitoring.
"""

from fastapi import FastAPI
from typing import Optional

app = FastAPI(title="Stamping Consumer Service")

# Global reference to consumer service
_consumer_service: Optional["StampingConsumerService"] = None


def set_consumer_service(service: "StampingConsumerService"):
    """Set global consumer service reference."""
    global _consumer_service
    _consumer_service = service


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        200: Service healthy
        503: Service unhealthy
    """
    if not _consumer_service:
        return {"status": "not_initialized"}, 503

    health = await _consumer_service.health_check()

    status_code = 200 if health["status"] in ("healthy", "degraded") else 503

    return health, status_code


@app.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint.

    Returns:
        Metrics in Prometheus format
    """
    if not _consumer_service:
        return {"error": "Service not initialized"}, 503

    metrics = _consumer_service.get_metrics()

    # Format as Prometheus metrics
    lines = [
        f"# HELP events_consumed_total Total events consumed from Kafka",
        f"# TYPE events_consumed_total counter",
        f"events_consumed_total {metrics['events_consumed_total']}",
        f"",
        f"# HELP events_processed_total Total events processed successfully",
        f"# TYPE events_processed_total counter",
        f"events_processed_total {metrics['events_processed_total']}",
        f"",
        f"# HELP events_failed_total Total events failed processing",
        f"# TYPE events_failed_total counter",
        f"events_failed_total {metrics['events_failed_total']}",
        f"",
        f"# HELP events_deduplicated_total Total events deduplicated",
        f"# TYPE events_deduplicated_total counter",
        f"events_deduplicated_total {metrics['events_deduplicated_total']}",
        f"",
        f"# HELP processing_success_rate Processing success rate (0-1)",
        f"# TYPE processing_success_rate gauge",
        f"processing_success_rate {metrics['processing_success_rate']:.4f}",
        f"",
        f"# HELP dlq_rate Dead letter queue rate (0-1)",
        f"# TYPE dlq_rate gauge",
        f"dlq_rate {metrics['dlq_rate']:.4f}",
    ]

    return "\n".join(lines), 200, {"Content-Type": "text/plain"}


@app.get("/health/kafka")
async def kafka_health():
    """Kafka-specific health check."""
    if not _consumer_service:
        return {"status": "not_initialized"}, 503

    health = await _consumer_service.health_check()
    kafka_health = health.get("dependencies", {}).get("kafka_consumer", {})

    status_code = 200 if kafka_health.get("status") == "healthy" else 503

    return kafka_health, status_code


@app.get("/health/database")
async def database_health():
    """PostgreSQL-specific health check."""
    if not _consumer_service:
        return {"status": "not_initialized"}, 503

    health = await _consumer_service.health_check()
    postgres_health = health.get("dependencies", {}).get("postgres", {})

    status_code = 200 if postgres_health.get("status") == "healthy" else 503

    return postgres_health, status_code
```

---

## Metrics & Monitoring

### Prometheus Metrics

```python
# Metrics tracked by consumer service
METRICS = {
    "events_consumed_total": {
        "type": "Counter",
        "description": "Total events consumed from Kafka",
    },
    "events_processed_total": {
        "type": "Counter",
        "description": "Total events processed successfully",
    },
    "events_failed_total": {
        "type": "Counter",
        "description": "Total events failed processing",
    },
    "events_deduplicated_total": {
        "type": "Counter",
        "description": "Total events deduplicated",
    },
    "processing_duration_ms": {
        "type": "Histogram",
        "description": "Event processing duration in milliseconds",
        "buckets": [10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
    },
    "batch_size": {
        "type": "Histogram",
        "description": "Batch size distribution",
        "buckets": [10, 25, 50, 100, 250, 500],
    },
    "consumer_lag": {
        "type": "Gauge",
        "description": "Consumer lag (messages behind)",
    },
    "dlq_size": {
        "type": "Gauge",
        "description": "Number of messages in DLQ",
    },
    "processing_success_rate": {
        "type": "Gauge",
        "description": "Processing success rate (0-1)",
    },
    "dlq_rate": {
        "type": "Gauge",
        "description": "DLQ rate (0-1)",
    },
}
```

### Monitoring Dashboard

**Grafana Dashboard Config** (JSON):

```json
{
  "dashboard": {
    "title": "Stamping Consumer Service",
    "panels": [
      {
        "title": "Event Throughput",
        "targets": [
          {
            "expr": "rate(events_consumed_total[5m])",
            "legendFormat": "Consumed"
          },
          {
            "expr": "rate(events_processed_total[5m])",
            "legendFormat": "Processed"
          }
        ]
      },
      {
        "title": "Consumer Lag",
        "targets": [
          {
            "expr": "consumer_lag",
            "legendFormat": "Lag"
          }
        ]
      },
      {
        "title": "Processing Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, processing_duration_ms)",
            "legendFormat": "p95"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "dlq_rate",
            "legendFormat": "DLQ Rate"
          }
        ]
      }
    ]
  }
}
```

---

## Configuration

### Environment Variables

```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=omninode-bridge-redpanda:9092
STAMPING_CONSUMER_GROUP=stamping-consumers
STAMPING_BATCH_SIZE=100
STAMPING_BATCH_TIMEOUT_MS=500

# Processing Configuration
STAMPING_MAX_WORKERS=10
STAMPING_PROCESSING_TIMEOUT=30
STAMPING_RETRY_ATTEMPTS=3
STAMPING_RETRY_BACKOFF=1.0

# DLQ Configuration
STAMPING_DLQ_ENABLED=true
STAMPING_DLQ_THRESHOLD=10

# Idempotency Configuration
STAMPING_IDEMPOTENCY_TTL=168  # 7 days in hours

# PostgreSQL Configuration
POSTGRES_HOST=omninode-bridge-postgres
POSTGRES_PORT=5432
POSTGRES_DATABASE=omninode_bridge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

# Logging
LOG_LEVEL=info
```

### config/consumer.yaml

```yaml
consumer:
  bootstrap_servers: ${KAFKA_BOOTSTRAP_SERVERS}
  group_id: stamping-consumers
  topics:
    - metadata-stamp-request
  batch_size: 100
  batch_timeout_ms: 500
  max_poll_interval_ms: 300000
  enable_auto_commit: false

processing:
  workers: 10
  timeout_seconds: 30
  retry_max_attempts: 3
  retry_backoff_base: 1.0

dlq:
  enabled: true
  topic_suffix: .dlq
  alert_threshold: 10
  retention_hours: 168

idempotency:
  enabled: true
  ttl_hours: 168
  cleanup_interval_hours: 24
```

---

## Graceful Shutdown

```python
# Signal handling implementation (already in main service class)

def _setup_signal_handlers(self) -> None:
    """Setup signal handlers for graceful shutdown."""
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(self._handle_shutdown_signal(s))
        )

async def _handle_shutdown_signal(self, sig: signal.Signals) -> None:
    """
    Handle shutdown signal with graceful cleanup.

    Shutdown Sequence:
    1. Set shutdown flag (prevents new batches)
    2. Wait for current batch to finish (max 30s)
    3. Commit offsets
    4. Close consumer
    5. Disconnect infrastructure
    """
    logger.info(f"Received shutdown signal: {sig.name}")
    self._shutdown_requested = True

    # Wait for current batch to finish
    await asyncio.sleep(1)

    # Stop service
    await self.stop()
```

---

## Deployment

### Dockerfile

```dockerfile
# docker/stamping-consumer/Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Poetry and dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run consumer service
CMD ["python", "-m", "omninode_bridge.consumers.stamping_consumer"]
```

### Docker Compose

```yaml
# deployment/docker-compose.yml (addition)

services:
  stamping-consumer:
    build:
      context: .
      dockerfile: docker/stamping-consumer/Dockerfile
    container_name: omninode-bridge-stamping-consumer
    networks:
      - omninode-bridge-network
    depends_on:
      redpanda:
        condition: service_healthy
      postgres:
        condition: service_healthy
    ports:
      - "8080:8080"  # Health check API
    environment:
      # Kafka Configuration
      KAFKA_BOOTSTRAP_SERVERS: omninode-bridge-redpanda:9092
      STAMPING_CONSUMER_GROUP: stamping-consumers
      STAMPING_BATCH_SIZE: 100
      STAMPING_BATCH_TIMEOUT_MS: 500

      # PostgreSQL Configuration
      POSTGRES_HOST: omninode-bridge-postgres
      POSTGRES_PORT: 5432
      POSTGRES_DATABASE: omninode_bridge
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}

      # Processing Configuration
      STAMPING_MAX_WORKERS: 10
      STAMPING_PROCESSING_TIMEOUT: 30
      STAMPING_RETRY_ATTEMPTS: 3

      # DLQ Configuration
      STAMPING_DLQ_ENABLED: true
      STAMPING_DLQ_THRESHOLD: 10

      # Logging
      LOG_LEVEL: info
      ENVIRONMENT: development

    deploy:
      replicas: 3  # Horizontal scaling
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

    restart: unless-stopped
```

---

## Error Handling Matrix

| Error Type | Retry? | DLQ? | Alert? | Action |
|------------|--------|------|--------|--------|
| **Validation Error** | ❌ No | ✅ Yes | ❌ No | Invalid schema, send to DLQ for review |
| **Database Timeout** | ✅ Yes (3x) | ✅ Yes | ✅ Yes | Retry with exponential backoff |
| **Kafka Timeout** | ✅ Yes (3x) | ❌ No | ✅ Yes | Kafka unavailable, retry without DLQ |
| **Intelligence Timeout** | ❌ No | ❌ No | ❌ No | Non-blocking, proceed without intelligence |
| **Duplicate Event** | ❌ No | ❌ No | ❌ No | Already processed, skip silently |
| **Hash Generation Error** | ✅ Yes (3x) | ✅ Yes | ⚠️ Maybe | Critical error, retry then DLQ |
| **Unexpected Error** | ✅ Yes (3x) | ✅ Yes | ✅ Yes | Unknown error, retry then DLQ with alert |
| **DLQ Publish Failure** | ❌ No | ❌ N/A | ✅ Yes | Log locally, critical alert |

### Retry Strategy

```python
async def _retry_with_backoff(
    self,
    operation: Callable,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
) -> Any:
    """
    Retry operation with exponential backoff.

    Backoff Formula: delay = backoff_base * (2 ** attempt)
    Example: 1s, 2s, 4s for base=1.0
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await operation()
        except Exception as e:
            last_exception = e

            if attempt < max_attempts - 1:
                delay = backoff_base * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)

    raise last_exception
```

---

## Testing Strategy

### Unit Tests

```python
# tests/consumers/test_stamping_consumer.py

import pytest
from omninode_bridge.consumers.stamping_consumer import StampingConsumerService


@pytest.mark.asyncio
async def test_event_processing_success():
    """Test successful event processing."""
    # Mock dependencies
    # Test event processing
    # Assert success metrics
    pass


@pytest.mark.asyncio
async def test_idempotency_duplicate_detection():
    """Test duplicate event detection."""
    # Process same event twice
    # Assert second is skipped
    pass


@pytest.mark.asyncio
async def test_dlq_publish_on_failure():
    """Test DLQ publish for failed events."""
    # Simulate processing failure
    # Assert DLQ publish
    pass
```

### Integration Tests

```python
# tests/integration/test_consumer_integration.py

@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_stamping_flow():
    """Test complete stamping flow from Kafka to PostgreSQL."""
    # 1. Publish test event to Kafka
    # 2. Wait for consumer to process
    # 3. Verify PostgreSQL record
    # 4. Verify success event published
    pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consumer_lag_under_load():
    """Test consumer lag under high load."""
    # Publish 10,000 events
    # Monitor consumer lag
    # Assert lag < 100 messages
    pass
```

### Load Tests

```python
# tests/performance/test_consumer_load.py

@pytest.mark.performance
@pytest.mark.asyncio
async def test_throughput_1000_events_per_second():
    """Test consumer can handle 1000+ events/sec."""
    # Publish events at 1000/sec rate
    # Monitor processing metrics
    # Assert throughput >= 1000/sec
    pass


@pytest.mark.performance
@pytest.mark.asyncio
async def test_processing_latency_p95():
    """Test p95 processing latency < 100ms."""
    # Process 10,000 events
    # Measure latencies
    # Assert p95 < 100ms
    pass
```

---

## Integration Steps

### Implementation Checklist

#### Phase 1: Core Infrastructure (Week 1)

- [ ] Create consumer service skeleton
  - [ ] `stamping_consumer.py` - Main service
  - [ ] `event_processor.py` - Event processing logic
  - [ ] `models/consumer_config.py` - Configuration

- [ ] Implement idempotency system
  - [ ] `idempotency_checker.py` - Duplicate detection
  - [ ] Create `processed_events` PostgreSQL table
  - [ ] Add TTL cleanup function

- [ ] Implement DLQ publisher
  - [ ] `dlq_publisher.py` - DLQ handling
  - [ ] Configure DLQ topics
  - [ ] Add alert threshold logic

#### Phase 2: Processing Pipeline (Week 2)

- [ ] Implement event processor
  - [ ] Payload validation
  - [ ] BLAKE3 hash integration
  - [ ] OnexTree intelligence integration (optional)
  - [ ] PostgreSQL persistence
  - [ ] Success event publishing

- [ ] Add batch processing
  - [ ] Configure batch size (100 events)
  - [ ] Configure timeout (500ms)
  - [ ] Parallel processing with asyncio.gather

- [ ] Add error handling
  - [ ] Retry with exponential backoff
  - [ ] DLQ routing for failures
  - [ ] Error metrics tracking

#### Phase 3: Health & Monitoring (Week 3)

- [ ] Add health check API
  - [ ] `/health` endpoint
  - [ ] `/health/kafka` endpoint
  - [ ] `/health/database` endpoint
  - [ ] `/metrics` Prometheus endpoint

- [ ] Implement metrics collection
  - [ ] Event counters
  - [ ] Processing latency histogram
  - [ ] Consumer lag gauge
  - [ ] DLQ size gauge

- [ ] Create Grafana dashboards
  - [ ] Throughput panel
  - [ ] Latency panel
  - [ ] Error rate panel
  - [ ] Consumer lag panel

#### Phase 4: Testing & Validation (Week 4)

- [ ] Write unit tests
  - [ ] Consumer service tests
  - [ ] Event processor tests
  - [ ] Idempotency tests
  - [ ] DLQ publisher tests

- [ ] Write integration tests
  - [ ] End-to-end flow tests
  - [ ] Kafka integration tests
  - [ ] PostgreSQL integration tests

- [ ] Write performance tests
  - [ ] Throughput load tests (1000+ events/sec)
  - [ ] Latency tests (p95 < 100ms)
  - [ ] Consumer lag tests

- [ ] Conduct load testing
  - [ ] Generate test events
  - [ ] Monitor metrics
  - [ ] Validate performance targets

#### Phase 5: Deployment (Week 5)

- [ ] Create Dockerfile
  - [ ] Multi-stage build
  - [ ] Health check configuration

- [ ] Update docker-compose.yml
  - [ ] Add consumer service
  - [ ] Configure environment variables
  - [ ] Set resource limits

- [ ] Deploy to development
  - [ ] Build container
  - [ ] Deploy with docker-compose
  - [ ] Validate health checks
  - [ ] Monitor metrics

- [ ] Documentation
  - [ ] Update CLAUDE.md
  - [ ] Create runbook
  - [ ] Document troubleshooting

---

## Summary

### Key Features

✅ **Batch Processing**: 100 events per batch with 500ms timeout
✅ **Idempotency**: PostgreSQL-based deduplication with 7-day TTL
✅ **Dead Letter Queue**: Failed events routed to DLQ with alert threshold
✅ **Health Checks**: Comprehensive health and metrics endpoints
✅ **Graceful Shutdown**: Signal handling with offset commit
✅ **Horizontal Scaling**: Consumer group coordination for 3+ replicas
✅ **Error Handling**: Retry with exponential backoff, comprehensive error matrix
✅ **Monitoring**: Prometheus metrics with Grafana dashboards

### Performance Targets

| Metric | Target |
|--------|--------|
| Throughput | >1,000 events/sec |
| Batch Size | 100 events |
| Processing Latency | <100ms (p95) |
| Consumer Lag | <100 events |
| DLQ Rate | <0.5% |
| Success Rate | >95% |

### Next Steps

1. Implement Phase 1 (Core Infrastructure)
2. Integrate with existing BLAKE3HashGenerator
3. Add OnexTree intelligence client
4. Create comprehensive tests
5. Deploy to development environment
6. Monitor and optimize performance

---

**Document Version**: 1.0.0
**Last Updated**: 2025-10-24
**Author**: Claude Code Agent
**Status**: Design Complete - Ready for Implementation
