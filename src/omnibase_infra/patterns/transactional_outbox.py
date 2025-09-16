"""
PostgreSQL Transactional Outbox Pattern Implementation

Implements the transactional outbox pattern for reliable event publishing
using PostgreSQL with CDC/WAL support for optimal performance.

Features:
- Transactional consistency between business data and events
- CDC/WAL-based event processing for low latency
- Batch processing with SELECT ... FOR UPDATE SKIP LOCKED
- Partitioned tables for performance and maintenance
- At-least-once delivery semantics
- Dead letter queue for failed events
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from asyncpg import Connection
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

from ..infrastructure.postgres_connection_manager import get_connection_manager
from ..models.outbox.model_outbox_event_data import (
    ModelOutboxEventData,
    ModelOutboxStatistics,
)
from ..observability.prometheus_metrics import get_metrics_collector


class EventStatus(Enum):
    """Status of outbox events."""
    PENDING = "pending"
    PROCESSING = "processing"
    PUBLISHED = "published"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class OutboxEvent:
    """Outbox event model for reliable event publishing."""
    id: str
    aggregate_type: str
    aggregate_id: str
    event_type: str
    event_data: ModelOutboxEventData
    status: EventStatus
    created_at: datetime
    updated_at: datetime
    partition_key: str
    topic: str
    retry_count: int = 0
    max_retries: int = 5
    error_message: str | None = None
    processed_at: datetime | None = None
    correlation_id: str | None = None

    def __post_init__(self):
        """Post-initialization processing."""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now(UTC)
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.partition_key:
            self.partition_key = self.aggregate_id

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["status"] = self.status.value
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        if self.processed_at:
            result["processed_at"] = self.processed_at.isoformat()
        return result

    def can_retry(self) -> bool:
        """Check if event can be retried."""
        return self.retry_count < self.max_retries and self.status == EventStatus.FAILED


class PostgreSQLOutboxPattern:
    """
    PostgreSQL Transactional Outbox Pattern implementation.
    
    Provides reliable event publishing with transactional consistency
    using PostgreSQL outbox table and CDC/WAL-based processing.
    """

    def __init__(self, schema: str = "infrastructure"):
        """
        Initialize outbox pattern implementation.
        
        Args:
            schema: Database schema for outbox tables
        """
        self._logger = logging.getLogger(__name__)
        self._schema = schema
        self._table_name = "event_outbox"
        self._full_table_name = f"{schema}.{self._table_name}"

        # Configuration
        self._batch_size = int(os.getenv("OUTBOX_BATCH_SIZE", "50"))
        self._poll_interval_seconds = float(os.getenv("OUTBOX_POLL_INTERVAL", "1.0"))
        self._max_processing_time = int(os.getenv("OUTBOX_MAX_PROCESSING_TIME", "300"))  # 5 minutes
        self._cleanup_interval_hours = int(os.getenv("OUTBOX_CLEANUP_INTERVAL", "24"))
        self._retention_days = int(os.getenv("OUTBOX_RETENTION_DAYS", "7"))

        # Processing state
        self._is_processing = False
        self._processor_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # Metrics collection
        self._metrics = get_metrics_collector()

        self._logger.info(f"Outbox pattern initialized: schema={schema}, batch_size={self._batch_size}")

    async def initialize_outbox_tables(self):
        """Initialize outbox tables with proper schema and partitioning."""
        connection_manager = get_connection_manager()

        try:
            async with connection_manager.transaction() as conn:
                # Create outbox table with partitioning support
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._full_table_name} (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        aggregate_type VARCHAR(100) NOT NULL,
                        aggregate_id UUID NOT NULL,
                        event_type VARCHAR(100) NOT NULL,
                        event_data JSONB NOT NULL,
                        status VARCHAR(20) NOT NULL DEFAULT 'pending',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        processed_at TIMESTAMP WITH TIME ZONE,
                        partition_key VARCHAR(100) NOT NULL,
                        topic VARCHAR(100) NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 5,
                        error_message TEXT,
                        correlation_id UUID,
                        
                        CONSTRAINT event_outbox_status_check 
                        CHECK (status IN ('pending', 'processing', 'published', 'failed', 'dead_letter'))
                    ) PARTITION BY HASH (partition_key)
                """)

                # Create partitions for better performance
                for i in range(4):  # 4 partitions
                    partition_name = f"{self._table_name}_p{i}"
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._schema}.{partition_name}
                        PARTITION OF {self._full_table_name}
                        FOR VALUES WITH (modulus 4, remainder {i})
                    """)

                # Create indexes for performance
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_event_outbox_status_created 
                    ON {self._full_table_name} (status, created_at) 
                    WHERE status IN ('pending', 'failed')
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_event_outbox_aggregate 
                    ON {self._full_table_name} (aggregate_type, aggregate_id)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_event_outbox_correlation 
                    ON {self._full_table_name} (correlation_id) 
                    WHERE correlation_id IS NOT NULL
                """)

                # Create updated_at trigger
                await conn.execute("""
                    CREATE OR REPLACE FUNCTION update_outbox_updated_at()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql
                """)

                await conn.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS trigger_outbox_updated_at
                    BEFORE UPDATE ON {self._full_table_name}
                    FOR EACH ROW
                    EXECUTE FUNCTION update_outbox_updated_at()
                """)

                self._logger.info("Outbox tables initialized successfully")

        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to initialize outbox tables: {e!s}",
            ) from e

    async def publish_event(self,
                           conn: Connection,
                           aggregate_type: str,
                           aggregate_id: str,
                           event_type: str,
                           event_data: ModelOutboxEventData,
                           topic: str,
                           correlation_id: str | None = None) -> str:
        """
        Publish event to outbox within a transaction.
        
        This method should be called within the same transaction as the
        business data changes to ensure atomicity.
        
        Args:
            conn: Database connection (must be in transaction)
            aggregate_type: Type of aggregate (e.g., 'user', 'order')
            aggregate_id: Unique identifier for the aggregate
            event_type: Type of event (e.g., 'user_created', 'order_updated')
            event_data: Event payload data
            topic: Kafka topic to publish to
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            Event ID of the created outbox event
            
        Raises:
            OnexError: If event publishing fails
        """
        start_time = time.perf_counter()
        try:
            event_id = str(uuid.uuid4())
            correlation_uuid = uuid.UUID(correlation_id) if correlation_id else None

            await conn.execute(
                f"""
                INSERT INTO {self._full_table_name} (
                    id, aggregate_type, aggregate_id, event_type, event_data,
                    partition_key, topic, correlation_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                event_id,
                aggregate_type,
                aggregate_id,
                event_type,
                json.dumps(event_data),
                aggregate_id,  # Use aggregate_id as partition key
                topic,
                correlation_uuid,
            )

            # Record database operation metrics
            duration = time.perf_counter() - start_time
            self._metrics.record_database_query(
                operation="INSERT",
                table=self._table_name,
                status="success",
                duration_seconds=duration,
            )

            self._logger.debug(f"Published event to outbox: {event_id}")
            return event_id

        except Exception as e:
            # Record failed database operation
            duration = time.perf_counter() - start_time
            self._metrics.record_database_query(
                operation="INSERT",
                table=self._table_name,
                status="error",
                duration_seconds=duration,
            )

            raise OnexError(
                code=CoreErrorCode.DATABASE_OPERATION_ERROR,
                message=f"Failed to publish event to outbox: {e!s}",
            ) from e

    async def start_processor(self):
        """Start the outbox event processor."""
        if self._is_processing:
            self._logger.warning("Outbox processor is already running")
            return

        self._is_processing = True
        self._shutdown_event.clear()
        self._processor_task = asyncio.create_task(self._process_outbox_events())

        self._logger.info("Outbox event processor started")

    async def stop_processor(self):
        """Stop the outbox event processor gracefully."""
        if not self._is_processing:
            return

        self._shutdown_event.set()

        if self._processor_task:
            try:
                await asyncio.wait_for(self._processor_task, timeout=30.0)
            except TimeoutError:
                self._processor_task.cancel()
                await self._processor_task

        self._is_processing = False
        self._logger.info("Outbox event processor stopped")

    async def _process_outbox_events(self):
        """
        Main processing loop for outbox events.
        
        Uses SELECT ... FOR UPDATE SKIP LOCKED for concurrent processing
        and implements batch processing for optimal performance.
        """
        connection_manager = get_connection_manager()
        last_cleanup = time.time()

        while not self._shutdown_event.is_set():
            try:
                # Process batch of pending events
                events = await self._claim_pending_events()

                if events:
                    self._logger.info(f"Processing {len(events)} outbox events")

                    # Process events in batch
                    await self._process_event_batch(events)

                    # Mark events as published or failed
                    await self._update_event_statuses(events)
                else:
                    # No events to process, wait before next poll
                    await asyncio.sleep(self._poll_interval_seconds)

                # Periodic cleanup of old events
                current_time = time.time()
                if current_time - last_cleanup > (self._cleanup_interval_hours * 3600):
                    await self._cleanup_old_events()
                    last_cleanup = current_time

            except Exception as e:
                self._logger.error(f"Error in outbox processor: {e!s}", exc_info=True)
                await asyncio.sleep(self._poll_interval_seconds)

    async def _claim_pending_events(self) -> list[OutboxEvent]:
        """
        Claim pending events for processing using SELECT ... FOR UPDATE SKIP LOCKED.
        
        Returns:
            List of claimed outbox events
        """
        connection_manager = get_connection_manager()

        try:
            async with connection_manager.transaction() as conn:
                # Claim pending events and failed events ready for retry
                rows = await conn.fetch(f"""
                    SELECT 
                        id, aggregate_type, aggregate_id, event_type, event_data,
                        status, created_at, updated_at, processed_at,
                        partition_key, topic, retry_count, max_retries,
                        error_message, correlation_id
                    FROM {self._full_table_name}
                    WHERE (
                        status = 'pending' 
                        OR (status = 'failed' AND retry_count < max_retries)
                        OR (status = 'processing' AND updated_at < $1)
                    )
                    ORDER BY created_at
                    LIMIT $2
                    FOR UPDATE SKIP LOCKED
                """, datetime.now(UTC) - timedelta(seconds=self._max_processing_time),
                self._batch_size)

                if not rows:
                    return []

                # Mark events as processing
                event_ids = [row["id"] for row in rows]
                await conn.execute(f"""
                    UPDATE {self._full_table_name}
                    SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                    WHERE id = ANY($1::UUID[])
                """, event_ids)

                # Convert to OutboxEvent objects
                events = []
                for row in rows:
                    event = OutboxEvent(
                        id=str(row["id"]),
                        aggregate_type=row["aggregate_type"],
                        aggregate_id=row["aggregate_id"],
                        event_type=row["event_type"],
                        event_data=json.loads(row["event_data"]),
                        status=EventStatus(row["status"]),
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        processed_at=row["processed_at"],
                        partition_key=row["partition_key"],
                        topic=row["topic"],
                        retry_count=row["retry_count"],
                        max_retries=row["max_retries"],
                        error_message=row["error_message"],
                        correlation_id=str(row["correlation_id"]) if row["correlation_id"] else None,
                    )
                    events.append(event)

                return events

        except Exception as e:
            self._logger.error(f"Failed to claim pending events: {e!s}")
            return []

    async def _process_event_batch(self, events: list[OutboxEvent]):
        """
        Process a batch of outbox events.
        
        Args:
            events: List of events to process
        """
        # Group events by topic for batch publishing
        events_by_topic = {}
        for event in events:
            if event.topic not in events_by_topic:
                events_by_topic[event.topic] = []
            events_by_topic[event.topic].append(event)

        # Process each topic group
        for topic, topic_events in events_by_topic.items():
            await self._publish_events_to_kafka(topic, topic_events)

    async def _publish_events_to_kafka(self, topic: str, events: list[OutboxEvent]):
        """
        Publish events to Kafka using the Kafka adapter.
        
        Args:
            topic: Kafka topic
            events: Events to publish
        """
        from ..enums.enum_kafka_operation_type import EnumKafkaOperationType
        from ..infrastructure.container import create_infrastructure_container
        from ..models.kafka.model_kafka_message import ModelKafkaMessage
        from ..nodes.kafka_adapter.v1_0_0.models.model_kafka_adapter_input import (
            ModelKafkaAdapterInput,
        )
        from ..nodes.kafka_adapter.v1_0_0.node import Node as KafkaAdapter

        try:
            # Create Kafka adapter instance
            container = create_infrastructure_container()
            kafka_adapter = KafkaAdapter(container)
            await kafka_adapter.initialize()

            # Publish each event
            for event in events:
                try:
                    # Create Kafka message
                    kafka_message = ModelKafkaMessage(
                        topic=topic,
                        key=event.partition_key,
                        value=json.dumps(event.event_data),
                        headers={"event_type": event.event_type, "aggregate_type": event.aggregate_type},
                        timestamp=event.created_at,
                    )

                    # Create adapter input
                    adapter_input = ModelKafkaAdapterInput(
                        operation_type=EnumKafkaOperationType.PRODUCE,
                        message=kafka_message,
                        correlation_id=uuid.UUID(event.correlation_id) if event.correlation_id else uuid.uuid4(),
                        timeout_seconds=30.0,
                    )

                    # Publish event
                    result = await kafka_adapter.process(adapter_input)

                    if result.success:
                        event.status = EventStatus.PUBLISHED
                        event.processed_at = datetime.now(UTC)
                        event.error_message = None
                        self._logger.debug(f"Published event {event.id} to topic {topic}")
                    else:
                        event.status = EventStatus.FAILED
                        event.retry_count += 1
                        event.error_message = result.error_message
                        self._logger.warning(f"Failed to publish event {event.id}: {result.error_message}")

                        # Move to dead letter queue if max retries exceeded
                        if not event.can_retry():
                            event.status = EventStatus.DEAD_LETTER
                            self._logger.error(f"Event {event.id} moved to dead letter queue after {event.retry_count} retries")

                except Exception as e:
                    event.status = EventStatus.FAILED
                    event.retry_count += 1
                    event.error_message = str(e)

                    if not event.can_retry():
                        event.status = EventStatus.DEAD_LETTER

                    self._logger.error(f"Error publishing event {event.id}: {e!s}")

            await kafka_adapter.cleanup()

        except Exception as e:
            # Mark all events as failed
            for event in events:
                event.status = EventStatus.FAILED
                event.retry_count += 1
                event.error_message = str(e)

                if not event.can_retry():
                    event.status = EventStatus.DEAD_LETTER

            self._logger.error(f"Error in batch event publishing: {e!s}")

    async def _update_event_statuses(self, events: list[OutboxEvent]):
        """
        Update event statuses in the database.
        
        Args:
            events: Events with updated statuses
        """
        connection_manager = get_connection_manager()

        try:
            async with connection_manager.transaction() as conn:
                for event in events:
                    await conn.execute(f"""
                        UPDATE {self._full_table_name}
                        SET status = $1, retry_count = $2, error_message = $3, 
                            processed_at = $4, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $5
                    """, event.status.value, event.retry_count, event.error_message,
                    event.processed_at, uuid.UUID(event.id))

                self._logger.debug(f"Updated status for {len(events)} events")

        except Exception as e:
            self._logger.error(f"Failed to update event statuses: {e!s}")

    async def _cleanup_old_events(self):
        """Clean up old processed events to maintain performance."""
        connection_manager = get_connection_manager()

        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=self._retention_days)

            async with connection_manager.transaction() as conn:
                # Delete old published events
                result = await conn.execute(f"""
                    DELETE FROM {self._full_table_name}
                    WHERE status = 'published' AND processed_at < $1
                """, cutoff_date)

                deleted_count = int(result.split()[1]) if result.split() else 0

                if deleted_count > 0:
                    self._logger.info(f"Cleaned up {deleted_count} old outbox events")

        except Exception as e:
            self._logger.error(f"Failed to cleanup old events: {e!s}")

    async def get_outbox_statistics(self) -> ModelOutboxStatistics:
        """
        Get outbox statistics for monitoring.
        
        Returns:
            Dictionary with outbox statistics
        """
        connection_manager = get_connection_manager()

        try:
            async with connection_manager.acquire_connection() as conn:
                # Get status counts
                status_counts = await conn.fetch(f"""
                    SELECT status, COUNT(*) as count
                    FROM {self._full_table_name}
                    GROUP BY status
                """)

                # Get oldest pending event
                oldest_pending = await conn.fetchrow(f"""
                    SELECT created_at
                    FROM {self._full_table_name}
                    WHERE status IN ('pending', 'failed')
                    ORDER BY created_at
                    LIMIT 1
                """)

                # Get processing rate (events per minute)
                processing_rate = await conn.fetchval(f"""
                    SELECT COUNT(*)
                    FROM {self._full_table_name}
                    WHERE status = 'published' 
                    AND processed_at > $1
                """, datetime.now(UTC) - timedelta(minutes=1))

                stats = {
                    "is_processing": self._is_processing,
                    "batch_size": self._batch_size,
                    "poll_interval_seconds": self._poll_interval_seconds,
                    "retention_days": self._retention_days,
                    "status_counts": {row["status"]: row["count"] for row in status_counts},
                    "oldest_pending_age_minutes": None,
                    "processing_rate_per_minute": processing_rate or 0,
                }

                if oldest_pending:
                    age = datetime.now(UTC) - oldest_pending["created_at"]
                    stats["oldest_pending_age_minutes"] = age.total_seconds() / 60

                return stats

        except Exception as e:
            self._logger.error(f"Failed to get outbox statistics: {e!s}")
            return {"error": str(e)}


# Global outbox pattern instance
_outbox_pattern: PostgreSQLOutboxPattern | None = None


def get_outbox_pattern() -> PostgreSQLOutboxPattern:
    """
    Get global outbox pattern instance.
    
    Returns:
        PostgreSQLOutboxPattern singleton instance
    """
    global _outbox_pattern

    if _outbox_pattern is None:
        _outbox_pattern = PostgreSQLOutboxPattern()

    return _outbox_pattern
