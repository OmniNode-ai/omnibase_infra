#!/usr/bin/env python3
"""
ProjectionMaterializerService - Projection Materializer Service.

Subscribes to StateCommitted events from Kafka and atomically updates
read-optimized projections with watermark tracking for eventual consistency.

ONEX v2.0 Compliance:
- Event-driven architecture with Kafka consumer
- Atomic database transactions for consistency
- Watermark-based progress tracking
- Metrics collection for lag and throughput

Pure Reducer Refactor - Wave 2, Workstream 2C
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Optional

from prometheus_client import Counter, Gauge
from pydantic import BaseModel, Field

from omninode_bridge.infrastructure.entities import ModelStateCommittedEvent
from omninode_bridge.infrastructure.kafka.kafka_consumer_wrapper import (
    KafkaConsumerWrapper,
)
from omninode_bridge.infrastructure.postgres_connection_manager import (
    ModelPostgresConfig,
    PostgresConnectionManager,
)

logger = logging.getLogger(__name__)

# Prometheus Metrics
projection_projections_materialized_total = Counter(
    "projection_materializer_projections_materialized_total",
    "Total number of projections materialized successfully",
    ["workflow_key"],
)

projection_projections_failed_total = Counter(
    "projection_materializer_projections_failed_total",
    "Total number of failed projection updates",
)

projection_watermark_updates_total = Counter(
    "projection_materializer_watermark_updates_total",
    "Total number of watermark updates",
)

projection_wm_regressions_total = Counter(
    "projection_materializer_wm_regressions_total",
    "Total number of watermark regressions detected",
)

projection_wm_lag_ms = Gauge(
    "projection_materializer_wm_lag_ms",
    "Current watermark lag in milliseconds",
    ["workflow_key"],
)

projection_max_lag_ms = Gauge(
    "projection_materializer_max_lag_ms",
    "Maximum watermark lag observed in milliseconds",
)

projection_events_processed_per_second = Gauge(
    "projection_materializer_events_processed_per_second",
    "Current event processing rate (events/second)",
)

projection_duplicate_events_skipped = Counter(
    "projection_materializer_duplicate_events_skipped",
    "Number of duplicate events skipped via idempotence",
)


class ProjectionMaterializerMetrics(BaseModel):
    """Metrics for projection materializer service."""

    # Projection metrics
    projections_materialized_total: int = Field(
        default=0,
        description="Total number of projections materialized",
    )
    projections_failed_total: int = Field(
        default=0,
        description="Total number of failed projection updates",
    )

    # Watermark metrics
    watermark_updates_total: int = Field(
        default=0,
        description="Total number of watermark updates",
    )
    wm_regressions_total: int = Field(
        default=0,
        description="Total number of watermark regressions detected",
    )

    # Lag metrics
    projection_wm_lag_ms: float = Field(
        default=0.0,
        description="Current watermark lag in milliseconds",
    )
    max_lag_ms: float = Field(
        default=0.0,
        description="Maximum watermark lag observed",
    )

    # Throughput metrics
    events_processed_per_second: float = Field(
        default=0.0,
        description="Current event processing rate",
    )

    # Idempotence metrics
    duplicate_events_skipped: int = Field(
        default=0,
        description="Number of duplicate events skipped",
    )


class ProjectionMaterializerService:
    """
    Projection materializer service for eventual consistency.

    Subscribes to StateCommitted events from Kafka and atomically updates
    workflow projections with watermark tracking. Ensures eventual consistency
    between canonical store and read-optimized projections.

    Architecture:
    - Kafka consumer for StateCommitted events
    - Atomic projection + watermark updates in database transaction
    - Idempotence checking to prevent duplicate processing
    - Lag monitoring with metrics collection

    Performance Targets:
    - Throughput: >1000 events/second
    - Watermark lag: <100ms under normal load
    - Update latency: <10ms per projection

    Example:
        >>> materializer = ProjectionMaterializerService(
        ...     bootstrap_servers="localhost:29092",
        ...     consumer_group="projection-materializer",
        ...     postgres_config=config
        ... )
        >>> await materializer.start()
        >>> # Service runs in background consuming events
        >>> await materializer.stop()
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:29092",
        consumer_group: str = "projection-materializer",
        postgres_config: Optional[dict[str, Any]] = None,
        enable_idempotence: bool = True,
    ):
        """
        Initialize projection materializer service.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            consumer_group: Kafka consumer group ID
            postgres_config: PostgreSQL configuration dict (optional, uses env vars if None)
            enable_idempotence: Enable duplicate event detection
        """
        self._bootstrap_servers = bootstrap_servers
        self._consumer_group = consumer_group
        self._enable_idempotence = enable_idempotence

        # Initialize Kafka consumer
        self._consumer = KafkaConsumerWrapper(
            bootstrap_servers=bootstrap_servers,
        )

        # Initialize PostgreSQL connection manager
        # If no config provided, ModelPostgresConfig will load from environment
        if postgres_config is None:
            pg_config = ModelPostgresConfig.from_environment()
        else:
            pg_config = ModelPostgresConfig(**postgres_config)

        self._db = PostgresConnectionManager(config=pg_config)

        # Metrics tracking
        self._metrics = ProjectionMaterializerMetrics()
        self._last_metrics_log = datetime.now(UTC)

        # Service state
        self._is_running = False
        self._consumer_task: Optional[asyncio.Task] = None

        logger.info(
            "ProjectionMaterializerService initialized",
            extra={
                "bootstrap_servers": bootstrap_servers,
                "consumer_group": consumer_group,
                "enable_idempotence": enable_idempotence,
            },
        )

    async def start(self) -> None:
        """
        Start the projection materializer service.

        Subscribes to StateCommitted Kafka topic and begins consuming events
        in the background. Call stop() to gracefully shutdown.

        Raises:
            Exception: If service is already running or subscription fails
        """
        if self._is_running:
            logger.warning("ProjectionMaterializerService already running")
            return

        logger.info("Starting ProjectionMaterializerService")

        # Initialize database connection pool
        await self._db.initialize()

        # Subscribe to StateCommitted events
        await self._consumer.subscribe_to_topics(
            topics=["state-committed"],
            group_id=self._consumer_group,
            topic_class="evt",
        )

        # Start background consumer task
        self._is_running = True
        self._consumer_task = asyncio.create_task(self._consume_events())

        logger.info(
            "ProjectionMaterializerService started",
            extra={
                "topics": self._consumer.subscribed_topics,
                "group_id": self._consumer_group,
            },
        )

    async def stop(self) -> None:
        """
        Stop the projection materializer service.

        Gracefully shuts down Kafka consumer and database connections.
        Waits for in-flight events to complete processing.
        """
        if not self._is_running:
            logger.warning("ProjectionMaterializerService not running")
            return

        logger.info("Stopping ProjectionMaterializerService")

        # Signal shutdown
        self._is_running = False

        # Cancel consumer task
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                logger.debug("Consumer task cancelled successfully")

        # Close Kafka consumer
        await self._consumer.close_consumer()

        # Close database connections
        await self._db.close()

        logger.info(
            "ProjectionMaterializerService stopped",
            extra={
                "total_projections": self._metrics.projections_materialized_total,
                "total_failures": self._metrics.projections_failed_total,
            },
        )

    async def _consume_events(self) -> None:
        """
        Background task for consuming StateCommitted events.

        Processes events in batches for efficiency while maintaining
        atomic per-event projection updates.
        """
        try:
            async for messages in self._consumer.consume_messages_stream(
                batch_timeout_ms=1000, max_records=500
            ):
                # Track batch start time
                batch_start = datetime.now(UTC)

                # Process each message in batch
                for msg in messages:
                    try:
                        await self._process_state_committed_event(msg)
                    except Exception as e:
                        logger.error(
                            f"Failed to process StateCommitted event: {e}",
                            exc_info=True,
                            extra={
                                "offset": msg.get("offset"),
                                "partition": msg.get("partition"),
                            },
                        )
                        self._metrics.projections_failed_total += 1
                        projection_projections_failed_total.inc()

                # Commit offsets after successful batch processing
                await self._consumer.commit_offsets()

                # Update throughput metrics
                batch_duration = (datetime.now(UTC) - batch_start).total_seconds()
                if batch_duration > 0:
                    events_per_sec = len(messages) / batch_duration
                    self._metrics.events_processed_per_second = events_per_sec
                    projection_events_processed_per_second.set(events_per_sec)

                # Log metrics periodically
                await self._log_metrics_if_needed()

        except asyncio.CancelledError:
            logger.info("Event consumption cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in event consumer: {e}", exc_info=True)
            self._is_running = False
            raise

    async def _process_state_committed_event(
        self, kafka_message: dict[str, Any]
    ) -> None:
        """
        Process a single StateCommitted event.

        Atomically updates projection and watermark in database transaction.
        Implements idempotence checking to prevent duplicate processing.

        Args:
            kafka_message: Kafka message dict with StateCommitted event

        Raises:
            Exception: If event processing fails
        """
        # Extract Kafka metadata
        partition_id = f"kafka-partition-{kafka_message['partition']}"
        offset = kafka_message["offset"]
        timestamp_ms = kafka_message["timestamp"]

        # Parse event payload
        event_data = kafka_message["value"]
        event = ModelStateCommittedEvent(**event_data)

        # Calculate watermark lag
        current_time_ms = datetime.now(UTC).timestamp() * 1000
        lag_ms = current_time_ms - timestamp_ms
        self._metrics.projection_wm_lag_ms = lag_ms
        self._metrics.max_lag_ms = max(self._metrics.max_lag_ms, lag_ms)

        # Update Prometheus metrics
        projection_wm_lag_ms.labels(workflow_key=event.workflow_key).set(lag_ms)
        projection_max_lag_ms.set(self._metrics.max_lag_ms)

        # Atomic projection + watermark update
        async with self._db.transaction() as conn:
            # Check for duplicate processing (idempotence)
            if self._enable_idempotence:
                is_duplicate = await self._check_duplicate(conn, partition_id, offset)
                if is_duplicate:
                    self._metrics.duplicate_events_skipped += 1
                    projection_duplicate_events_skipped.inc()
                    logger.debug(
                        "Skipping duplicate event",
                        extra={
                            "partition_id": partition_id,
                            "offset": offset,
                            "workflow_key": event.workflow_key,
                        },
                    )
                    return

            # Upsert workflow projection
            await self._upsert_projection(conn, event)

            # Advance watermark atomically
            await self._advance_watermark(conn, partition_id, offset)

            # Record processing in idempotence log (if enabled)
            if self._enable_idempotence:
                await self._record_processing(conn, partition_id, offset)

        # Update metrics
        self._metrics.projections_materialized_total += 1
        self._metrics.watermark_updates_total += 1

        # Update Prometheus metrics
        projection_projections_materialized_total.labels(
            workflow_key=event.workflow_key
        ).inc()
        projection_watermark_updates_total.inc()

        logger.debug(
            "Projection materialized successfully",
            extra={
                "workflow_key": event.workflow_key,
                "version": event.version,
                "partition_id": partition_id,
                "offset": offset,
                "lag_ms": lag_ms,
            },
        )

    async def _upsert_projection(
        self, conn: Any, event: ModelStateCommittedEvent
    ) -> None:
        """
        Upsert workflow projection with new state.

        Uses INSERT ... ON CONFLICT DO UPDATE for atomic upsert.

        Args:
            conn: Database connection from transaction context
            event: StateCommitted event with new state
        """
        import json

        query = """
            INSERT INTO workflow_projection (
                workflow_key, version, tag, last_action, namespace,
                updated_at, indices, extras
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb)
            ON CONFLICT (workflow_key) DO UPDATE
            SET
                version = EXCLUDED.version,
                tag = EXCLUDED.tag,
                last_action = EXCLUDED.last_action,
                namespace = EXCLUDED.namespace,
                updated_at = EXCLUDED.updated_at,
                indices = EXCLUDED.indices,
                extras = EXCLUDED.extras
            WHERE workflow_projection.version < EXCLUDED.version
        """

        # Serialize JSONB fields to JSON strings
        indices_json = json.dumps(event.indices) if event.indices is not None else None
        extras_json = json.dumps(event.extras) if event.extras is not None else None

        await conn.execute(
            query,
            event.workflow_key,
            event.version,
            event.tag,
            event.last_action,
            event.namespace,
            event.committed_at,
            indices_json,
            extras_json,
        )

    async def _advance_watermark(
        self, conn: Any, partition_id: str, offset: int
    ) -> None:
        """
        Advance watermark for partition.

        Uses GREATEST to prevent watermark regressions. Only updates if
        new offset is greater than current watermark.

        Args:
            conn: Database connection from transaction context
            partition_id: Kafka partition identifier
            offset: Kafka offset to advance to
        """
        # Check current watermark
        current_offset = await conn.fetchval(
            """
            SELECT "offset" FROM projection_watermarks
            WHERE partition_id = $1
            """,
            partition_id,
        )

        # Detect watermark regressions
        if current_offset is not None and offset <= current_offset:
            self._metrics.wm_regressions_total += 1
            projection_wm_regressions_total.inc()
            logger.warning(
                "Watermark regression detected",
                extra={
                    "partition_id": partition_id,
                    "current_offset": current_offset,
                    "new_offset": offset,
                },
            )
            return

        # Upsert watermark with GREATEST to prevent regressions
        query = """
            INSERT INTO projection_watermarks (partition_id, "offset", updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (partition_id) DO UPDATE
            SET
                "offset" = GREATEST(projection_watermarks."offset", EXCLUDED."offset"),
                updated_at = NOW()
        """

        await conn.execute(query, partition_id, offset)

    async def _check_duplicate(self, conn: Any, partition_id: str, offset: int) -> bool:
        """
        Check if event has already been processed (idempotence).

        Args:
            conn: Database connection from transaction context
            partition_id: Kafka partition identifier
            offset: Kafka offset to check

        Returns:
            True if event already processed, False otherwise
        """
        # Check watermark - if offset <= watermark, already processed
        current_offset = await conn.fetchval(
            """
            SELECT "offset" FROM projection_watermarks
            WHERE partition_id = $1
            """,
            partition_id,
        )

        return current_offset is not None and offset <= current_offset

    async def _record_processing(
        self, conn: Any, partition_id: str, offset: int
    ) -> None:
        """
        Record event processing for idempotence tracking.

        This is handled implicitly by watermark updates, so this method
        is a no-op placeholder for future enhancements.

        Args:
            conn: Database connection from transaction context
            partition_id: Kafka partition identifier
            offset: Kafka offset processed
        """
        # Watermark tracking provides sufficient idempotence guarantee
        pass

    async def _log_metrics_if_needed(self) -> None:
        """Log metrics every 60 seconds."""
        now = datetime.now(UTC)
        elapsed = (now - self._last_metrics_log).total_seconds()

        if elapsed >= 60:
            logger.info(
                "Projection Materializer Metrics",
                extra={
                    "projections_materialized": self._metrics.projections_materialized_total,
                    "projections_failed": self._metrics.projections_failed_total,
                    "watermark_updates": self._metrics.watermark_updates_total,
                    "wm_regressions": self._metrics.wm_regressions_total,
                    "lag_ms": round(self._metrics.projection_wm_lag_ms, 2),
                    "max_lag_ms": round(self._metrics.max_lag_ms, 2),
                    "events_per_second": round(
                        self._metrics.events_processed_per_second, 2
                    ),
                    "duplicates_skipped": self._metrics.duplicate_events_skipped,
                },
            )
            self._last_metrics_log = now

    @property
    def metrics(self) -> ProjectionMaterializerMetrics:
        """Get current metrics snapshot."""
        return self._metrics.model_copy()

    @property
    def is_running(self) -> bool:
        """Check if service is currently running."""
        return self._is_running


# Async context manager support
@asynccontextmanager
async def create_projection_materializer(
    bootstrap_servers: str = "localhost:29092",
    consumer_group: str = "projection-materializer",
    postgres_config: Optional[dict[str, Any]] = None,
    enable_idempotence: bool = True,
):
    """
    Create and manage projection materializer service lifecycle.

    Example:
        >>> async with create_projection_materializer() as materializer:
        ...     # Service is running
        ...     await asyncio.sleep(60)
        ...     # Service will be stopped on exit
    """
    service = ProjectionMaterializerService(
        bootstrap_servers=bootstrap_servers,
        consumer_group=consumer_group,
        postgres_config=postgres_config,
        enable_idempotence=enable_idempotence,
    )

    try:
        await service.start()
        yield service
    finally:
        await service.stop()
