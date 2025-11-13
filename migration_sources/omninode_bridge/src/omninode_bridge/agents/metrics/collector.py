"""
High-performance metrics collector with <10ms overhead.

Provides ring buffer-based collection with async batch flushing.
"""

import asyncio
import logging
from typing import Optional

from omninode_bridge.agents.metrics.alerting.rules import AlertRuleEngine
from omninode_bridge.agents.metrics.models import Metric, MetricType
from omninode_bridge.agents.metrics.ring_buffer import RingBuffer
from omninode_bridge.agents.metrics.storage.kafka import KafkaMetricsWriter
from omninode_bridge.agents.metrics.storage.postgres import PostgreSQLMetricsWriter

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    High-performance metrics collector with <10ms overhead guarantee.

    Design:
    - Ring buffer for O(1) writes
    - Batch flushing (100 metrics or 1s)
    - Async I/O for non-blocking storage
    - Pre-allocated memory to avoid GC pauses

    Performance:
    - Emit metric: <1ms (direct buffer write, no I/O)
    - Batch flush: <50ms (amortized)
    - Total overhead: <10ms per 100 metrics (<0.1ms per metric)

    Usage:
        collector = MetricsCollector()
        await collector.record_timing("operation_time_ms", 45.2)
        await collector.record_counter("operation_count", 1)
        await collector.flush()
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        batch_size: int = 100,
        flush_interval_ms: int = 1000,
        kafka_enabled: bool = True,
        postgres_enabled: bool = True,
    ):
        """
        Initialize metrics collector.

        Args:
            buffer_size: Ring buffer size (default 10000)
            batch_size: Metrics per batch (default 100)
            flush_interval_ms: Flush interval in ms (default 1000ms)
            kafka_enabled: Enable Kafka publishing
            postgres_enabled: Enable PostgreSQL persistence
        """
        self._buffer = RingBuffer[Metric](capacity=buffer_size)
        self._batch_size = batch_size
        self._flush_interval_ms = flush_interval_ms
        self._kafka_enabled = kafka_enabled
        self._postgres_enabled = postgres_enabled

        # Storage writers
        self._kafka_writer: Optional[KafkaMetricsWriter] = None
        self._postgres_writer: Optional[PostgreSQLMetricsWriter] = None

        # Alert engine
        self._alert_engine: Optional[AlertRuleEngine] = None

        # Background flush task
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics counter for batch triggering
        self._metrics_since_flush = 0
        self._flush_lock = asyncio.Lock()
        self._flush_in_progress = False

        logger.info(
            f"MetricsCollector initialized: buffer_size={buffer_size}, "
            f"batch_size={batch_size}, flush_interval_ms={flush_interval_ms}"
        )

    async def start(
        self,
        kafka_bootstrap_servers: Optional[str] = None,
        postgres_url: Optional[str] = None,
        alert_engine: Optional[AlertRuleEngine] = None,
    ) -> None:
        """
        Start metrics collector and initialize storage.

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            postgres_url: PostgreSQL connection URL
            alert_engine: Optional alert rule engine
        """
        # Initialize Kafka writer
        if self._kafka_enabled and kafka_bootstrap_servers:
            self._kafka_writer = KafkaMetricsWriter(kafka_bootstrap_servers)
            await self._kafka_writer.start()
            logger.info("Kafka metrics writer started")

        # Initialize PostgreSQL writer
        if self._postgres_enabled and postgres_url:
            self._postgres_writer = PostgreSQLMetricsWriter(postgres_url)
            await self._postgres_writer.start()
            logger.info("PostgreSQL metrics writer started")

        # Set alert engine
        self._alert_engine = alert_engine

        # Start background flush task
        self._running = True
        self._flush_task = asyncio.create_task(self._background_flush())
        logger.info("Background flush task started")

    async def stop(self) -> None:
        """Stop metrics collector and flush remaining metrics."""
        self._running = False

        # Cancel background flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self.flush()

        # Stop writers
        if self._kafka_writer:
            await self._kafka_writer.stop()
        if self._postgres_writer:
            await self._postgres_writer.stop()

        logger.info("MetricsCollector stopped")

    async def record_timing(
        self,
        metric_name: str,
        duration_ms: float,
        tags: Optional[dict[str, str]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Record timing metric (e.g., operation duration).

        Performance: <1ms (direct buffer write, no I/O)

        Args:
            metric_name: Metric name (e.g., "routing_decision_time_ms")
            duration_ms: Duration in milliseconds
            tags: Optional tags for filtering/grouping
            correlation_id: Optional correlation ID for tracing
        """
        await self._record(
            metric_name=metric_name,
            metric_type=MetricType.TIMING,
            value=duration_ms,
            unit="ms",
            tags=tags or {},
            correlation_id=correlation_id,
        )

    async def record_counter(
        self,
        metric_name: str,
        count: int = 1,
        tags: Optional[dict[str, str]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Record counter metric (e.g., operation count).

        Performance: <1ms

        Args:
            metric_name: Metric name (e.g., "state_operation_count")
            count: Count to increment (default 1)
            tags: Optional tags
            correlation_id: Optional correlation ID
        """
        await self._record(
            metric_name=metric_name,
            metric_type=MetricType.COUNTER,
            value=float(count),
            unit="count",
            tags=tags or {},
            correlation_id=correlation_id,
        )

    async def record_gauge(
        self,
        metric_name: str,
        value: float,
        unit: str,
        tags: Optional[dict[str, str]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Record gauge metric (e.g., current value snapshot).

        Performance: <1ms

        Args:
            metric_name: Metric name (e.g., "coordination_agent_count")
            value: Current value
            unit: Unit of measurement (e.g., "count", "KB")
            tags: Optional tags
            correlation_id: Optional correlation ID
        """
        await self._record(
            metric_name=metric_name,
            metric_type=MetricType.GAUGE,
            value=value,
            unit=unit,
            tags=tags or {},
            correlation_id=correlation_id,
        )

    async def record_rate(
        self,
        metric_name: str,
        rate_percent: float,
        tags: Optional[dict[str, str]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Record rate metric (e.g., success rate, cache hit rate).

        Performance: <1ms

        Args:
            metric_name: Metric name (e.g., "routing_cache_hit_rate")
            rate_percent: Rate as percentage (0-100)
            tags: Optional tags
            correlation_id: Optional correlation ID
        """
        await self._record(
            metric_name=metric_name,
            metric_type=MetricType.RATE,
            value=rate_percent,
            unit="%",
            tags=tags or {},
            correlation_id=correlation_id,
        )

    async def _record(
        self,
        metric_name: str,
        metric_type: MetricType,
        value: float,
        unit: str,
        tags: dict[str, str],
        correlation_id: Optional[str],
    ) -> None:
        """
        Internal record method with buffer write.

        Performance: <1ms (O(1) ring buffer write)
        """
        metric = Metric(
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            tags=tags,
            correlation_id=correlation_id,
        )

        # Write to ring buffer (fast, no I/O)
        await self._buffer.write(metric)

        # Increment counter and check flush threshold (with synchronization)
        async with self._flush_lock:
            self._metrics_since_flush += 1
            should_flush = (
                self._metrics_since_flush >= self._batch_size
                and not self._flush_in_progress
            )

        # Trigger flush if batch size reached (outside lock to avoid deadlock)
        if should_flush:
            asyncio.create_task(self.flush())

        # Evaluate alert rules (async, non-blocking)
        if self._alert_engine:
            asyncio.create_task(self._alert_engine.evaluate_metric(metric))

    async def flush(self) -> None:
        """
        Flush buffered metrics to storage.

        Performance: <50ms (batch write, async I/O)
        """
        # Acquire flush lock to prevent concurrent flushes
        async with self._flush_lock:
            # Skip if flush already in progress
            if self._flush_in_progress:
                return

            # Set flush in progress flag
            self._flush_in_progress = True

        batch_size = 0
        try:
            # Read batch from ring buffer
            batch = await self._buffer.read_batch(max_size=self._batch_size)

            if not batch:
                return

            batch_size = len(batch)
            logger.debug(f"Flushing {batch_size} metrics")

            # Parallel async writes (don't wait, fire and forget with error handling)
            tasks = []

            if self._kafka_writer:
                tasks.append(self._kafka_writer.write_batch(batch))

            if self._postgres_writer:
                tasks.append(self._postgres_writer.write_batch(batch))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Storage write failed: {result}")
        finally:
            # Reset counter and clear flush flag
            async with self._flush_lock:
                self._metrics_since_flush = max(
                    0, self._metrics_since_flush - batch_size
                )
                self._flush_in_progress = False

    async def _background_flush(self) -> None:
        """Background task for periodic flushing."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval_ms / 1000.0)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background flush error: {e}")

    async def get_stats(self) -> dict[str, int]:
        """
        Get collector statistics.

        Returns:
            Dictionary with buffer size, capacity, metrics pending
        """
        return {
            "buffer_size": await self._buffer.size(),
            "buffer_capacity": await self._buffer.capacity(),
            "metrics_since_flush": self._metrics_since_flush,
        }
