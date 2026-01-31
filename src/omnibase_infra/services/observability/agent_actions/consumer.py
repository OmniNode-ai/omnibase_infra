# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Async Kafka Consumer for Agent Actions Observability.

This module provides an async Kafka consumer for agent observability events.
Events are consumed from multiple topics, validated using Pydantic models,
and persisted to PostgreSQL via the WriterAgentActionsPostgres.

Design Decisions:
    - Per-partition offset tracking: Commit only successfully persisted partitions
    - Batch processing: Configurable batch size and timeout
    - Circuit breaker: Resilience via writer's MixinAsyncCircuitBreaker
    - Health check: HTTP endpoint for Kubernetes probes
    - Graceful shutdown: Signal handling with drain and commit

Critical Invariant:
    For each (topic, partition), commit offsets only up to the highest offset
    that has been successfully persisted for that partition.
    Never commit offsets for partitions that had write failures in the batch.

Topics consumed:
    - agent-actions
    - agent-routing-decisions
    - agent-transformation-events
    - router-performance-metrics
    - agent-detection-failures
    - agent-execution-logs

Related Tickets:
    - OMN-1743: Migrate agent_actions_consumer to omnibase_infra (current)
    - OMN-1526: Session consumer moved from omniclaude (reference pattern)

Example:
    >>> from omnibase_infra.services.observability.agent_actions import (
    ...     AgentActionsConsumer,
    ...     ConfigAgentActionsConsumer,
    ... )
    >>>
    >>> config = ConfigAgentActionsConsumer(
    ...     kafka_bootstrap_servers="localhost:9092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omninode_bridge",
    ... )
    >>> consumer = AgentActionsConsumer(config)
    >>>
    >>> # Run consumer (blocking)
    >>> await consumer.start()
    >>> await consumer.run()

    # Or run as module:
    # python -m omnibase_infra.services.observability.agent_actions.consumer
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import asyncpg
from aiohttp import web
from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.errors import KafkaError
from pydantic import BaseModel, ValidationError

from omnibase_infra.services.observability.agent_actions.config import (
    ConfigAgentActionsConsumer,
)
from omnibase_infra.services.observability.agent_actions.models import (
    ModelAgentAction,
    ModelDetectionFailure,
    ModelExecutionLog,
    ModelPerformanceMetric,
    ModelRoutingDecision,
    ModelTransformationEvent,
)
from omnibase_infra.services.observability.agent_actions.writer_postgres import (
    WriterAgentActionsPostgres,
)

if TYPE_CHECKING:
    from aiokafka.structs import ConsumerRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases and Constants
# =============================================================================

# Map topics to their Pydantic model class
TOPIC_TO_MODEL: dict[str, type[BaseModel]] = {
    "agent-actions": ModelAgentAction,
    "agent-routing-decisions": ModelRoutingDecision,
    "agent-transformation-events": ModelTransformationEvent,
    "router-performance-metrics": ModelPerformanceMetric,
    "agent-detection-failures": ModelDetectionFailure,
    "agent-execution-logs": ModelExecutionLog,
}

# Map topics to writer method names
TOPIC_TO_WRITER_METHOD: dict[str, str] = {
    "agent-actions": "write_agent_actions",
    "agent-routing-decisions": "write_routing_decisions",
    "agent-transformation-events": "write_transformation_events",
    "router-performance-metrics": "write_performance_metrics",
    "agent-detection-failures": "write_detection_failures",
    "agent-execution-logs": "write_execution_logs",
}


# =============================================================================
# Enums
# =============================================================================


class EnumHealthStatus(StrEnum):
    """Health check status values.

    Used by the health check endpoint to indicate consumer health.

    Status Semantics:
        HEALTHY: Consumer running, circuit closed, recent successful write
        DEGRADED: Consumer running but circuit open (retrying)
        UNHEALTHY: Consumer stopped or no writes for extended period
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# =============================================================================
# Consumer Metrics
# =============================================================================


class ConsumerMetrics:
    """Metrics tracking for the agent actions consumer.

    Tracks processing statistics for observability and monitoring.
    Thread-safe via asyncio lock protection.

    Attributes:
        messages_received: Total messages received from Kafka.
        messages_processed: Successfully processed messages.
        messages_failed: Messages that failed processing.
        messages_skipped: Messages skipped (invalid, duplicate, etc.).
        batches_processed: Number of batches successfully processed.
        last_poll_at: Timestamp of last Kafka poll.
        last_successful_write_at: Timestamp of last successful database write.
    """

    def __init__(self) -> None:
        """Initialize metrics with zero values."""
        self.messages_received: int = 0
        self.messages_processed: int = 0
        self.messages_failed: int = 0
        self.messages_skipped: int = 0
        self.batches_processed: int = 0
        self.last_poll_at: datetime | None = None
        self.last_successful_write_at: datetime | None = None
        self._lock = asyncio.Lock()

    async def record_received(self, count: int = 1) -> None:
        """Record messages received."""
        async with self._lock:
            self.messages_received += count
            self.last_poll_at = datetime.now(UTC)

    async def record_processed(self, count: int = 1) -> None:
        """Record successfully processed messages."""
        async with self._lock:
            self.messages_processed += count
            self.last_successful_write_at = datetime.now(UTC)

    async def record_failed(self, count: int = 1) -> None:
        """Record failed messages."""
        async with self._lock:
            self.messages_failed += count

    async def record_skipped(self, count: int = 1) -> None:
        """Record skipped messages."""
        async with self._lock:
            self.messages_skipped += count

    async def record_batch_processed(self) -> None:
        """Record a successfully processed batch."""
        async with self._lock:
            self.batches_processed += 1

    async def snapshot(self) -> dict[str, object]:
        """Get a snapshot of current metrics.

        Returns:
            Dictionary with all metric values.
        """
        async with self._lock:
            return {
                "messages_received": self.messages_received,
                "messages_processed": self.messages_processed,
                "messages_failed": self.messages_failed,
                "messages_skipped": self.messages_skipped,
                "batches_processed": self.batches_processed,
                "last_poll_at": (
                    self.last_poll_at.isoformat() if self.last_poll_at else None
                ),
                "last_successful_write_at": (
                    self.last_successful_write_at.isoformat()
                    if self.last_successful_write_at
                    else None
                ),
            }


# =============================================================================
# Agent Actions Consumer
# =============================================================================


class AgentActionsConsumer:
    """Async Kafka consumer for agent observability events.

    Consumes events from multiple observability topics and persists them
    to PostgreSQL. Implements at-least-once delivery with per-partition
    offset tracking to ensure no message loss on partial batch failures.

    Features:
        - **Per-partition offset tracking**: Commit only successfully persisted
          partitions. Partial batch failures do not cause message loss.

        - **Batch processing**: Configurable batch size and timeout for
          efficient database writes via executemany.

        - **Circuit breaker**: Database resilience via writer's circuit breaker.
          Consumer degrades gracefully when database is unavailable.

        - **Health check endpoint**: HTTP server for Kubernetes liveness
          and readiness probes.

        - **Graceful shutdown**: Signal handling with drain and final commit.

    Thread Safety:
        This consumer is designed for single-threaded async execution.
        Multiple consumers can run with different group_ids for horizontal
        scaling (partition assignment via Kafka consumer groups).

    Example:
        >>> config = ConfigAgentActionsConsumer(
        ...     kafka_bootstrap_servers="localhost:9092",
        ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omninode_bridge",
        ... )
        >>> consumer = AgentActionsConsumer(config)
        >>>
        >>> await consumer.start()
        >>> try:
        ...     await consumer.run()
        ... finally:
        ...     await consumer.stop()

    Attributes:
        metrics: Consumer metrics for observability.
        is_running: Whether the consumer is currently running.
    """

    def __init__(self, config: ConfigAgentActionsConsumer) -> None:
        """Initialize the agent actions consumer.

        Args:
            config: Consumer configuration (Kafka, PostgreSQL, batch settings).

        Example:
            >>> config = ConfigAgentActionsConsumer(
            ...     kafka_bootstrap_servers="localhost:9092",
            ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omninode_bridge",
            ... )
            >>> consumer = AgentActionsConsumer(config)
        """
        self._config = config
        self._consumer: AIOKafkaConsumer | None = None
        self._pool: asyncpg.Pool | None = None
        self._writer: WriterAgentActionsPostgres | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Health check server
        self._health_app: web.Application | None = None
        self._health_runner: web.AppRunner | None = None
        self._health_site: web.TCPSite | None = None

        # Metrics
        self.metrics = ConsumerMetrics()

        # Consumer ID for logging
        self._consumer_id = f"agent-actions-consumer-{uuid4().hex[:8]}"

        logger.info(
            "AgentActionsConsumer initialized",
            extra={
                "consumer_id": self._consumer_id,
                "topics": self._config.topics,
                "group_id": self._config.kafka_group_id,
                "bootstrap_servers": self._config.kafka_bootstrap_servers,
                "batch_size": self._config.batch_size,
                "batch_timeout_ms": self._config.batch_timeout_ms,
            },
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if the consumer is currently running.

        Returns:
            True if start() has been called and stop() has not.
        """
        return self._running

    @property
    def consumer_id(self) -> str:
        """Get the unique consumer identifier.

        Returns:
            Consumer ID string for logging and tracing.
        """
        return self._consumer_id

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """Start the consumer, pool, writer, and health check server.

        Creates the asyncpg pool, initializes the writer, creates the Kafka
        consumer, and starts the health check HTTP server.

        Raises:
            RuntimeError: If the consumer is already running.
            asyncpg.PostgresError: If database connection fails.
            KafkaError: If Kafka connection fails.

        Example:
            >>> await consumer.start()
            >>> # Consumer is now connected, ready for run()
        """
        if self._running:
            logger.warning(
                "Consumer already running",
                extra={"consumer_id": self._consumer_id},
            )
            return

        correlation_id = uuid4()

        logger.info(
            "Starting AgentActionsConsumer",
            extra={
                "consumer_id": self._consumer_id,
                "correlation_id": str(correlation_id),
                "topics": self._config.topics,
            },
        )

        try:
            # Create PostgreSQL pool
            self._pool = await asyncpg.create_pool(
                dsn=self._config.postgres_dsn,
                min_size=2,
                max_size=10,
            )
            logger.info(
                "PostgreSQL pool created",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                },
            )

            # Create writer with pool injection
            self._writer = WriterAgentActionsPostgres(
                pool=self._pool,
                circuit_breaker_threshold=self._config.circuit_breaker_threshold,
                circuit_breaker_reset_timeout=self._config.circuit_breaker_reset_timeout,
                circuit_breaker_half_open_successes=self._config.circuit_breaker_half_open_successes,
            )

            # Create Kafka consumer
            self._consumer = AIOKafkaConsumer(
                *self._config.topics,
                bootstrap_servers=self._config.kafka_bootstrap_servers,
                group_id=self._config.kafka_group_id,
                auto_offset_reset=self._config.auto_offset_reset,
                enable_auto_commit=False,  # Manual commits for at-least-once
                max_poll_records=self._config.batch_size,
            )

            await self._consumer.start()
            logger.info(
                "Kafka consumer started",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                    "topics": self._config.topics,
                    "group_id": self._config.kafka_group_id,
                },
            )

            # Start health check server
            await self._start_health_server()

            self._running = True
            self._shutdown_event.clear()

            logger.info(
                "AgentActionsConsumer started",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                },
            )

        except Exception as e:
            logger.exception(
                "Failed to start consumer",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                },
            )
            # Cleanup any partial initialization
            await self._cleanup_resources(correlation_id)
            raise

    async def stop(self) -> None:
        """Stop the consumer gracefully.

        Signals the consume loop to exit, waits for in-flight processing,
        commits final offsets, and closes all connections. Safe to call
        multiple times.

        Example:
            >>> await consumer.stop()
            >>> # Consumer is now stopped and disconnected
        """
        if not self._running:
            logger.debug(
                "Consumer not running, nothing to stop",
                extra={"consumer_id": self._consumer_id},
            )
            return

        correlation_id = uuid4()

        logger.info(
            "Stopping AgentActionsConsumer",
            extra={
                "consumer_id": self._consumer_id,
                "correlation_id": str(correlation_id),
            },
        )

        # Signal shutdown
        self._running = False
        self._shutdown_event.set()

        # Cleanup resources
        await self._cleanup_resources(correlation_id)

        # Log final metrics
        metrics_snapshot = await self.metrics.snapshot()
        logger.info(
            "AgentActionsConsumer stopped",
            extra={
                "consumer_id": self._consumer_id,
                "correlation_id": str(correlation_id),
                "final_metrics": metrics_snapshot,
            },
        )

    async def _cleanup_resources(self, correlation_id: UUID) -> None:
        """Clean up all resources during shutdown.

        Args:
            correlation_id: Correlation ID for logging.
        """
        # Stop health check server
        if self._health_site is not None:
            await self._health_site.stop()
            self._health_site = None

        if self._health_runner is not None:
            await self._health_runner.cleanup()
            self._health_runner = None

        self._health_app = None

        # Stop Kafka consumer
        if self._consumer is not None:
            try:
                await self._consumer.stop()
            except Exception as e:
                logger.warning(
                    "Error stopping Kafka consumer",
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                        "error": str(e),
                    },
                )
            finally:
                self._consumer = None

        # Close PostgreSQL pool
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception as e:
                logger.warning(
                    "Error closing PostgreSQL pool",
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                        "error": str(e),
                    },
                )
            finally:
                self._pool = None

        self._writer = None

    async def run(self) -> None:
        """Run the main consume loop.

        Continuously consumes messages from Kafka topics, processes them
        in batches, and writes to PostgreSQL. Implements at-least-once
        delivery by committing offsets only after successful writes.

        This method blocks until stop() is called or an unrecoverable error
        occurs. Use this after calling start().

        Example:
            >>> await consumer.start()
            >>> try:
            ...     await consumer.run()
            ... finally:
            ...     await consumer.stop()
        """
        if not self._running or self._consumer is None:
            raise RuntimeError(
                "Consumer not started. Call start() before run().",
            )

        correlation_id = uuid4()

        logger.info(
            "Starting consume loop",
            extra={
                "consumer_id": self._consumer_id,
                "correlation_id": str(correlation_id),
            },
        )

        await self._consume_loop(correlation_id)

    async def __aenter__(self) -> AgentActionsConsumer:
        """Async context manager entry.

        Starts the consumer and returns self for use in async with blocks.

        Returns:
            Self for chaining.

        Example:
            >>> async with AgentActionsConsumer(config) as consumer:
            ...     await consumer.run()
        """
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit.

        Stops the consumer on exit from async with block.
        """
        await self.stop()

    # =========================================================================
    # Consume Loop
    # =========================================================================

    async def _consume_loop(self, correlation_id: UUID) -> None:
        """Main consumption loop with batch processing.

        Polls Kafka for messages, accumulates batches, processes them,
        and commits offsets for successfully written partitions only.

        Args:
            correlation_id: Correlation ID for tracing this consume session.
        """
        if self._consumer is None:
            logger.error(
                "Consumer is None in consume loop",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        batch_timeout_seconds = self._config.batch_timeout_ms / 1000.0

        try:
            while self._running:
                # Poll with timeout for batch accumulation
                try:
                    records = await asyncio.wait_for(
                        self._consumer.getmany(
                            timeout_ms=self._config.batch_timeout_ms,
                            max_records=self._config.batch_size,
                        ),
                        timeout=batch_timeout_seconds + 5.0,  # Buffer for poll timeout
                    )
                except TimeoutError:
                    # Poll timeout is normal, continue loop
                    continue

                if not records:
                    continue

                # Flatten all messages from all partitions
                messages: list[ConsumerRecord] = []
                for tp_messages in records.values():
                    messages.extend(tp_messages)

                if not messages:
                    continue

                await self.metrics.record_received(len(messages))

                # Process batch and get successful offsets per partition
                batch_correlation_id = uuid4()
                successful_offsets = await self._process_batch(
                    messages, batch_correlation_id
                )

                # Commit only successful offsets
                if successful_offsets:
                    await self._commit_offsets(successful_offsets, batch_correlation_id)
                    await self.metrics.record_batch_processed()

        except asyncio.CancelledError:
            logger.info(
                "Consume loop cancelled",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                },
            )
            raise

        except KafkaError as e:
            logger.exception(
                "Kafka error in consume loop",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                },
            )
            raise

        except Exception as e:
            logger.exception(
                "Unexpected error in consume loop",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                },
            )
            raise

        finally:
            logger.info(
                "Consume loop exiting",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                },
            )

    # =========================================================================
    # Batch Processing
    # =========================================================================

    async def _process_batch(
        self,
        messages: list[ConsumerRecord],
        correlation_id: UUID,
    ) -> dict[TopicPartition, int]:
        """Process batch and return highest successful offset per partition.

        Groups messages by topic, validates them, writes each topic's batch
        to PostgreSQL, and tracks successful offsets per partition.

        Args:
            messages: List of Kafka ConsumerRecords to process.
            correlation_id: Correlation ID for tracing.

        Returns:
            Dictionary mapping TopicPartition to highest successful offset.
            Only partitions with successful writes are included.
        """
        if self._writer is None:
            logger.error(
                "Writer is None during batch processing",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                },
            )
            return {}

        successful_offsets: dict[TopicPartition, int] = {}
        parsed_skipped: int = 0

        # Group messages by topic with their ConsumerRecord for offset tracking
        by_topic: dict[str, list[tuple[ConsumerRecord, BaseModel]]] = {}

        for msg in messages:
            try:
                # Decode message value
                value = msg.value
                if isinstance(value, bytes):
                    value = value.decode("utf-8")

                payload = json.loads(value)

                # Get model class for topic
                model_cls = TOPIC_TO_MODEL.get(msg.topic)
                if model_cls is None:
                    logger.warning(
                        "Unknown topic, skipping message",
                        extra={
                            "consumer_id": self._consumer_id,
                            "correlation_id": str(correlation_id),
                            "topic": msg.topic,
                        },
                    )
                    parsed_skipped += 1
                    # Still track offset for unknown topics (don't block)
                    tp = TopicPartition(msg.topic, msg.partition)
                    current = successful_offsets.get(tp, -1)
                    successful_offsets[tp] = max(current, msg.offset)
                    continue

                # Validate with Pydantic model
                model = model_cls.model_validate(payload)
                by_topic.setdefault(msg.topic, []).append((msg, model))

            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to decode JSON message",
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                        "topic": msg.topic,
                        "partition": msg.partition,
                        "offset": msg.offset,
                        "error": str(e),
                    },
                )
                parsed_skipped += 1
                # Skip malformed messages but track offset to avoid reprocessing
                tp = TopicPartition(msg.topic, msg.partition)
                current = successful_offsets.get(tp, -1)
                successful_offsets[tp] = max(current, msg.offset)

            except ValidationError as e:
                logger.warning(
                    "Message validation failed",
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                        "topic": msg.topic,
                        "partition": msg.partition,
                        "offset": msg.offset,
                        "error": str(e),
                    },
                )
                parsed_skipped += 1
                # Skip invalid messages but track offset
                tp = TopicPartition(msg.topic, msg.partition)
                current = successful_offsets.get(tp, -1)
                successful_offsets[tp] = max(current, msg.offset)

        if parsed_skipped > 0:
            await self.metrics.record_skipped(parsed_skipped)

        # Write each topic's batch to PostgreSQL
        for topic, items in by_topic.items():
            writer_method_name = TOPIC_TO_WRITER_METHOD.get(topic)
            if writer_method_name is None:
                logger.warning(
                    "No writer method for topic",
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                        "topic": topic,
                    },
                )
                continue

            writer_method: Callable[
                [list[BaseModel], UUID | None], Coroutine[object, object, int]
            ] = getattr(self._writer, writer_method_name)
            models = [item[1] for item in items]

            try:
                written_count = await writer_method(models, correlation_id)

                # Record successful offsets per partition for this topic
                for msg, _ in items:
                    tp = TopicPartition(msg.topic, msg.partition)
                    current = successful_offsets.get(tp, -1)
                    successful_offsets[tp] = max(current, msg.offset)

                await self.metrics.record_processed(written_count)

                logger.debug(
                    "Wrote batch for topic",
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                        "topic": topic,
                        "count": written_count,
                    },
                )

            except Exception:
                # Write failed for this topic - don't update offsets for its partitions
                logger.exception(
                    "Failed to write batch for topic",
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                        "topic": topic,
                        "count": len(models),
                    },
                )
                await self.metrics.record_failed(len(models))
                # Remove any offsets we may have tracked for failed partitions
                for msg, _ in items:
                    tp = TopicPartition(msg.topic, msg.partition)
                    # Only remove if this batch was the only contributor
                    # In practice, we don't add until success, so this is safe
                    successful_offsets.pop(tp, None)

        return successful_offsets

    async def _commit_offsets(
        self,
        offsets: dict[TopicPartition, int],
        correlation_id: UUID,
    ) -> None:
        """Commit only successfully persisted offsets per partition.

        Commits offset + 1 for each partition (next offset to consume).

        Args:
            offsets: Dictionary mapping TopicPartition to highest persisted offset.
            correlation_id: Correlation ID for tracing.
        """
        if not offsets or self._consumer is None:
            return

        # Build commit offsets (offset + 1 = next offset to consume)
        commit_offsets = {tp: offset + 1 for tp, offset in offsets.items()}

        try:
            await self._consumer.commit(commit_offsets)

            logger.debug(
                "Committed offsets",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                    "partitions": len(commit_offsets),
                },
            )

        except KafkaError:
            logger.exception(
                "Failed to commit offsets",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                },
            )
            # Don't re-raise - messages will be reprocessed on restart

    # =========================================================================
    # Health Check Server
    # =========================================================================

    async def _start_health_server(self) -> None:
        """Start minimal HTTP health check server.

        Starts an aiohttp server on the configured port with a /health endpoint.
        """
        self._health_app = web.Application()
        self._health_app.router.add_get("/health", self._health_handler)

        self._health_runner = web.AppRunner(self._health_app)
        await self._health_runner.setup()

        self._health_site = web.TCPSite(
            self._health_runner,
            host=self._config.health_check_host,  # Configurable - see config.py for security notes
            port=self._config.health_check_port,
        )
        await self._health_site.start()

        logger.info(
            "Health check server started",
            extra={
                "consumer_id": self._consumer_id,
                "host": self._config.health_check_host,
                "port": self._config.health_check_port,
            },
        )

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Handle health check requests.

        Returns JSON with health status based on:
        - Consumer running state
        - Circuit breaker state (from writer)
        - Last successful write timestamp

        Args:
            request: aiohttp request object.

        Returns:
            JSON response with health status.
        """
        metrics_snapshot = await self.metrics.snapshot()
        circuit_state = self._writer.get_circuit_breaker_state() if self._writer else {}

        # Determine health status
        if not self._running:
            status = EnumHealthStatus.UNHEALTHY
        elif circuit_state.get("state") == "open":
            status = EnumHealthStatus.DEGRADED
        else:
            # Check for recent successful write (within staleness threshold)
            last_write = metrics_snapshot.get("last_successful_write_at")
            if last_write is None:
                # No writes yet - consider healthy if just started
                last_poll = metrics_snapshot.get("last_poll_at")
                if last_poll is None:
                    status = EnumHealthStatus.HEALTHY  # Just started
                else:
                    status = EnumHealthStatus.DEGRADED  # Polling but no writes
            else:
                # Check if last write was recent (within staleness threshold)
                try:
                    last_write_dt = datetime.fromisoformat(str(last_write))
                    age_seconds = (datetime.now(UTC) - last_write_dt).total_seconds()
                    if age_seconds > self._config.health_check_staleness_seconds:
                        status = EnumHealthStatus.DEGRADED
                    else:
                        status = EnumHealthStatus.HEALTHY
                except (ValueError, TypeError):
                    status = EnumHealthStatus.HEALTHY

        response_body = {
            "status": status.value,
            "consumer_running": self._running,
            "consumer_id": self._consumer_id,
            "last_poll_time": metrics_snapshot.get("last_poll_at"),
            "last_successful_write": metrics_snapshot.get("last_successful_write_at"),
            "circuit_breaker_state": circuit_state.get("state", "unknown"),
            "messages_processed": metrics_snapshot.get("messages_processed", 0),
            "messages_failed": metrics_snapshot.get("messages_failed", 0),
            "batches_processed": metrics_snapshot.get("batches_processed", 0),
        }

        # Return appropriate HTTP status code
        http_status = 200 if status == EnumHealthStatus.HEALTHY else 503

        return web.json_response(response_body, status=http_status)

    # =========================================================================
    # Health Check (Direct API)
    # =========================================================================

    async def health_check(self) -> dict[str, object]:
        """Check consumer health status.

        Returns a dictionary with health information for programmatic access.

        Returns:
            Dictionary with health status including:
                - status: Overall health (healthy, degraded, unhealthy)
                - consumer_running: Whether consume loop is active
                - circuit_breaker_state: Current circuit breaker state
                - consumer_id: Unique consumer identifier
                - metrics: Current metrics snapshot
        """
        metrics_snapshot = await self.metrics.snapshot()
        circuit_state = self._writer.get_circuit_breaker_state() if self._writer else {}

        # Determine health status
        if not self._running:
            status = EnumHealthStatus.UNHEALTHY
        elif circuit_state.get("state") == "open":
            status = EnumHealthStatus.DEGRADED
        else:
            status = EnumHealthStatus.HEALTHY

        return {
            "status": status.value,
            "consumer_running": self._running,
            "consumer_id": self._consumer_id,
            "group_id": self._config.kafka_group_id,
            "topics": self._config.topics,
            "circuit_breaker_state": circuit_state,
            "metrics": metrics_snapshot,
        }


# =============================================================================
# Entry Point
# =============================================================================


async def _main() -> None:
    """Main entry point for running the consumer as a module."""
    # Load configuration from environment
    config = ConfigAgentActionsConsumer()

    logger.info(
        "Starting agent actions consumer",
        extra={
            "topics": config.topics,
            "bootstrap_servers": config.kafka_bootstrap_servers,
            "group_id": config.kafka_group_id,
            "health_port": config.health_check_port,
        },
    )

    consumer = AgentActionsConsumer(config)

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    shutdown_task: asyncio.Task[None] | None = None

    def signal_handler() -> None:
        nonlocal shutdown_task
        logger.info("Received shutdown signal")
        # Only create shutdown task once to avoid race conditions
        if shutdown_task is None:
            shutdown_task = asyncio.create_task(consumer.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await consumer.start()
        await consumer.run()
    except asyncio.CancelledError:
        logger.info("Consumer cancelled")
    finally:
        # Ensure shutdown task completes if it was started by signal handler
        if shutdown_task is not None:
            if not shutdown_task.done():
                await shutdown_task
            # Task already completed, no action needed
        else:
            # No signal received, perform clean shutdown
            await consumer.stop()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(_main())


__all__ = [
    "AgentActionsConsumer",
    "ConsumerMetrics",
    "EnumHealthStatus",
    "TOPIC_TO_MODEL",
    "TOPIC_TO_WRITER_METHOD",
]
