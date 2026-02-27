# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Async Kafka Consumer for Skill Lifecycle Observability (OMN-2934).

Consumes skill-started and skill-completed events from Kafka and persists
them to PostgreSQL via WriterSkillLifecyclePostgres.

Design Decisions:
    - Per-partition offset tracking: Commit only successfully persisted partitions.
    - Batch processing: Configurable batch size and timeout.
    - Circuit breaker: Resilience via writer's MixinAsyncCircuitBreaker.
    - Health check: HTTP endpoint for Kubernetes probes.
    - Graceful shutdown: Signal handling with drain and commit.

Critical Invariant:
    For each (topic, partition), commit offsets only up to the highest offset
    that has been successfully persisted for that partition.
    Never commit offsets for partitions that had write failures in the batch.

Topics consumed (OMN-2934):
    - onex.evt.omniclaude.skill-started.v1
    - onex.evt.omniclaude.skill-completed.v1

Related Tickets:
    - OMN-2934: Add provisioning and consumer for skill lifecycle events
    - OMN-2773: Original emission introduced in skill node runtime

Example:
    >>> from omnibase_infra.services.observability.skill_lifecycle import (
    ...     SkillLifecycleConsumer,
    ...     ConfigSkillLifecycleConsumer,
    ... )
    >>>
    >>> config = ConfigSkillLifecycleConsumer(
    ...     kafka_bootstrap_servers="localhost:9092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
    ... )
    >>> consumer = SkillLifecycleConsumer(config)
    >>> await consumer.start()
    >>> await consumer.run()

    # Or run as module:
    # python -m omnibase_infra.services.observability.skill_lifecycle.consumer
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import asyncpg
from aiohttp import web
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import KafkaError

from omnibase_infra.services.observability.skill_lifecycle.config import (
    ConfigSkillLifecycleConsumer,
)
from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
    WriterSkillLifecyclePostgres,
)

if TYPE_CHECKING:
    from aiokafka.structs import ConsumerRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


def mask_dsn_password(dsn: str) -> str:
    """Mask password in a PostgreSQL DSN for safe logging.

    Args:
        dsn: PostgreSQL connection string.

    Returns:
        DSN with password replaced by '***'.
    """
    try:
        parsed = urlparse(dsn)
        if not parsed.password:
            return dsn
        if parsed.port:
            masked_netloc = f"{parsed.username}:***@{parsed.hostname}:{parsed.port}"
        else:
            masked_netloc = f"{parsed.username}:***@{parsed.hostname}"
        return urlunparse(
            (
                parsed.scheme,
                masked_netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
    except Exception:
        logger.debug("Failed to parse DSN for masking, returning as-is")
        return dsn


# =============================================================================
# Enums
# =============================================================================


class EnumHealthStatus(StrEnum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# =============================================================================
# Consumer Metrics
# =============================================================================


class ConsumerMetrics:
    """Metrics tracking for the skill lifecycle consumer."""

    MAX_LATENCY_SAMPLES: int = 100

    def __init__(self) -> None:
        self.messages_received: int = 0
        self.messages_processed: int = 0
        self.messages_failed: int = 0
        self.messages_skipped: int = 0
        self.messages_sent_to_dlq: int = 0
        self.batches_processed: int = 0
        self.last_poll_at: datetime | None = None
        self.last_successful_write_at: datetime | None = None
        self.started_at: datetime = datetime.now(UTC)
        self._lock = asyncio.Lock()
        self.per_topic_received: dict[str, int] = {}
        self.per_topic_processed: dict[str, int] = {}
        self.per_topic_failed: dict[str, int] = {}
        self.batch_latency_ms: list[float] = []
        self._export_hooks: list[Callable[[str, float, dict[str, str]], None]] = []

    def _export(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        resolved_labels = labels or {}
        for hook in self._export_hooks:
            try:
                hook(name, value, resolved_labels)
            except Exception:
                logger.debug("Metrics export hook failed", exc_info=True)

    async def record_received(self, count: int = 1, topic: str | None = None) -> None:
        async with self._lock:
            self.messages_received += count
            self.last_poll_at = datetime.now(UTC)
            if topic is not None:
                self.per_topic_received[topic] = (
                    self.per_topic_received.get(topic, 0) + count
                )
        self._export(
            "consumer_messages_received_total", float(count), {"topic": topic or ""}
        )

    async def record_processed(self, count: int = 1, topic: str | None = None) -> None:
        async with self._lock:
            self.messages_processed += count
            self.last_successful_write_at = datetime.now(UTC)
            if topic is not None:
                self.per_topic_processed[topic] = (
                    self.per_topic_processed.get(topic, 0) + count
                )
        self._export(
            "consumer_messages_processed_total", float(count), {"topic": topic or ""}
        )

    async def record_failed(self, count: int = 1, topic: str | None = None) -> None:
        async with self._lock:
            self.messages_failed += count
            if topic is not None:
                self.per_topic_failed[topic] = (
                    self.per_topic_failed.get(topic, 0) + count
                )
        self._export(
            "consumer_messages_failed_total", float(count), {"topic": topic or ""}
        )

    async def record_skipped(self, count: int = 1) -> None:
        async with self._lock:
            self.messages_skipped += count
        self._export("consumer_messages_skipped_total", float(count))

    async def record_sent_to_dlq(self, count: int = 1) -> None:
        async with self._lock:
            self.messages_sent_to_dlq += count
        self._export("consumer_messages_dlq_total", float(count))

    async def record_batch_processed(self, latency_ms: float | None = None) -> None:
        async with self._lock:
            self.batches_processed += 1
            if latency_ms is not None:
                self.batch_latency_ms.append(latency_ms)
                if len(self.batch_latency_ms) > self.MAX_LATENCY_SAMPLES:
                    self.batch_latency_ms = self.batch_latency_ms[
                        -self.MAX_LATENCY_SAMPLES :
                    ]
        if latency_ms is not None:
            self._export("consumer_batch_latency_ms", latency_ms)

    async def record_polled(self) -> None:
        async with self._lock:
            self.last_poll_at = datetime.now(UTC)

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            latency_stats: dict[str, object] = {}
            if self.batch_latency_ms:
                sorted_latencies = sorted(self.batch_latency_ms)
                latency_stats = {
                    "count": len(sorted_latencies),
                    "min_ms": sorted_latencies[0],
                    "max_ms": sorted_latencies[-1],
                    "avg_ms": sum(sorted_latencies) / len(sorted_latencies),
                    "p50_ms": sorted_latencies[len(sorted_latencies) // 2],
                    "p99_ms": sorted_latencies[
                        min(
                            int(len(sorted_latencies) * 0.99),
                            len(sorted_latencies) - 1,
                        )
                    ],
                }
            return {
                "messages_received": self.messages_received,
                "messages_processed": self.messages_processed,
                "messages_failed": self.messages_failed,
                "messages_skipped": self.messages_skipped,
                "messages_sent_to_dlq": self.messages_sent_to_dlq,
                "batches_processed": self.batches_processed,
                "last_poll_at": (
                    self.last_poll_at.isoformat() if self.last_poll_at else None
                ),
                "last_successful_write_at": (
                    self.last_successful_write_at.isoformat()
                    if self.last_successful_write_at
                    else None
                ),
                "started_at": self.started_at.isoformat(),
                "uptime_seconds": (datetime.now(UTC) - self.started_at).total_seconds(),
                "per_topic_received": dict(self.per_topic_received),
                "per_topic_processed": dict(self.per_topic_processed),
                "per_topic_failed": dict(self.per_topic_failed),
                "batch_latency_stats": latency_stats,
            }


# =============================================================================
# Skill Lifecycle Consumer
# =============================================================================

TOPIC_STARTED = "onex.evt.omniclaude.skill-started.v1"
TOPIC_COMPLETED = "onex.evt.omniclaude.skill-completed.v1"


class SkillLifecycleConsumer:
    """Async Kafka consumer for skill lifecycle observability events.

    Subscribes to skill-started and skill-completed topics and persists
    events to PostgreSQL via WriterSkillLifecyclePostgres.

    Features:
        - Batch processing with configurable size and timeout
        - Per-partition offset tracking (at-least-once delivery)
        - Circuit breaker for database resilience
        - DLQ forwarding for permanently failed messages
        - HTTP health check endpoint for Kubernetes probes
        - Graceful shutdown with SIGTERM/SIGINT handling

    Attributes:
        config: Consumer configuration.
        metrics: Processing metrics.
    """

    def __init__(self, config: ConfigSkillLifecycleConsumer) -> None:
        """Initialize the consumer.

        Args:
            config: Consumer configuration loaded from environment.
        """
        self.config = config
        self.metrics = ConsumerMetrics()

        self._consumer: AIOKafkaConsumer | None = None
        self._producer: AIOKafkaProducer | None = None
        self._pool: asyncpg.Pool | None = None
        self._writer: WriterSkillLifecyclePostgres | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._retry_counts: dict[bytes | None, int] = {}

    async def start(self) -> None:
        """Initialize Kafka consumer, producer, and PostgreSQL pool.

        Raises:
            Exception: If connection to Kafka or PostgreSQL fails.
        """
        logger.info(
            "Starting SkillLifecycleConsumer",
            extra={
                "bootstrap_servers": self.config.kafka_bootstrap_servers,
                "group_id": self.config.kafka_group_id,
                "topics": self.config.topics,
                "postgres_dsn": mask_dsn_password(self.config.postgres_dsn),
                "batch_size": self.config.batch_size,
            },
        )

        # PostgreSQL pool
        self._pool = await asyncpg.create_pool(
            dsn=self.config.postgres_dsn,
            min_size=1,
            max_size=5,
        )

        # Writer
        self._writer = WriterSkillLifecyclePostgres(
            pool=self._pool,
            circuit_breaker_threshold=self.config.circuit_breaker_threshold,
            circuit_breaker_reset_timeout=self.config.circuit_breaker_reset_timeout,
            circuit_breaker_half_open_successes=self.config.circuit_breaker_half_open_successes,
        )

        # Kafka consumer
        self._consumer = AIOKafkaConsumer(
            *self.config.topics,
            bootstrap_servers=self.config.kafka_bootstrap_servers,
            group_id=self.config.kafka_group_id,
            auto_offset_reset=self.config.auto_offset_reset,
            enable_auto_commit=self.config.enable_auto_commit,
            value_deserializer=lambda v: v,
        )
        await self._consumer.start()

        # Kafka producer (for DLQ)
        if self.config.dlq_enabled:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: v,
            )
            await self._producer.start()

        self._running = True
        logger.info("SkillLifecycleConsumer started successfully")

    async def stop(self) -> None:
        """Gracefully stop consumer, producer, and pool."""
        logger.info("Stopping SkillLifecycleConsumer")
        self._running = False
        self._shutdown_event.set()

        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception:
                logger.warning("Error stopping Kafka consumer", exc_info=True)

        if self._producer:
            try:
                await self._producer.stop()
            except Exception:
                logger.warning("Error stopping Kafka producer", exc_info=True)

        if self._pool:
            try:
                await self._pool.close()
            except Exception:
                logger.warning("Error closing PostgreSQL pool", exc_info=True)

        logger.info("SkillLifecycleConsumer stopped")

    def _parse_message(self, record: ConsumerRecord) -> dict[str, object] | None:
        """Parse and decode a Kafka message.

        Args:
            record: Raw Kafka consumer record.

        Returns:
            Parsed dict or None if invalid.
        """
        try:
            raw = json.loads(record.value.decode("utf-8"))
            return dict(raw) if isinstance(raw, dict) else None
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            logger.warning(
                "Failed to parse Kafka message",
                extra={
                    "topic": record.topic,
                    "partition": record.partition,
                    "offset": record.offset,
                },
            )
            return None

    async def _send_to_dlq(self, record: ConsumerRecord, reason: str) -> None:
        """Forward a failed message to the dead letter queue.

        Args:
            record: Original Kafka consumer record.
            reason: Failure reason for DLQ metadata.
        """
        if not self.config.dlq_enabled or not self._producer:
            logger.warning(
                "DLQ disabled, dropping message",
                extra={
                    "topic": record.topic,
                    "offset": record.offset,
                    "reason": reason,
                },
            )
            return

        dlq_payload = json.dumps(
            {
                "original_topic": record.topic,
                "original_partition": record.partition,
                "original_offset": record.offset,
                "failure_reason": reason,
                "failed_at": datetime.now(UTC).isoformat(),
                "payload": record.value.decode("utf-8", errors="replace"),
            }
        ).encode("utf-8")

        try:
            await self._producer.send_and_wait(self.config.dlq_topic, value=dlq_payload)
            await self.metrics.record_sent_to_dlq()
            logger.warning(
                "Message sent to DLQ",
                extra={
                    "dlq_topic": self.config.dlq_topic,
                    "original_topic": record.topic,
                    "offset": record.offset,
                    "reason": reason,
                },
            )
        except Exception:
            logger.exception("Failed to send message to DLQ")

    async def _process_batch(
        self,
        batch: list[ConsumerRecord],
    ) -> dict[TopicPartition, int]:
        """Process a batch of Kafka messages.

        Groups messages by topic type, writes to PostgreSQL, returns
        per-partition max offsets for successful commits.

        Args:
            batch: List of Kafka consumer records.

        Returns:
            Dict mapping TopicPartition to max committed offset.
        """
        started_events: list[dict[str, object]] = []
        completed_events: list[dict[str, object]] = []
        partition_offsets: dict[TopicPartition, int] = {}
        failed_partitions: set[TopicPartition] = set()

        # Parse all messages
        for record in batch:
            await self.metrics.record_received(topic=record.topic)
            parsed = self._parse_message(record)
            if parsed is None:
                await self._send_to_dlq(record, "parse_error")
                await self.metrics.record_skipped()
                continue

            if record.topic == TOPIC_STARTED:
                started_events.append(parsed)
            elif record.topic == TOPIC_COMPLETED:
                completed_events.append(parsed)
            else:
                logger.warning("Unexpected topic: %s", record.topic)
                await self.metrics.record_skipped()
                continue

            tp = TopicPartition(record.topic, record.partition)
            current = partition_offsets.get(tp, -1)
            if record.offset > current:
                partition_offsets[tp] = record.offset

        assert self._writer is not None

        batch_start = time.monotonic()

        # Write started events
        if started_events:
            try:
                await self._writer.write_started(started_events)
                await self.metrics.record_processed(
                    count=len(started_events), topic=TOPIC_STARTED
                )
            except Exception:
                logger.exception("Failed to write skill-started batch")
                await self.metrics.record_failed(
                    count=len(started_events), topic=TOPIC_STARTED
                )
                # Mark all started-topic partitions as failed
                for record in batch:
                    if record.topic == TOPIC_STARTED:
                        failed_partitions.add(
                            TopicPartition(record.topic, record.partition)
                        )

        # Write completed events
        if completed_events:
            try:
                await self._writer.write_completed(completed_events)
                await self.metrics.record_processed(
                    count=len(completed_events), topic=TOPIC_COMPLETED
                )
            except Exception:
                logger.exception("Failed to write skill-completed batch")
                await self.metrics.record_failed(
                    count=len(completed_events), topic=TOPIC_COMPLETED
                )
                for record in batch:
                    if record.topic == TOPIC_COMPLETED:
                        failed_partitions.add(
                            TopicPartition(record.topic, record.partition)
                        )

        batch_latency = (time.monotonic() - batch_start) * 1000
        await self.metrics.record_batch_processed(latency_ms=batch_latency)

        # Exclude failed partitions from commit
        return {
            tp: offset
            for tp, offset in partition_offsets.items()
            if tp not in failed_partitions
        }

    async def run(self) -> None:
        """Main consumer loop.

        Polls Kafka in batches, writes to PostgreSQL, commits offsets.
        Runs until stop() is called or SIGTERM received.
        """
        assert self._consumer is not None

        batch_timeout_s = self.config.batch_timeout_ms / 1000.0
        poll_timeout_s = batch_timeout_s + self.config.poll_timeout_buffer_seconds

        logger.info(
            "SkillLifecycleConsumer entering run loop",
            extra={
                "batch_size": self.config.batch_size,
                "batch_timeout_s": batch_timeout_s,
            },
        )

        while self._running:
            batch: list[ConsumerRecord] = []
            deadline = time.monotonic() + batch_timeout_s

            # Accumulate messages until batch_size or timeout
            while len(batch) < self.config.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    records = await asyncio.wait_for(
                        self._consumer.getmany(
                            timeout_ms=int(remaining * 1000),
                            max_records=self.config.batch_size - len(batch),
                        ),
                        timeout=poll_timeout_s,
                    )
                    await self.metrics.record_polled()
                    for tp_records in records.values():
                        batch.extend(tp_records)
                except TimeoutError:
                    await self.metrics.record_polled()
                    break
                except KafkaError:
                    logger.exception("Kafka poll error")
                    await asyncio.sleep(1.0)
                    break

            if not batch:
                continue

            # Process batch
            try:
                committed = await self._process_batch(batch)
                if committed and self._consumer:
                    # Commit offsets (offset + 1 per Kafka convention)
                    await self._consumer.commit(
                        {tp: offset + 1 for tp, offset in committed.items()}
                    )
            except Exception:
                logger.exception("Unhandled error in batch processing")

        logger.info("SkillLifecycleConsumer run loop exited")

    # =========================================================================
    # Health Check
    # =========================================================================

    def _build_health_response(self) -> tuple[dict[str, object], int]:
        """Build health check response dict and HTTP status code.

        Returns:
            Tuple of (response_dict, http_status_code).
        """
        now = datetime.now(UTC)
        last_write = self.metrics.last_successful_write_at
        last_poll = self.metrics.last_poll_at

        write_age = (now - last_write).total_seconds() if last_write else float("inf")
        poll_age = (now - last_poll).total_seconds() if last_poll else float("inf")

        if not self._running:
            status = EnumHealthStatus.UNHEALTHY
        elif (
            write_age > self.config.health_check_staleness_seconds
            or poll_age > self.config.health_check_poll_staleness_seconds
        ):
            status = EnumHealthStatus.DEGRADED
        else:
            status = EnumHealthStatus.HEALTHY

        http_code = 200 if status == EnumHealthStatus.HEALTHY else 503
        response: dict[str, object] = {
            "status": str(status),
            "running": self._running,
            "last_successful_write_at": (
                last_write.isoformat() if last_write else None
            ),
            "last_poll_at": last_poll.isoformat() if last_poll else None,
            "write_age_seconds": write_age if write_age != float("inf") else None,
            "poll_age_seconds": poll_age if poll_age != float("inf") else None,
        }
        if self._writer:
            response["circuit_breaker"] = self._writer.get_circuit_breaker_state()
        return response, http_code

    async def _health_handler(self, request: web.Request) -> web.Response:
        """HTTP handler for GET /health.

        Args:
            request: Incoming aiohttp request.

        Returns:
            JSON response with health status.
        """
        response, http_code = self._build_health_response()
        return web.json_response(response, status=http_code)

    async def run_with_health_check(self) -> None:
        """Run consumer and health check server concurrently.

        Handles SIGTERM/SIGINT for graceful shutdown.
        """
        loop = asyncio.get_running_loop()

        # Setup signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        app = web.Application()
        app.router.add_get("/health", self._health_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(
            runner,
            host=self.config.health_check_host,
            port=self.config.health_check_port,
        )
        await site.start()

        logger.info(
            "Health check server started",
            extra={
                "host": self.config.health_check_host,
                "port": self.config.health_check_port,
            },
        )

        try:
            await self.run()
        finally:
            await runner.cleanup()


# =============================================================================
# Entry Point
# =============================================================================


async def _main() -> None:
    """Main entry point for running the consumer as a module."""
    import os

    logging.basicConfig(
        level=getattr(
            logging, os.getenv("ONEX_LOG_LEVEL", "INFO").upper(), logging.INFO
        ),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = ConfigSkillLifecycleConsumer()  # type: ignore[call-arg]
    consumer = SkillLifecycleConsumer(config)

    try:
        await consumer.start()
        await consumer.run_with_health_check()
    except Exception:
        logger.exception("SkillLifecycleConsumer terminated with error")
        raise
    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(_main())


__all__ = ["SkillLifecycleConsumer", "ConsumerMetrics", "mask_dsn_password"]
