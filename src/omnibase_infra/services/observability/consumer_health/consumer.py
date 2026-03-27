# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Async Kafka Consumer for Consumer Health read-model projection (OMN-6757).

Consumes consumer health events from Kafka and persists them to PostgreSQL
via WriterConsumerHealthPostgres for omnidash /consumer-health dashboard.

Topics consumed:
    - onex.evt.omnibase-infra.consumer-health.v1

Related Tickets:
    - OMN-6757: Wire consumer-health topic with read-model projection
    - OMN-5529: Runtime Health Event Pipeline (epic)

Example:
    >>> from omnibase_infra.services.observability.consumer_health import (
    ...     ConsumerHealthProjectionConsumer,
    ...     ConfigConsumerHealthProjection,
    ... )
    >>>
    >>> config = ConfigConsumerHealthProjection(
    ...     kafka_bootstrap_servers="localhost:19092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
    ... )
    >>> consumer = ConsumerHealthProjectionConsumer(config)
    >>> await consumer.start()
    >>> await consumer.run()

    # Or run as module:
    # python -m omnibase_infra.services.observability.consumer_health.consumer
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import asyncpg
from aiohttp import web
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

from omnibase_infra.services.observability.consumer_health.config import (
    ConfigConsumerHealthProjection,
)
from omnibase_infra.services.observability.consumer_health.writer_postgres import (
    WriterConsumerHealthPostgres,
)

if TYPE_CHECKING:
    from aiokafka.structs import ConsumerRecord

logger = logging.getLogger(__name__)


def _mask_dsn_password(dsn: str) -> str:
    """Mask password in a PostgreSQL DSN for safe logging."""
    try:
        parsed = urlparse(dsn)
        if not parsed.password:
            return dsn
        if parsed.port:
            masked = f"{parsed.username}:***@{parsed.hostname}:{parsed.port}"
        else:
            masked = f"{parsed.username}:***@{parsed.hostname}"
        return urlunparse(
            (
                parsed.scheme,
                masked,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
    except Exception:  # noqa: BLE001 — boundary: returns redacted fallback
        return "***redacted***"


class EnumHealthStatus(StrEnum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ConsumerHealthProjectionConsumer:
    """Kafka consumer that projects consumer-health events into PostgreSQL.

    Follows the same pattern as SkillLifecycleConsumer and AgentActionsConsumer:
    batch processing, per-partition offset tracking, circuit breaker resilience,
    health check endpoint, and graceful shutdown.
    """

    def __init__(self, config: ConfigConsumerHealthProjection) -> None:
        self._config = config
        self._consumer: AIOKafkaConsumer | None = None
        self._writer: WriterConsumerHealthPostgres | None = None
        self._pool: asyncpg.Pool | None = None  # type: ignore[type-arg]
        self._health_app: web.Application | None = None
        self._health_runner: web.AppRunner | None = None
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._messages_received: int = 0
        self._messages_processed: int = 0
        self._messages_failed: int = 0
        self._batches_processed: int = 0
        self._last_poll_at: datetime | None = None
        self._last_write_at: datetime | None = None
        self._started_at: datetime | None = None

    async def start(self) -> None:
        """Initialize Kafka consumer, PostgreSQL pool, and health check server."""
        self._started_at = datetime.now(UTC)

        # PostgreSQL pool
        logger.info(
            "Connecting to PostgreSQL: %s",
            _mask_dsn_password(self._config.postgres_dsn),
        )
        self._pool = await asyncpg.create_pool(
            dsn=self._config.postgres_dsn, min_size=2, max_size=10
        )

        # Writer
        self._writer = WriterConsumerHealthPostgres(
            self._pool,
            circuit_breaker_threshold=self._config.circuit_breaker_threshold,
            circuit_breaker_reset_timeout=self._config.circuit_breaker_reset_timeout,
            circuit_breaker_half_open_successes=self._config.circuit_breaker_half_open_successes,
        )

        # Kafka consumer
        self._consumer = AIOKafkaConsumer(
            *self._config.topics,
            bootstrap_servers=self._config.kafka_bootstrap_servers,
            group_id=self._config.kafka_group_id,
            auto_offset_reset=self._config.auto_offset_reset,
            enable_auto_commit=self._config.enable_auto_commit,
            session_timeout_ms=self._config.session_timeout_ms,
            heartbeat_interval_ms=self._config.heartbeat_interval_ms,
            max_poll_interval_ms=self._config.max_poll_interval_ms,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")) if v else None,
        )
        await self._consumer.start()
        logger.info(
            "Consumer started: topics=%s group=%s",
            self._config.topics,
            self._config.kafka_group_id,
        )

        # Health check
        await self._start_health_check()

    async def _start_health_check(self) -> None:
        """Start aiohttp health check server."""
        self._health_app = web.Application()
        self._health_app.router.add_get("/health", self._handle_health)
        self._health_runner = web.AppRunner(self._health_app)
        await self._health_runner.setup()
        site = web.TCPSite(
            self._health_runner,
            self._config.health_check_host,
            self._config.health_check_port,
        )
        await site.start()
        logger.info(
            "Health check: http://%s:%d/health",
            self._config.health_check_host,
            self._config.health_check_port,
        )

    async def _handle_health(self, _request: web.Request) -> web.Response:
        """HTTP health check endpoint."""
        status = self._determine_health()
        body = {
            "status": status.value,
            "consumer_group": self._config.kafka_group_id,
            "topics": self._config.topics,
            "messages_received": self._messages_received,
            "messages_processed": self._messages_processed,
            "messages_failed": self._messages_failed,
            "batches_processed": self._batches_processed,
            "last_poll_at": self._last_poll_at.isoformat()
            if self._last_poll_at
            else None,
            "last_write_at": self._last_write_at.isoformat()
            if self._last_write_at
            else None,
            "started_at": self._started_at.isoformat() if self._started_at else None,
        }
        http_status = 200 if status == EnumHealthStatus.HEALTHY else 503
        return web.json_response(body, status=http_status)

    def _determine_health(self) -> EnumHealthStatus:
        """Determine consumer health based on poll and write recency."""
        now = datetime.now(UTC)

        if self._last_poll_at is not None:
            poll_age = (now - self._last_poll_at).total_seconds()
            if poll_age > self._config.health_check_poll_staleness_seconds:
                return EnumHealthStatus.UNHEALTHY

        if self._last_write_at is not None:
            write_age = (now - self._last_write_at).total_seconds()
            if write_age > self._config.health_check_staleness_seconds:
                return EnumHealthStatus.DEGRADED

        return EnumHealthStatus.HEALTHY

    async def run(self) -> None:
        """Main consumer loop: poll, batch, write, commit."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        logger.info("Consumer run loop started")
        batch: list[tuple[ConsumerRecord, dict[str, object]]] = []
        batch_deadline = time.monotonic() + self._config.batch_timeout_ms / 1000.0

        while not self._shutdown_event.is_set():
            try:
                timeout_s = max(0.1, batch_deadline - time.monotonic())
                result = await asyncio.wait_for(
                    self._consumer.getmany(  # type: ignore[union-attr]
                        timeout_ms=int(timeout_s * 1000),
                        max_records=self._config.batch_size,
                    ),
                    timeout=timeout_s + self._config.poll_timeout_buffer_seconds,
                )
                self._last_poll_at = datetime.now(UTC)

                for _tp, records in result.items():
                    for record in records:
                        self._messages_received += 1
                        if record.value is not None:
                            batch.append((record, record.value))

                # Flush batch on size or timeout
                now_mono = time.monotonic()
                if len(batch) >= self._config.batch_size or now_mono >= batch_deadline:
                    if batch:
                        await self._flush_batch(batch)
                        batch = []
                    batch_deadline = now_mono + self._config.batch_timeout_ms / 1000.0

            except TimeoutError:
                # Poll timeout — flush any partial batch
                if batch:
                    await self._flush_batch(batch)
                    batch = []
                batch_deadline = (
                    time.monotonic() + self._config.batch_timeout_ms / 1000.0
                )
            except KafkaError:
                logger.exception("Kafka error in consumer loop")
                await asyncio.sleep(1.0)
            except Exception:
                logger.exception("Unexpected error in consumer loop")
                await asyncio.sleep(1.0)

        # Drain remaining
        if batch:
            await self._flush_batch(batch)
        logger.info("Consumer run loop exited")

    async def _flush_batch(
        self,
        batch: list[tuple[ConsumerRecord, dict[str, object]]],
    ) -> None:
        """Write batch to PostgreSQL and commit offsets."""
        if not batch or self._writer is None or self._consumer is None:
            return

        events = [event for _, event in batch]
        start = time.monotonic()

        try:
            written = await self._writer.write_batch(events)
            elapsed_ms = (time.monotonic() - start) * 1000
            self._messages_processed += written
            self._messages_failed += len(events) - written
            self._batches_processed += 1
            self._last_write_at = datetime.now(UTC)
            logger.debug(
                "Flushed batch: %d/%d written in %.1fms",
                written,
                len(events),
                elapsed_ms,
            )
            # Commit offsets
            await self._consumer.commit()
        except Exception:
            self._messages_failed += len(events)
            logger.exception("Failed to flush batch of %d events", len(events))

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._shutdown_event.set()
        if self._consumer is not None:
            await self._consumer.stop()
        if self._pool is not None:
            await self._pool.close()
        if self._health_runner is not None:
            await self._health_runner.cleanup()
        logger.info("Consumer stopped")

    async def health_check(self) -> dict[str, object]:
        """Programmatic health check for monitoring."""
        status = self._determine_health()
        return {
            "status": status.value,
            "messages_received": self._messages_received,
            "messages_processed": self._messages_processed,
            "messages_failed": self._messages_failed,
        }


async def _main() -> None:
    """Entry point for running as a module."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    config = ConfigConsumerHealthProjection()  # type: ignore[call-arg]
    consumer = ConsumerHealthProjectionConsumer(config)
    await consumer.start()
    try:
        await consumer.run()
    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(_main())
