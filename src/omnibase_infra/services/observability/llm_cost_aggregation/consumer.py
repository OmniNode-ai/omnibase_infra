# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Async Kafka Consumer for LLM Cost Aggregation.

ServiceLlmCostAggregator consumes ``onex.evt.omniintelligence.llm-call-completed.v1``
events from Kafka, writes raw call metrics to ``llm_call_metrics``, and aggregates
costs into ``llm_cost_aggregates`` across multiple dimensions and rolling windows.

Design Decisions:
    - Per-partition offset tracking: Commit only successfully persisted partitions
    - Batch processing: Configurable batch size and timeout
    - Circuit breaker: Resilience via writer's MixinAsyncCircuitBreaker
    - Health check: HTTP endpoint for Kubernetes probes
    - Graceful shutdown: Signal handling with drain and commit
    - Event deduplication: Writer-level dedup cache prevents double-counting on replay

Critical Invariant:
    For each (topic, partition), commit offsets only up to the highest offset
    that has been successfully persisted for that partition.
    Never commit offsets for partitions that had write failures in the batch.

Topics consumed:
    - onex.evt.omniintelligence.llm-call-completed.v1

Related Tickets:
    - OMN-2240: E1-T4 LLM cost aggregation service
    - OMN-2236: llm_call_metrics + llm_cost_aggregates migration 031
    - OMN-2238: Extract and normalize token usage from LLM API responses

Example:
    >>> from omnibase_infra.services.observability.llm_cost_aggregation import (
    ...     ServiceLlmCostAggregator,
    ...     ConfigLlmCostAggregation,
    ... )
    >>>
    >>> config = ConfigLlmCostAggregation(
    ...     kafka_bootstrap_servers="localhost:9092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
    ... )
    >>> service = ServiceLlmCostAggregator(config)
    >>>
    >>> # Run consumer (blocking)
    >>> await service.start()
    >>> await service.run()

    # Or run as module:
    # python -m omnibase_infra.services.observability.llm_cost_aggregation.consumer
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import signal
import sys
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse
from uuid import UUID, uuid4

import asyncpg
from aiohttp import web
from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.errors import KafkaError
from aiokafka.structs import OffsetAndMetadata
from pydantic import ValidationError

from omnibase_core.errors import OnexError
from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.services.observability.llm_cost_aggregation.config import (
    ConfigLlmCostAggregation,
)
from omnibase_infra.services.observability.llm_cost_aggregation.writer_postgres import (
    WriterLlmCostAggregationPostgres,
)

if TYPE_CHECKING:
    from aiokafka.structs import ConsumerRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


# Pre-compiled pattern/replacement pairs for credential masking in DSN
# query strings and non-standard formats that urlparse may not detect.
#
# These patterns fire only as a **fallback** when urlparse does not detect
# a password (e.g., password passed as a query parameter or in a
# non-standard format).  The :secret@ pattern could theoretically match
# non-credential text in a user:port@ scenario, but since it only runs
# when urlparse failed to find any password, the false-positive risk is
# acceptable for this defense-in-depth masking layer.
_DSN_PASSWORD_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # password=value, pwd=value, passwd=value in query params (key=value&...)
    # Groups: (1) key=, (2) value  ->  replacement preserves key, masks value
    (
        re.compile(r"((?:password|passwd|pwd)\s*=\s*)([^&\s]+)", re.IGNORECASE),
        r"\g<1>***",
    ),
    # :secret@ pattern in netloc that urlparse missed
    # Groups: (1) ':', (2) secret, (3) '@'  ->  replacement preserves : and @
    (
        re.compile(r"(:)([^:@/]+)(@)"),
        r"\g<1>***\g<3>",
    ),
)


def mask_dsn_password(dsn: str) -> str:
    """Mask password in a PostgreSQL DSN for safe logging.

    First attempts structured masking via ``urlparse``. If ``urlparse``
    does not detect a password (e.g., password passed as a query parameter
    or in a non-standard format), falls back to regex-based masking of
    common credential patterns (``password=``, ``pwd=``, ``passwd=``,
    ``:secret@``).

    Args:
        dsn: PostgreSQL connection string.

    Returns:
        DSN with password replaced by '***'.
    """
    _MASKING_FAILED = "***DSN_MASKING_FAILED***"

    # Extract the original password (if any) up front so we can verify
    # it was actually removed from the masked result.
    try:
        _original_password = urlparse(dsn).password
    except Exception:
        _original_password = None

    try:
        parsed = urlparse(dsn)
        if parsed.password:
            user_part = parsed.username or ""
            host_part = (
                f"{parsed.hostname}:{parsed.port}"
                if parsed.port
                else str(parsed.hostname)
            )
            masked_netloc = f"{user_part}:***@{host_part}"
            masked = urlunparse(
                (
                    parsed.scheme,
                    masked_netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )
        else:
            # urlparse did not detect a password.  Apply regex fallback to
            # catch password=<value>, pwd=<value>, passwd=<value> in query
            # strings, and :secret@ patterns in the netloc.
            masked = dsn
            for pattern, replacement in _DSN_PASSWORD_PATTERNS:
                masked = pattern.sub(replacement, masked)

    except (ValueError, AttributeError):
        # Even on parse failure, attempt regex masking on the raw string
        masked = dsn
        for pattern, replacement in _DSN_PASSWORD_PATTERNS:
            masked = pattern.sub(replacement, masked)
    except Exception:
        logger.warning(
            "Unexpected error masking DSN password; returning placeholder to prevent credential leak",
        )
        return _MASKING_FAILED

    # Defense-in-depth: verify the original password was actually removed.
    # If masking somehow failed to strip it, fall back to the safe placeholder.
    if _original_password and _original_password in masked:
        logger.warning("DSN masking did not fully remove password; using fallback")
        return _MASKING_FAILED

    return masked


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
    """Metrics tracking for the LLM cost aggregation consumer.

    Coroutine-safe via asyncio.Lock protection (single event loop only).
    """

    def __init__(self) -> None:
        """Initialize metrics with zero values."""
        self.messages_received: int = 0
        self.messages_processed: int = 0
        self.messages_failed: int = 0
        self.messages_skipped: int = 0
        self.batches_processed: int = 0
        self.aggregations_written: int = 0
        self.commit_failures: int = 0
        self.last_poll_at: datetime | None = None
        self.last_successful_write_at: datetime | None = None
        self.last_commit_failure_at: datetime | None = None
        self.started_at: datetime = datetime.now(UTC)
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

    async def record_aggregations(self, count: int = 1) -> None:
        """Record aggregation rows written."""
        async with self._lock:
            self.aggregations_written += count

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

    async def record_polled(self) -> None:
        """Record a poll attempt."""
        async with self._lock:
            self.last_poll_at = datetime.now(UTC)

    async def record_commit_failure(self) -> None:
        """Record an offset commit failure."""
        async with self._lock:
            self.commit_failures += 1
            self.last_commit_failure_at = datetime.now(UTC)

    async def reset_commit_failures(self) -> None:
        """Reset consecutive commit failure counter after successful commit."""
        async with self._lock:
            self.commit_failures = 0

    async def snapshot(self) -> dict[str, object]:
        """Get a snapshot of current metrics."""
        async with self._lock:
            return {
                "messages_received": self.messages_received,
                "messages_processed": self.messages_processed,
                "messages_failed": self.messages_failed,
                "messages_skipped": self.messages_skipped,
                "batches_processed": self.batches_processed,
                "aggregations_written": self.aggregations_written,
                "commit_failures": self.commit_failures,
                "last_poll_at": (
                    self.last_poll_at.isoformat() if self.last_poll_at else None
                ),
                "last_successful_write_at": (
                    self.last_successful_write_at.isoformat()
                    if self.last_successful_write_at
                    else None
                ),
                "last_commit_failure_at": (
                    self.last_commit_failure_at.isoformat()
                    if self.last_commit_failure_at
                    else None
                ),
                "started_at": self.started_at.isoformat(),
            }


# =============================================================================
# ServiceLlmCostAggregator
# =============================================================================


class ServiceLlmCostAggregator:
    """Async Kafka consumer for LLM cost aggregation.

    Consumes LLM call completed events, writes raw metrics to
    ``llm_call_metrics``, and aggregates costs into ``llm_cost_aggregates``
    across session, model, repo, and pattern dimensions with 24h, 7d,
    and 30d rolling windows.

    Features:
        - **Per-partition offset tracking**: Commit only successfully persisted
          partitions. Partial batch failures do not cause message loss.
        - **Dual write**: Both raw metrics and aggregated costs in one batch.
        - **Event deduplication**: In-memory bounded cache prevents double-counting.
        - **Rolling windows**: 24h, 7d, 30d aggregation via additive upserts.
        - **Estimated coverage tracking**: Weighted average of API vs estimated usage.
        - **Circuit breaker**: Database resilience via writer's circuit breaker.
        - **Health check endpoint**: HTTP server for Kubernetes probes.
        - **Graceful shutdown**: Signal handling with drain and final commit.

    Attributes:
        metrics: Consumer metrics for observability.
        is_running: Whether the consumer is currently running.
    """

    def __init__(self, config: ConfigLlmCostAggregation) -> None:
        """Initialize the LLM cost aggregation consumer.

        Args:
            config: Consumer configuration (Kafka, PostgreSQL, batch settings).
        """
        self._config = config
        self._consumer: AIOKafkaConsumer | None = None
        self._pool: asyncpg.Pool | None = None
        self._writer: WriterLlmCostAggregationPostgres | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Health check server
        self._health_app: web.Application | None = None
        self._health_runner: web.AppRunner | None = None
        self._health_site: web.TCPSite | None = None

        # Metrics
        self.metrics = ConsumerMetrics()

        # Consumer ID for logging
        self._consumer_id = f"llm-cost-aggregation-{uuid4().hex[:8]}"

        logger.info(
            "ServiceLlmCostAggregator initialized",
            extra={
                "consumer_id": self._consumer_id,
                "topics": self._config.topics,
                "group_id": self._config.kafka_group_id,
                "bootstrap_servers": self._config.kafka_bootstrap_servers,
                "postgres_dsn": mask_dsn_password(self._config.postgres_dsn),
                "batch_size": self._config.batch_size,
                "batch_timeout_ms": self._config.batch_timeout_ms,
            },
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if the consumer is currently running."""
        return self._running

    @property
    def consumer_id(self) -> str:
        """Get the unique consumer identifier."""
        return self._consumer_id

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """Start the consumer, pool, writer, and health check server.

        Raises:
            RuntimeError: If the consumer is already running.
            asyncpg.PostgresError: If database connection fails.
            KafkaError: If Kafka connection fails.
        """
        if self._running:
            logger.warning(
                "Consumer already running",
                extra={"consumer_id": self._consumer_id},
            )
            return

        correlation_id = uuid4()

        logger.info(
            "Starting ServiceLlmCostAggregator",
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
                min_size=self._config.pool_min_size,
                max_size=self._config.pool_max_size,
            )

            # Create writer with pool injection
            self._writer = WriterLlmCostAggregationPostgres(
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
                enable_auto_commit=False,
                max_poll_records=self._config.batch_size,
            )

            await self._consumer.start()

            # Start health check server
            await self._start_health_server()

            self._running = True
            self._shutdown_event.clear()

            logger.info(
                "ServiceLlmCostAggregator started",
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
            await self._cleanup_resources(correlation_id)
            raise

    async def stop(self) -> None:
        """Stop the consumer gracefully."""
        if not self._running:
            return

        correlation_id = uuid4()

        logger.info(
            "Stopping ServiceLlmCostAggregator",
            extra={
                "consumer_id": self._consumer_id,
                "correlation_id": str(correlation_id),
            },
        )

        self._running = False
        self._shutdown_event.set()

        await self._cleanup_resources(correlation_id)

        metrics_snapshot = await self.metrics.snapshot()
        logger.info(
            "ServiceLlmCostAggregator stopped",
            extra={
                "consumer_id": self._consumer_id,
                "correlation_id": str(correlation_id),
                "final_metrics": metrics_snapshot,
            },
        )

    async def _cleanup_resources(self, correlation_id: UUID) -> None:
        """Clean up all resources during shutdown."""
        if self._health_site is not None:
            await self._health_site.stop()
            self._health_site = None

        if self._health_runner is not None:
            await self._health_runner.cleanup()
            self._health_runner = None

        self._health_app = None

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

        This method blocks until stop() is called or an unrecoverable error
        occurs. Use this after calling start().
        """
        if not self._running or self._consumer is None:
            raise OnexError(
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

    async def __aenter__(self) -> ServiceLlmCostAggregator:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.stop()

    # =========================================================================
    # Consume Loop
    # =========================================================================

    async def _consume_loop(self, correlation_id: UUID) -> None:
        """Main consumption loop with batch processing."""
        if self._consumer is None:
            return

        batch_timeout_seconds = self._config.batch_timeout_ms / 1000.0

        try:
            while self._running:
                try:
                    # Buffer accounts for event loop scheduling latency; increase if TimeoutError is frequent.
                    records = await asyncio.wait_for(
                        self._consumer.getmany(
                            timeout_ms=self._config.batch_timeout_ms,
                            max_records=self._config.batch_size,
                        ),
                        timeout=batch_timeout_seconds
                        + self._config.poll_timeout_buffer_seconds,
                    )
                except TimeoutError:
                    continue

                await self.metrics.record_polled()

                if not records:
                    continue

                messages: list[ConsumerRecord] = []
                for tp_messages in records.values():
                    messages.extend(tp_messages)

                if not messages:
                    continue

                await self.metrics.record_received(len(messages))

                batch_correlation_id = uuid4()
                successful_offsets = await self._process_batch(
                    messages, batch_correlation_id
                )

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

    @staticmethod
    def _track_skipped_offset(
        skipped_offsets: dict[TopicPartition, int],
        msg: ConsumerRecord,
    ) -> None:
        """Track offset for a skipped message to enable commit after processing."""
        tp = TopicPartition(msg.topic, msg.partition)
        current = skipped_offsets.get(tp, -1)
        skipped_offsets[tp] = max(current, msg.offset)

    async def _process_batch(
        self,
        messages: list[ConsumerRecord],
        correlation_id: UUID,
    ) -> dict[TopicPartition, int]:
        """Process batch: parse events, write metrics, write aggregates.

        Returns highest successful offset per partition.
        """
        if self._writer is None:
            return {}

        successful_offsets: dict[TopicPartition, int] = {}
        skipped_offsets: dict[TopicPartition, int] = {}
        parsed_skipped: int = 0

        # Parse all messages into event dictionaries
        parsed_events: list[tuple[ConsumerRecord, dict[str, object]]] = []

        for msg in messages:
            if msg.value is None:
                parsed_skipped += 1
                self._track_skipped_offset(skipped_offsets, msg)
                continue

            try:
                value = msg.value
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8")
                    except UnicodeDecodeError:
                        parsed_skipped += 1
                        self._track_skipped_offset(skipped_offsets, msg)
                        continue

                payload = json.loads(value)
                if not isinstance(payload, dict):
                    parsed_skipped += 1
                    self._track_skipped_offset(skipped_offsets, msg)
                    continue

                parsed_events.append((msg, payload))

            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode JSON message",
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                        "topic": msg.topic,
                        "partition": msg.partition,
                        "offset": msg.offset,
                    },
                )
                parsed_skipped += 1
                self._track_skipped_offset(skipped_offsets, msg)

        if parsed_skipped > 0:
            await self.metrics.record_skipped(parsed_skipped)

        if not parsed_events:
            # Merge skipped offsets even when no events parsed
            for tp, offset in skipped_offsets.items():
                current = successful_offsets.get(tp, -1)
                successful_offsets[tp] = max(current, offset)
            return successful_offsets

        event_dicts = [ev for _, ev in parsed_events]

        # Write raw call metrics
        try:
            metrics_written = await self._writer.write_call_metrics(
                event_dicts, correlation_id
            )

            # Track offsets for successful writes
            for msg, _ in parsed_events:
                tp = TopicPartition(msg.topic, msg.partition)
                current = successful_offsets.get(tp, -1)
                successful_offsets[tp] = max(current, msg.offset)

            await self.metrics.record_processed(metrics_written)

        except Exception:
            logger.exception(
                "Failed to write call metrics batch",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                    "count": len(event_dicts),
                },
            )
            await self.metrics.record_failed(len(event_dicts))
            # On metrics write failure, skip aggregation too
            for tp, offset in skipped_offsets.items():
                current = successful_offsets.get(tp, -1)
                successful_offsets[tp] = max(current, offset)
            return successful_offsets

        # Write cost aggregates
        try:
            agg_written = await self._writer.write_cost_aggregates(
                event_dicts, correlation_id
            )
            await self.metrics.record_aggregations(agg_written)

        except Exception:
            # Aggregation failure is non-fatal -- raw metrics were already written.
            # Aggregates will catch up on the next batch.
            logger.exception(
                "Failed to write cost aggregates (non-fatal, raw metrics persisted)",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                    "count": len(event_dicts),
                },
            )

        # Merge skipped offsets
        for tp, offset in skipped_offsets.items():
            current = successful_offsets.get(tp, -1)
            successful_offsets[tp] = max(current, offset)

        return successful_offsets

    async def _commit_offsets(
        self,
        offsets: dict[TopicPartition, int],
        correlation_id: UUID,
    ) -> None:
        """Commit only successfully persisted offsets per partition."""
        if not offsets or self._consumer is None:
            return

        commit_map: dict[TopicPartition, OffsetAndMetadata] = {
            tp: OffsetAndMetadata(offset + 1, "") for tp, offset in offsets.items()
        }

        try:
            await self._consumer.commit(commit_map)
            await self.metrics.reset_commit_failures()

        except KafkaError as exc:
            await self.metrics.record_commit_failure()

            # Classify the error: fatal errors indicate the consumer's
            # group membership is invalid and it must rejoin.  Retriable
            # errors (e.g., transient coordinator issues) are expected to
            # self-resolve.
            _FATAL_ERROR_NAMES = {
                "UnknownMemberIdError",
                "RebalanceInProgressError",
                "IllegalGenerationError",
                "FencedInstanceIdError",
            }
            error_name = type(exc).__name__
            is_fatal = error_name in _FATAL_ERROR_NAMES

            metrics_snapshot = await self.metrics.snapshot()
            commit_failures = metrics_snapshot.get("commit_failures", 0)

            if is_fatal:
                # Fatal: consumer group membership is stale.  Log at ERROR
                # and re-raise so the consume loop can trigger reconnection.
                logger.exception(
                    "Fatal commit error (%s) -- consumer must rejoin group",
                    error_name,
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                        "error_name": error_name,
                    },
                )
                raise
            if isinstance(commit_failures, int) and commit_failures >= 5:
                logger.exception(
                    "Persistent commit failures detected",
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                        "commit_failures": commit_failures,
                    },
                )
            else:
                logger.warning(
                    "Retriable commit error (%s), will retry on next batch",
                    error_name,
                    exc_info=True,
                    extra={
                        "consumer_id": self._consumer_id,
                        "correlation_id": str(correlation_id),
                    },
                )

    # =========================================================================
    # Health Check Server
    # =========================================================================

    async def _start_health_server(self) -> None:
        """Start minimal HTTP health check server.

        Raises:
            InfraConnectionError: If the health check port is already in use
                or otherwise unavailable (wraps ``OSError``).
        """
        self._health_app = web.Application()
        self._health_app.router.add_get("/health", self._health_handler)
        self._health_app.router.add_get("/health/live", self._liveness_handler)
        self._health_app.router.add_get("/health/ready", self._readiness_handler)

        self._health_runner = web.AppRunner(self._health_app)
        await self._health_runner.setup()

        self._health_site = web.TCPSite(
            self._health_runner,
            host=self._config.health_check_host,
            port=self._config.health_check_port,
        )

        try:
            await self._health_site.start()
        except OSError as exc:
            port = self._config.health_check_port
            host = self._config.health_check_host
            logger.exception(
                "Health check port %d already in use (host=%s)",
                port,
                host,
                extra={
                    "consumer_id": self._consumer_id,
                    "host": host,
                    "port": port,
                    "error": str(exc),
                },
            )
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="start_health_server",
            )
            raise InfraConnectionError(
                f"Health check port {port} already in use (host={host})",
                context=context,
            ) from exc

        logger.info(
            "Health check server started",
            extra={
                "consumer_id": self._consumer_id,
                "host": self._config.health_check_host,
                "port": self._config.health_check_port,
            },
        )

    def _determine_health_status(
        self,
        metrics_snapshot: dict[str, object],
        circuit_state: dict[str, JsonType],
    ) -> EnumHealthStatus:
        """Determine consumer health status based on current state."""
        if not self._running:
            return EnumHealthStatus.UNHEALTHY

        circuit_breaker_state = circuit_state.get("state")
        if circuit_breaker_state in ("open", "half_open"):
            return EnumHealthStatus.DEGRADED

        last_poll = metrics_snapshot.get("last_poll_at")
        if last_poll is not None:
            try:
                last_poll_dt = datetime.fromisoformat(str(last_poll))
                poll_age_seconds = (datetime.now(UTC) - last_poll_dt).total_seconds()
                if poll_age_seconds > self._config.health_check_poll_staleness_seconds:
                    return EnumHealthStatus.DEGRADED
            except (ValueError, TypeError):
                pass

        last_write = metrics_snapshot.get("last_successful_write_at")
        messages_received = metrics_snapshot.get("messages_received", 0)

        if last_write is None:
            started_at_str = metrics_snapshot.get("started_at")
            if started_at_str is not None:
                try:
                    started_at_dt = datetime.fromisoformat(str(started_at_str))
                    age_seconds = (datetime.now(UTC) - started_at_dt).total_seconds()
                    if age_seconds <= self._config.startup_grace_period_seconds:
                        return EnumHealthStatus.HEALTHY
                    return EnumHealthStatus.DEGRADED
                except (ValueError, TypeError):
                    return EnumHealthStatus.HEALTHY
            return EnumHealthStatus.HEALTHY

        try:
            last_write_dt = datetime.fromisoformat(str(last_write))
            write_age_seconds = (datetime.now(UTC) - last_write_dt).total_seconds()
            if (
                write_age_seconds > self._config.health_check_staleness_seconds
                and isinstance(messages_received, int)
                and messages_received > 0
            ):
                return EnumHealthStatus.DEGRADED
            return EnumHealthStatus.HEALTHY
        except (ValueError, TypeError):
            return EnumHealthStatus.HEALTHY

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        metrics_snapshot = await self.metrics.snapshot()
        circuit_state = self._writer.get_circuit_breaker_state() if self._writer else {}

        status = self._determine_health_status(metrics_snapshot, circuit_state)

        response_body = {
            "status": status.value,
            "consumer_running": self._running,
            "consumer_id": self._consumer_id,
            "last_poll_time": metrics_snapshot.get("last_poll_at"),
            "last_successful_write": metrics_snapshot.get("last_successful_write_at"),
            "circuit_breaker_state": circuit_state.get("state", "unknown"),
            "messages_processed": metrics_snapshot.get("messages_processed", 0),
            "messages_failed": metrics_snapshot.get("messages_failed", 0),
            "aggregations_written": metrics_snapshot.get("aggregations_written", 0),
            "batches_processed": metrics_snapshot.get("batches_processed", 0),
        }

        # Return HTTP 200 for HEALTHY and DEGRADED so that Kubernetes
        # readiness probes continue routing traffic when the service is
        # functional but experiencing minor staleness. DEGRADED indicates
        # slightly stale data, not inability to serve. Only UNHEALTHY
        # returns 503 to stop traffic routing.  The "status" field in the
        # JSON body allows monitoring to differentiate the actual state.
        http_status = 200 if status != EnumHealthStatus.UNHEALTHY else 503
        return web.json_response(response_body, status=http_status)

    async def _liveness_handler(self, request: web.Request) -> web.Response:
        """Handle Kubernetes liveness probe requests."""
        is_alive = self._running
        response_body = {
            "status": "alive" if is_alive else "dead",
            "consumer_id": self._consumer_id,
        }
        return web.json_response(response_body, status=200 if is_alive else 503)

    async def _readiness_handler(self, request: web.Request) -> web.Response:
        """Handle Kubernetes readiness probe requests."""
        dependencies_ready = {
            "postgres_pool": self._pool is not None,
            "kafka_consumer": self._consumer is not None,
            "writer": self._writer is not None,
        }

        circuit_state = self._writer.get_circuit_breaker_state() if self._writer else {}
        circuit_ready = circuit_state.get("state") != "open"
        dependencies_ready["circuit_breaker"] = circuit_ready

        all_ready = all(dependencies_ready.values()) and self._running

        response_body = {
            "status": "ready" if all_ready else "not_ready",
            "consumer_id": self._consumer_id,
            "consumer_running": self._running,
            "dependencies": dependencies_ready,
            "circuit_breaker_state": circuit_state.get("state", "unknown"),
        }

        return web.json_response(response_body, status=200 if all_ready else 503)

    # =========================================================================
    # Health Check (Direct API)
    # =========================================================================

    async def health_check(self) -> dict[str, object]:
        """Check consumer health status for programmatic access."""
        metrics_snapshot = await self.metrics.snapshot()
        circuit_state = self._writer.get_circuit_breaker_state() if self._writer else {}

        status = self._determine_health_status(metrics_snapshot, circuit_state)

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
    try:
        config = ConfigLlmCostAggregation()
    except ValidationError as exc:
        # Translate Pydantic's raw ValidationError into a user-friendly
        # message that tells the operator which env vars to set.
        missing = [str(e["loc"][-1]) for e in exc.errors() if e["type"] == "missing"]
        prefix = "OMNIBASE_INFRA_LLM_COST_"
        if missing:
            env_vars = ", ".join(f"{prefix}{f.upper()}" for f in missing)
            print(
                f"ERROR: Missing required configuration. "
                f"Set the following environment variable(s): {env_vars}",
                file=sys.stderr,
            )
        else:
            print(
                f"ERROR: Invalid configuration: {exc}",
                file=sys.stderr,
            )
        sys.exit(1)

    logger.info(
        "Starting LLM cost aggregation consumer",
        extra={
            "topics": config.topics,
            "bootstrap_servers": config.kafka_bootstrap_servers,
            "postgres_dsn": mask_dsn_password(config.postgres_dsn),
            "group_id": config.kafka_group_id,
            "health_port": config.health_check_port,
        },
    )

    consumer = ServiceLlmCostAggregator(config)

    loop = asyncio.get_running_loop()
    shutdown_task: asyncio.Task[None] | None = None

    def _shutdown_task_done(task: asyncio.Task[None]) -> None:
        """Log errors from the shutdown task so they are not silently swallowed."""
        if task.cancelled():
            logger.warning("Shutdown task was cancelled")
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Shutdown task raised an exception",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def signal_handler() -> None:
        nonlocal shutdown_task
        logger.info("Received shutdown signal")
        if shutdown_task is None:
            shutdown_task = asyncio.create_task(consumer.stop())
            shutdown_task.add_done_callback(_shutdown_task_done)

    # Note: add_signal_handler is Unix-only; this consumer targets Linux/Docker.
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await consumer.start()
        await consumer.run()
    except asyncio.CancelledError:
        logger.info("Consumer cancelled")
    finally:
        if shutdown_task is not None:
            if not shutdown_task.done():
                await shutdown_task
        else:
            await consumer.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(_main())


__all__ = [
    "ConsumerMetrics",
    "EnumHealthStatus",
    "ServiceLlmCostAggregator",
    "mask_dsn_password",
]
