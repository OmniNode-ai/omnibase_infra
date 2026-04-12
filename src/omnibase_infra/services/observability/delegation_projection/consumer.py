# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Async Kafka consumer for delegation projection.

ServiceDelegationProjectionConsumer consumes ``onex.evt.omniclaude.task-delegated.v1``
events from Kafka and UPSERTs them into the ``delegation_events`` table.

Topics consumed:
    - onex.evt.omniclaude.task-delegated.v1

Related Tickets:
    - OMN-8532: delegation projector missing consumer service

Example:
    # Run as module:
    # python -m omnibase_infra.services.observability.delegation_projection.consumer
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import asyncpg
from aiohttp import web
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import KafkaError
from aiokafka.structs import OffsetAndMetadata
from pydantic import ValidationError

from omnibase_core.errors import OnexError
from omnibase_infra.event_bus.consumer_health_emitter import ConsumerHealthEmitter
from omnibase_infra.event_bus.mixin_consumer_health import MixinConsumerHealth
from omnibase_infra.services.observability.delegation_projection.config import (
    ConfigDelegationProjection,
)
from omnibase_infra.services.observability.delegation_projection.writer_postgres import (
    WriterDelegationProjectionPostgres,
)

if TYPE_CHECKING:
    from aiokafka.structs import ConsumerRecord

logger = logging.getLogger(__name__)

_FATAL_COMMIT_ERROR_NAMES: frozenset[str] = frozenset(
    {
        "UnknownMemberIdError",
        "RebalanceInProgressError",
        "IllegalGenerationError",
        "FencedInstanceIdError",
    }
)

_CONSECUTIVE_TIMEOUT_LOG_INTERVAL: int = 5


class EnumHealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class HealthSnapshot:
    last_poll_at: datetime | None
    last_successful_write_at: datetime | None
    messages_received: int
    started_at: datetime


class ConsumerMetrics:
    """Metrics tracking for the delegation projection consumer."""

    def __init__(self) -> None:
        self.messages_received: int = 0
        self.messages_processed: int = 0
        self.messages_failed: int = 0
        self.messages_skipped: int = 0
        self.batches_processed: int = 0
        self.rows_written: int = 0
        self.consecutive_commit_failures: int = 0
        self.last_poll_at: datetime | None = None
        self.last_successful_write_at: datetime | None = None
        self.last_commit_failure_at: datetime | None = None
        self.started_at: datetime = datetime.now(UTC)
        self._lock = asyncio.Lock()

    async def record_received(self, count: int = 1) -> None:
        async with self._lock:
            self.messages_received += count
            self.last_poll_at = datetime.now(UTC)

    async def record_processed(self, count: int = 1) -> None:
        async with self._lock:
            self.messages_processed += count
            self.last_successful_write_at = datetime.now(UTC)

    async def record_rows_written(self, count: int = 1) -> None:
        async with self._lock:
            self.rows_written += count

    async def record_failed(self, count: int = 1) -> None:
        async with self._lock:
            self.messages_failed += count

    async def record_skipped(self, count: int = 1) -> None:
        async with self._lock:
            self.messages_skipped += count

    async def record_batch_processed(self) -> None:
        async with self._lock:
            self.batches_processed += 1

    async def record_polled(self) -> None:
        async with self._lock:
            self.last_poll_at = datetime.now(UTC)

    async def record_commit_failure(self) -> None:
        async with self._lock:
            self.consecutive_commit_failures += 1
            self.last_commit_failure_at = datetime.now(UTC)

    async def reset_consecutive_commit_failures(self) -> None:
        async with self._lock:
            self.consecutive_commit_failures = 0

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            return {
                "messages_received": self.messages_received,
                "messages_processed": self.messages_processed,
                "messages_failed": self.messages_failed,
                "messages_skipped": self.messages_skipped,
                "batches_processed": self.batches_processed,
                "rows_written": self.rows_written,
                "consecutive_commit_failures": self.consecutive_commit_failures,
                "last_poll_at": self.last_poll_at.isoformat()
                if self.last_poll_at
                else None,
                "last_successful_write_at": (
                    self.last_successful_write_at.isoformat()
                    if self.last_successful_write_at
                    else None
                ),
                "started_at": self.started_at.isoformat(),
            }

    async def health_snapshot(self) -> HealthSnapshot:
        async with self._lock:
            return HealthSnapshot(
                last_poll_at=self.last_poll_at,
                last_successful_write_at=self.last_successful_write_at,
                messages_received=self.messages_received,
                started_at=self.started_at,
            )


class ServiceDelegationProjectionConsumer(MixinConsumerHealth):
    """Async Kafka consumer for delegation projection.

    Consumes task-delegated events and UPSERTs them into delegation_events.
    """

    def __init__(self, config: ConfigDelegationProjection) -> None:
        self._config = config
        self._consumer: AIOKafkaConsumer | None = None
        self._pool: asyncpg.Pool | None = None
        self._writer: WriterDelegationProjectionPostgres | None = None
        self._health_producer: AIOKafkaProducer | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        self._health_app: web.Application | None = None
        self._health_runner: web.AppRunner | None = None
        self._health_site: web.TCPSite | None = None

        self.metrics = ConsumerMetrics()
        self._consumer_id = f"delegation-projection-{uuid4().hex[:8]}"

        logger.info(
            "ServiceDelegationProjectionConsumer initialized",
            extra={
                "consumer_id": self._consumer_id,
                "topics": self._config.topics,
                "group_id": self._config.kafka_group_id,
            },
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    async def start(self) -> None:
        if self._running:
            return

        correlation_id = uuid4()
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self._config.postgres_dsn,
                min_size=self._config.pool_min_size,
                max_size=self._config.pool_max_size,
            )

            self._writer = WriterDelegationProjectionPostgres(
                pool=self._pool,
                circuit_breaker_threshold=self._config.circuit_breaker_threshold,
                circuit_breaker_reset_timeout=self._config.circuit_breaker_reset_timeout,
                circuit_breaker_half_open_successes=self._config.circuit_breaker_half_open_successes,
            )

            self._consumer = AIOKafkaConsumer(
                *self._config.topics,
                bootstrap_servers=self._config.kafka_bootstrap_servers,
                group_id=self._config.kafka_group_id,
                auto_offset_reset=self._config.auto_offset_reset,
                enable_auto_commit=False,
                session_timeout_ms=self._config.session_timeout_ms,
                heartbeat_interval_ms=self._config.heartbeat_interval_ms,
                max_poll_interval_ms=self._config.max_poll_interval_ms,
                max_poll_records=self._config.batch_size,
            )

            await self._consumer.start()
            await self._start_health_server()

            self._running = True
            self._shutdown_event.clear()

            if ConsumerHealthEmitter.is_enabled():
                self._health_producer = AIOKafkaProducer(
                    bootstrap_servers=self._config.kafka_bootstrap_servers,
                )
                await self._health_producer.start()
                self._init_health_emitter(
                    self._health_producer,
                    consumer_identity="delegation-projection-consumer",
                    consumer_group=self._config.kafka_group_id,
                    topic=",".join(self._config.topics),
                    service_label="ServiceDelegationProjectionConsumer",
                )

            logger.info(
                "ServiceDelegationProjectionConsumer started",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                },
            )

        except Exception as e:
            logger.exception(
                "Failed to start consumer",
                extra={"consumer_id": self._consumer_id, "error": str(e)},
            )
            await self._cleanup_resources(correlation_id)
            raise

    async def stop(self) -> None:
        if not self._running:
            return

        correlation_id = uuid4()
        self._running = False
        self._shutdown_event.set()
        await self._cleanup_resources(correlation_id)

        metrics_snapshot = await self.metrics.snapshot()
        logger.info(
            "ServiceDelegationProjectionConsumer stopped",
            extra={
                "consumer_id": self._consumer_id,
                "correlation_id": str(correlation_id),
                "final_metrics": metrics_snapshot,
            },
        )

    async def _cleanup_resources(self, correlation_id: UUID) -> None:
        if self._health_site is not None:
            await self._health_site.stop()
            self._health_site = None

        if self._health_runner is not None:
            await self._health_runner.cleanup()
            self._health_runner = None

        self._health_app = None

        if self._health_producer is not None:
            try:
                await self._health_producer.stop()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Error stopping health producer",
                    extra={"consumer_id": self._consumer_id},
                )
            finally:
                self._health_producer = None

        if self._consumer is not None:
            try:
                await self._consumer.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error stopping Kafka consumer",
                    extra={"consumer_id": self._consumer_id, "error": str(e)},
                )
            finally:
                self._consumer = None

        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error closing PostgreSQL pool",
                    extra={"consumer_id": self._consumer_id, "error": str(e)},
                )
            finally:
                self._pool = None

        self._writer = None

    async def run(self) -> None:
        if not self._running or self._consumer is None:
            raise OnexError("Consumer not started. Call start() before run().")
        await self._consume_loop(uuid4())

    async def __aenter__(self) -> ServiceDelegationProjectionConsumer:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.stop()

    async def _consume_loop(self, correlation_id: UUID) -> None:
        if self._consumer is None:
            return

        batch_timeout_seconds = self._config.batch_timeout_ms / 1000.0
        consecutive_timeouts: int = 0

        try:
            while self._running:
                try:
                    records = await asyncio.wait_for(
                        self._consumer.getmany(
                            timeout_ms=self._config.batch_timeout_ms,
                            max_records=self._config.batch_size,
                        ),
                        timeout=batch_timeout_seconds
                        + self._config.poll_timeout_buffer_seconds,
                    )
                except TimeoutError:
                    consecutive_timeouts += 1
                    if consecutive_timeouts % _CONSECUTIVE_TIMEOUT_LOG_INTERVAL == 0:
                        logger.warning(
                            "Kafka getmany() timed out %d consecutive times",
                            consecutive_timeouts,
                            extra={"consumer_id": self._consumer_id},
                        )
                    continue

                consecutive_timeouts = 0
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
                "Consume loop cancelled", extra={"consumer_id": self._consumer_id}
            )
            raise
        except KafkaError as e:
            logger.exception(
                "Kafka error in consume loop",
                extra={"consumer_id": self._consumer_id, "error": str(e)},
            )
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error in consume loop",
                extra={"consumer_id": self._consumer_id, "error": str(e)},
            )
            raise
        finally:
            logger.info(
                "Consume loop exiting", extra={"consumer_id": self._consumer_id}
            )

    @staticmethod
    def _track_skipped_offset(
        skipped_offsets: dict[TopicPartition, int],
        msg: ConsumerRecord,
    ) -> None:
        tp = TopicPartition(msg.topic, msg.partition)
        current = skipped_offsets.get(tp, -1)
        skipped_offsets[tp] = max(current, msg.offset)

    async def _process_batch(
        self,
        messages: list[ConsumerRecord],
        correlation_id: UUID,
    ) -> dict[TopicPartition, int]:
        if self._writer is None:
            return {}

        successful_offsets: dict[TopicPartition, int] = {}
        skipped_offsets: dict[TopicPartition, int] = {}
        parsed_skipped: int = 0

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
            for tp, offset in skipped_offsets.items():
                current = successful_offsets.get(tp, -1)
                successful_offsets[tp] = max(current, offset)
            return successful_offsets

        event_dicts = [ev for _, ev in parsed_events]

        try:
            rows_written = await self._writer.write_events(event_dicts, correlation_id)

            for msg, _ in parsed_events:
                tp = TopicPartition(msg.topic, msg.partition)
                current = successful_offsets.get(tp, -1)
                successful_offsets[tp] = max(current, msg.offset)

            await self.metrics.record_processed(len(parsed_events))
            await self.metrics.record_rows_written(rows_written)

        except Exception:
            logger.exception(
                "Failed to write delegation_events batch",
                extra={
                    "consumer_id": self._consumer_id,
                    "correlation_id": str(correlation_id),
                    "count": len(event_dicts),
                },
            )
            await self.metrics.record_failed(len(event_dicts))

        for tp, offset in skipped_offsets.items():
            current = successful_offsets.get(tp, -1)
            successful_offsets[tp] = max(current, offset)

        return successful_offsets

    async def _commit_offsets(
        self,
        offsets: dict[TopicPartition, int],
        correlation_id: UUID,
    ) -> None:
        if not offsets or self._consumer is None:
            return

        commit_map: dict[TopicPartition, OffsetAndMetadata] = {
            tp: OffsetAndMetadata(offset + 1, "") for tp, offset in offsets.items()
        }

        try:
            await self._consumer.commit(commit_map)
            await self.metrics.reset_consecutive_commit_failures()

        except KafkaError as exc:
            await self.metrics.record_commit_failure()
            error_name = type(exc).__name__
            is_fatal = error_name in _FATAL_COMMIT_ERROR_NAMES

            if is_fatal:
                logger.exception(
                    "Fatal commit error (%s) -- consumer must rejoin group",
                    error_name,
                    extra={"consumer_id": self._consumer_id},
                )
                raise
            logger.warning(
                "Retriable commit error (%s), will retry on next batch",
                error_name,
                exc_info=True,
                extra={"consumer_id": self._consumer_id},
            )

    async def _start_health_server(self) -> None:
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext

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
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="start_health_server",
            )
            raise InfraConnectionError(
                f"Health check port {port} already in use (host={host})",
                context=context,
            ) from exc

    async def _determine_health_status(self) -> EnumHealthStatus:
        if not self._running:
            return EnumHealthStatus.UNHEALTHY

        circuit_state = self._writer.get_circuit_breaker_state() if self._writer else {}
        if circuit_state.get("state") in ("open", "half_open"):
            return EnumHealthStatus.DEGRADED

        snap = await self.metrics.health_snapshot()
        now = datetime.now(UTC)

        if snap.last_poll_at is not None:
            poll_age = (now - snap.last_poll_at).total_seconds()
            if poll_age > self._config.health_check_poll_staleness_seconds:
                return EnumHealthStatus.DEGRADED

        if snap.last_successful_write_at is None:
            age = (now - snap.started_at).total_seconds()
            if age <= self._config.startup_grace_period_seconds:
                return EnumHealthStatus.HEALTHY
            return EnumHealthStatus.DEGRADED

        write_age = (now - snap.last_successful_write_at).total_seconds()
        if (
            write_age > self._config.health_check_staleness_seconds
            and snap.messages_received > 0
        ):
            return EnumHealthStatus.DEGRADED

        return EnumHealthStatus.HEALTHY

    async def _health_handler(self, request: web.Request) -> web.Response:
        metrics_snapshot = await self.metrics.snapshot()
        circuit_state = self._writer.get_circuit_breaker_state() if self._writer else {}
        status = await self._determine_health_status()

        body = {
            "status": status.value,
            "consumer_running": self._running,
            "consumer_id": self._consumer_id,
            "last_poll_time": metrics_snapshot.get("last_poll_at"),
            "last_successful_write": metrics_snapshot.get("last_successful_write_at"),
            "circuit_breaker_state": circuit_state.get("state", "unknown"),
            "messages_processed": metrics_snapshot.get("messages_processed", 0),
            "messages_failed": metrics_snapshot.get("messages_failed", 0),
            "rows_written": metrics_snapshot.get("rows_written", 0),
        }
        http_status = 200 if status != EnumHealthStatus.UNHEALTHY else 503
        return web.json_response(body, status=http_status)

    async def _liveness_handler(self, request: web.Request) -> web.Response:
        body = {
            "status": "alive" if self._running else "dead",
            "consumer_id": self._consumer_id,
        }
        return web.json_response(body, status=200 if self._running else 503)

    async def _readiness_handler(self, request: web.Request) -> web.Response:
        deps = {
            "postgres_pool": self._pool is not None,
            "kafka_consumer": self._consumer is not None,
            "writer": self._writer is not None,
        }
        circuit_state = self._writer.get_circuit_breaker_state() if self._writer else {}
        deps["circuit_breaker"] = circuit_state.get("state") != "open"
        all_ready = all(deps.values()) and self._running

        body = {
            "status": "ready" if all_ready else "not_ready",
            "consumer_id": self._consumer_id,
            "consumer_running": self._running,
            "dependencies": deps,
        }
        return web.json_response(body, status=200 if all_ready else 503)

    async def health_check(self) -> dict[str, object]:
        metrics_snapshot = await self.metrics.snapshot()
        circuit_state = self._writer.get_circuit_breaker_state() if self._writer else {}
        status = await self._determine_health_status()
        return {
            "status": status.value,
            "consumer_running": self._running,
            "consumer_id": self._consumer_id,
            "group_id": self._config.kafka_group_id,
            "topics": self._config.topics,
            "circuit_breaker_state": circuit_state,
            "metrics": metrics_snapshot,
        }


async def _main() -> None:
    from omnibase_infra.utils.util_consumer_restart import run_with_restart

    try:
        ConfigDelegationProjection()
    except ValidationError as exc:
        missing = [str(e["loc"][-1]) for e in exc.errors() if e["type"] == "missing"]
        prefix = "OMNIBASE_INFRA_DELEGATION_"
        if missing:
            env_vars = ", ".join(f"{prefix}{f.upper()}" for f in missing)
            print(
                f"ERROR: Missing required configuration. "
                f"Set the following environment variable(s): {env_vars}",
                file=sys.stderr,
            )
        else:
            print(f"ERROR: Invalid configuration: {exc}", file=sys.stderr)
        sys.exit(1)

    shutdown_event = asyncio.Event()
    active_consumer: list[ServiceDelegationProjectionConsumer | None] = [None]

    def _on_signal() -> None:
        shutdown_event.set()
        if active_consumer[0] is not None:
            asyncio.get_running_loop().create_task(active_consumer[0].stop())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal)

    async def _run_once() -> None:
        config = ConfigDelegationProjection()
        logger.info(
            "Starting delegation projection consumer",
            extra={
                "topics": config.topics,
                "bootstrap_servers": config.kafka_bootstrap_servers,
                "group_id": config.kafka_group_id,
                "health_port": config.health_check_port,
            },
        )
        consumer = ServiceDelegationProjectionConsumer(config)
        active_consumer[0] = consumer
        try:
            await consumer.start()
            await consumer.run()
        finally:
            active_consumer[0] = None
            try:
                await asyncio.wait_for(consumer.stop(), timeout=10.0)
            except TimeoutError:
                logger.warning("consumer.stop() timed out after 10s")

    await run_with_restart(
        _run_once,
        name="ServiceDelegationProjectionConsumer",
        shutdown_event=shutdown_event,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(_main())


__all__ = [
    "ConsumerMetrics",
    "EnumHealthStatus",
    "HealthSnapshot",
    "ServiceDelegationProjectionConsumer",
]
