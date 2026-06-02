# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reusable DLQ replay engine for the contract-native replay node (OMN-12619).

This module owns the Kafka I/O surface and the replay eligibility predicate.
It is the relocated, importable home of the engine that previously lived in
``scripts/dlq_replay.py`` (``DLQConsumer``, ``DLQProducer``, ``should_replay``,
``generate_replay_correlation_id``). The CLI script now imports from here, and
the node handler drives this engine end-to-end.

What is net-new for OMN-12619:
    - ``DLQQuarantineProducer``: publishes non-replayable messages to
      ``onex.dlq.omnibase-infra.quarantine.v1`` instead of the legacy skip-and-drop path.

Eligibility (``should_replay``) is unchanged from the legacy implementation —
it is not reimplemented; it is moved.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, KafkaError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums import EnumNonRetryableErrorCategory
from omnibase_infra.event_bus.topic_constants import build_dlq_topic
from omnibase_infra.nodes.node_dlq_replay_effect.models.enum_dlq_replay_filter_type import (
    EnumDlqReplayFilterType,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.utils.util_datetime import is_timezone_aware

logger = logging.getLogger(__name__)

# Centralized non-retryable error types (consistent with event_bus_kafka.py).
NON_RETRYABLE_ERRORS = EnumNonRetryableErrorCategory.get_all_values()

# Consumer-group identity for the persistent replay consumer. The runtime kernel
# reads ONEX_GROUP_ID (service_kernel.py:639); this default documents the value
# the dlq-replay-consumer service must set. Replaces the legacy ephemeral
# "dlq-replay-{pid}" group that left no durable read position.
DLQ_REPLAY_CONSUMER_GROUP: str = "onex-dlq-replay"


def generate_replay_correlation_id() -> UUID:
    """Mint a new correlation ID for a replay/quarantine outcome."""
    return uuid4()


def parse_datetime_with_timezone(dt_string: str) -> datetime:
    """Parse an ISO-8601 string, normalising 'Z' and assuming UTC if naive."""
    normalized = dt_string.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if not is_timezone_aware(dt):
        dt = dt.replace(tzinfo=UTC)
    return dt


def safe_truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate text to max_chars (characters, not bytes) with a suffix."""
    if len(text) <= max_chars:
        return text
    suffix_len = len(suffix)
    if max_chars <= suffix_len:
        return suffix[:max_chars]
    return text[: max_chars - suffix_len] + suffix


def sanitize_bootstrap_servers(servers: str) -> str:
    """Return host:port entries safe for logging, or '[redacted]' if unexpected."""
    if "@" in servers or "://" in servers:
        return "[redacted]"
    try:
        sanitized_parts = []
        for raw in servers.split(","):
            part = raw.strip()
            if ":" in part:
                host, port = part.rsplit(":", 1)
                if port.isdigit():
                    sanitized_parts.append(f"{host}:{port}")
                else:
                    return "[redacted]"
            else:
                sanitized_parts.append(part)
        return ",".join(sanitized_parts)
    except Exception:  # noqa: BLE001 — boundary: returns degraded response
        return "[redacted]"


class ModelDlqReplayEngineConfig(BaseModel):
    """Runtime configuration for the DLQ replay engine.

    Carries the eligibility predicate inputs (mirrors the legacy
    ``ModelReplayConfig`` fields that ``should_replay`` reads) plus Kafka
    connection details. ``bootstrap_servers`` and ``consumer_group`` are
    required; there is no silent default for the broker address.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bootstrap_servers: str = Field(..., min_length=1)
    dlq_topic: str = Field(..., min_length=1)
    quarantine_topic: str = Field(
        default_factory=lambda: build_dlq_topic("quarantine"), min_length=1
    )
    consumer_group: str = Field(default=DLQ_REPLAY_CONSUMER_GROUP, min_length=1)
    max_replay_count: int = Field(default=5, gt=0)
    rate_limit_per_second: float = Field(default=100.0, gt=0.0)
    dry_run: bool = Field(default=False)
    filter_type: EnumDlqReplayFilterType = Field(default=EnumDlqReplayFilterType.ALL)
    filter_topics: tuple[str, ...] = Field(default=())
    filter_error_types: tuple[str, ...] = Field(default=())
    filter_correlation_ids: tuple[UUID, ...] = Field(default=())
    filter_start_time: datetime | None = Field(default=None)
    filter_end_time: datetime | None = Field(default=None)
    add_replay_headers: bool = Field(default=True)
    limit: int | None = Field(default=None, gt=0)
    max_request_size: int = Field(default=10485760, gt=0)
    request_timeout_ms: int = Field(default=30000, gt=0)

    @field_validator("bootstrap_servers")
    @classmethod
    def _bootstrap_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "bootstrap_servers cannot be empty; provide a Kafka broker address"
            )
        return stripped


def should_replay(
    message: ModelDlqMessage, config: ModelDlqReplayEngineConfig
) -> tuple[bool, str]:
    """Eligibility predicate — relocated unchanged from scripts/dlq_replay.py.

    Evaluation order: max-retry count, non-retryable error type, time-range
    (orthogonal), then the type-specific filter. Returns (eligible, reason).
    """
    if message.retry_count >= config.max_replay_count:
        return (
            False,
            f"Exceeded max replay count: {message.retry_count} >= {config.max_replay_count}",
        )

    if message.error_type in NON_RETRYABLE_ERRORS:
        return (False, f"Non-retryable error type: {message.error_type}")

    if config.filter_start_time or config.filter_end_time:
        try:
            failure_dt = parse_datetime_with_timezone(message.failure_timestamp)
            if config.filter_start_time and failure_dt < config.filter_start_time:
                return (False, f"Before start time: {config.filter_start_time}")
            if config.filter_end_time and failure_dt > config.filter_end_time:
                return (False, f"After end time: {config.filter_end_time}")
        except ValueError as exc:
            logger.warning(
                "Failed to parse failure_timestamp, skipping time filter",
                extra={
                    "correlation_id": str(message.correlation_id),
                    "failure_timestamp": message.failure_timestamp,
                    "parse_error": str(exc),
                },
            )

    if config.filter_type == EnumDlqReplayFilterType.BY_TOPIC:
        if message.original_topic not in config.filter_topics:
            return (False, f"Topic not in filter: {message.original_topic}")
    elif config.filter_type == EnumDlqReplayFilterType.BY_ERROR_TYPE:
        if message.error_type not in config.filter_error_types:
            return (False, f"Error type not in filter: {message.error_type}")
    elif config.filter_type == EnumDlqReplayFilterType.BY_CORRELATION_ID:
        if message.correlation_id not in config.filter_correlation_ids:
            return (False, f"Correlation ID not in filter: {message.correlation_id}")
    elif config.filter_type == EnumDlqReplayFilterType.BY_TIME_RANGE:
        # Time windows are applied orthogonally above; this branch makes the
        # time-range-only filter explicit in logs and future maintenance.
        return (True, "Eligible for replay within time range")

    return (True, "Eligible for replay")


class DLQConsumer:
    """Reads and parses messages from a DLQ topic with a persistent group."""

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config
        self._consumer: AIOKafkaConsumer | None = None
        self._started = False

    async def start(self) -> None:
        logger.info(
            "Starting DLQ consumer for topic: %s",
            self.config.dlq_topic,
            extra={
                "bootstrap_servers": sanitize_bootstrap_servers(
                    self.config.bootstrap_servers
                ),
                "consumer_group": self.config.consumer_group,
            },
        )
        try:
            self._consumer = AIOKafkaConsumer(
                self.config.dlq_topic,
                bootstrap_servers=self.config.bootstrap_servers,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                group_id=self.config.consumer_group,
                consumer_timeout_ms=5000,
            )
            await self._consumer.start()
            self._started = True
        except KafkaConnectionError:
            logger.exception("Failed to connect DLQ consumer to Kafka")
            raise
        except KafkaError:
            logger.exception("Kafka error during DLQ consumer start")
            raise

    async def stop(self) -> None:
        if self._started and self._consumer is not None:
            try:
                await self._consumer.stop()
            except KafkaError as exc:
                logger.warning("Error stopping DLQ consumer: %s", exc)
            finally:
                self._started = False
                self._consumer = None

    async def commit(self) -> None:
        """Commit consumed offsets — only call after durable handling."""
        if self._started and self._consumer is not None:
            await self._consumer.commit()

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        if not self._started or self._consumer is None:
            raise RuntimeError("Consumer not started")
        try:
            async for msg in self._consumer:
                if msg.value is None:
                    logger.warning("Null DLQ message value at offset %s", msg.offset)
                    continue
                try:
                    decoded_value = msg.value.decode("utf-8")
                except UnicodeDecodeError:
                    decoded_value = msg.value.decode("utf-8", errors="replace")
                try:
                    payload = json.loads(decoded_value)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Failed to parse DLQ message at offset %s: %s", msg.offset, exc
                    )
                    continue
                yield ModelDlqMessage.from_kafka_message(
                    payload=payload,
                    dlq_offset=msg.offset,
                    dlq_partition=msg.partition,
                )
        except asyncio.CancelledError:
            logger.info("DLQ consumption cancelled")
            raise
        except KafkaError:
            logger.exception("Kafka error during DLQ consumption")
            raise


class DLQProducer:
    """Replays messages back to their original topic with replay headers."""

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config
        self._producer: AIOKafkaProducer | None = None
        self._started = False
        self._last_publish: datetime | None = None
        self._interval = 1.0 / config.rate_limit_per_second

    async def start(self) -> None:
        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                acks="all",
                enable_idempotence=True,
                max_request_size=self.config.max_request_size,
                request_timeout_ms=self.config.request_timeout_ms,
            )
            await self._producer.start()
            self._started = True
        except KafkaConnectionError:
            logger.exception("Failed to connect DLQ producer to Kafka")
            raise
        except KafkaError:
            logger.exception("Kafka error during DLQ producer start")
            raise

    async def stop(self) -> None:
        if self._started and self._producer is not None:
            try:
                await self._producer.stop()
            except KafkaError as exc:
                logger.warning("Error stopping DLQ producer: %s", exc)
            finally:
                self._started = False
                self._producer = None

    async def replay_message(
        self, message: ModelDlqMessage, replay_correlation_id: UUID
    ) -> None:
        """Replay a message to its original topic. Raises on publish failure."""
        if not self._started or self._producer is None:
            raise RuntimeError("Producer not started")

        if self._last_publish is not None:
            elapsed = (datetime.now(UTC) - self._last_publish).total_seconds()
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)

        headers: list[tuple[str, bytes]] = []
        if self.config.add_replay_headers:
            headers = [
                ("x-replay-count", str(message.retry_count + 1).encode("utf-8")),
                ("x-replayed-at", datetime.now(UTC).isoformat().encode("utf-8")),
                ("x-replayed-by", b"node_dlq_replay_effect"),
                ("x-original-dlq-offset", str(message.dlq_offset).encode("utf-8")),
                ("x-replay-correlation-id", str(replay_correlation_id).encode("utf-8")),
                ("correlation_id", str(message.correlation_id).encode("utf-8")),
            ]

        key = (
            message.original_key.encode("utf-8", errors="replace")
            if message.original_key
            else None
        )
        value = message.original_value.encode("utf-8", errors="replace")

        await self._producer.send_and_wait(
            message.original_topic, value=value, key=key, headers=headers
        )
        self._last_publish = datetime.now(UTC)


class DLQQuarantineProducer:
    """Publishes non-replayable messages to onex.dlq.omnibase-infra.quarantine.v1 (OMN-12619).

    This is the durable replacement for the legacy skip-and-drop path. A
    non-replayable message is wrapped with the quarantine decision (reason,
    original DLQ coordinates, and a quarantine correlation ID) and published so
    a quarantine owner can later reclassify or re-enter it. Publish failures
    propagate so the caller never records a false QUARANTINED success.
    """

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config
        self._producer: AIOKafkaProducer | None = None
        self._started = False

    async def start(self) -> None:
        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                acks="all",
                enable_idempotence=True,
                max_request_size=self.config.max_request_size,
                request_timeout_ms=self.config.request_timeout_ms,
            )
            await self._producer.start()
            self._started = True
        except KafkaError:
            logger.exception("Kafka error during quarantine producer start")
            raise

    async def stop(self) -> None:
        if self._started and self._producer is not None:
            try:
                await self._producer.stop()
            except KafkaError as exc:
                logger.warning("Error stopping quarantine producer: %s", exc)
            finally:
                self._started = False
                self._producer = None

    @staticmethod
    def build_quarantine_payload(
        message: ModelDlqMessage,
        reason: str,
        quarantine_correlation_id: UUID,
        source_dlq_topic: str,
    ) -> dict[str, object]:
        """Build the durable quarantine record payload."""
        return {
            "quarantine_correlation_id": str(quarantine_correlation_id),
            "quarantined_at": datetime.now(UTC).isoformat(),
            "quarantined_by": "node_dlq_replay_effect",
            "reason": reason,
            "original_topic": message.original_topic,
            "original_correlation_id": str(message.correlation_id),
            "error_type": message.error_type,
            "retry_count": message.retry_count,
            "source_dlq_topic": source_dlq_topic,
            "source_dlq_offset": message.dlq_offset,
            "source_dlq_partition": message.dlq_partition,
            "original_payload": message.raw_payload,
        }

    async def quarantine_message(
        self,
        message: ModelDlqMessage,
        reason: str,
        quarantine_correlation_id: UUID,
    ) -> None:
        """Publish a non-replayable message to the quarantine topic.

        Raises on publish failure; the caller must NOT record QUARANTINED
        success unless this returns normally.
        """
        if not self._started or self._producer is None:
            raise RuntimeError("Quarantine producer not started")

        payload = self.build_quarantine_payload(
            message, reason, quarantine_correlation_id, self.config.dlq_topic
        )
        headers: list[tuple[str, bytes]] = [
            ("x-quarantine-reason", reason.encode("utf-8")),
            (
                "x-quarantine-correlation-id",
                str(quarantine_correlation_id).encode("utf-8"),
            ),
            ("correlation_id", str(message.correlation_id).encode("utf-8")),
        ]
        key = str(message.correlation_id).encode("utf-8")
        value = json.dumps(payload).encode("utf-8")
        await self._producer.send_and_wait(
            self.config.quarantine_topic, value=value, key=key, headers=headers
        )


__all__ = [
    "DLQ_REPLAY_CONSUMER_GROUP",
    "DLQConsumer",
    "DLQProducer",
    "DLQQuarantineProducer",
    "ModelDlqReplayEngineConfig",
    "generate_replay_correlation_id",
    "parse_datetime_with_timezone",
    "safe_truncate",
    "sanitize_bootstrap_servers",
    "should_replay",
]
