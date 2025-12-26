#!/usr/bin/env python3
"""DLQ Replay Utility - Replay failed messages from Dead Letter Queue.

This script provides a command-line interface for replaying messages from
the Dead Letter Queue (DLQ) back to their original topics.

Usage:
    python scripts/dlq_replay.py --help
    python scripts/dlq_replay.py list --dlq-topic dlq-events
    python scripts/dlq_replay.py replay --dlq-topic dlq-events --dry-run
    python scripts/dlq_replay.py replay --dlq-topic dlq-events --filter-topic dev.orders
    python scripts/dlq_replay.py replay --dlq-topic dlq-events --start-time 2025-01-01T00:00:00Z
    python scripts/dlq_replay.py replay --dlq-topic dlq-events --enable-tracking

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker addresses (REQUIRED - no default)
    POSTGRES_HOST: PostgreSQL host for tracking (required if --enable-tracking)
    POSTGRES_PORT: PostgreSQL port (default: 5432)
    POSTGRES_DATABASE: PostgreSQL database name (default: omninode_bridge)
    POSTGRES_USER: PostgreSQL username (required if --enable-tracking)
    POSTGRES_PASSWORD: PostgreSQL password (required if --enable-tracking)

See Also:
    docs/operations/DLQ_REPLAY_RUNBOOK.md - Complete replay documentation
    OMN-949 - DLQ configuration ticket
    OMN-1032 - PostgreSQL tracking integration (integrated)

SPDX-License-Identifier: MIT
Copyright (c) 2025 OmniNode Team
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, KafkaError
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from omnibase_infra.dlq import (
    DLQTrackingService,
    EnumReplayStatus,
    ModelDlqReplayRecord,
    ModelDlqTrackingConfig,
)
from omnibase_infra.enums import EnumNonRetryableErrorCategory

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dlq_replay")


# =============================================================================
# Helper Functions
# =============================================================================


def sanitize_bootstrap_servers(servers: str) -> str:
    """Sanitize bootstrap servers for logging (remove potential credentials).

    Kafka bootstrap servers typically don't contain credentials, but some
    configurations or URL-like formats might. This function extracts just
    the host:port parts for safe logging.

    Args:
        servers: The bootstrap_servers connection string

    Returns:
        Sanitized string with just host:port entries, or "[redacted]" if
        the format is unexpected and might contain sensitive data.

    Example:
        >>> sanitize_bootstrap_servers("kafka:9092,kafka2:9092")
        'kafka:9092,kafka2:9092'
        >>> sanitize_bootstrap_servers("user:pass@kafka:9092")
        '[redacted]'
    """
    # Check for common credential patterns
    if "@" in servers or "://" in servers:
        # Might contain credentials - redact entirely
        return "[redacted]"

    # For standard host:port format, extract just the hosts
    try:
        parts = servers.split(",")
        sanitized_parts = []
        for part in parts:
            part = part.strip()
            # Validate it looks like host:port
            if ":" in part:
                host, port = part.rsplit(":", 1)
                # Ensure port is numeric
                if port.isdigit():
                    sanitized_parts.append(f"{host}:{port}")
                else:
                    return "[redacted]"
            else:
                # Just a hostname without port - allow it
                sanitized_parts.append(part)
        return ",".join(sanitized_parts)
    except Exception:
        return "[redacted]"


def safe_truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    """Safely truncate text to max_chars, handling multi-byte UTF-8 characters.

    This function ensures that truncation does not break multi-byte UTF-8
    characters by operating on string characters rather than bytes.

    Args:
        text: The text to truncate
        max_chars: Maximum number of characters (not bytes) to keep
        suffix: Suffix to append when truncating (default: "...")

    Returns:
        Truncated text with suffix if truncated, or original text if shorter

    Example:
        >>> safe_truncate("Hello World", 8)
        'Hello...'
        >>> safe_truncate("Hello", 10)
        'Hello'
    """
    if len(text) <= max_chars:
        return text
    # Reserve space for suffix
    suffix_len = len(suffix)
    if max_chars <= suffix_len:
        return suffix[:max_chars]
    return text[: max_chars - suffix_len] + suffix


def generate_replay_correlation_id() -> UUID:
    """Generate a new correlation ID for replay tracking.

    This function provides a single point of correlation ID generation
    for all replay outcomes (completed, failed, skipped), ensuring
    consistency in tracking.

    Returns:
        UUID: A new unique correlation ID for the replay attempt.
    """
    return uuid4()


# =============================================================================
# Enums and Models
# =============================================================================


class EnumFilterType(str, Enum):
    """Filter criteria for selective replay."""

    ALL = "all"
    BY_TOPIC = "by_topic"
    BY_ERROR_TYPE = "by_error_type"
    BY_TIME_RANGE = "by_time_range"
    BY_CORRELATION_ID = "by_correlation_id"


class ModelDlqMessage(BaseModel):
    """Parsed DLQ message with metadata."""

    original_topic: str
    original_key: str | None
    original_value: str
    original_offset: (
        str | None
    )  # Kafka offsets are int, converted to str in from_kafka_message
    original_partition: int | None
    failure_reason: str
    failure_timestamp: str
    correlation_id: UUID
    retry_count: int
    error_type: str
    dlq_offset: int
    dlq_partition: int
    raw_payload: dict[str, object]

    @classmethod
    def from_kafka_message(
        cls,
        payload: dict[str, object],
        dlq_offset: int,
        dlq_partition: int,
    ) -> ModelDlqMessage:
        """Parse DLQ message from Kafka payload.

        Raises:
            ValueError: If retry_count is present but not a valid integer.
        """
        original_message = payload.get("original_message", {})
        if not isinstance(original_message, dict):
            original_message = {}

        correlation_id_str = payload.get("correlation_id", "")
        try:
            correlation_id = UUID(str(correlation_id_str))
        except (ValueError, AttributeError):
            correlation_id = uuid4()

        # Explicit validation for retry_count instead of silent coercion
        retry_count = cls._parse_retry_count(payload.get("retry_count", 0))

        return cls(
            original_topic=str(payload.get("original_topic", "unknown")),
            original_key=original_message.get("key"),
            original_value=str(original_message.get("value", "")),
            # Convert offset to str for type consistency (Kafka offsets are int in JSON)
            original_offset=str(original_message["offset"])
            if "offset" in original_message and original_message["offset"] is not None
            else None,
            original_partition=original_message.get("partition"),
            failure_reason=str(payload.get("failure_reason", "")),
            failure_timestamp=str(payload.get("failure_timestamp", "")),
            correlation_id=correlation_id,
            retry_count=retry_count,
            error_type=str(payload.get("error_type", "Unknown")),
            dlq_offset=dlq_offset,
            dlq_partition=dlq_partition,
            raw_payload=payload,
        )

    @staticmethod
    def _parse_retry_count(value: object) -> int:
        """Parse retry_count with explicit validation.

        Args:
            value: The retry_count value from the payload.

        Returns:
            The parsed integer retry count.

        Raises:
            ValueError: If value cannot be parsed as a valid integer.
        """
        if isinstance(value, int):
            return value
        if value is None:
            return 0
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return 0
            try:
                return int(stripped)
            except ValueError as e:
                raise ValueError(
                    f"Invalid retry_count value: '{value}' is not a valid integer"
                ) from e
        raise ValueError(
            f"Invalid retry_count type: expected int or str, got {type(value).__name__}"
        )


class ModelReplayConfig(BaseModel):
    """Configuration for DLQ replay operation."""

    bootstrap_servers: str  # REQUIRED - no default, must be provided via env or CLI
    dlq_topic: str = "dlq-events"
    max_replay_count: int = 5
    rate_limit_per_second: float = 100.0
    dry_run: bool = False
    filter_type: EnumFilterType = EnumFilterType.ALL
    filter_topics: list[str] = Field(default_factory=list)
    filter_error_types: list[str] = Field(default_factory=list)
    filter_correlation_ids: list[UUID] = Field(default_factory=list)
    add_replay_headers: bool = True
    limit: int | None = None
    # Time-range filtering (OMN-1032)
    filter_start_time: datetime | None = None
    filter_end_time: datetime | None = None
    # PostgreSQL tracking configuration (OMN-1032)
    enable_tracking: bool = False
    postgres_host: str | None = None
    postgres_port: int = 5432
    postgres_database: str = "omninode_bridge"
    postgres_user: str | None = None
    postgres_password: str | None = None
    # Kafka producer settings - extracted from hardcoded values for configurability
    max_request_size: int = Field(
        default=10485760,  # 10MB
        description="Maximum size of a Kafka request in bytes (default: 10MB)",
    )
    request_timeout_ms: int = Field(
        default=30000,  # 30 seconds
        description="Kafka producer request timeout in milliseconds (default: 30s)",
    )

    @field_validator("rate_limit_per_second")
    @classmethod
    def validate_rate_limit(cls, v: float) -> float:
        """Validate rate_limit_per_second is positive to prevent division by zero."""
        if v <= 0:
            raise ValueError(
                f"rate_limit_per_second must be > 0, got {v}. "
                "A zero or negative rate limit would cause division by zero."
            )
        return v

    @field_validator("filter_end_time", mode="after")
    @classmethod
    def validate_time_range(
        cls, v: datetime | None, info: ValidationInfo
    ) -> datetime | None:
        """Validate that filter_end_time is after filter_start_time."""
        if v is not None and info.data:
            start_time = info.data.get("filter_start_time")
            if start_time is not None and v < start_time:
                raise ValueError(
                    f"filter_end_time ({v}) must be after filter_start_time ({start_time})"
                )
        return v

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ModelReplayConfig:
        """Create config from command line arguments.

        Raises:
            ValueError: If bootstrap_servers is not provided via env var or CLI.
        """
        # Bootstrap servers: env var takes precedence, then CLI arg, no default
        bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
        if bootstrap_servers is None:
            cli_bootstrap = getattr(args, "bootstrap_servers", None)
            if cli_bootstrap is not None:
                bootstrap_servers = cli_bootstrap
            else:
                raise ValueError(
                    "KAFKA_BOOTSTRAP_SERVERS environment variable or "
                    "--bootstrap-servers argument is required. "
                    "No default value is provided for security reasons."
                )

        filter_type = EnumFilterType.ALL
        filter_topics: list[str] = []
        filter_error_types: list[str] = []
        filter_correlation_ids: list[UUID] = []
        filter_start_time: datetime | None = None
        filter_end_time: datetime | None = None

        # Parse time filters first (can combine with other filters)
        start_time_str = getattr(args, "start_time", None)
        end_time_str = getattr(args, "end_time", None)

        if start_time_str:
            try:
                # Handle 'Z' suffix for UTC timezone
                filter_start_time = datetime.fromisoformat(
                    start_time_str.replace("Z", "+00:00")
                )
                # Ensure timezone-aware (assume UTC if not specified)
                if filter_start_time.tzinfo is None:
                    filter_start_time = filter_start_time.replace(tzinfo=UTC)
            except ValueError as e:
                raise ValueError(
                    f"Invalid start_time format: {start_time_str}. "
                    "Use ISO 8601 format (e.g., 2025-01-01T00:00:00Z)"
                ) from e

        if end_time_str:
            try:
                # Handle 'Z' suffix for UTC timezone
                filter_end_time = datetime.fromisoformat(
                    end_time_str.replace("Z", "+00:00")
                )
                # Ensure timezone-aware (assume UTC if not specified)
                if filter_end_time.tzinfo is None:
                    filter_end_time = filter_end_time.replace(tzinfo=UTC)
            except ValueError as e:
                raise ValueError(
                    f"Invalid end_time format: {end_time_str}. "
                    "Use ISO 8601 format (e.g., 2025-01-01T23:59:59Z)"
                ) from e

        # Determine filter type based on provided filters
        if args.filter_topic:
            filter_type = EnumFilterType.BY_TOPIC
            filter_topics = [args.filter_topic]
        elif args.filter_error_type:
            filter_type = EnumFilterType.BY_ERROR_TYPE
            filter_error_types = [args.filter_error_type]
        elif args.filter_correlation_id:
            filter_type = EnumFilterType.BY_CORRELATION_ID
            try:
                filter_correlation_ids = [UUID(args.filter_correlation_id)]
            except ValueError as e:
                raise ValueError(
                    f"Invalid correlation ID format: {args.filter_correlation_id}"
                ) from e
        elif filter_start_time or filter_end_time:
            # Only set BY_TIME_RANGE if no other filter is specified
            filter_type = EnumFilterType.BY_TIME_RANGE

        # Parse PostgreSQL tracking configuration
        enable_tracking = getattr(args, "enable_tracking", False)
        postgres_host = os.environ.get("POSTGRES_HOST")
        postgres_port_str = os.environ.get("POSTGRES_PORT", "5432")
        postgres_database = os.environ.get("POSTGRES_DATABASE", "omninode_bridge")
        postgres_user = os.environ.get("POSTGRES_USER")
        postgres_password = os.environ.get("POSTGRES_PASSWORD")

        try:
            postgres_port = int(postgres_port_str)
        except ValueError:
            postgres_port = 5432

        return cls(
            bootstrap_servers=bootstrap_servers,
            dlq_topic=args.dlq_topic,
            max_replay_count=args.max_replay_count,
            rate_limit_per_second=getattr(args, "rate_limit", 100.0),
            dry_run=getattr(args, "dry_run", False),
            filter_type=filter_type,
            filter_topics=filter_topics,
            filter_error_types=filter_error_types,
            filter_correlation_ids=filter_correlation_ids,
            filter_start_time=filter_start_time,
            filter_end_time=filter_end_time,
            enable_tracking=enable_tracking,
            postgres_host=postgres_host,
            postgres_port=postgres_port,
            postgres_database=postgres_database,
            postgres_user=postgres_user,
            postgres_password=postgres_password,
            limit=getattr(args, "limit", None),
        )

    def build_tracking_dsn(self) -> str | None:
        """Build PostgreSQL DSN from config fields.

        Returns:
            DSN string if all required fields are present and tracking is enabled,
            None otherwise.
        """
        if not self.enable_tracking:
            return None
        if not all([self.postgres_host, self.postgres_user, self.postgres_password]):
            return None
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )


class ModelReplayResult(BaseModel):
    """Result of a replay operation."""

    correlation_id: UUID
    original_topic: str
    status: EnumReplayStatus
    message: str
    replay_correlation_id: UUID | None = None


# =============================================================================
# Non-Retryable Error Types
# =============================================================================

# Use centralized enum for non-retryable error types.
# This ensures consistency with kafka_event_bus.py and other retry logic.
# See: src/omnibase_infra/enums/enum_non_retryable_error_category.py
# Related: OMN-1032
NON_RETRYABLE_ERRORS = EnumNonRetryableErrorCategory.get_all_values()


# =============================================================================
# DLQ Consumer
# =============================================================================


class DLQConsumer:
    """Consumer for reading DLQ messages using aiokafka.

    This consumer reads messages from the Dead Letter Queue topic and yields
    parsed ModelDlqMessage instances for processing.

    Attributes:
        config: Replay configuration containing bootstrap servers and topic
        _consumer: The underlying AIOKafkaConsumer instance
        _started: Whether the consumer has been started
    """

    def __init__(self, config: ModelReplayConfig) -> None:
        """Initialize DLQ consumer.

        Args:
            config: Replay configuration with Kafka connection details
        """
        self.config = config
        self._consumer: AIOKafkaConsumer | None = None
        self._started = False

    async def start(self) -> None:
        """Start the consumer and connect to Kafka.

        Raises:
            KafkaConnectionError: If unable to connect to Kafka brokers
            KafkaError: For other Kafka-related errors
        """
        logger.info(
            f"Starting DLQ consumer for topic: {self.config.dlq_topic}",
            extra={
                "bootstrap_servers": sanitize_bootstrap_servers(
                    self.config.bootstrap_servers
                )
            },
        )
        try:
            self._consumer = AIOKafkaConsumer(
                self.config.dlq_topic,
                bootstrap_servers=self.config.bootstrap_servers,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                group_id=f"dlq-replay-{os.getpid()}",
                consumer_timeout_ms=5000,
            )
            await self._consumer.start()
            self._started = True
            logger.info("DLQ consumer started successfully")
        except KafkaConnectionError:
            logger.exception(
                "Failed to connect to Kafka",
                extra={
                    "bootstrap_servers": sanitize_bootstrap_servers(
                        self.config.bootstrap_servers
                    )
                },
            )
            raise
        except KafkaError:
            logger.exception("Kafka error during consumer start")
            raise

    async def stop(self) -> None:
        """Stop the consumer and disconnect from Kafka."""
        if self._started and self._consumer is not None:
            try:
                await self._consumer.stop()
                logger.info("DLQ consumer stopped")
            except KafkaError as e:
                logger.warning(f"Error stopping consumer: {e}")
            finally:
                self._started = False
                self._consumer = None

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        """Consume and yield DLQ messages.

        Yields:
            ModelDlqMessage instances parsed from DLQ topic

        Raises:
            RuntimeError: If consumer has not been started
        """
        if not self._started or self._consumer is None:
            raise RuntimeError("Consumer not started")

        try:
            async for msg in self._consumer:
                try:
                    if msg.value is None:
                        logger.warning(
                            f"Received message with null value at offset {msg.offset}"
                        )
                        continue

                    # Decode with error handling for malformed UTF-8
                    try:
                        decoded_value = msg.value.decode("utf-8")
                    except UnicodeDecodeError:
                        logger.warning(
                            f"Failed to decode message at offset {msg.offset} as UTF-8, "
                            "using replacement characters"
                        )
                        decoded_value = msg.value.decode("utf-8", errors="replace")

                    payload = json.loads(decoded_value)
                    yield ModelDlqMessage.from_kafka_message(
                        payload=payload,
                        dlq_offset=msg.offset,
                        dlq_partition=msg.partition,
                    )
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse DLQ message at offset {msg.offset}: {e}"
                    )
                    continue
                except Exception as e:
                    logger.warning(
                        f"Error processing message at offset {msg.offset}: {e}"
                    )
                    continue
        except asyncio.CancelledError:
            logger.info("Message consumption cancelled")
            raise
        except KafkaError:
            logger.exception("Kafka error during message consumption")
            raise


# =============================================================================
# DLQ Producer
# =============================================================================


class DLQProducer:
    """Producer for replaying messages to original topics using aiokafka.

    This producer publishes DLQ messages back to their original topics
    with replay tracking headers and rate limiting support.

    Rate Limiting:
        Uses a token bucket approach that allows the first message to be sent
        immediately without delay. Subsequent messages are rate-limited based
        on the configured rate_limit_per_second.

    Attributes:
        config: Replay configuration containing bootstrap servers and rate limits
        _producer: The underlying AIOKafkaProducer instance
        _started: Whether the producer has been started
        _last_publish: Timestamp of last publish for rate limiting (None until first publish)
        _interval: Minimum interval between publishes
    """

    def __init__(self, config: ModelReplayConfig) -> None:
        """Initialize DLQ producer.

        Args:
            config: Replay configuration with Kafka connection details
        """
        self.config = config
        self._producer: AIOKafkaProducer | None = None
        self._started = False
        # Initialize to None to allow first message immediately (no delay)
        self._last_publish: datetime | None = None
        self._interval = 1.0 / config.rate_limit_per_second

    async def start(self) -> None:
        """Start the producer and connect to Kafka.

        Raises:
            KafkaConnectionError: If unable to connect to Kafka brokers
            KafkaError: For other Kafka-related errors
        """
        logger.info(
            "Starting DLQ producer",
            extra={
                "bootstrap_servers": sanitize_bootstrap_servers(
                    self.config.bootstrap_servers
                ),
                "max_request_size": self.config.max_request_size,
                "request_timeout_ms": self.config.request_timeout_ms,
            },
        )
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
            logger.info("DLQ producer started successfully")
        except KafkaConnectionError:
            logger.exception(
                "Failed to connect to Kafka",
                extra={
                    "bootstrap_servers": sanitize_bootstrap_servers(
                        self.config.bootstrap_servers
                    )
                },
            )
            raise
        except KafkaError:
            logger.exception("Kafka error during producer start")
            raise

    async def stop(self) -> None:
        """Stop the producer and disconnect from Kafka."""
        if self._started and self._producer is not None:
            try:
                await self._producer.stop()
                logger.info("DLQ producer stopped")
            except KafkaError as e:
                logger.warning(f"Error stopping producer: {e}")
            finally:
                self._started = False
                self._producer = None

    async def replay_message(
        self,
        message: ModelDlqMessage,
        replay_correlation_id: UUID,
    ) -> None:
        """Replay a message to its original topic with rate limiting.

        Args:
            message: DLQ message to replay
            replay_correlation_id: New correlation ID for replay tracking

        Raises:
            RuntimeError: If producer has not been started
            KafkaError: If unable to publish to Kafka
        """
        # TODO(OMN-1032): Integrate with PostgreSQL tracking for replay state persistence.
        # Currently stores replay state in memory only; state is lost on script restart.
        if not self._started or self._producer is None:
            raise RuntimeError("Producer not started")

        # Rate limiting - first message is sent immediately (no delay)
        if self._last_publish is not None:
            elapsed = (datetime.now(UTC) - self._last_publish).total_seconds()
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)

        # Build replay headers
        headers: list[tuple[str, bytes]] = []
        if self.config.add_replay_headers:
            headers = [
                ("x-replay-count", str(message.retry_count + 1).encode("utf-8")),
                ("x-replayed-at", datetime.now(UTC).isoformat().encode("utf-8")),
                ("x-replayed-by", b"dlq_replay_script"),
                ("x-original-dlq-offset", str(message.dlq_offset).encode("utf-8")),
                ("x-replay-correlation-id", str(replay_correlation_id).encode("utf-8")),
                ("correlation_id", str(message.correlation_id).encode("utf-8")),
            ]

        # Prepare key and value with UTF-8 encoding error handling
        key = None
        if message.original_key:
            try:
                key = message.original_key.encode("utf-8")
            except UnicodeEncodeError:
                logger.warning(
                    "Failed to encode message key as UTF-8, using replacement chars",
                    extra={"correlation_id": str(message.correlation_id)},
                )
                key = message.original_key.encode("utf-8", errors="replace")

        try:
            value = message.original_value.encode("utf-8")
        except UnicodeEncodeError:
            logger.warning(
                "Failed to encode message value as UTF-8, using replacement chars",
                extra={"correlation_id": str(message.correlation_id)},
            )
            value = message.original_value.encode("utf-8", errors="replace")

        try:
            await self._producer.send_and_wait(
                message.original_topic,
                value=value,
                key=key,
                headers=headers,
            )
            self._last_publish = datetime.now(UTC)
            logger.debug(
                f"Replayed message to {message.original_topic}",
                extra={
                    "correlation_id": str(message.correlation_id),
                    "replay_correlation_id": str(replay_correlation_id),
                },
            )
        except KafkaError:
            logger.exception(
                f"Failed to replay message to {message.original_topic}",
                extra={
                    "correlation_id": str(message.correlation_id),
                },
            )
            raise


# =============================================================================
# Replay Filter
# =============================================================================


def should_replay(
    message: ModelDlqMessage, config: ModelReplayConfig
) -> tuple[bool, str]:
    """Determine if a DLQ message should be replayed based on configured filters.

    This function evaluates a DLQ message against multiple filter criteria to
    determine replay eligibility. Filters are applied in a specific order with
    short-circuit evaluation.

    Filter Evaluation Order:
        1. **Max replay count**: Rejects messages that have exceeded the maximum
           retry attempts (fail-fast check).
        2. **Non-retryable errors**: Rejects messages with error types that are
           inherently non-retryable (e.g., schema validation errors).
        3. **Time range filter**: Applied ORTHOGONALLY to other filters - messages
           must fall within the configured time range regardless of filter_type.
        4. **Type-specific filter**: Applied based on config.filter_type (topic,
           error type, correlation ID, or no additional filter).

    Orthogonal Time Range Filtering:
        Time range filters (filter_start_time, filter_end_time) are applied
        independently of the filter_type setting. This means:

        - When filter_type=BY_TOPIC and time range is set:
          Message must match the topic filter AND fall within the time range.

        - When filter_type=BY_ERROR_TYPE and time range is set:
          Message must match the error type filter AND fall within the time range.

        - When filter_type=BY_TIME_RANGE (no other filter specified):
          Only the time range constraint is applied.

        This orthogonal design allows operators to combine time-based filtering
        with any other filter criterion for precise replay targeting.

    Example:
        To replay only connection errors from the last 24 hours::

            config = ModelReplayConfig(
                filter_type=EnumFilterType.BY_ERROR_TYPE,
                filter_error_types=["InfraConnectionError"],
                filter_start_time=datetime.now(UTC) - timedelta(hours=24),
                filter_end_time=datetime.now(UTC),
            )
            # A message is replayed only if:
            # 1. retry_count < max_replay_count
            # 2. error_type is not in NON_RETRYABLE_ERRORS
            # 3. failure_timestamp is within the last 24 hours (time range)
            # 4. error_type == "InfraConnectionError" (type-specific filter)

    Args:
        message: DLQ message to evaluate for replay eligibility.
        config: Replay configuration containing filter criteria.

    Returns:
        Tuple of (should_replay, reason) where:
            - should_replay: True if message passes all filter criteria
            - reason: Human-readable explanation of the decision
    """
    # Check max replay count
    if message.retry_count >= config.max_replay_count:
        return (
            False,
            f"Exceeded max replay count: {message.retry_count} >= {config.max_replay_count}",
        )

    # Check non-retryable error types
    if message.error_type in NON_RETRYABLE_ERRORS:
        return (False, f"Non-retryable error type: {message.error_type}")

    # Apply time-range filter (can apply regardless of filter_type)
    if config.filter_start_time or config.filter_end_time:
        try:
            # Parse message timestamp, handling 'Z' suffix for UTC
            failure_dt = datetime.fromisoformat(
                message.failure_timestamp.replace("Z", "+00:00")
            )
            if config.filter_start_time and failure_dt < config.filter_start_time:
                return (False, f"Before start time: {config.filter_start_time}")
            if config.filter_end_time and failure_dt > config.filter_end_time:
                return (False, f"After end time: {config.filter_end_time}")
        except ValueError as e:
            # If timestamp can't be parsed, log a warning and don't filter by time
            logger.warning(
                "Failed to parse failure_timestamp, skipping time filter",
                extra={
                    "correlation_id": str(message.correlation_id),
                    "failure_timestamp": message.failure_timestamp,
                    "parse_error": str(e),
                },
            )

    # Apply filters
    if config.filter_type == EnumFilterType.BY_TOPIC:
        if message.original_topic not in config.filter_topics:
            return (False, f"Topic not in filter: {message.original_topic}")

    elif config.filter_type == EnumFilterType.BY_ERROR_TYPE:
        if message.error_type not in config.filter_error_types:
            return (False, f"Error type not in filter: {message.error_type}")

    elif config.filter_type == EnumFilterType.BY_CORRELATION_ID:
        if message.correlation_id not in config.filter_correlation_ids:
            return (False, f"Correlation ID not in filter: {message.correlation_id}")

    # BY_TIME_RANGE is handled above (time filtering applies independently of filter_type)
    # Time filters are orthogonal and can combine with topic/error/correlation filters

    return (True, "Eligible for replay")


# =============================================================================
# Replay Executor
# =============================================================================


class DLQReplayExecutor:
    """Executor for DLQ replay operations."""

    def __init__(self, config: ModelReplayConfig) -> None:
        """Initialize replay executor."""
        self.config = config
        self.consumer = DLQConsumer(config)
        self.producer = DLQProducer(config)
        self.results: list[ModelReplayResult] = []
        self._tracking_service: DLQTrackingService | None = None

    async def start(self) -> None:
        """Start consumer, producer, and optionally tracking service."""
        await self.consumer.start()
        if not self.config.dry_run:
            await self.producer.start()

        # Initialize tracking if enabled
        dsn = self.config.build_tracking_dsn()
        if dsn:
            tracking_config = ModelDlqTrackingConfig(dsn=dsn)
            self._tracking_service = DLQTrackingService(tracking_config)
            try:
                await self._tracking_service.initialize()
                logger.info("DLQ tracking service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize tracking service: {e}")
                self._tracking_service = None

    async def stop(self) -> None:
        """Stop consumer, producer, and tracking service."""
        await self.consumer.stop()
        if not self.config.dry_run:
            await self.producer.stop()
        if self._tracking_service is not None:
            await self._tracking_service.shutdown()
            logger.info("DLQ tracking service stopped")

    async def _record_tracking(
        self,
        message: ModelDlqMessage,
        status: EnumReplayStatus,
        replay_correlation_id: UUID,
        error_message: str | None = None,
    ) -> None:
        """Record replay attempt to PostgreSQL if tracking is enabled.

        Args:
            message: The DLQ message being replayed.
            status: The replay status (from local EnumReplayStatus).
            replay_correlation_id: New correlation ID for replay tracking.
            error_message: Error details if replay failed.
        """
        if self._tracking_service is None:
            return

        try:
            record = ModelDlqReplayRecord(
                id=uuid4(),
                original_message_id=message.correlation_id,
                replay_correlation_id=replay_correlation_id,
                original_topic=message.original_topic,
                target_topic=message.original_topic,  # Same as original for replay
                replay_status=status,
                replay_timestamp=datetime.now(UTC),
                success=status == EnumReplayStatus.COMPLETED,
                error_message=error_message,
                dlq_offset=message.dlq_offset,
                dlq_partition=message.dlq_partition,
                retry_count=message.retry_count,
            )
            await self._tracking_service.record_replay_attempt(record)
        except Exception as e:
            logger.warning(f"Failed to record tracking: {e}")

    async def execute(self) -> list[ModelReplayResult]:
        """Execute the replay operation.

        Returns:
            List of replay results
        """
        count = 0
        limit = self.config.limit

        async for message in self.consumer.consume_messages():
            if limit is not None and count >= limit:
                break

            should, reason = should_replay(message, self.config)

            if not should:
                skip_correlation_id = generate_replay_correlation_id()
                result = ModelReplayResult(
                    correlation_id=message.correlation_id,
                    original_topic=message.original_topic,
                    status=EnumReplayStatus.SKIPPED,
                    message=reason,
                    replay_correlation_id=skip_correlation_id,
                )
                self.results.append(result)
                await self._record_tracking(
                    message, EnumReplayStatus.SKIPPED, skip_correlation_id, reason
                )
                logger.info(
                    f"SKIP: {message.correlation_id} - {reason}",
                    extra={
                        "original_topic": message.original_topic,
                        "error_type": message.error_type,
                    },
                )
                continue

            replay_correlation_id = generate_replay_correlation_id()

            if self.config.dry_run:
                result = ModelReplayResult(
                    correlation_id=message.correlation_id,
                    original_topic=message.original_topic,
                    status=EnumReplayStatus.PENDING,
                    message="DRY RUN - would replay",
                    replay_correlation_id=replay_correlation_id,
                )
                logger.info(
                    f"DRY RUN: Would replay {message.correlation_id} -> {message.original_topic}",
                    extra={
                        "retry_count": message.retry_count,
                        "error_type": message.error_type,
                    },
                )
            else:
                try:
                    await self.producer.replay_message(message, replay_correlation_id)
                    result = ModelReplayResult(
                        correlation_id=message.correlation_id,
                        original_topic=message.original_topic,
                        status=EnumReplayStatus.COMPLETED,
                        message="Replayed successfully",
                        replay_correlation_id=replay_correlation_id,
                    )
                    await self._record_tracking(
                        message, EnumReplayStatus.COMPLETED, replay_correlation_id
                    )
                    logger.info(
                        f"REPLAYED: {message.correlation_id} -> {message.original_topic}",
                        extra={
                            "replay_correlation_id": str(replay_correlation_id),
                        },
                    )
                except Exception as e:
                    result = ModelReplayResult(
                        correlation_id=message.correlation_id,
                        original_topic=message.original_topic,
                        status=EnumReplayStatus.FAILED,
                        message=f"Replay failed: {e}",
                        replay_correlation_id=replay_correlation_id,
                    )
                    await self._record_tracking(
                        message, EnumReplayStatus.FAILED, replay_correlation_id, str(e)
                    )
                    logger.exception(
                        f"FAILED: {message.correlation_id}",
                        extra={
                            "original_topic": message.original_topic,
                        },
                    )

            self.results.append(result)
            count += 1

        return self.results


# =============================================================================
# CLI Commands
# =============================================================================


async def cmd_list(args: argparse.Namespace) -> int:
    """List messages in the DLQ."""
    config = ModelReplayConfig.from_args(args)
    consumer = DLQConsumer(config)

    try:
        await consumer.start()

        count = 0
        limit = args.limit if hasattr(args, "limit") and args.limit else 100

        print(f"\n{'=' * 80}")
        print(f"DLQ Messages from: {config.dlq_topic}")
        print(f"{'=' * 80}\n")

        async for message in consumer.consume_messages():
            if count >= limit:
                print(f"\n... (limited to {limit} messages)")
                break

            should, reason = should_replay(message, config)
            status = "ELIGIBLE" if should else f"SKIP: {reason}"

            print(f"[{count + 1}] {message.correlation_id}")
            print(f"    Topic:     {message.original_topic}")
            print(f"    Error:     {message.error_type}")
            print(f"    Reason:    {safe_truncate(message.failure_reason, 80)}")
            print(f"    Timestamp: {message.failure_timestamp}")
            print(f"    Retries:   {message.retry_count}")
            print(f"    Status:    {status}")
            print()

            count += 1

        print(f"Total messages listed: {count}")
        return 0

    finally:
        await consumer.stop()


async def cmd_replay(args: argparse.Namespace) -> int:
    """Execute DLQ replay operation."""
    config = ModelReplayConfig.from_args(args)
    executor = DLQReplayExecutor(config)

    try:
        await executor.start()
        results = await executor.execute()

        # Summary
        completed = sum(1 for r in results if r.status == EnumReplayStatus.COMPLETED)
        skipped = sum(1 for r in results if r.status == EnumReplayStatus.SKIPPED)
        failed = sum(1 for r in results if r.status == EnumReplayStatus.FAILED)
        pending = sum(1 for r in results if r.status == EnumReplayStatus.PENDING)

        print(f"\n{'=' * 80}")
        print("Replay Summary")
        print(f"{'=' * 80}")
        print(f"Total processed: {len(results)}")
        print(f"  Completed:     {completed}")
        print(f"  Skipped:       {skipped}")
        print(f"  Failed:        {failed}")
        if config.dry_run:
            print(f"  Pending (dry): {pending}")
        if config.enable_tracking:
            tracking_status = (
                "enabled" if executor._tracking_service else "failed to initialize"
            )
            print(f"  Tracking:      {tracking_status}")
        print()

        return 0 if failed == 0 else 1

    finally:
        await executor.stop()


async def cmd_stats(args: argparse.Namespace) -> int:
    """Show DLQ statistics."""
    config = ModelReplayConfig.from_args(args)
    consumer = DLQConsumer(config)

    try:
        await consumer.start()

        stats: dict[str, dict[str, int]] = {
            "by_topic": {},
            "by_error_type": {},
            "by_retry_count": {},
        }
        total = 0

        async for message in consumer.consume_messages():
            total += 1

            # Count by topic
            topic = message.original_topic
            stats["by_topic"][topic] = stats["by_topic"].get(topic, 0) + 1

            # Count by error type
            error_type = message.error_type
            stats["by_error_type"][error_type] = (
                stats["by_error_type"].get(error_type, 0) + 1
            )

            # Count by retry count
            retry_key = str(message.retry_count)
            stats["by_retry_count"][retry_key] = (
                stats["by_retry_count"].get(retry_key, 0) + 1
            )

        print(f"\n{'=' * 80}")
        print(f"DLQ Statistics: {config.dlq_topic}")
        print(f"{'=' * 80}")
        print(f"\nTotal messages: {total}\n")

        print("By Original Topic:")
        for topic, count in sorted(
            stats["by_topic"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {topic}: {count}")

        print("\nBy Error Type:")
        for error_type, count in sorted(
            stats["by_error_type"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {error_type}: {count}")

        print("\nBy Retry Count:")
        for retry_count, count in sorted(stats["by_retry_count"].items()):
            print(f"  {retry_count} retries: {count}")

        return 0

    finally:
        await consumer.stop()


# =============================================================================
# CLI Parser
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="DLQ Replay Utility - Replay failed messages from Dead Letter Queue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List DLQ messages
  python scripts/dlq_replay.py list --dlq-topic dlq-events

  # Replay with dry run (no actual publish)
  python scripts/dlq_replay.py replay --dlq-topic dlq-events --dry-run

  # Replay only messages from specific topic
  python scripts/dlq_replay.py replay --dlq-topic dlq-events --filter-topic dev.orders

  # Replay only connection errors
  python scripts/dlq_replay.py replay --dlq-topic dlq-events --filter-error-type InfraConnectionError

  # Show DLQ statistics
  python scripts/dlq_replay.py stats --dlq-topic dlq-events

See docs/operations/DLQ_REPLAY_RUNBOOK.md for complete documentation.
        """,
    )

    # Global options
    parser.add_argument(
        "--bootstrap-servers",
        default=None,
        help="Kafka bootstrap servers (REQUIRED via env KAFKA_BOOTSTRAP_SERVERS or this flag)",
    )
    parser.add_argument(
        "--dlq-topic",
        default="dlq-events",
        help="DLQ topic name (default: dlq-events)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Enable PostgreSQL replay tracking (requires POSTGRES_* env vars)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List messages in the DLQ")
    list_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum messages to list (default: 100)",
    )
    list_parser.add_argument(
        "--max-replay-count",
        type=int,
        default=5,
        help="Max replay attempts for eligibility check (default: 5)",
    )
    list_parser.add_argument("--filter-topic", help="Filter by original topic")
    list_parser.add_argument("--filter-error-type", help="Filter by error type")
    list_parser.add_argument("--filter-correlation-id", help="Filter by correlation ID")
    list_parser.add_argument(
        "--start-time",
        type=str,
        help="Filter messages after this time (ISO 8601 format, e.g., 2025-01-01T00:00:00Z)",
    )
    list_parser.add_argument(
        "--end-time",
        type=str,
        help="Filter messages before this time (ISO 8601 format, e.g., 2025-01-01T23:59:59Z)",
    )

    # replay command
    replay_parser = subparsers.add_parser("replay", help="Replay DLQ messages")
    replay_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be replayed without actually publishing",
    )
    replay_parser.add_argument(
        "--max-replay-count",
        type=int,
        default=5,
        help="Maximum total replay attempts per message (default: 5)",
    )
    replay_parser.add_argument(
        "--rate-limit",
        type=float,
        default=100.0,
        help="Maximum messages to replay per second (default: 100)",
    )
    replay_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum messages to replay (default: unlimited)",
    )
    replay_parser.add_argument("--filter-topic", help="Only replay from specific topic")
    replay_parser.add_argument(
        "--filter-error-type", help="Only replay specific error types"
    )
    replay_parser.add_argument(
        "--filter-correlation-id", help="Only replay specific correlation ID"
    )
    replay_parser.add_argument(
        "--start-time",
        type=str,
        help="Filter messages after this time (ISO 8601 format, e.g., 2025-01-01T00:00:00Z)",
    )
    replay_parser.add_argument(
        "--end-time",
        type=str,
        help="Filter messages before this time (ISO 8601 format, e.g., 2025-01-01T23:59:59Z)",
    )

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show DLQ statistics")
    stats_parser.add_argument("--max-replay-count", type=int, default=5)
    stats_parser.add_argument("--filter-topic", default=None)
    stats_parser.add_argument("--filter-error-type", default=None)
    stats_parser.add_argument("--filter-correlation-id", default=None)

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> int:
    """Main entry point for DLQ replay CLI.

    Signal handling relies on asyncio's default behavior:
    - SIGINT (Ctrl+C) raises asyncio.CancelledError in async tasks
    - KeyboardInterrupt is caught as a fallback

    Returns:
        Exit code (0 for success, 1 for error, 130 for interrupt)
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return 1

    command_handlers = {
        "list": cmd_list,
        "replay": cmd_replay,
        "stats": cmd_stats,
    }

    handler = command_handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return await handler(args)
    except asyncio.CancelledError:
        logger.info("Operation cancelled")
        return 130  # Standard exit code for SIGINT
    except KafkaConnectionError:
        logger.exception("Failed to connect to Kafka")
        print(
            f"\nError: Could not connect to Kafka at {args.bootstrap_servers}",
            file=sys.stderr,
        )
        print("Please verify:", file=sys.stderr)
        print("  1. Kafka/Redpanda is running", file=sys.stderr)
        print("  2. Bootstrap servers are correct", file=sys.stderr)
        print(
            "  3. Network connectivity to the broker",
            file=sys.stderr,
        )
        return 1
    except KafkaError:
        logger.exception("Kafka error occurred")
        print("\nKafka error occurred. See log for details.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
