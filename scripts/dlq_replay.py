#!/usr/bin/env python3
"""DLQ Replay Utility - Replay failed messages from Dead Letter Queue.

This script provides a command-line interface for replaying messages from
the Dead Letter Queue (DLQ) back to their original topics.

Usage:
    python scripts/dlq_replay.py --help
    python scripts/dlq_replay.py list --dlq-topic dlq-events
    python scripts/dlq_replay.py replay --dlq-topic dlq-events --dry-run
    python scripts/dlq_replay.py replay --dlq-topic dlq-events --filter-topic dev.orders

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker addresses (default: localhost:9092)
    POSTGRES_HOST: PostgreSQL host for replay tracking (optional)
    POSTGRES_PORT: PostgreSQL port (default: 5432)
    POSTGRES_DATABASE: Database name (default: omninode_bridge)
    POSTGRES_USER: Database user (default: postgres)
    POSTGRES_PASSWORD: Database password (required if tracking enabled)

See Also:
    docs/operations/DLQ_REPLAY_GUIDE.md - Complete replay documentation
    OMN-949 - DLQ Replay Mechanism ticket

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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dlq_replay")


# =============================================================================
# Enums and Models
# =============================================================================


class EnumReplayStatus(str, Enum):
    """Status of a replay operation."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EnumFilterType(str, Enum):
    """Filter criteria for selective replay."""

    ALL = "all"
    BY_TOPIC = "by_topic"
    BY_ERROR_TYPE = "by_error_type"
    BY_TIME_RANGE = "by_time_range"
    BY_CORRELATION_ID = "by_correlation_id"


@dataclass
class DLQMessage:
    """Parsed DLQ message with metadata."""

    original_topic: str
    original_key: str | None
    original_value: str
    original_offset: str | None
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
    ) -> DLQMessage:
        """Parse DLQ message from Kafka payload."""
        original_message = payload.get("original_message", {})
        if not isinstance(original_message, dict):
            original_message = {}

        correlation_id_str = payload.get("correlation_id", "")
        try:
            correlation_id = UUID(str(correlation_id_str))
        except (ValueError, AttributeError):
            correlation_id = uuid4()

        return cls(
            original_topic=str(payload.get("original_topic", "unknown")),
            original_key=original_message.get("key"),
            original_value=str(original_message.get("value", "")),
            original_offset=original_message.get("offset"),
            original_partition=original_message.get("partition"),
            failure_reason=str(payload.get("failure_reason", "")),
            failure_timestamp=str(payload.get("failure_timestamp", "")),
            correlation_id=correlation_id,
            retry_count=int(payload.get("retry_count", 0)),
            error_type=str(payload.get("error_type", "Unknown")),
            dlq_offset=dlq_offset,
            dlq_partition=dlq_partition,
            raw_payload=payload,
        )


@dataclass
class ReplayConfig:
    """Configuration for DLQ replay operation."""

    bootstrap_servers: str = "localhost:9092"
    dlq_topic: str = "dlq-events"
    max_replay_count: int = 5
    rate_limit_per_second: float = 100.0
    dry_run: bool = False
    filter_type: EnumFilterType = EnumFilterType.ALL
    filter_topics: list[str] = field(default_factory=list)
    filter_error_types: list[str] = field(default_factory=list)
    filter_correlation_ids: list[UUID] = field(default_factory=list)
    add_replay_headers: bool = True
    limit: int | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ReplayConfig:
        """Create config from command line arguments."""
        bootstrap_servers = os.environ.get(
            "KAFKA_BOOTSTRAP_SERVERS",
            args.bootstrap_servers,
        )

        filter_type = EnumFilterType.ALL
        filter_topics: list[str] = []
        filter_error_types: list[str] = []
        filter_correlation_ids: list[UUID] = []

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
            except ValueError:
                logger.warning(
                    f"Invalid correlation ID format: {args.filter_correlation_id}"
                )
                sys.exit(1)

        return cls(
            bootstrap_servers=bootstrap_servers,
            dlq_topic=args.dlq_topic,
            max_replay_count=args.max_replay_count,
            rate_limit_per_second=args.rate_limit,
            dry_run=args.dry_run,
            filter_type=filter_type,
            filter_topics=filter_topics,
            filter_error_types=filter_error_types,
            filter_correlation_ids=filter_correlation_ids,
            limit=args.limit if hasattr(args, "limit") else None,
        )


@dataclass
class ReplayResult:
    """Result of a replay operation."""

    correlation_id: UUID
    original_topic: str
    status: EnumReplayStatus
    message: str
    replay_correlation_id: UUID | None = None


# =============================================================================
# Non-Retryable Error Types
# =============================================================================

NON_RETRYABLE_ERRORS = frozenset(
    {
        "InfraAuthenticationError",
        "ProtocolConfigurationError",
        "SecretResolutionError",
        "ValidationError",
        "SchemaValidationError",
        "PermissionDeniedError",
    }
)


# =============================================================================
# DLQ Consumer (Stub Implementation)
# =============================================================================


class DLQConsumer:
    """Consumer for reading DLQ messages.

    This is a stub implementation. In production, use aiokafka.AIOKafkaConsumer.
    """

    def __init__(self, config: ReplayConfig) -> None:
        """Initialize DLQ consumer."""
        self.config = config
        self._started = False

    async def start(self) -> None:
        """Start the consumer."""
        logger.info(
            f"Starting DLQ consumer for topic: {self.config.dlq_topic}",
            extra={"bootstrap_servers": self.config.bootstrap_servers},
        )
        # TODO: Initialize aiokafka.AIOKafkaConsumer
        # self._consumer = AIOKafkaConsumer(
        #     self.config.dlq_topic,
        #     bootstrap_servers=self.config.bootstrap_servers,
        #     auto_offset_reset="earliest",
        #     enable_auto_commit=False,
        # )
        # await self._consumer.start()
        self._started = True

    async def stop(self) -> None:
        """Stop the consumer."""
        if self._started:
            # TODO: await self._consumer.stop()
            self._started = False
            logger.info("DLQ consumer stopped")

    async def consume_messages(self) -> AsyncIterator[DLQMessage]:
        """Consume and yield DLQ messages.

        Yields:
            DLQMessage instances parsed from DLQ topic
        """
        if not self._started:
            raise RuntimeError("Consumer not started")

        # TODO: Replace with actual Kafka consumption
        # async for msg in self._consumer:
        #     try:
        #         payload = json.loads(msg.value.decode("utf-8"))
        #         yield DLQMessage.from_kafka_message(
        #             payload=payload,
        #             dlq_offset=msg.offset,
        #             dlq_partition=msg.partition,
        #         )
        #     except json.JSONDecodeError as e:
        #         logger.warning(f"Failed to parse DLQ message: {e}")
        #         continue

        # Stub: Yield no messages (for skeleton)
        return
        yield  # Make this a generator


# =============================================================================
# DLQ Producer (Stub Implementation)
# =============================================================================


class DLQProducer:
    """Producer for replaying messages to original topics.

    This is a stub implementation. In production, use aiokafka.AIOKafkaProducer.
    """

    def __init__(self, config: ReplayConfig) -> None:
        """Initialize DLQ producer."""
        self.config = config
        self._started = False
        self._last_publish = datetime.now(UTC)
        self._interval = 1.0 / config.rate_limit_per_second

    async def start(self) -> None:
        """Start the producer."""
        logger.info(
            "Starting DLQ producer",
            extra={"bootstrap_servers": self.config.bootstrap_servers},
        )
        # TODO: Initialize aiokafka.AIOKafkaProducer
        # self._producer = AIOKafkaProducer(
        #     bootstrap_servers=self.config.bootstrap_servers,
        #     acks="all",
        #     enable_idempotence=True,
        # )
        # await self._producer.start()
        self._started = True

    async def stop(self) -> None:
        """Stop the producer."""
        if self._started:
            # TODO: await self._producer.stop()
            self._started = False
            logger.info("DLQ producer stopped")

    async def replay_message(
        self,
        message: DLQMessage,
        replay_correlation_id: UUID,
    ) -> None:
        """Replay a message to its original topic with rate limiting.

        Args:
            message: DLQ message to replay
            replay_correlation_id: New correlation ID for replay tracking
        """
        if not self._started:
            raise RuntimeError("Producer not started")

        # Rate limiting
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

        # TODO: Replace with actual Kafka publish
        # key = message.original_key.encode("utf-8") if message.original_key else None
        # value = message.original_value.encode("utf-8")
        # await self._producer.send(
        #     message.original_topic,
        #     value=value,
        #     key=key,
        #     headers=headers,
        # )

        self._last_publish = datetime.now(UTC)
        logger.debug(
            f"Replayed message to {message.original_topic}",
            extra={
                "correlation_id": str(message.correlation_id),
                "replay_correlation_id": str(replay_correlation_id),
            },
        )


# =============================================================================
# Replay Filter
# =============================================================================


def should_replay(message: DLQMessage, config: ReplayConfig) -> tuple[bool, str]:
    """Determine if a DLQ message should be replayed.

    Args:
        message: DLQ message to evaluate
        config: Replay configuration

    Returns:
        Tuple of (should_replay, reason)
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

    return (True, "Eligible for replay")


# =============================================================================
# Replay Executor
# =============================================================================


class DLQReplayExecutor:
    """Executor for DLQ replay operations."""

    def __init__(self, config: ReplayConfig) -> None:
        """Initialize replay executor."""
        self.config = config
        self.consumer = DLQConsumer(config)
        self.producer = DLQProducer(config)
        self.results: list[ReplayResult] = []

    async def start(self) -> None:
        """Start consumer and producer."""
        await self.consumer.start()
        if not self.config.dry_run:
            await self.producer.start()

    async def stop(self) -> None:
        """Stop consumer and producer."""
        await self.consumer.stop()
        if not self.config.dry_run:
            await self.producer.stop()

    async def execute(self) -> list[ReplayResult]:
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
                result = ReplayResult(
                    correlation_id=message.correlation_id,
                    original_topic=message.original_topic,
                    status=EnumReplayStatus.SKIPPED,
                    message=reason,
                )
                self.results.append(result)
                logger.info(
                    f"SKIP: {message.correlation_id} - {reason}",
                    extra={
                        "original_topic": message.original_topic,
                        "error_type": message.error_type,
                    },
                )
                continue

            replay_correlation_id = uuid4()

            if self.config.dry_run:
                result = ReplayResult(
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
                    result = ReplayResult(
                        correlation_id=message.correlation_id,
                        original_topic=message.original_topic,
                        status=EnumReplayStatus.COMPLETED,
                        message="Replayed successfully",
                        replay_correlation_id=replay_correlation_id,
                    )
                    logger.info(
                        f"REPLAYED: {message.correlation_id} -> {message.original_topic}",
                        extra={
                            "replay_correlation_id": str(replay_correlation_id),
                        },
                    )
                except Exception as e:
                    result = ReplayResult(
                        correlation_id=message.correlation_id,
                        original_topic=message.original_topic,
                        status=EnumReplayStatus.FAILED,
                        message=f"Replay failed: {e}",
                        replay_correlation_id=replay_correlation_id,
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
    config = ReplayConfig.from_args(args)
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
            print(f"    Reason:    {message.failure_reason[:80]}...")
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
    config = ReplayConfig.from_args(args)
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
        print()

        return 0 if failed == 0 else 1

    finally:
        await executor.stop()


async def cmd_stats(args: argparse.Namespace) -> int:
    """Show DLQ statistics."""
    config = ReplayConfig.from_args(args)
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

See docs/operations/DLQ_REPLAY_GUIDE.md for complete documentation.
        """,
    )

    # Global options
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092, env: KAFKA_BOOTSTRAP_SERVERS)",
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
    list_parser.add_argument("--dry-run", action="store_true", default=True)
    list_parser.add_argument("--rate-limit", type=float, default=100.0)

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

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show DLQ statistics")
    stats_parser.add_argument("--max-replay-count", type=int, default=5)
    stats_parser.add_argument("--filter-topic", default=None)
    stats_parser.add_argument("--filter-error-type", default=None)
    stats_parser.add_argument("--filter-correlation-id", default=None)
    stats_parser.add_argument("--dry-run", action="store_true", default=True)
    stats_parser.add_argument("--rate-limit", type=float, default=100.0)

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> int:
    """Main entry point."""
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

    return await handler(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
