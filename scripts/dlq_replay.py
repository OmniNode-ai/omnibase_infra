#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""DLQ Replay CLI — thin shim over the contract-native DLQ replay node.

As of OMN-12619 the replay engine is owned by ``NodeDlqReplayEffect``
(``src/omnibase_infra/nodes/node_dlq_replay_effect``). This script is now a
thin CLI adapter: it parses args, builds the node's engine config, and drives
``HandlerDlqReplay``. The Kafka I/O classes, the eligibility predicate
(``should_replay``), and ``ModelReplayResult`` were relocated into the node and
are imported back here — they are no longer defined in this file.

Non-replayable messages are QUARANTINED to ``onex.dlq.quarantine.v1`` (never
dropped). The node uses the persistent ``onex-dlq-replay`` consumer group; this
CLI shares that engine. Prefer running the node via the
``dlq-replay-consumer`` service for steady-state operation — this CLI is for
operator-initiated, ad-hoc runs.

Usage:
    python scripts/dlq_replay.py list --dlq-topic onex.dlq.events.v1
    python scripts/dlq_replay.py replay --dlq-topic onex.dlq.events.v1 --dry-run
    python scripts/dlq_replay.py replay --dlq-topic onex.dlq.events.v1 --enable-tracking

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker addresses (REQUIRED - no default)
    OMNIBASE_INFRA_DB_URL: PostgreSQL DSN for tracking (required with --enable-tracking)

See Also:
    docs/operations/DLQ_REPLAY_RUNBOOK.md
    docs/operations/DLQ_QUARANTINE_OWNERSHIP.md
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from uuid import UUID

from aiokafka.errors import KafkaConnectionError, KafkaError
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from omnibase_infra.dlq import (
    EnumReplayStatus,
    ModelDlqTrackingConfig,
    ServiceDlqTracking,
)
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQ_REPLAY_CONSUMER_GROUP,
    DLQConsumer,
    DLQProducer,
    DLQQuarantineProducer,
    ModelDlqReplayEngineConfig,
    parse_datetime_with_timezone,
    safe_truncate,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.enum_dlq_replay_filter_type import (
    EnumDlqReplayFilterType,
)
from omnibase_infra.utils.util_datetime import is_timezone_aware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dlq_replay")


class ModelReplayConfig(BaseModel):
    """CLI-facing replay configuration parsed from argparse.

    This is the operator/CLI surface (validation of bootstrap servers, rate
    limit, time range). It is converted to the node's
    ``ModelDlqReplayEngineConfig`` via ``to_engine_config()`` before driving the
    node engine. The engine/eligibility logic itself lives in the node.
    """

    bootstrap_servers: str
    dlq_topic: str = "onex.dlq.events.v1"
    max_replay_count: int = 5
    rate_limit_per_second: float = 100.0
    dry_run: bool = False
    filter_type: EnumDlqReplayFilterType = EnumDlqReplayFilterType.ALL
    filter_topics: list[str] = Field(default_factory=list)
    filter_error_types: list[str] = Field(default_factory=list)
    filter_correlation_ids: list[UUID] = Field(default_factory=list)
    add_replay_headers: bool = True
    limit: int | None = None
    filter_start_time: datetime | None = None
    filter_end_time: datetime | None = None
    enable_tracking: bool = False
    postgres_dsn: str | None = None
    max_request_size: int = Field(default=10485760)
    request_timeout_ms: int = Field(default=30000)

    @field_validator("bootstrap_servers", mode="before")
    @classmethod
    def validate_bootstrap_servers(cls, v: object) -> str:
        """Validate bootstrap_servers host:port format."""
        if v is None:
            raise ValueError(
                "bootstrap_servers cannot be None. "
                "Set KAFKA_BOOTSTRAP_SERVERS or use --bootstrap-servers."
            )
        if not isinstance(v, str):
            raise ValueError(
                f"bootstrap_servers must be a string, got {type(v).__name__}"
            )
        stripped = v.strip()
        if not stripped:
            raise ValueError(
                "bootstrap_servers cannot be empty. "
                "Provide a valid Kafka broker address (e.g., 'localhost:9092')."
            )
        for raw in stripped.split(","):
            server = raw.strip()
            if not server:
                raise ValueError(
                    f"bootstrap_servers cannot contain empty entries. Got: '{v}'"
                )
            if ":" not in server:
                raise ValueError(
                    f"Invalid bootstrap server format '{server}'. "
                    "Expected 'host:port' (e.g., 'localhost:9092')."
                )
            host, port_str = server.rsplit(":", 1)
            if not host:
                raise ValueError(
                    f"Invalid bootstrap server format '{server}'. Host cannot be empty."
                )
            try:
                port = int(port_str)
                if port < 1 or port > 65535:
                    raise ValueError(
                        f"Invalid port {port} in '{server}'. "
                        "Port must be between 1 and 65535."
                    )
            except ValueError as e:
                if "Invalid port" in str(e):
                    raise
                raise ValueError(
                    f"Invalid port '{port_str}' in '{server}'. "
                    "Port must be a valid integer."
                ) from e
        return stripped

    @field_validator("rate_limit_per_second")
    @classmethod
    def validate_rate_limit(cls, v: float) -> float:
        """Validate rate_limit_per_second is positive (no division by zero)."""
        if v <= 0:
            raise ValueError(
                f"rate_limit_per_second must be > 0, got {v}. "
                "A zero or negative rate limit would cause division by zero."
            )
        return v

    @field_validator("postgres_dsn", mode="before")
    @classmethod
    def validate_postgres_dsn_scheme(cls, v: object) -> object:
        """Validate postgres_dsn uses a postgresql scheme when provided."""
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError(f"postgres_dsn must be a string, got {type(v).__name__}")
        stripped = v.strip()
        if stripped and not stripped.startswith("postgresql"):
            raise ValueError(
                f"postgres_dsn must start with 'postgresql' scheme, got '{stripped[:20]}...'"
            )
        return stripped

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
        """Build CLI config from parsed args.

        Raises:
            ValueError: If bootstrap_servers is not provided via env or CLI.
        """
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

        filter_type = EnumDlqReplayFilterType.ALL
        filter_topics: list[str] = []
        filter_error_types: list[str] = []
        filter_correlation_ids: list[UUID] = []
        filter_start_time: datetime | None = None
        filter_end_time: datetime | None = None

        start_time_str = getattr(args, "start_time", None)
        end_time_str = getattr(args, "end_time", None)
        if start_time_str:
            try:
                filter_start_time = parse_datetime_with_timezone(start_time_str)
            except ValueError as e:
                raise ValueError(
                    f"Invalid start_time format: {start_time_str}. "
                    "Use ISO 8601 format (e.g., 2025-01-01T00:00:00Z)"
                ) from e
        if end_time_str:
            try:
                filter_end_time = parse_datetime_with_timezone(end_time_str)
            except ValueError as e:
                raise ValueError(
                    f"Invalid end_time format: {end_time_str}. "
                    "Use ISO 8601 format (e.g., 2025-01-01T23:59:59Z)"
                ) from e

        if args.filter_topic:
            filter_type = EnumDlqReplayFilterType.BY_TOPIC
            filter_topics = [args.filter_topic]
        elif args.filter_error_type:
            filter_type = EnumDlqReplayFilterType.BY_ERROR_TYPE
            filter_error_types = [args.filter_error_type]
        elif args.filter_correlation_id:
            filter_type = EnumDlqReplayFilterType.BY_CORRELATION_ID
            try:
                filter_correlation_ids = [UUID(args.filter_correlation_id)]
            except ValueError as e:
                raise ValueError(
                    f"Invalid correlation ID format: {args.filter_correlation_id}"
                ) from e
        elif filter_start_time or filter_end_time:
            filter_type = EnumDlqReplayFilterType.BY_TIME_RANGE

        enable_tracking = getattr(args, "enable_tracking", False)
        postgres_dsn = os.environ.get("OMNIBASE_INFRA_DB_URL")

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
            postgres_dsn=postgres_dsn,
            limit=getattr(args, "limit", None),
        )

    def build_tracking_dsn(self) -> str | None:
        """Return PostgreSQL DSN for tracking, or None when disabled."""
        if not self.enable_tracking:
            return None
        if not self.postgres_dsn:
            logger.warning("OMNIBASE_INFRA_DB_URL not set; tracking will be disabled")
            return None
        return self.postgres_dsn

    def to_engine_config(self) -> ModelDlqReplayEngineConfig:
        """Convert the CLI config into the node engine config.

        Uses the persistent consumer group (ONEX_GROUP_ID / onex-dlq-replay)
        rather than the legacy ephemeral dlq-replay-{pid} group.
        """
        # scripts/ is allowlisted for env reads; the node engine never reads env.
        consumer_group = os.environ.get("ONEX_GROUP_ID") or DLQ_REPLAY_CONSUMER_GROUP
        return ModelDlqReplayEngineConfig(
            bootstrap_servers=self.bootstrap_servers,
            dlq_topic=self.dlq_topic,
            consumer_group=consumer_group,
            max_replay_count=self.max_replay_count,
            rate_limit_per_second=self.rate_limit_per_second,
            dry_run=self.dry_run,
            filter_type=self.filter_type,
            filter_topics=tuple(self.filter_topics),
            filter_error_types=tuple(self.filter_error_types),
            filter_correlation_ids=tuple(self.filter_correlation_ids),
            filter_start_time=self.filter_start_time,
            filter_end_time=self.filter_end_time,
            add_replay_headers=self.add_replay_headers,
            limit=self.limit,
            max_request_size=self.max_request_size,
            request_timeout_ms=self.request_timeout_ms,
        )


async def cmd_list(args: argparse.Namespace) -> int:
    """List messages in the DLQ (eligibility decided by the node predicate)."""
    config = ModelReplayConfig.from_args(args)
    engine_config = config.to_engine_config()
    consumer = DLQConsumer(engine_config)
    try:
        await consumer.start()
        count = 0
        limit = args.limit if getattr(args, "limit", None) else 100
        print(f"\n{'=' * 80}")
        print(f"DLQ Messages from: {engine_config.dlq_topic}")
        print(f"{'=' * 80}\n")
        async for message in consumer.consume_messages():
            if count >= limit:
                print(f"\n... (limited to {limit} messages)")
                break
            should, reason = should_replay(message, engine_config)
            status = "ELIGIBLE" if should else f"QUARANTINE: {reason}"
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
    """Execute a DLQ replay run via the node handler."""
    config = ModelReplayConfig.from_args(args)
    engine_config = config.to_engine_config()

    consumer = DLQConsumer(engine_config)
    producer = DLQProducer(engine_config)
    quarantine_producer = DLQQuarantineProducer(engine_config)
    tracking: ServiceDlqTracking | None = None

    try:
        await consumer.start()
        if not engine_config.dry_run:
            await producer.start()
            await quarantine_producer.start()

        dsn = config.build_tracking_dsn()
        if dsn:
            tracking = ServiceDlqTracking(ModelDlqTrackingConfig(dsn=dsn))
            try:
                await tracking.initialize()
            except Exception as e:  # noqa: BLE001 — boundary: degrade tracking
                logger.warning("Failed to initialize tracking service: %s", e)
                tracking = None

        handler = HandlerDlqReplay(
            consumer=consumer,
            producer=producer,
            quarantine_producer=quarantine_producer,
            tracking=tracking,
        )
        result = await handler.run()

        print(f"\n{'=' * 80}")
        print("Replay Summary")
        print(f"{'=' * 80}")
        print(f"Total processed: {result.total_processed}")
        print(f"  Completed:     {result.completed}")
        print(f"  Quarantined:   {result.quarantined}")
        print(f"  Failed:        {result.failed}")
        if engine_config.dry_run:
            print(f"  Pending (dry): {result.pending}")
        if config.enable_tracking:
            tracking_status = (
                "enabled"
                if tracking is not None and tracking.is_tracking_enabled
                else "failed to initialize"
            )
            print(f"  Tracking:      {tracking_status}")
        print()

        return 0 if result.failed == 0 else 1
    finally:
        await consumer.stop()
        if not engine_config.dry_run:
            await producer.stop()
            await quarantine_producer.stop()
        if tracking is not None and tracking.is_tracking_enabled:
            await tracking.shutdown()


async def cmd_stats(args: argparse.Namespace) -> int:
    """Show DLQ statistics."""
    config = ModelReplayConfig.from_args(args)
    engine_config = config.to_engine_config()
    consumer = DLQConsumer(engine_config)
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
            topic = message.original_topic
            stats["by_topic"][topic] = stats["by_topic"].get(topic, 0) + 1
            error_type = message.error_type
            stats["by_error_type"][error_type] = (
                stats["by_error_type"].get(error_type, 0) + 1
            )
            retry_key = str(message.retry_count)
            stats["by_retry_count"][retry_key] = (
                stats["by_retry_count"].get(retry_key, 0) + 1
            )
        print(f"\n{'=' * 80}")
        print(f"DLQ Statistics: {engine_config.dlq_topic}")
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


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="DLQ Replay CLI — thin shim over NodeDlqReplayEffect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/dlq_replay.py list --dlq-topic onex.dlq.events.v1
  python scripts/dlq_replay.py replay --dlq-topic onex.dlq.events.v1 --dry-run
  python scripts/dlq_replay.py stats --dlq-topic onex.dlq.events.v1

Non-replayable messages are quarantined to onex.dlq.quarantine.v1 (never dropped).
See docs/operations/DLQ_QUARANTINE_OWNERSHIP.md.
        """,
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=None,
        help="Kafka bootstrap servers (REQUIRED via env KAFKA_BOOTSTRAP_SERVERS or this flag)",
    )
    parser.add_argument(
        "--dlq-topic",
        default="onex.dlq.events.v1",
        help="DLQ topic name (default: onex.dlq.events.v1)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Enable PostgreSQL replay tracking (requires OMNIBASE_INFRA_DB_URL)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    list_parser = subparsers.add_parser("list", help="List messages in the DLQ")
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.add_argument("--max-replay-count", type=int, default=5)
    list_parser.add_argument("--filter-topic")
    list_parser.add_argument("--filter-error-type")
    list_parser.add_argument("--filter-correlation-id")
    list_parser.add_argument("--start-time", type=str)
    list_parser.add_argument("--end-time", type=str)

    replay_parser = subparsers.add_parser("replay", help="Replay DLQ messages")
    replay_parser.add_argument("--dry-run", action="store_true")
    replay_parser.add_argument("--max-replay-count", type=int, default=5)
    replay_parser.add_argument("--rate-limit", type=float, default=100.0)
    replay_parser.add_argument("--limit", type=int)
    replay_parser.add_argument("--filter-topic")
    replay_parser.add_argument("--filter-error-type")
    replay_parser.add_argument("--filter-correlation-id")
    replay_parser.add_argument("--start-time", type=str)
    replay_parser.add_argument("--end-time", type=str)

    stats_parser = subparsers.add_parser("stats", help="Show DLQ statistics")
    stats_parser.add_argument("--max-replay-count", type=int, default=5)
    stats_parser.add_argument("--filter-topic", default=None)
    stats_parser.add_argument("--filter-error-type", default=None)
    stats_parser.add_argument("--filter-correlation-id", default=None)

    return parser


async def main() -> int:
    """CLI entry point."""
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
        return 130
    except KafkaConnectionError:
        logger.exception("Failed to connect to Kafka")
        print(
            f"\nError: Could not connect to Kafka at {args.bootstrap_servers}",
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


# Re-exported for callers/tests that import the eligibility helpers from the CLI.
__all__ = [
    "EnumReplayStatus",
    "ModelReplayConfig",
    "is_timezone_aware",
    "parse_datetime_with_timezone",
    "should_replay",
]


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
