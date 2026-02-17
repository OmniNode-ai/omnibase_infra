# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Demo Reset -- scoped environment reset for safe pre-demo cleanup.

Provides a safe, scoped reset of demo-related infrastructure without
affecting shared resources. This module resets only demo-scoped resources:

1. **Consumer group offsets** -- Projector starts fresh on next run
2. **Projector state** -- Registration projection rows cleared
3. **Topic data** (optional) -- Clean slate for event monitor

Shared infrastructure is explicitly preserved:
- PostgreSQL table schemas and indexes
- Non-demo Kafka topics and consumer groups
- Consul/Vault configuration
- Application code and contracts

The reset is idempotent: running twice produces the same result.

Usage:
    CLI entry point::

        uv run omni-infra demo reset [--dry-run] [--purge-topics]

    Programmatic::

        from omnibase_infra.cli.demo_reset import DemoResetEngine

        engine = DemoResetEngine(config)
        report = await engine.execute(dry_run=True)
        print(report.format_summary())

Related Tickets:
    - OMN-2299: Demo Reset scoped command for safe environment reset

.. versionadded:: 0.9.1
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

if TYPE_CHECKING:
    import asyncpg

from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

# =============================================================================
# Constants -- Demo-Scoped Resources
# =============================================================================

# Table that holds projector state for the registration domain.
# Only the ROWS are deleted, never the table or schema.
DEMO_PROJECTION_TABLE: Final[str] = "registration_projections"

# Allowlist of table names permitted in SQL interpolation.
# This prevents SQL injection via the ``projection_table`` config field.
# Add new table names here as new projection domains are created.
_ALLOWED_PROJECTION_TABLES: Final[frozenset[str]] = frozenset(
    {
        "registration_projections",
    }
)

# Consumer group pattern: any group containing "registration" or "projector".
# These are the groups whose offsets are reset so projectors start fresh.
DEMO_CONSUMER_GROUP_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(registration|projector|introspection)", re.IGNORECASE
)

# Topics that are considered demo-scoped. Only these may be purged.
# Platform topics that carry demo event data.
DEMO_TOPIC_PREFIXES: Final[tuple[str, ...]] = (
    "onex.evt.platform.",
    "onex.cmd.platform.",
    "onex.evt.omniintelligence.",
    "onex.cmd.omniintelligence.",
    "onex.evt.omniclaude.",
    "onex.evt.agent.",
)

# Resources that are NEVER touched, listed for the summary report.
PRESERVED_RESOURCES: Final[tuple[str, ...]] = (
    "PostgreSQL table schemas and indexes",
    "Consul KV configuration",
    "Vault secrets",
    "Non-demo Kafka topics",
    "Non-demo consumer groups",
    "Application code and contracts",
    "Docker container state",
)


# =============================================================================
# Result Models
# =============================================================================


class EnumResetAction(str, Enum):
    """Classification of a reset action."""

    RESET = "reset"
    PRESERVED = "preserved"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class ResetActionResult:
    """Result of a single reset action.

    Attributes:
        resource: Name of the resource affected.
        action: What was done (reset, preserved, skipped, error).
        detail: Human-readable description of what happened.
    """

    resource: str
    action: EnumResetAction
    detail: str


@dataclass
class DemoResetReport:
    """Aggregate report of all demo reset actions.

    Attributes:
        actions: List of individual action results.
        dry_run: Whether this was a dry-run (no changes made).
    """

    actions: list[ResetActionResult] = field(default_factory=list)
    dry_run: bool = False

    @property
    def reset_count(self) -> int:
        """Number of resources that were reset."""
        return sum(1 for a in self.actions if a.action == EnumResetAction.RESET)

    @property
    def preserved_count(self) -> int:
        """Number of resources explicitly preserved."""
        return sum(1 for a in self.actions if a.action == EnumResetAction.PRESERVED)

    @property
    def error_count(self) -> int:
        """Number of actions that failed."""
        return sum(1 for a in self.actions if a.action == EnumResetAction.ERROR)

    @property
    def skipped_count(self) -> int:
        """Number of actions skipped (e.g., already clean)."""
        return sum(1 for a in self.actions if a.action == EnumResetAction.SKIPPED)

    def format_summary(self) -> str:
        """Format the report as a human-readable summary.

        Returns:
            Multi-line string suitable for CLI output.
        """
        lines: list[str] = []
        mode = "DRY RUN" if self.dry_run else "EXECUTED"
        lines.append(f"Demo Reset Report ({mode})")
        lines.append("=" * 60)

        # Group by action type
        for action_type in EnumResetAction:
            group = [a for a in self.actions if a.action == action_type]
            if not group:
                continue

            label = action_type.value.upper()
            lines.append(f"\n  [{label}]")
            for item in group:
                lines.append(f"    {item.resource}: {item.detail}")

        lines.append("")
        lines.append(
            f"Summary: {self.reset_count} reset, "
            f"{self.preserved_count} preserved, "
            f"{self.skipped_count} skipped, "
            f"{self.error_count} errors"
        )

        return "\n".join(lines)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class DemoResetConfig:
    """Configuration for the demo reset engine.

    Note:
        ``consumer_group_pattern`` is typed as ``re.Pattern`` which is
        technically a mutable object (compiled regex patterns have mutable
        internal caching).  In practice ``re.Pattern`` is effectively
        immutable -- its public API is read-only -- so ``frozen=True``
        is safe here despite the dataclass not performing a deep-freeze.

    Attributes:
        postgres_dsn: PostgreSQL connection string.
        kafka_bootstrap_servers: Kafka broker address(es).
        purge_topics: Whether to delete messages from demo topics.
        projection_table: Table name for projector state.
        consumer_group_pattern: Regex to match demo consumer groups.
        demo_topic_prefixes: Topic prefixes considered demo-scoped.
    """

    postgres_dsn: str = ""
    kafka_bootstrap_servers: str = ""
    purge_topics: bool = False
    projection_table: str = DEMO_PROJECTION_TABLE
    consumer_group_pattern: re.Pattern[str] = DEMO_CONSUMER_GROUP_PATTERN
    demo_topic_prefixes: tuple[str, ...] = DEMO_TOPIC_PREFIXES

    @classmethod
    def from_env(cls, *, purge_topics: bool = False) -> DemoResetConfig:
        """Create config from environment variables.

        Reads OMNIBASE_INFRA_DB_URL and KAFKA_BOOTSTRAP_SERVERS from
        the environment. Falls back to empty strings if not set.

        Args:
            purge_topics: Whether to purge demo topic data.

        Returns:
            DemoResetConfig populated from environment.
        """
        return cls(
            postgres_dsn=os.environ.get("OMNIBASE_INFRA_DB_URL", ""),
            kafka_bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", ""),
            purge_topics=purge_topics,
        )


# =============================================================================
# Postgres Connection Context Manager
# =============================================================================


class PostgresConnectionContext:
    """Async context manager for a single PostgreSQL connection with timeout.

    This replaces the previous pattern of creating a full ``asyncpg`` pool
    for each one-shot query.  A single connection avoids pool churn and
    eliminates the leak that occurred when ``asyncio.wait_for`` timed out
    during ``asyncpg.create_pool()``: the partially-initialized pool was
    never closed.

    The connection is established inside ``__aenter__`` via
    ``asyncio.wait_for(asyncpg.connect(...), timeout=...)``.  If the
    timeout fires before ``asyncpg.connect()`` returns, the assignment to
    ``self._conn`` never occurs, so ``__aexit__`` sees ``None`` and skips
    cleanup.  This is safe because ``asyncio.wait_for`` cancels the
    wrapped coroutine on timeout, and asyncpg internally closes any
    underlying socket resources when the coroutine is cancelled.
    ``__aexit__`` only closes connections that were successfully
    established and assigned to ``self._conn``.
    """

    def __init__(self, dsn: str, timeout: float) -> None:
        self._dsn = dsn
        self._timeout = timeout
        self._conn: asyncpg.Connection | None = None

    async def __aenter__(self) -> asyncpg.Connection:
        import asyncpg as _asyncpg

        self._conn = await asyncio.wait_for(
            _asyncpg.connect(self._dsn),
            timeout=self._timeout,
        )
        return self._conn

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                # Best-effort cleanup; the connection may already be dead
                # after a timeout or network error.
                logger.debug(
                    "Failed to close PostgreSQL connection during cleanup",
                    exc_info=True,
                )


# =============================================================================
# Engine
# =============================================================================


class DemoResetEngine:
    """Scoped demo reset engine.

    Executes a series of reset operations on demo-scoped resources
    while explicitly preserving shared infrastructure.

    The engine is designed to be:
    - **Safe**: Only demo-scoped resources are affected
    - **Idempotent**: Running twice produces the same result
    - **Observable**: Every action is reported with detail
    - **Reversible**: Topic purge is the only destructive operation

    Note:
        The Kafka consumer group and topic operations use
        ``confluent-kafka``'s ``AdminClient``, which performs blocking I/O
        internally.  This is acceptable for CLI use where ``execute()``
        is invoked via ``asyncio.run()`` and the event loop is otherwise
        idle, but the engine **must not** be used within an event loop
        that serves concurrent requests (e.g. a FastAPI endpoint) because
        the blocking calls would stall the loop.

    Example:
        >>> config = DemoResetConfig.from_env(purge_topics=False)
        >>> engine = DemoResetEngine(config)
        >>> report = await engine.execute(dry_run=True)
        >>> print(report.format_summary())
    """

    def __init__(self, config: DemoResetConfig) -> None:
        self._config = config

    @staticmethod
    def _validate_table_name(table: str, *, correlation_id: UUID | None = None) -> None:
        """Validate that a table name is in the allowlist.

        Prevents SQL injection by ensuring only known-safe table names
        are interpolated into SQL statements.

        Args:
            table: Table name to validate.
            correlation_id: Optional trace identifier included in the error
                message for diagnostic consistency with defense-in-depth
                checks downstream.

        Raises:
            ValueError: If the table name is not in ``_ALLOWED_PROJECTION_TABLES``.
        """
        if table not in _ALLOWED_PROJECTION_TABLES:
            msg = (
                f"Table name {table!r} is not in the allowed projection tables: "
                f"{sorted(_ALLOWED_PROJECTION_TABLES)}"
            )
            if correlation_id is not None:
                msg += f" (correlation_id={correlation_id})"
            raise ValueError(msg)

    async def execute(self, *, dry_run: bool = False) -> DemoResetReport:
        """Execute the demo reset sequence.

        Operations are executed in order:
        1. Clear projector state (DELETE rows from projection table)
        2. Reset consumer group offsets (delete demo consumer groups)
        3. Optionally purge demo topic data
        4. Record preserved resources

        Args:
            dry_run: If True, report what would happen without making changes.

        Returns:
            DemoResetReport with all actions taken and their results.
        """
        report = DemoResetReport(dry_run=dry_run)

        correlation_id = uuid4()

        # Step 1: Clear projector state
        await self._reset_projector_state(
            report, dry_run=dry_run, correlation_id=correlation_id
        )

        # Step 2: Reset consumer group offsets
        await self._reset_consumer_groups(
            report, dry_run=dry_run, correlation_id=correlation_id
        )

        # Step 3: Optionally purge demo topics
        if self._config.purge_topics:
            await self._purge_demo_topics(
                report, dry_run=dry_run, correlation_id=correlation_id
            )
        else:
            report.actions.append(
                ResetActionResult(
                    resource="Demo topic data",
                    action=EnumResetAction.SKIPPED,
                    detail="Topic purge not requested (use --purge-topics to enable)",
                )
            )

        # Step 4: Record preserved resources
        for resource in PRESERVED_RESOURCES:
            report.actions.append(
                ResetActionResult(
                    resource=resource,
                    action=EnumResetAction.PRESERVED,
                    detail="Explicitly preserved (not demo-scoped)",
                )
            )

        return report

    # -------------------------------------------------------------------------
    # Step 1: Projector State
    # -------------------------------------------------------------------------

    async def _reset_projector_state(
        self,
        report: DemoResetReport,
        *,
        dry_run: bool,
        correlation_id: UUID,
    ) -> None:
        """Clear all rows from the demo projection table.

        The table schema and indexes are preserved; only data rows are deleted.
        This is equivalent to TRUNCATE but uses DELETE for transaction safety.

        Args:
            report: Report to append results to.
            dry_run: If True, only report what would happen.
            correlation_id: Trace identifier for error diagnostics.
        """
        table = self._config.projection_table

        if not self._config.postgres_dsn:
            report.actions.append(
                ResetActionResult(
                    resource=f"Projector state ({table})",
                    action=EnumResetAction.SKIPPED,
                    detail="OMNIBASE_INFRA_DB_URL not configured",
                )
            )
            return

        if dry_run:
            try:
                row_count = await self._count_projection_rows(
                    correlation_id=correlation_id,
                )
                report.actions.append(
                    ResetActionResult(
                        resource=f"Projector state ({table})",
                        action=EnumResetAction.RESET,
                        detail=f"Would delete {row_count} row(s) from {table}",
                    )
                )
            except Exception as exc:
                logger.exception(
                    "Failed to count projector rows (dry run), correlation_id=%s",
                    correlation_id,
                )
                report.actions.append(
                    ResetActionResult(
                        resource=f"Projector state ({table})",
                        action=EnumResetAction.ERROR,
                        detail=f"Failed: {sanitize_error_message(exc)}",
                    )
                )
            return

        try:
            deleted = await self._delete_projection_rows(
                correlation_id=correlation_id,
            )
            report.actions.append(
                ResetActionResult(
                    resource=f"Projector state ({table})",
                    action=EnumResetAction.RESET,
                    detail=f"Deleted {deleted} row(s) from {table}",
                )
            )
        except Exception as exc:
            logger.exception(
                "Failed to clear projector state, correlation_id=%s",
                correlation_id,
            )
            report.actions.append(
                ResetActionResult(
                    resource=f"Projector state ({table})",
                    action=EnumResetAction.ERROR,
                    detail=f"Failed: {sanitize_error_message(exc)}",
                )
            )

    @staticmethod
    def _postgres_connection_timeout() -> float:
        """Return the connection timeout in seconds for PostgreSQL."""
        return 10.0

    def _postgres_connection(self) -> PostgresConnectionContext:
        """Create a single PostgreSQL connection with proper timeout handling.

        Uses ``asyncpg.connect()`` instead of a connection pool because each
        reset operation only needs a single query. This avoids pool churn
        (creating and tearing down a pool for every operation) and eliminates
        the risk of leaking a partially-initialized pool if the connection
        attempt times out.

        The connection is established inside the context manager's
        ``__aenter__`` with a timeout. ``__aexit__`` always closes the
        connection if it was opened, even after a timeout or error.

        Returns:
            Async context manager yielding an ``asyncpg.Connection``.

        Raises:
            asyncio.TimeoutError: If the connection cannot be established
                within the configured timeout.
        """
        return PostgresConnectionContext(
            dsn=self._config.postgres_dsn,
            timeout=self._postgres_connection_timeout(),
        )

    async def _count_projection_rows(
        self,
        *,
        correlation_id: UUID,
    ) -> int:
        """Count rows in the projection table.

        Args:
            correlation_id: Trace identifier for error diagnostics.
        """
        table = self._config.projection_table
        self._validate_table_name(table, correlation_id=correlation_id)

        # SAFETY: The f-string SQL interpolation below is safe because
        # ``_validate_table_name`` restricts ``table`` to the frozen
        # allowlist ``_ALLOWED_PROJECTION_TABLES``.  Any expansion of
        # that allowlist requires coordinated security review.
        if table not in _ALLOWED_PROJECTION_TABLES:
            raise ValueError(
                f"Table {table!r} passed validation but is not in the allowlist "
                f"(correlation_id={correlation_id})"
            )

        async with self._postgres_connection() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) as cnt FROM {table}"  # noqa: S608 -- table name validated against allowlist
            )
            return int(row["cnt"]) if row else 0

    async def _delete_projection_rows(
        self,
        *,
        correlation_id: UUID,
    ) -> int:
        """Delete all rows from the projection table. Returns count deleted.

        Args:
            correlation_id: Trace identifier for error diagnostics.
        """
        table = self._config.projection_table
        self._validate_table_name(table, correlation_id=correlation_id)

        # SAFETY: The f-string SQL interpolation below is safe because
        # ``_validate_table_name`` restricts ``table`` to the frozen
        # allowlist ``_ALLOWED_PROJECTION_TABLES``.  Any expansion of
        # that allowlist requires coordinated security review.
        if table not in _ALLOWED_PROJECTION_TABLES:
            raise ValueError(
                f"Table {table!r} passed validation but is not in the allowlist "
                f"(correlation_id={correlation_id})"
            )

        async with self._postgres_connection() as conn:
            result = await conn.execute(
                f"DELETE FROM {table}"  # noqa: S608 -- table name validated against allowlist
            )
            # asyncpg returns "DELETE N" where N is the row count
            match = re.search(r"\d+", result)
            return int(match.group()) if match else 0

    # -------------------------------------------------------------------------
    # Step 2: Consumer Groups
    # -------------------------------------------------------------------------

    async def _reset_consumer_groups(
        self,
        report: DemoResetReport,
        *,
        dry_run: bool,
        correlation_id: UUID,
    ) -> None:
        """Delete demo-scoped consumer groups so projectors start fresh.

        Consumer groups matching the demo pattern are deleted entirely.
        Kafka recreates them automatically when consumers reconnect.

        Args:
            report: Report to append results to.
            dry_run: If True, only report what would happen.
            correlation_id: Trace identifier for error diagnostics.
        """
        if not self._config.kafka_bootstrap_servers:
            report.actions.append(
                ResetActionResult(
                    resource="Consumer group offsets",
                    action=EnumResetAction.SKIPPED,
                    detail="KAFKA_BOOTSTRAP_SERVERS not configured",
                )
            )
            return

        try:
            from confluent_kafka.admin import AdminClient

            # NOTE: confluent-kafka's AdminClient uses blocking I/O internally.
            # This is acceptable for CLI use where the event loop is otherwise
            # idle, but these calls should not be used in a concurrent server.
            admin = AdminClient(
                {"bootstrap.servers": self._config.kafka_bootstrap_servers}
            )

            # List all consumer groups (modern API, confluent-kafka >= 2.x)
            list_result = admin.list_consumer_groups(
                request_timeout=10,
            ).result()
            all_groups: list[str] = [
                g.group_id for g in list_result.valid if g.group_id is not None
            ]

            # Filter to demo-scoped groups
            demo_groups: list[str] = [
                g for g in all_groups if self._config.consumer_group_pattern.search(g)
            ]

            if not demo_groups:
                report.actions.append(
                    ResetActionResult(
                        resource="Consumer group offsets",
                        action=EnumResetAction.SKIPPED,
                        detail="No demo consumer groups found",
                    )
                )
                # Record non-demo groups as preserved (demo_groups is empty,
                # so all_groups are non-demo)
                if all_groups:
                    report.actions.append(
                        ResetActionResult(
                            resource=f"Non-demo consumer groups ({len(all_groups)})",
                            action=EnumResetAction.PRESERVED,
                            detail=f"Groups preserved: {', '.join(all_groups[:5])}"
                            + (
                                f" (+{len(all_groups) - 5} more)"
                                if len(all_groups) > 5
                                else ""
                            ),
                        )
                    )
                return

            if dry_run:
                report.actions.append(
                    ResetActionResult(
                        resource="Consumer group offsets",
                        action=EnumResetAction.RESET,
                        detail=(
                            f"Would delete {len(demo_groups)} consumer group(s): "
                            + ", ".join(demo_groups)
                        ),
                    )
                )
            else:
                # Delete demo consumer groups
                futures = admin.delete_consumer_groups(demo_groups)
                deleted: list[str] = []
                errors: list[str] = []

                for group_id, future in futures.items():
                    try:
                        future.result(timeout=10)
                        deleted.append(str(group_id))
                    except Exception as exc:
                        errors.append(f"{group_id}: {sanitize_error_message(exc)}")

                if deleted:
                    report.actions.append(
                        ResetActionResult(
                            resource="Consumer group offsets",
                            action=EnumResetAction.RESET,
                            detail=(
                                f"Deleted {len(deleted)} consumer group(s): "
                                + ", ".join(deleted)
                            ),
                        )
                    )
                if errors:
                    report.actions.append(
                        ResetActionResult(
                            resource="Consumer group offsets (partial failure)",
                            action=EnumResetAction.ERROR,
                            detail="; ".join(errors),
                        )
                    )

            # Record non-demo groups as preserved
            demo_groups_set = set(demo_groups)
            non_demo = [g for g in all_groups if g not in demo_groups_set]
            if non_demo:
                report.actions.append(
                    ResetActionResult(
                        resource=f"Non-demo consumer groups ({len(non_demo)})",
                        action=EnumResetAction.PRESERVED,
                        detail=f"Groups preserved: {', '.join(non_demo[:5])}"
                        + (
                            f" (+{len(non_demo) - 5} more)" if len(non_demo) > 5 else ""
                        ),
                    )
                )

        except ImportError:
            report.actions.append(
                ResetActionResult(
                    resource="Consumer group offsets",
                    action=EnumResetAction.ERROR,
                    detail="confluent-kafka not installed",
                )
            )
        except Exception as exc:
            logger.exception(
                "Failed to reset consumer groups, correlation_id=%s",
                correlation_id,
            )
            report.actions.append(
                ResetActionResult(
                    resource="Consumer group offsets",
                    action=EnumResetAction.ERROR,
                    detail=f"Failed: {sanitize_error_message(exc)}",
                )
            )

    # -------------------------------------------------------------------------
    # Step 3: Topic Purge (Optional)
    # -------------------------------------------------------------------------

    async def _purge_demo_topics(
        self,
        report: DemoResetReport,
        *,
        dry_run: bool,
        correlation_id: UUID,
    ) -> None:
        """Purge messages from demo-scoped Kafka topics.

        Uses Kafka's delete-records API to remove all messages from
        demo topics. The topics themselves are preserved; only messages
        are deleted.

        Args:
            report: Report to append results to.
            dry_run: If True, only report what would happen.
            correlation_id: Trace identifier for error diagnostics.
        """
        if not self._config.kafka_bootstrap_servers:
            report.actions.append(
                ResetActionResult(
                    resource="Demo topic data",
                    action=EnumResetAction.SKIPPED,
                    detail="KAFKA_BOOTSTRAP_SERVERS not configured",
                )
            )
            return

        try:
            from confluent_kafka import TopicPartition
            from confluent_kafka.admin import AdminClient

            admin = AdminClient(
                {"bootstrap.servers": self._config.kafka_bootstrap_servers}
            )

            # Get cluster metadata to find topics
            metadata = admin.list_topics(timeout=10)

            # Filter to demo-scoped topics
            demo_topics: list[str] = []
            non_demo_topics: list[str] = []

            for topic_name in metadata.topics:
                if topic_name.startswith("_"):
                    # Skip internal Kafka topics (__consumer_offsets, etc.)
                    continue

                is_demo = any(
                    topic_name.startswith(prefix)
                    for prefix in self._config.demo_topic_prefixes
                )
                if is_demo:
                    demo_topics.append(topic_name)
                else:
                    non_demo_topics.append(topic_name)

            if not demo_topics:
                report.actions.append(
                    ResetActionResult(
                        resource="Demo topic data",
                        action=EnumResetAction.SKIPPED,
                        detail="No demo topics found",
                    )
                )
                return

            if dry_run:
                report.actions.append(
                    ResetActionResult(
                        resource="Demo topic data",
                        action=EnumResetAction.RESET,
                        detail=(
                            f"Would purge messages from {len(demo_topics)} topic(s): "
                            + ", ".join(sorted(demo_topics))
                        ),
                    )
                )
            else:
                # Query the actual high-watermark offset for each partition
                # using a Consumer, then pass those explicit offsets to
                # delete_records.  The previous approach used offset -1
                # (OFFSET_END) which is not reliably interpreted as "delete
                # all" across confluent-kafka versions.
                from confluent_kafka import Consumer as _KafkaConsumer

                # 16 hex chars from uuid4 = 64 bits of entropy, sufficient for
                # single-CLI-invocation uniqueness but not globally collision-proof
                # under high concurrency (birthday bound ~2^32 simultaneous resets).
                ephemeral_group_id = f"_demo-reset-watermark-query-{uuid4().hex[:16]}"
                consumer = _KafkaConsumer(
                    {
                        "bootstrap.servers": self._config.kafka_bootstrap_servers,
                        "group.id": ephemeral_group_id,
                        "enable.auto.commit": False,
                    }
                )
                try:
                    partitions_to_delete: list[TopicPartition] = []
                    watermark_errors: int = 0
                    for topic_name in demo_topics:
                        topic_metadata = metadata.topics[topic_name]
                        for partition_id in topic_metadata.partitions:
                            try:
                                _low, high = consumer.get_watermark_offsets(
                                    TopicPartition(topic_name, partition_id),
                                    timeout=5,
                                )
                            except Exception as exc:
                                watermark_errors += 1
                                logger.debug(
                                    "Failed to get watermarks for %s[%d]: %s",
                                    topic_name,
                                    partition_id,
                                    sanitize_error_message(exc),
                                )
                                continue
                            if high > 0:
                                partitions_to_delete.append(
                                    TopicPartition(topic_name, partition_id, high)
                                )
                finally:
                    consumer.close()

                # Clean up the ephemeral consumer group so it does not
                # remain as an orphan in Kafka after watermark queries.
                try:
                    delete_futures = admin.delete_consumer_groups([ephemeral_group_id])
                    delete_futures[ephemeral_group_id].result(timeout=10)
                    logger.debug(
                        "Deleted ephemeral consumer group %s",
                        ephemeral_group_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to delete ephemeral consumer group %s: %s",
                        ephemeral_group_id,
                        sanitize_error_message(exc),
                    )
                    report.actions.append(
                        ResetActionResult(
                            resource="Ephemeral consumer group cleanup",
                            action=EnumResetAction.ERROR,
                            detail=(
                                f"Failed to delete ephemeral group "
                                f"{ephemeral_group_id}: "
                                f"{sanitize_error_message(exc)} "
                                f"(orphan may remain in Kafka)"
                            ),
                        )
                    )

                if partitions_to_delete:
                    futures = admin.delete_records(partitions_to_delete)
                    purged: list[str] = []
                    errors: list[str] = []

                    for tp, future in futures.items():
                        try:
                            future.result(timeout=10)
                            if tp.topic not in purged:
                                purged.append(tp.topic)
                        except Exception as exc:
                            errors.append(
                                f"{tp.topic}[{tp.partition}]: "
                                f"{sanitize_error_message(exc)}"
                            )

                    if purged:
                        report.actions.append(
                            ResetActionResult(
                                resource="Demo topic data",
                                action=EnumResetAction.RESET,
                                detail=(
                                    f"Purged messages from {len(purged)} topic(s): "
                                    + ", ".join(sorted(purged))
                                ),
                            )
                        )
                    if errors:
                        report.actions.append(
                            ResetActionResult(
                                resource="Demo topic data (partial failure)",
                                action=EnumResetAction.ERROR,
                                detail="; ".join(errors[:5]),
                            )
                        )
                elif watermark_errors > 0:
                    # All watermark lookups failed -- we cannot
                    # determine whether the topics are truly empty.
                    report.actions.append(
                        ResetActionResult(
                            resource="Demo topic data",
                            action=EnumResetAction.ERROR,
                            detail=(
                                f"Failed to read watermark offsets for "
                                f"{watermark_errors} partition(s) — cannot "
                                f"determine if topics are empty"
                            ),
                        )
                    )
                else:
                    report.actions.append(
                        ResetActionResult(
                            resource="Demo topic data",
                            action=EnumResetAction.SKIPPED,
                            detail=(
                                "Demo topics exist but all partitions are "
                                "already empty — nothing to purge"
                            ),
                        )
                    )

            # Record non-demo topics as preserved
            if non_demo_topics:
                report.actions.append(
                    ResetActionResult(
                        resource=f"Non-demo topics ({len(non_demo_topics)})",
                        action=EnumResetAction.PRESERVED,
                        detail=f"Topics preserved: {', '.join(sorted(non_demo_topics)[:5])}"
                        + (
                            f" (+{len(non_demo_topics) - 5} more)"
                            if len(non_demo_topics) > 5
                            else ""
                        ),
                    )
                )

        except ImportError:
            report.actions.append(
                ResetActionResult(
                    resource="Demo topic data",
                    action=EnumResetAction.ERROR,
                    detail="confluent-kafka not installed",
                )
            )
        except Exception as exc:
            logger.exception(
                "Failed to purge demo topics, correlation_id=%s",
                correlation_id,
            )
            report.actions.append(
                ResetActionResult(
                    resource="Demo topic data",
                    action=EnumResetAction.ERROR,
                    detail=f"Failed: {sanitize_error_message(exc)}",
                )
            )


# =============================================================================
# Module Exports
# =============================================================================

__all__: list[str] = [
    "DEMO_CONSUMER_GROUP_PATTERN",
    "DEMO_PROJECTION_TABLE",
    "DEMO_TOPIC_PREFIXES",
    "DemoResetConfig",
    "DemoResetEngine",
    "DemoResetReport",
    "EnumResetAction",
    "PRESERVED_RESOURCES",
    "ResetActionResult",
]
